from __future__ import annotations

import copy
import logging

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from pytorch_metric_learning.losses import ArcFaceLoss
from pytorch_metric_learning.samplers import MPerClassSampler

from src.core.evaluation.metrics import (
    compute_per_row_metrics,
    summarize_metrics,
)
from src.core.embeddings.faiss_utils import build_gpu_index, search_and_format
from src.core.embeddings.loader import filter_embeddings_by_split, load_embeddings
from src.services.metric_learning.dataset import EmbeddingDataset
from src.services.metric_learning.projection_head import ProjectionHead
from src.schemas.metric_learning.metric_learning import MetricLearningConfig

logger = logging.getLogger(__name__)

RETRIEVAL_K = 51
RETRIEVAL_THRESHOLD = 0.5
RETRIEVAL_RECALL_KS = [5, 10, 50]


class MetricLearningService:
    """Train a projection head with ArcFace loss for metric learning.

    Loads pre-computed frozen embeddings, applies optional fusion, trains
    a projection head (dense 512) supervised by ArcFace loss, and evaluates
    retrieval metrics on the validation set after each epoch. Logs results
    to MLflow and saves the best projection head weights.
    """

    def __init__(self, config_path: str) -> None:
        """Initialize the service and validate the config file.

        Args:
            config_path: Path to the JSON config file.
        """
        with open(config_path) as f:
            self.config = MetricLearningConfig.model_validate_json(f.read())

        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = (
            torch.device(self.config.device)
            if self.config.device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info("Using device: %s", self.device)

    def run(self) -> None:
        """Run the full training pipeline."""
        embeddings, index = load_embeddings(
            self.config.embeddings_dir, self.config.fusion_type
        )

        assignments = pd.read_csv(filepath_or_buffer=self.config.assignments_path)
        train_df = assignments[assignments["split"] == "train"].copy()
        val_df = assignments[assignments["split"] == "val"].copy()
        logger.info("Train rows: %d, Val rows: %d", len(train_df), len(val_df))

        unique_labels = sorted(
            pd.concat([train_df["label_group"], val_df["label_group"]]).unique()
        )
        label_to_class = {label: i for i, label in enumerate(unique_labels)}
        num_classes = len(unique_labels)
        logger.info("Number of classes: %d", num_classes)

        train_emb, train_labels, _ = self._prepare_split(
            embeddings=embeddings,
            index=index,
            split_df=train_df,
            label_to_class=label_to_class,
        )
        val_emb, val_labels, val_pids = self._prepare_split(
            embeddings=embeddings,
            index=index,
            split_df=val_df,
            label_to_class=label_to_class,
        )
        input_dim = embeddings.shape[1]

        train_dataset = EmbeddingDataset(train_emb, train_labels)
        val_dataset = EmbeddingDataset(val_emb, val_labels)

        train_sampler = MPerClassSampler(
            labels=train_labels.tolist(),
            m=self.config.samples_per_class,
            batch_size=self.config.batch_size,
        )
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.config.batch_size,
            sampler=train_sampler,
            drop_last=False,
        )
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        model = ProjectionHead(
            input_dim=input_dim,
            projection_dim=self.config.projection_dim,
        ).to(self.device)

        loss_fn = ArcFaceLoss(
            num_classes=num_classes,
            embedding_size=self.config.projection_dim,
            margin=self.config.margin,
            scale=self.config.scale,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.config.num_epochs,
        )

        mlflow.set_tracking_uri(uri=self.config.mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name=self.config.mlflow_experiment_name)

        best_val_f1 = 0.0
        patience_counter = 0
        best_model_state: dict[str, torch.Tensor] | None = None

        with mlflow.start_run(run_name="metric_learning"):
            mlflow.log_params(
                {
                    "input_dim": input_dim,
                    "projection_dim": self.config.projection_dim,
                    "samples_per_class": self.config.samples_per_class,
                    "num_classes": num_classes,
                    "learning_rate": self.config.learning_rate,
                    "weight_decay": self.config.weight_decay,
                    "batch_size": self.config.batch_size,
                    "num_epochs": self.config.num_epochs,
                    "margin": self.config.margin,
                    "scale": self.config.scale,
                    "early_stopping_patience": self.config.early_stopping_patience,
                    "fusion_type": self.config.fusion_type or "none",
                }
            )

            for epoch in range(1, self.config.num_epochs + 1):
                train_loss = self._run_epoch(
                    model=model,
                    loss_fn=loss_fn,
                    loader=train_loader,
                    optimizer=optimizer,
                    desc="  train",
                )
                val_loss = self._run_epoch(
                    model=model,
                    loss_fn=loss_fn,
                    loader=val_loader,
                    desc="  val",
                )
                retrieval_metrics = self._evaluate_retrieval(
                    model=model,
                    val_embeddings=val_emb,
                    val_df=val_df,
                    val_index=val_pids,
                )

                current_lr = scheduler.get_last_lr()[0]
                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "avg_f1": retrieval_metrics["avg_f1"],
                        "avg_precision": retrieval_metrics["avg_precision"],
                        "avg_recall": retrieval_metrics["avg_recall"],
                        "learning_rate": current_lr,
                    },
                    step=epoch,
                )
                for k, v in retrieval_metrics["recall-k"].items():
                    mlflow.log_metric(key=f"recall-{k}", value=v, step=epoch)

                logger.info(
                    "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | "
                    "val_f1=%.4f | val_prec=%.4f | val_rec=%.4f",
                    epoch,
                    self.config.num_epochs,
                    train_loss,
                    val_loss,
                    retrieval_metrics["avg_f1"],
                    retrieval_metrics["avg_precision"],
                    retrieval_metrics["avg_recall"],
                )

                if retrieval_metrics["avg_f1"] > best_val_f1:
                    best_val_f1 = retrieval_metrics["avg_f1"]
                    patience_counter = 0
                    best_model_state = copy.deepcopy(model.state_dict())
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        logger.info(
                            "Early stopping at epoch %d "
                            "(no improvement for %d epochs)",
                            epoch,
                            self.config.early_stopping_patience,
                        )
                        break

                scheduler.step()

            best_path = self.config.output_dir / "projection_head.pt"
            torch.save(best_model_state, best_path)
            mlflow.log_artifact(str(best_path))

            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            final_metrics = self._evaluate_retrieval(
                model=model,
                val_embeddings=val_emb,
                val_df=val_df,
                val_index=val_pids,
            )

            metrics_path = self.config.output_dir / "metrics_summary.csv"
            metrics_row = {
                "avg_precision": final_metrics["avg_precision"],
                "avg_recall": final_metrics["avg_recall"],
                "avg_f1": final_metrics["avg_f1"],
            }
            for k, v in final_metrics["recall-k"].items():
                metrics_row[f"recall-{k}"] = v
            pd.DataFrame([metrics_row]).to_csv(
                path_or_buf=metrics_path, index=False
            )
            mlflow.log_artifact(str(metrics_path))

            logger.info(
                "Training complete. Best val_f1=%.4f. "
                "Artifacts saved to %s",
                best_val_f1,
                self.config.output_dir,
            )

    def _run_epoch(
        self,
        model: nn.Module,
        loss_fn: ArcFaceLoss,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer | None = None,
        desc: str = "",
    ) -> float:
        """Run one training or validation epoch.

        Args:
            model: ProjectionHead module.
            loss_fn: ArcFace loss function.
            loader: DataLoader for the split.
            optimizer: AdamW optimizer (None for eval-only).
            desc: tqdm description prefix.

        Returns:
            Mean loss for the epoch.
        """
        is_train = optimizer is not None
        model.train() if is_train else model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.set_grad_enabled(is_train):
            for emb_batch, label_batch in tqdm(
                loader, desc=desc, leave=False
            ):
                emb_batch = emb_batch.to(self.device)
                label_batch = label_batch.to(self.device)

                projected = model(emb_batch)
                loss = loss_fn(embeddings=projected, labels=label_batch)

                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def _prepare_split(
        self,
        embeddings: np.ndarray,
        index: dict[str, str],
        split_df: pd.DataFrame,
        label_to_class: dict,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
        """Filter embeddings and map labels for a given split.

        Args:
            embeddings: Full embedding array of shape (N, D).
            index: Dict mapping row-key to posting_id.
            split_df: DataFrame for the target split.
            label_to_class: Mapping from label_group to contiguous class ID.

        Returns:
            Tuple of (filtered_embeddings, class_id_labels, local_index).
        """
        split_emb, split_index = filter_embeddings_by_split(
            embeddings=embeddings, index=index, split_df=split_df,
        )
        labels = np.array(
            [label_to_class[row["label_group"]] for _, row in split_df.iterrows()]
        )
        return split_emb, labels, split_index

    def _evaluate_retrieval(
        self,
        model: nn.Module,
        val_embeddings: np.ndarray,
        val_df: pd.DataFrame,
        val_index: dict[str, str],
    ) -> dict:
        """Run retrieval evaluation on the validation set.

        Projects validation embeddings through the model, builds a FAISS
        GPU index, performs self k-NN search, and computes retrieval metrics.

        Args:
            model: ProjectionHead module (in eval mode).
            val_embeddings: Validation embeddings of shape (N, D).
            val_df: Validation DataFrame with label_group and stratum.
            val_index: Dict mapping local row-key to posting_id.

        Returns:
            Dict with avg_f1, avg_precision, avg_recall, and
            recall_at_k mapping.
        """
        model.eval()
        val_tensor = torch.from_numpy(val_embeddings).float().to(self.device)

        with torch.no_grad():
            projected = model(val_tensor).cpu().numpy()

        index = build_gpu_index(projected)
        results = search_and_format(
            index=index,
            query_embeddings=projected,
            index_map=val_index,
            k=RETRIEVAL_K,
            threshold=RETRIEVAL_THRESHOLD,
        )

        val_df = val_df.copy()
        val_df["retrieved_products"] = results

        df_metrics = compute_per_row_metrics(
            df=val_df, recall_ks=RETRIEVAL_RECALL_KS
        )
        summary = summarize_metrics(df_metrics=df_metrics)

        return {
            "avg_f1": float(summary["f1"]),
            "avg_precision": float(summary["precision"]),
            "avg_recall": float(summary["recall"]),
            "recall-k": {k: float(summary[f"recall-{k}"]) for k in RETRIEVAL_RECALL_KS},
        }


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    MetricLearningService(config_path=args.config).run()

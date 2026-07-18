from __future__ import annotations

import argparse
import logging

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from src.core.evaluation.metrics import (
    compute_per_row_metrics,
    summarize_metrics,
)
from src.schemas.grid_search.grid_search import GridSearchConfig
from src.services.retrieval.retrieval import RetrievalService

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]

METRIC_COLUMNS = ["avg_f1", "avg_precision", "avg_recall"]


class GridSearchService:
    """Grid-search over the retrieval threshold hyperparameter.

    Builds the FAISS index once and reuses it across all thresholds.
    Logs per-threshold metrics to MLFlow and saves a summary plot as JPEG.
    """

    def __init__(self, config_path: str) -> None:
        """Initialize the service and validate the config file.

        Args:
            config_path: Path to the JSON config file.
        """
        with open(config_path) as f:
            self.config = GridSearchConfig.model_validate_json(f.read())

        self.config.grid_output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        """Run grid search over thresholds and log results to MLFlow."""
        retrieval_service = RetrievalService(
            config_path=str(self.config.base_retrieval_config)
        )

        logger.info("Building FAISS index once for all thresholds")
        split_df, split_embeddings, split_index, faiss_index = (
            retrieval_service.prepare()
        )

        mlflow.set_tracking_uri(uri=self.config.mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name=self.config.mlflow_experiment_name)

        rows = []

        with mlflow.start_run(run_name="grid_search"):
            for threshold in THRESHOLDS:
                logger.info("Running threshold = %.1f", threshold)
                grid_dir = self.config.grid_output_dir / f"threshold_{threshold}"
                grid_dir.mkdir(parents=True, exist_ok=True)

                retrieval_service.search_and_save(
                    faiss_index=faiss_index,
                    split_df=split_df,
                    split_embeddings=split_embeddings,
                    split_index=split_index,
                    threshold=threshold,
                    output_dir=grid_dir,
                )

                df_retrieval = pd.read_csv(
                    filepath_or_buffer=grid_dir / "retrieval_results.csv"
                )
                df_metrics = compute_per_row_metrics(
                    df=df_retrieval, recall_ks=self.config.recall_ks
                )
                summary = summarize_metrics(df_metrics=df_metrics)

                df_metrics.to_csv(
                    path_or_buf=grid_dir / "evaluation_results.csv", index=False
                )
                summary.to_csv(
                    path_or_buf=grid_dir / "evaluation_summary.csv", header=False
                )

                metrics_dict = {
                    "avg_precision": float(summary["precision"]),
                    "avg_recall": float(summary["recall"]),
                    "avg_f1": float(summary["f1"]),
                }
                for k in self.config.recall_ks:
                    metrics_dict[f"avg_recall-{k}"] = float(summary[f"recall-{k}"])

                with mlflow.start_run(
                    run_name=f"threshold_{threshold}", nested=True
                ):
                    mlflow.log_param("threshold", threshold)
                    mlflow.log_metrics(metrics_dict)

                row = {"threshold": threshold}
                row.update(metrics_dict)
                rows.append(row)

                logger.info(
                    "Threshold=%.1f  F1=%.4f  Precision=%.4f  Recall=%.4f",
                    threshold,
                    metrics_dict["avg_f1"],
                    metrics_dict["avg_precision"],
                    metrics_dict["avg_recall"],
                )

            results_df = pd.DataFrame(rows)
            results_df.to_csv(
                path_or_buf=self.config.grid_output_dir / "results_summary.csv",
                index=False,
            )

            fig, ax = plt.subplots(figsize=(8, 5))
            for metric in METRIC_COLUMNS:
                ax.plot(
                    results_df["threshold"],
                    results_df[metric],
                    marker="o",
                    label=metric.replace("avg_", ""),
                )
            ax.set_xlabel("Threshold")
            ax.set_ylabel("Score")
            ax.set_title("Grid Search: Threshold vs Metrics")
            ax.legend()
            ax.grid(True)
            fig.tight_layout()

            plot_path = self.config.grid_output_dir / "threshold_metrics.jpg"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)

            mlflow.log_artifact(str(plot_path))
            mlflow.log_artifact(
                str(self.config.grid_output_dir / "results_summary.csv")
            )

            logger.info("Grid search complete. Results saved to %s", self.config.grid_output_dir)
            logger.info(
                "MLFlow UI: mlflow ui --backend-store-uri %s",
                self.config.mlflow_tracking_uri,
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    GridSearchService(config_path=args.config).run()

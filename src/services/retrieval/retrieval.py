from __future__ import annotations

import logging
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch

from src.core.embeddings.faiss_utils import build_gpu_index, search_and_format
from src.core.embeddings.loader import filter_embeddings_by_split, load_embeddings
from src.core.models.projection_head import ProjectionHead
from src.schemas.retrieval.retrieval import RetrievalConfig

logger = logging.getLogger(__name__)


class RetrievalService:
    """Retrieve nearest neighbors for a dataset split using FAISS.

    Loads pre-computed embeddings (safetensors) and an index mapping,
    filters the assignments CSV to the requested split, then runs
    k-nearest-neighbor search with a cosine-similarity threshold.
    Results are saved as a CSV with one row per query product.
    """

    def __init__(self, config_path: str) -> None:
        """Initialize the service and validate the config file.

        Args:
            config_path: Path to the JSON config file.
        """
        with open(config_path) as f:
            self.config = RetrievalConfig.model_validate_json(f.read())

        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        """Load embeddings, filter assignments, run FAISS k-NN, and save results."""
        split_df, split_embeddings, split_index, faiss_index = self.prepare()
        self.search_and_save(
            faiss_index=faiss_index,
            split_df=split_df,
            split_embeddings=split_embeddings,
            split_index=split_index,
            threshold=self.config.threshold,
            output_dir=self.config.output_dir,
        )

    def _load_projection_head(
        self, input_dim: int, projection_dim: int
    ) -> ProjectionHead:
        """Load a trained projection head from a checkpoint.

        Args:
            input_dim: Dimensionality of the input embeddings.
            projection_dim: Dimensionality of the projected embeddings.

        Returns:
            Loaded ProjectionHead in eval mode.
        """
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ProjectionHead(
            input_dim=input_dim, projection_dim=projection_dim
        ).to(self._device)
        state_dict = torch.load(
            self.config.projection_head_path,
            map_location=self._device,
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        model.eval()
        logger.info(
            "Loaded projection head on %s from %s",
            self._device,
            self.config.projection_head_path,
        )
        return model

    def prepare(self) -> tuple[pd.DataFrame, np.ndarray, dict[str, str], faiss.Index]:
        """Load data, filter by split, optionally apply projection head, and build FAISS index.

        Returns:
            Tuple of (split_df, split_embeddings, split_index, faiss_index).
        """
        assignments = pd.read_csv(filepath_or_buffer=self.config.assignments_path)
        split_df = assignments[assignments["split"] == self.config.split].copy()
        logger.info(
            "Loaded %d rows for split '%s'",
            len(split_df),
            self.config.split,
        )

        embeddings, index = load_embeddings(
            self.config.embeddings_dir, self.config.fusion_type
        )
        split_embeddings, split_index = filter_embeddings_by_split(
            embeddings=embeddings, index=index, split_df=split_df,
        )

        if self.config.projection_head_path is not None:
            model = self._load_projection_head(
                input_dim=embeddings.shape[1],
                projection_dim=self.config.projection_dim,
            )
            emb_tensor = (
                torch.from_numpy(split_embeddings).float().to(self._device)
            )
            with torch.no_grad():
                split_embeddings = model(emb_tensor).cpu().numpy()

        faiss_index = build_gpu_index(split_embeddings)
        return split_df, split_embeddings, split_index, faiss_index

    def search_and_save(
        self,
        faiss_index: faiss.Index,
        split_df: pd.DataFrame,
        split_embeddings: np.ndarray,
        split_index: dict[str, str],
        threshold: float,
        output_dir: Path,
    ) -> None:
        """Run k-NN search with a given threshold and save results.

        Args:
            faiss_index: Pre-built FAISS index.
            split_df: Filtered assignment rows (from prepare()).
            split_embeddings: Filtered embeddings for the split.
            split_index: Map from local row-key to posting_id.
            threshold: Cosine-similarity threshold.
            output_dir: Directory to save retrieval_results.csv.
        """
        results = search_and_format(
            index=faiss_index,
            query_embeddings=split_embeddings,
            index_map=split_index,
            k=self.config.k,
            threshold=threshold,
        )

        split_df["retrieved_products"] = results
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "retrieval_results.csv"
        split_df[
            [
                "posting_id",
                "image",
                "title",
                "label_group",
                "stratum",
                "retrieved_products"
            ]
        ].to_csv(
            path_or_buf=out_path, index=False
        )
        logger.info(
            "Saved retrieval results for %d queries to %s",
            len(split_df),
            out_path,
        )

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    RetrievalService(config_path=args.config).run()

from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from safetensors.numpy import load_file

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

    def prepare(self) -> tuple[pd.DataFrame, np.ndarray, dict[str, str], faiss.Index]:
        """Load data, filter by split, and build FAISS index.

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

        embeddings, index = self._load_embeddings()
        split_posting_ids = set(split_df["posting_id"].tolist())
        split_indices = [
            int(k) for k, v in index.items() if v in split_posting_ids
        ]
        split_indices.sort()
        split_embeddings = embeddings[split_indices]
        split_index = {
            str(i): index[str(orig_idx)]
            for i, orig_idx in enumerate(split_indices)
        }

        faiss_index = self._build_faiss_index(split_embeddings)
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
        results = self._search(
            index=faiss_index,
            query_embeddings=split_embeddings,
            index_map=split_index,
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

    def _load_embeddings(self) -> tuple[np.ndarray, dict[str, str]]:
        """Load embeddings from all configured directories and apply fusion.

        Returns:
            Tuple of (embeddings array, index dict mapping row-key to posting_id).
        """
        embeddings_list = []
        index: dict[str, str] | None = None

        for dir_path in self.config.embeddings_dir:
            embeddings_path = dir_path / "embedding.safetensors"
            index_path = dir_path / "index.json"

            data = load_file(filename=str(embeddings_path))
            embeddings = data["embeddings"]

            with open(index_path) as f:
                current_index = json.load(f)

            if index is None:
                index = current_index
            elif index != current_index:
                raise ValueError(
                    f"Index mismatch: {index_path} differs from previously loaded index."
                )

            embeddings_list.append(embeddings)
            logger.info("Loaded embeddings from %s shape %s", dir_path, embeddings.shape)

        if len(embeddings_list) == 1:
            fused = embeddings_list[0]
        else:
            fusion_type = self.config.fusion_type
            if fusion_type == "concat":
                fused = np.concatenate(embeddings_list, axis=1)
            elif fusion_type == "sum":
                fused = sum(embeddings_list)
            else:
                raise ValueError(
                    f"fusion_type must be 'concat' or 'sum' when multiple "
                    f"embedding dirs are provided, got: {fusion_type}"
                )

            norms = np.linalg.norm(fused, axis=1, keepdims=True)
            fused = fused / np.maximum(norms, 1e-12)

        logger.info("Fused embeddings shape %s", fused.shape)
        return fused, index

    def _build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build a FAISS index on GPU using inner product (cosine similarity).

        Args:
            embeddings: L2-normalized array of shape (N, D).

        Returns:
            FAISS Index with vectors added.
        """
        dim = embeddings.shape[1]
        cpu_index = faiss.IndexFlatIP(dim)
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        gpu_index.add(embeddings)
        logger.info("Built FAISS GPU index with %d vectors", gpu_index.ntotal)
        return gpu_index

    def _search(
        self,
        index: faiss.Index,
        query_embeddings: np.ndarray,
        index_map: dict[str, str],
        threshold: float | None = None,
    ) -> list[str]:
        """Run k-NN search for each query and format results.

        Args:
            index: FAISS index containing all embeddings.
            query_embeddings: Embedding array for queries (N, D).
            index_map: Dict mapping row-key to posting_id.
            threshold: Cosine-similarity threshold. Defaults to config.threshold.

        Returns:
            List of JSON string lists of retrieved posting_ids.
        """
        if threshold is None:
            threshold = self.config.threshold
        distances, indices = index.search(query_embeddings, self.config.k)
        mask = distances >= threshold

        results = []
        for row_indices, row_mask in zip(indices, mask):
            retrieved = [index_map[str(i)] for i in row_indices[row_mask]]
            results.append(json.dumps(retrieved))

        return results


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    RetrievalService(config_path=args.config).run()

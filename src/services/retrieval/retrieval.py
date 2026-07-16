from __future__ import annotations

import json
import logging

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

        index_ip = self._build_faiss_index(split_embeddings)
        results = self._search(
            index=index_ip,
            query_embeddings=split_embeddings,
            index_map=split_index,
        )

        split_df["retrieved_products"] = results
        out_path = self.config.output_dir / "retrieval_results.csv"
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
        """Load embeddings and index from the configured directory.

        Returns:
            Tuple of (embeddings array, index dict mapping row-key to posting_id).
        """
        embeddings_path = self.config.embeddings_dir / "embedding.safetensors"
        index_path = self.config.embeddings_dir / "index.json"

        data = load_file(filename=str(embeddings_path))
        embeddings = data["embeddings"]
        logger.info("Loaded embeddings with shape %s", embeddings.shape)

        with open(index_path) as f:
            index = json.load(f)

        return embeddings, index

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
    ) -> list[str]:
        """Run k-NN search for each query and format results.

        Args:
            index: FAISS index containing all embeddings.
            query_embeddings: Embedding array for queries (N, D).
            index_map: Dict mapping row-key to posting_id.

        Returns:
            List of JSON string lists of retrieved posting_ids.
        """
        distances, indices = index.search(query_embeddings, self.config.k)
        mask = distances >= self.config.threshold

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

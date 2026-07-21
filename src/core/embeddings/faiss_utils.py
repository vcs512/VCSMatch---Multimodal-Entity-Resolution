from __future__ import annotations

import json
import logging

import faiss
import numpy as np

logger = logging.getLogger(__name__)


def build_gpu_index(embeddings: np.ndarray) -> faiss.Index:
    """Build a FAISS GPU index using inner product (cosine similarity).

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


def search_and_format(
    index: faiss.Index,
    query_embeddings: np.ndarray,
    index_map: dict[str, str],
    k: int,
    threshold: float,
) -> list[str]:
    """Run k-NN search for each query and format results as JSON strings.

    Args:
        index: FAISS index containing all embeddings.
        query_embeddings: Embedding array for queries (N, D).
        index_map: Dict mapping local row-key to posting_id.
        k: Number of nearest neighbors to retrieve.
        threshold: Similarity threshold (cosine).

    Returns:
        List of JSON string lists of retrieved posting_ids.
    """
    distances, indices = index.search(query_embeddings, k)
    mask = distances >= threshold

    results = []
    for row_indices, row_mask in zip(indices, mask):
        retrieved = [index_map[str(i)] for i in row_indices[row_mask]]
        results.append(json.dumps(retrieved))

    return results

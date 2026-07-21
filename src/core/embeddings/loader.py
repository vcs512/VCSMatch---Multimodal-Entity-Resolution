from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors.numpy import load_file

logger = logging.getLogger(__name__)


def load_embeddings(
    embeddings_dir: list[Path],
    fusion_type: str | None = None,
) -> tuple[np.ndarray, dict[str, str]]:
    """Load embeddings from safetensors files and apply fusion.

    Args:
        embeddings_dir: Ordered list of directories containing
            embedding.safetensors and index.json (one per modality).
        fusion_type: Fusion strategy when multiple dirs:
            "concat" concatenates along axis 1, "sum" averages.
            Required if >1 dir.

    Returns:
        Tuple of (fused embeddings array, index dict mapping row-key
        to posting_id).

    Raises:
        ValueError: If index files differ between directories or
            fusion_type is invalid.
    """
    embeddings_list = []
    index: dict[str, str] | None = None

    for dir_path in embeddings_dir:
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
                f"Index mismatch: {index_path} differs from "
                f"previously loaded index."
            )

        embeddings_list.append(embeddings)
        logger.info("Loaded embeddings from %s shape %s", dir_path, embeddings.shape)

    if len(embeddings_list) == 1:
        fused = embeddings_list[0]
    else:
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


def filter_embeddings_by_split(
    embeddings: np.ndarray,
    index: dict[str, str],
    split_df: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, str]]:
    """Filter embeddings and build a local index for a given split.

    Returns embeddings in the same row order as split_df.

    Args:
        embeddings: Full embedding array of shape (N, D).
        index: Dict mapping row-key to posting_id.
        split_df: DataFrame with a posting_id column.

    Returns:
        Tuple of (filtered_embeddings, local_index) where local_index
        maps new row-key to posting_id.
    """
    pid_to_orig_idx = {v: int(k) for k, v in index.items()}
    orig_indices = [pid_to_orig_idx[pid] for pid in split_df["posting_id"]]
    filtered = embeddings[orig_indices]
    local_index = {
        str(i): index[str(orig_idx)] for i, orig_idx in enumerate(orig_indices)
    }
    return filtered, local_index

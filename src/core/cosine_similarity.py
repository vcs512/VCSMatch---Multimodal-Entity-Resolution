from __future__ import annotations

import numpy as np


def classify_by_prototypes(
    embeddings: np.ndarray,
    prototypes: np.ndarray,
    scale: int = 1,
) -> np.ndarray:
    """Soft-classify embeddings by cosine similarity to prototype vectors.

    Uses a softmax over prototype similarities to produce class
    probabilities.

    Args:
        embeddings: Unit-length array of shape (n, d) to classify.
        prototypes: Unit-length array of shape (c, d), one row per class.
        scale: Scale to multiply similarity before softmax.

    Returns:
        Array of shape (n, c) with softmax probabilities.
    """
    similarities = scale * (prototypes @ embeddings.T)
    exp_scores = np.exp(similarities - similarities.max(axis=0, keepdims=True))
    probs = (exp_scores / exp_scores.sum(axis=0, keepdims=True)).T
    return probs

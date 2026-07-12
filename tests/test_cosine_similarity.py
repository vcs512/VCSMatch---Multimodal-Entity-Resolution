from __future__ import annotations

import numpy as np

from src.core.cosine_similarity import classify_by_prototypes


class TestClassifyByPrototypes:
    def test_two_prototypes_returns_probabilities(self):
        np.random.seed(42)
        embeddings = np.random.randn(10, 8).astype(np.float32)
        prototypes = np.random.randn(2, 8).astype(np.float32)
        probs = classify_by_prototypes(
            embeddings=embeddings, prototypes=prototypes
        )
        assert probs.shape == (10, 2)
        assert np.allclose(probs.sum(axis=1), 1.0)
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_identical_embeddings_yield_equal_scores(self):
        emb = np.ones((3, 4), dtype=np.float32)
        protos = np.ones((2, 4), dtype=np.float32)
        probs = classify_by_prototypes(embeddings=emb, prototypes=protos)
        assert np.allclose(probs[:, 0], probs[:, 1])

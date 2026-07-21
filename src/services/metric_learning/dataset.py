from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    """PyTorch dataset wrapping pre-computed embeddings and labels."""

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray) -> None:
        """Initialize dataset with embeddings and labels.

        Args:
            embeddings: Array of shape (N, D).
            labels: Array of class IDs of shape (N,).
        """
        self.embeddings = torch.from_numpy(embeddings).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            Number of samples.
        """
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return embedding and label at the given index.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (embedding tensor, label tensor).
        """
        return self.embeddings[idx], self.labels[idx]

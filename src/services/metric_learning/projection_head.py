from __future__ import annotations

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """Projection head mapping frozen embeddings to a metric learning space.

    Architecture: Linear -> BatchNorm -> Dropout, output L2-normalized.
    """

    def __init__(self, input_dim: int, projection_dim: int = 512) -> None:
        """Initialize the projection head.

        Args:
            input_dim: Dimensionality of input embeddings.
            projection_dim: Dimensionality of output embeddings.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and L2-normalize input embeddings.

        Args:
            x: Input tensor of shape (N, input_dim).

        Returns:
            L2-normalized tensor of shape (N, projection_dim).
        """
        return nn.functional.normalize(self.net(x), dim=1)

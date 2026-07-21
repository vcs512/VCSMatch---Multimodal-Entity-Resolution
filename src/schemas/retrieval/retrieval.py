from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class RetrievalConfig(BaseModel):
    """Configuration for the retrieval service."""

    embeddings_dir: list[Path]
    fusion_type: str | None = None
    assignments_path: Path
    output_dir: Path
    split: str = "test"
    k: int = 50
    threshold: float = 0.5
    projection_head_path: Path | None = None
    projection_dim: int = 512

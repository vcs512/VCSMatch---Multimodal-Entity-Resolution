from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class RetrievalConfig(BaseModel):
    """Configuration for the retrieval service."""

    embeddings_dir: Path
    assignments_path: Path
    output_dir: Path
    split: str = "test"
    k: int = 50
    threshold: float = 0.5

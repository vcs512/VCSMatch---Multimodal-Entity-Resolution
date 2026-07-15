from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class TextEmbeddingConfig(BaseModel):
    """Configuration for text embedding service."""

    model_path: Path
    csv_path: Path
    column: str
    output_dir: Path
    id_column: str
    batch_size: int = 128
    device: str | None = None

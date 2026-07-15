from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class VisionEmbeddingConfig(BaseModel):
    """Configuration for vision embedding service."""

    model_path: Path
    image_dir: Path
    csv_path: Path
    column: str
    output_dir: Path
    id_column: str
    batch_size: int = 64
    device: str | None = None

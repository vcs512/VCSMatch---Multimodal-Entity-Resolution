from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class VisionEmbeddingConfig(BaseModel):
    """Configuration for vision embedding service."""

    model_path: Path
    image_dir: Path
    output_dir: Path
    csv_path: Path | None = None
    image_column: str = "image"
    batch_size: int = 64
    device: str | None = None
    num_workers: int = 0

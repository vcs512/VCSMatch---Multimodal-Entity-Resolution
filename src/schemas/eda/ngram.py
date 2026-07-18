from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class NGramDetectionConfig(BaseModel):
    """Configuration for the common n-gram detection service."""

    csv_path: Path
    text_column: str = "title"
    output_path: Path
    min_frequency: int = 100
    n_top: int = 200
    ngram_min_length: int = 2
    ngram_max_length: int = 4

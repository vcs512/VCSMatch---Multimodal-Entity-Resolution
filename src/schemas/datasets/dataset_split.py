from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class DatasetSplitConfig(BaseModel):
    """Configuration for the dataset split service."""

    eda_dir: Path
    output_dir: Path

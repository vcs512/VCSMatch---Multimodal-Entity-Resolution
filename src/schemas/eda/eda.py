from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class EDAConfig(BaseModel):
    """Configuration for the EDA orchestrator service."""

    train_csv_path: Path
    text_column: str = "title"
    vision_embeddings_path: Path
    text_prompt_embeddings_path: Path
    augmented_csv_path: Path

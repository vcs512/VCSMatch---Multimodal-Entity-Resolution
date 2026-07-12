from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.schemas.embedding.text import TextEmbeddingConfig


class TestTextEmbeddingConfig:
    def test_valid_minimal(self):
        config = TextEmbeddingConfig(
            model_path="checkpoints/siglip2-base-patch16-224",
            csv_path="data/train.csv",
            column="title",
            output_dir="data/embeddings",
        )
        assert config.model_path == Path("checkpoints/siglip2-base-patch16-224")
        assert config.csv_path == Path("data/train.csv")
        assert config.column == "title"
        assert config.output_dir == Path("data/embeddings")
        assert config.id_column is None
        assert config.batch_size == 128
        assert config.device is None

    def test_valid_full(self):
        config = TextEmbeddingConfig(
            model_path="checkpoints/siglip2-base-patch16-224",
            csv_path="data/train.csv",
            column="title",
            output_dir="data/embeddings",
            id_column="posting_id",
            batch_size=64,
            device="cuda",
        )
        assert config.id_column == "posting_id"
        assert config.batch_size == 64
        assert config.device == "cuda"

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            TextEmbeddingConfig(
                csv_path="data/train.csv",
                column="title",
                output_dir="data/embeddings",
            )

    def test_valid_from_json(self):
        data = {
            "model_path": "checkpoints/siglip2-base-patch16-224",
            "csv_path": "data/train.csv",
            "column": "title",
            "output_dir": "data/embeddings",
            "id_column": "posting_id",
            "batch_size": 128,
            "device": "cuda",
        }
        config = TextEmbeddingConfig.model_validate(data)
        assert config.model_path == Path("checkpoints/siglip2-base-patch16-224")
        assert config.batch_size == 128

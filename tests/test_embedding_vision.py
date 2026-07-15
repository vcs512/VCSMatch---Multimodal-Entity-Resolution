from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.schemas.embedding.vision import VisionEmbeddingConfig


class TestVisionEmbeddingConfig:
    def test_valid_minimal(self):
        config = VisionEmbeddingConfig(
            model_path="checkpoints/siglip2-base-patch16-224",
            image_dir="data/train_images",
            csv_path="data/train.csv",
            column="image",
            output_dir="data/embeddings",
            id_column="posting_id",
        )
        assert config.model_path == Path("checkpoints/siglip2-base-patch16-224")
        assert config.image_dir == Path("data/train_images")
        assert config.csv_path == Path("data/train.csv")
        assert config.column == "image"
        assert config.output_dir == Path("data/embeddings")
        assert config.id_column == "posting_id"
        assert config.batch_size == 64
        assert config.device is None

    def test_valid_full(self):
        config = VisionEmbeddingConfig(
            model_path="checkpoints/siglip2-base-patch16-224",
            image_dir="data/train_images",
            csv_path="data/train.csv",
            column="photo",
            output_dir="data/embeddings",
            id_column="posting_id",
            batch_size=32,
            device="cuda",
        )
        assert config.csv_path == Path("data/train.csv")
        assert config.column == "photo"
        assert config.id_column == "posting_id"
        assert config.batch_size == 32
        assert config.device == "cuda"

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            VisionEmbeddingConfig(
                image_dir="data/train_images",
                output_dir="data/embeddings",
            )

    def test_valid_from_json(self):
        data = {
            "model_path": "checkpoints/siglip2-base-patch16-224",
            "image_dir": "data/train_images",
            "csv_path": "data/train.csv",
            "column": "image",
            "output_dir": "data/embeddings",
            "id_column": "posting_id",
            "batch_size": 64,
            "device": "cuda",
        }
        config = VisionEmbeddingConfig.model_validate(data)
        assert config.model_path == Path("checkpoints/siglip2-base-patch16-224")
        assert config.batch_size == 64

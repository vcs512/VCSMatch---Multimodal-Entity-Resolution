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
            output_dir="data/embeddings",
        )
        assert config.model_path == Path("checkpoints/siglip2-base-patch16-224")
        assert config.image_dir == Path("data/train_images")
        assert config.output_dir == Path("data/embeddings")
        assert config.csv_path is None
        assert config.image_column == "image"
        assert config.batch_size == 64
        assert config.device is None
        assert config.num_workers == 0

    def test_valid_full(self):
        config = VisionEmbeddingConfig(
            model_path="checkpoints/siglip2-base-patch16-224",
            image_dir="data/train_images",
            output_dir="data/embeddings",
            csv_path="data/train.csv",
            image_column="photo",
            batch_size=32,
            device="cuda",
            num_workers=4,
        )
        assert config.csv_path == Path("data/train.csv")
        assert config.image_column == "photo"
        assert config.batch_size == 32
        assert config.device == "cuda"
        assert config.num_workers == 4

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
            "output_dir": "data/embeddings",
            "csv_path": "data/train.csv",
            "image_column": "image",
            "batch_size": 64,
            "device": "cuda",
            "num_workers": 0,
        }
        config = VisionEmbeddingConfig.model_validate(data)
        assert config.model_path == Path("checkpoints/siglip2-base-patch16-224")
        assert config.batch_size == 64

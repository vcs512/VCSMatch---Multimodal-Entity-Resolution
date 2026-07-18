from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.schemas.embedding.text import TextEmbeddingConfig
from src.services.embedding.text import clean_text


class TestTextEmbeddingConfig:
    def test_valid_minimal(self):
        config = TextEmbeddingConfig(
            model_path="checkpoints/siglip2-base-patch16-224",
            csv_path="data/train.csv",
            column="title",
            output_dir="data/embeddings",
            id_column="posting_id",
        )
        assert config.model_path == Path("checkpoints/siglip2-base-patch16-224")
        assert config.csv_path == Path("data/train.csv")
        assert config.column == "title"
        assert config.output_dir == Path("data/embeddings")
        assert config.id_column == "posting_id"
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


class TestCleanText:
    def test_removes_emoji(self):
        result = clean_text("iphone 12 case \U0001F60A")
        assert result == "iphone 12 case"

    def test_removes_multiple_emojis(self):
        result = clean_text("\U0001F4F1 samsung \U0001F4F1 \U00002764")
        assert result == "samsung"

    def test_removes_readystock(self):
        result = clean_text("baju anak readystock")
        assert result == "baju anak"

    def test_removes_ready_stock(self):
        result = clean_text("baju anak ready stock")
        assert result == "baju anak"

    def test_removes_ready(self):
        result = clean_text("case iphone 12 ready")
        assert result == "case iphone 12"

    def test_removes_original(self):
        result = clean_text("charger original samsung")
        assert result == "charger samsung"

    def test_removes_best_seller(self):
        result = clean_text("best seller parfum pria")
        assert result == "parfum pria"

    def test_removes_promo(self):
        result = clean_text("sepatu promo diskon")
        assert result == "sepatu"

    def test_removes_promotion(self):
        result = clean_text("handuk promotion besar")
        assert result == "handuk besar"

    def test_removes_grosir(self):
        result = clean_text("grosir baju anak murah")
        assert result == "baju anak"

    def test_removes_murah(self):
        result = clean_text("baju murah grosir")
        assert result == "baju"

    def test_removes_termurah(self):
        result = clean_text("termurah case iphone")
        assert result == "case iphone"

    def test_removes_diskon(self):
        result = clean_text("tas branded diskon 50%")
        assert result == "tas branded 50%"

    def test_removes_gratis_ongkir(self):
        result = clean_text("gratis ongkir jam tangan")
        assert result == "jam tangan"

    def test_removes_cod(self):
        result = clean_text("cod jam tangan pria")
        assert result == "jam tangan pria"

    def test_removes_limited(self):
        result = clean_text("limited edition sneakers")
        assert result == "edition sneakers"

    def test_removes_brand(self):
        result = clean_text("brand import jam tangan")
        assert result == "jam tangan"

    def test_removes_import(self):
        result = clean_text("jam tangan import original")
        assert result == "jam tangan"

    def test_removes_viral(self):
        result = clean_text("viral sepatu wanita")
        assert result == "sepatu wanita"

    def test_removes_obral(self):
        result = clean_text("obral baju bayi")
        assert result == "baju bayi"

    def test_removes_murah_meriah(self):
        result = clean_text("baju murah meriah import")
        assert result == "baju"

    def test_case_insensitive(self):
        result = clean_text("BEST SELLER ORIGINAL READY")
        assert result == ""

    def test_no_slang(self):
        result = clean_text("samsung galaxy s24 ultra 512gb")
        assert result == "samsung galaxy s24 ultra 512gb"

    def test_collapses_whitespace(self):
        result = clean_text("baju    murah   grosir")
        assert result == "baju"

    def test_emoji_and_slang_combined(self):
        result = clean_text("\U0001F4F1 iphone 12 \U0001F60A ready original grosir")
        assert result == "iphone 12"

    def test_removes_escaped_unicode(self):
        result = clean_text("baju \\xe2\\x9a\\xa0 murah")
        assert result == "baju"

    def test_removes_parentheses(self):
        result = clean_text("baju (original) murah")
        assert result == "baju"

    def test_removes_square_brackets(self):
        result = clean_text("baju [ready] grosir")
        assert result == "baju"

    def test_removes_exclamation(self):
        result = clean_text("baju murah! grosir")
        assert result == "baju"

    def test_removes_noise_only_text_preserved(self):
        result = clean_text("(hello world)! [test] \\xe2foo")
        assert result == "hello world test foo"

    def test_removes_all_noise_combined(self):
        result = clean_text("baju (\\xe2\\x9a\\xa0)[ready]! murah")
        assert result == "baju"

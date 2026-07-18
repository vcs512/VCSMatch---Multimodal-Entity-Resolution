from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from src.services.eda.ngram_detection import NGramDetectionService


class TestNGramDetectionService:
    @pytest.fixture
    def config_path(self, tmp_path: Path) -> str:
        csv_path = tmp_path / "test.csv"
        output_path = tmp_path / "common_ngrams.json"
        config_path = tmp_path / "config.json"

        titles = [
            "free shipping murah grosir produk",
            "free shipping murah grosir barang",
            "free shipping diskon besar produk",
            "free shipping grosir murah barang",
            "limited edition produk baru",
            "limited edition barang murah",
            "obral murah grosir produk",
            "obral besar diskon gratis",
            "murah meriah produk grosir",
            "gratis ongkir murah grosir",
        ]
        df = pd.DataFrame({"title": titles})
        df.to_csv(csv_path, index=False)

        config = {
            "csv_path": str(csv_path),
            "text_column": "title",
            "output_path": str(output_path),
            "min_frequency": 3,
            "n_top": 10,
            "ngram_min_length": 2,
            "ngram_max_length": 3,
        }
        with open(config_path, "w") as f:
            json.dump(config, f)

        return str(config_path)

    def test_process_returns_ngrams(self, config_path: str):
        service = NGramDetectionService(config_path=config_path)
        ngrams = service.process()

        assert len(ngrams) > 0
        assert all(isinstance(ng, str) for ng in ngrams)

    def test_output_file_exists(self, config_path: str):
        service = NGramDetectionService(config_path=config_path)
        ngrams = service.process()

        output_path = Path(service.config.output_path)
        assert output_path.exists()

        with open(output_path) as f:
            data = json.load(f)

        assert "ngrams" in data
        assert data["ngrams"] == ngrams
        assert data["min_frequency"] == 3
        assert data["total_titles"] == 10

    def test_stop_words_excluded(self, config_path: str):
        service = NGramDetectionService(config_path=config_path)
        ngrams = service.process()

        stop_words = service._get_stop_words()
        for ng in ngrams:
            words = set(re.findall(r"\b\w+\b", ng.lower()))
            assert words.isdisjoint(stop_words), f"N-gram '{ng}' contains a stop word"

    def test_only_frequent_ngrams(self, config_path: str):
        service = NGramDetectionService(config_path=config_path)
        ngrams = service.process()

        df = pd.read_csv(service.config.csv_path)
        texts = df[service.config.text_column].astype(str).tolist()

        for ng in ngrams:
            count = sum(1 for text in texts if ng in text.lower())
            assert count >= 3, (
                f"N-gram '{ng}' appears {count} times, "
                f"below min_frequency={service.config.min_frequency}"
            )

    def test_n_top_respected(self, config_path: str):
        service = NGramDetectionService(config_path=config_path)
        ngrams = service.process()

        assert len(ngrams) <= 10

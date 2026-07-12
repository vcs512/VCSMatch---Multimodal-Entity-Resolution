from __future__ import annotations

import pandas as pd

from src.services.eda.language_detection import LanguageDetectionService


class TestLanguageDetectionService:
    def test_english_text_scores_low_indonesian(self):
        service = LanguageDetectionService()
        df = pd.DataFrame(
            {
                "posting_id": ["test1"],
                "title": ["Paper Bag Victoria Secret"],
            }
        )
        scores = service.process(train_df=df, text_column="title")
        assert scores.iloc[0] < 0.5

    def test_indonesian_text_scores_high_indonesian(self):
        service = LanguageDetectionService()
        df = pd.DataFrame(
            {
                "posting_id": ["test2"],
                "title": ["Tas Wanita Fashion Import Termurah"],
            }
        )
        scores = service.process(train_df=df, text_column="title")
        assert scores.iloc[0] > 0.5

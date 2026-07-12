from __future__ import annotations

import logging

import pandas as pd
from lingua import Language, LanguageDetectorBuilder

logger = logging.getLogger(__name__)


class LanguageDetectionService:
    """Detect whether each product title is English or Indonesian.

    Uses the lingua library restricted to EN/ID, computing a normalized
    ``indonesian_score`` in [0, 1] for every row.
    """

    def __init__(self) -> None:
        """Initialize the service language detector."""
        self._detector = LanguageDetectorBuilder.from_languages(
            Language.ENGLISH, Language.INDONESIAN
        ).build()

    def process(self, train_df: pd.DataFrame, text_column: str) -> pd.Series:
        """Run language detection on each title and return indonesian_score.

        Args:
            train_df: DataFrame containing the product titles.
            text_column: Name of the column in ``train_df`` with the titles.

        Returns:
            A Series of indonesian_score values, one per input row.
        """
        texts = train_df[text_column].astype(str).tolist()
        logger.info("Detecting language for %d titles", len(texts))

        scores = []
        for text in texts:
            indonesian_score = self._detector.compute_language_confidence(
                text=text, language=Language.INDONESIAN
            )
            scores.append(indonesian_score)

        return pd.Series(data=scores, name="indonesian_score")

from __future__ import annotations

import json
import logging

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from spacy.lang.en import English
from spacy.lang.id import Indonesian

from src.schemas.eda.ngram import NGramDetectionConfig

logger = logging.getLogger(__name__)


class NGramDetectionService:
    """Detect common n-grams in a text column of a CSV.

    Strips English and Indonesian stop words, then extracts the most
    frequent n-grams (bigrams, trigrams, 4-grams) and saves them as a
    JSON list for downstream removal in the embedding pipeline.
    """

    def __init__(self, config_path: str) -> None:
        """Initialize the service and validate the config file.

        Args:
            config_path: Path to the JSON config file.
        """
        with open(config_path) as f:
            self.config = NGramDetectionConfig.model_validate_json(f.read())

    @staticmethod
    def _get_stop_words() -> set[str]:
        """Combine English and Indonesian stop words from spaCy.

        Returns:
            A set of stop words.
        """
        en_stop = English.Defaults.stop_words
        id_stop = Indonesian.Defaults.stop_words
        return en_stop | id_stop

    def process(self) -> list[str]:
        """Read texts, detect common n-grams, and save to output_path.

        Returns:
            The list of common n-grams found.
        """
        df = pd.read_csv(filepath_or_buffer=self.config.csv_path)
        texts = df[self.config.text_column].astype(str).tolist()
        logger.info("Loaded %d texts from column '%s'", len(texts), self.config.text_column)

        stop_words = self._get_stop_words()
        logger.info("Using %d combined stop words", len(stop_words))

        vectorizer = CountVectorizer(
            ngram_range=(self.config.ngram_min_length, self.config.ngram_max_length),
            stop_words=sorted(stop_words),
            min_df=self.config.min_frequency,
            token_pattern=r"(?u)\b\w+\b",
        )

        matrix = vectorizer.fit_transform(texts)
        sums = matrix.sum(axis=0).A1
        feature_names = vectorizer.get_feature_names_out()

        ngram_freq = list(zip(feature_names, sums, strict=True))
        ngram_freq.sort(key=lambda x: x[1], reverse=True)

        ngrams = [item[0] for item in ngram_freq[: self.config.n_top]]

        output = {
            "ngrams": ngrams,
            "min_frequency": self.config.min_frequency,
            "total_titles": len(texts),
        }
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file=self.config.output_path, mode="w") as f:
            json.dump(obj=output, fp=f, indent=2)
        logger.info("Saved %d common n-grams to %s", len(ngrams), self.config.output_path)

        return ngrams


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    NGramDetectionService(config_path=args.config).process()

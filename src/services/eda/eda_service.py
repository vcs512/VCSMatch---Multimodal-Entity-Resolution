from __future__ import annotations

import logging

import pandas as pd
from safetensors.numpy import load_file

from src.core.cosine_similarity import classify_by_prototypes
from src.schemas.eda.eda import EDAConfig
from src.services.eda.language_detection import LanguageDetectionService

logger = logging.getLogger(__name__)


class EDAOrchestratorService:
    """Orchestrate the full EDA pipeline: zero-shot classification and language
        detection.

    Loads pre-computed vision and prompt embeddings, runs cosine-similarity
    zero-shot classification to produce amateur_score, runs language
    detection to produce indonesian_score, then merges both into the
    original CSV and saves the augmented result.
    """

    def __init__(self, config_path: str) -> None:
        """Initialize the service and validate the config file.

        Args:
            config_path: Path to the JSON config file.
        """
        with open(config_path) as f:
            self.config = EDAConfig.model_validate_json(f.read())

    def run(self) -> pd.DataFrame:
        """Execute the EDA pipeline and return the augmented DataFrame.

        Loads embeddings, runs zero-shot classification and language
        detection, merges scores into the source CSV, and saves the
        result to ``augmented_csv_path``.

        Returns:
            The augmented DataFrame with multiples_score and indonesian_score columns.
        """
        df = pd.read_csv(filepath_or_buffer=self.config.train_csv_path)
        logger.info("Loaded %d rows from %s", len(df), self.config.train_csv_path)

        image_data = load_file(filename=str(self.config.vision_embeddings_path))
        image_embeddings = image_data["embeddings"]
        logger.info(
            "Loaded image embeddings: %s", image_embeddings.shape
        )

        prompt_data = load_file(
            filename=str(self.config.text_prompt_embeddings_path)
        )
        prompt_embeddings = prompt_data["embeddings"]
        logger.info(
            "Loaded prompt embeddings: %s", prompt_embeddings.shape
        )

        probs = classify_by_prototypes(
            embeddings=image_embeddings, prototypes=prompt_embeddings, scale=30
        )
        df["multiples_score"] = probs[:, 1]
        logger.info("Computed zero-shot multiple products together scores")

        lang_service = LanguageDetectionService()
        df["indonesian_score"] = lang_service.process(
            train_df=df, text_column=self.config.text_column
        )
        logger.info("Computed language scores")

        self.config.augmented_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path_or_buf=self.config.augmented_csv_path, index=False)
        logger.info(
            "Saved augmented CSV to %s", self.config.augmented_csv_path
        )

        return df


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/eda/eda.json"
    )
    args = parser.parse_args()
    EDAOrchestratorService(config_path=args.config).run()

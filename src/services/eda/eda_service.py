from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from safetensors.numpy import load_file

from src.core.cosine_similarity import classify_by_prototypes
from src.schemas.eda.eda import EDAConfig
from src.services.eda.language_detection import LanguageDetectionService

logger = logging.getLogger(__name__)


class EDAOrchestratorService:
    """Orchestrate the full EDA pipeline.

    Loads pre-computed vision and prompt embeddings, runs cosine-similarity
    zero-shot classification to produce multiples_score, runs language
    detection to produce indonesian_score, merges both into the
    original CSV, and saves the augmented result along with row-level
    and group-level statistics.
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
        detection, merges scores into the source CSV, saves the augmented
        result, and computes/saves row-level and group-level statistics
        with box plots.

        Returns:
            The augmented DataFrame with multiples_score and indonesian_score
            columns.
        """
        df = pd.read_csv(filepath_or_buffer=self.config.train_csv_path)
        logger.info("Loaded %d rows from %s", len(df), self.config.train_csv_path)

        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

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

        augmented_csv_path = output_dir / "train_augmented.csv"
        df.to_csv(path_or_buf=augmented_csv_path, index=False)
        logger.info("Saved augmented CSV to %s", augmented_csv_path)

        row_dir = output_dir / "row"
        group_dir = output_dir / "group"
        self._compute_and_save_rows_statistics(df=df, row_dir=row_dir)
        self._compute_and_save_grouped_statistics(df=df, group_dir=group_dir)

        return df

    @staticmethod
    def _save_boxplot(
        data: pd.Series,
        title: str,
        xlabel: str,
        save_path: Path,
    ) -> None:
        """Save a single box plot as JPG.

        Args:
            data: Series of values to plot.
            title: Plot title.
            xlabel: Label for the x-axis.
            save_path: Destination path for the JPG file.
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(x=data, ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        fig.tight_layout()
        fig.savefig(save_path, dpi=100, format="jpg")
        plt.close(fig)
        logger.info("Saved box plot to %s", save_path)

    def _compute_and_save_rows_statistics(
        self, df: pd.DataFrame, row_dir: Path
    ) -> None:
        """Compute describe() on score columns and save CSV + box plots.

        Args:
            df: Augmented DataFrame with multiples_score and indonesian_score.
            row_dir: Directory to write the statistics CSV and box plot JPGs into.
        """
        row_dir.mkdir(parents=True, exist_ok=True)

        score_columns = ["multiples_score", "indonesian_score"]
        stats_path = row_dir / "train_augmented_rows_statistics.csv"
        df[score_columns].describe().to_csv(path_or_buf=stats_path)
        logger.info("Saved row statistics to %s", stats_path)

        for col in score_columns:
            plot_path = row_dir / f"{col}_boxplot.jpg"
            self._save_boxplot(
                data=df[col],
                title=f"Box Plot — {col}",
                xlabel=col,
                save_path=plot_path,
            )

    def _compute_and_save_grouped_statistics(
        self, df: pd.DataFrame, group_dir: Path
    ) -> None:
        """Group by label_group, compute aggregates, save CSV + box plots.

        Args:
            df: Augmented DataFrame with multiples_score and indonesian_score.
            group_dir: Directory to write the aggregate CSV, describe CSV, and
                box plot JPGs into.
        """
        group_dir.mkdir(parents=True, exist_ok=True)

        grouped = (
            df.groupby(by="label_group", as_index=False)
            .agg(
                product_count=("posting_id", "count"),
                multiples_score_mean=("multiples_score", "mean"),
                multiples_score_std=("multiples_score", "std"),
                indonesian_score_mean=("indonesian_score", "mean"),
                indonesian_score_std=("indonesian_score", "std"),
            )
        )
        stats_path = group_dir / "train_augmented_grouped_statistics.csv"
        grouped.to_csv(path_or_buf=stats_path, index=False)
        logger.info("Saved grouped statistics to %s", stats_path)

        describe_path = group_dir / "train_augmented_grouped_statistics_describe.csv"
        agg_columns = [
            "product_count",
            "multiples_score_std",
            "indonesian_score_std",
        ]
        grouped[agg_columns].describe().to_csv(path_or_buf=describe_path)
        logger.info("Saved grouped statistics describe to %s", describe_path)

        boxplot_configs = [
            ("product_count", "Box Plot — Product Count per Group", "Product count"),
            (
                "multiples_score_std",
                "Box Plot — Std Multiples Score per Group",
                "Std multiples score",
            ),
            (
                "indonesian_score_std",
                "Box Plot — Std Indonesian Score per Group",
                "Std indonesian score",
            ),
        ]
        for col, title, xlabel in boxplot_configs:
            plot_path = group_dir / f"{col}_boxplot.jpg"
            self._save_boxplot(
                data=grouped[col], title=title, xlabel=xlabel, save_path=plot_path,
            )


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/eda/eda.json"
    )
    args = parser.parse_args()
    EDAOrchestratorService(config_path=args.config).run()

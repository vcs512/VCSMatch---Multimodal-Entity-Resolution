from __future__ import annotations

import logging

import pandas as pd

from src.core.datasets.dataset_split import assign_stratum, stratified_group_split
from src.schemas.datasets.dataset_split import DatasetSplitConfig

logger = logging.getLogger(__name__)


class DatasetSplitService:
    """Split the augmented dataset into train/val/test by label_group.

    Loads pre-computed group statistics from the EDA output, stratifies
    groups by variance, then splits preserving stratum proportions.
    Both per-group and per-row assignment files are saved.
    """

    def __init__(self, config_path: str) -> None:
        """Initialize the service and validate the config file.

        Args:
            config_path: Path to the JSON config file.
        """
        with open(config_path) as f:
            self.config = DatasetSplitConfig.model_validate_json(f.read())

        self.output_dir = self.config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        """Load augmented CSV and grouped statistics from EDA output, stratify, split, and save."""
        input_csv_path = self.config.eda_dir / "train_augmented.csv"
        df = pd.read_csv(filepath_or_buffer=input_csv_path)
        logger.info("Loaded %d rows from %s", len(df), input_csv_path)

        grouped_csv_path = (
            self.config.eda_dir / "group" / "train_augmented_grouped_statistics.csv"
        )
        group_summary = pd.read_csv(filepath_or_buffer=grouped_csv_path)
        logger.info(
            "Loaded %d groups from %s", len(group_summary), grouped_csv_path
        )

        group_summary = assign_stratum(group_summary=group_summary)
        stratum_counts = group_summary["stratum"].value_counts()
        logger.info("Stratum distribution:\n%s", stratum_counts)
        stratum_counts.reset_index().to_csv(
            path_or_buf=self.output_dir / "stratum_distribution.csv",
            index=False,
            header=["stratum", "count"],
        )

        group_summary = stratified_group_split(groups=group_summary)
        split_counts = group_summary["split"].value_counts()
        logger.info("Split distribution (groups):\n%s", split_counts)
        split_counts.reset_index().to_csv(
            path_or_buf=self.output_dir / "group_split_distribution.csv",
            index=False,
            header=["split", "count"],
        )

        group_to_split = group_summary.set_index("label_group")["split"].to_dict()
        df["split"] = df["label_group"].map(group_to_split)
        df["stratum"] = df["label_group"].map(
            group_summary.set_index("label_group")["stratum"].to_dict()
        )

        group_summary.to_csv(
            path_or_buf=self.output_dir / "group_summary.csv", index=False
        )
        df.to_csv(path_or_buf=self.output_dir / "assignments.csv", index=False)

        row_split_counts = df["split"].value_counts()
        logger.info("Split distribution (rows):\n%s", row_split_counts)
        logger.info("All outputs saved to %s", self.output_dir)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/dataset_split.json"
    )
    args = parser.parse_args()
    DatasetSplitService(config_path=args.config).run()

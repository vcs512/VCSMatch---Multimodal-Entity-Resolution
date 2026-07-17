from __future__ import annotations

import logging

import pandas as pd

from src.core.evaluation.metrics import (
    compute_per_row_metrics,
    summarize_by_stratum,
    summarize_metrics,
)
from src.schemas.evaluation.evaluation import EvaluationConfig

logger = logging.getLogger(__name__)


class EvaluationService:
    """Compute retrieval metrics from retrieval results.

    Loads retrieval_results.csv, computes per-row metrics (recall@k,
    precision, recall, f1), and saves per-row results plus summaries
    (overall and by stratum).
    """

    def __init__(self, config_path: str) -> None:
        """Initialize the service and validate the config file.

        Args:
            config_path: Path to the JSON config file.
        """
        with open(config_path) as f:
            self.config = EvaluationConfig.model_validate_json(f.read())

        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        """Load retrieval results, compute metrics, and save outputs."""
        df = pd.read_csv(filepath_or_buffer=self.config.retrieval_results_path)
        logger.info(
            "Loaded %d retrieval results from %s",
            len(df),
            self.config.retrieval_results_path,
        )

        df_metrics = compute_per_row_metrics(
            df=df, recall_ks=self.config.recall_ks
        )

        summary = summarize_metrics(df_metrics=df_metrics)
        by_stratum = summarize_by_stratum(df_metrics=df_metrics)

        out_dir = self.config.output_dir
        df_metrics.to_csv(path_or_buf=out_dir / "evaluation_results.csv", index=False)
        summary.to_csv(path_or_buf=out_dir / "evaluation_summary.csv", header=False)
        by_stratum.to_csv(path_or_buf=out_dir / "evaluation_by_stratum.csv", index=False)

        logger.info("Saved per-row metrics (%d rows) to %s", len(df_metrics), out_dir / "evaluation_results.csv")
        logger.info("Saved overall summary to %s", out_dir / "evaluation_summary.csv")
        logger.info("Saved stratum summary to %s", out_dir / "evaluation_by_stratum.csv")
        logger.info("Overall metrics:\n%s", summary)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/evaluation/evaluation.json")
    args = parser.parse_args()
    EvaluationService(config_path=args.config).run()

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class EvaluationConfig(BaseModel):
    """Configuration for the evaluation service.

    Attributes:
        retrieval_results_path: Path to the retrieval results CSV.
        output_dir: Directory to write evaluation outputs.
        recall_ks: List of k values for recall@k computation.
    """

    retrieval_results_path: Path
    output_dir: Path
    recall_ks: list[int] = [5, 10, 50]

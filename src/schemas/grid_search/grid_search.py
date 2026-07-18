from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class GridSearchConfig(BaseModel):
    """Configuration for the grid-search service.

    Attributes:
        base_retrieval_config: Path to the base retrieval JSON config.
        recall_ks: k values for recall@k computation.
        mlflow_tracking_uri: Tracking URI for MLFlow.
        mlflow_experiment_name: MLFlow experiment name.
        grid_output_dir: Root directory for grid-search outputs.
    """

    base_retrieval_config: Path
    recall_ks: list[int] = [5, 10, 50]
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "retrieval_grid_search"
    grid_output_dir: Path = Path("data/grid_search")

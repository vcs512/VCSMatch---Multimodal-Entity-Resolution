from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class MetricLearningConfig(BaseModel):
    """Configuration for the metric learning training service.

    Attributes:
        embeddings_dir: List of directories containing embedding.safetensors
            and index.json (one per modality).
        fusion_type: Fusion strategy when multiple embedding dirs:
            "concat" or "sum". Required if >1 dir.
        assignments_path: Path to assignments.csv with split column.
        output_dir: Directory to write training artifacts (projection head,
            metrics summary).
        mlflow_tracking_uri: Tracking URI for MLFlow.
        mlflow_experiment_name: MLFlow experiment name.
        projection_dim: Output dimension of the projection head.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay for L2 regularization.
        batch_size: Batch size for training and validation loaders.
        num_epochs: Maximum number of training epochs.
        margin: ArcFace margin parameter.
        scale: ArcFace scale parameter.
        early_stopping_patience: Stop training if val loss does not improve
            for this many consecutive epochs.
        device: Torch device override (auto-detected if None).
    """

    embeddings_dir: list[Path]
    fusion_type: str | None = None
    assignments_path: Path
    output_dir: Path
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "metric_learning"
    projection_dim: int = 512
    samples_per_class: int = 4
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    batch_size: int = 256
    num_epochs: int = 50
    margin: float = 28.6
    scale: float = 64.0
    early_stopping_patience: int = 5
    device: str | None = None

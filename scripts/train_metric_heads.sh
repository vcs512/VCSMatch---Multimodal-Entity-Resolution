#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
GEN_DIR="configs/_generated"

cd "$PROJECT_ROOT"


mkdir -p "$GEN_DIR"

echo "========================================"
echo " Pipeline: train_metric_heads"
echo "   Train projection heads for 4 variants"
echo "========================================"

train_head() {
  local name="$1"
  local config_path="$GEN_DIR/train_${name}.json"

  python -c "
import json

config = {
    'embeddings_dir': $2,
    'fusion_type': json.loads('$3'),
    'assignments_path': 'data/splits/assignments.csv',
    'output_dir': 'experiments/metric_learning/${name}',
    'mlflow_tracking_uri': 'sqlite:///mlflow.db',
    'mlflow_experiment_name': 'metric_learning',
    'projection_dim': 512,
    'samples_per_class': 4,
    'learning_rate': 0.0001,
    'weight_decay': 0.01,
    'batch_size': 256,
    'num_epochs': 50,
    'margin': 0.20,
    'scale': 20.0,
    'early_stopping_patience': 5,
    'device': 'cuda'
}
with open('${config_path}', 'w') as f:
    json.dump(config, f, indent=2)
"

  echo "  [${name}] Training metric head..."
  docker compose run --rm metric-learning uv run python -m src.services.metric_learning.train --config "$config_path"
  echo "  [${name}] Done → experiments/metric_learning/${name}"
  echo ""
}

train_head "images"   '["data/embeddings/images"]'   "null"

train_head "titles"   '["data/embeddings/titles"]'   "null"

train_head "multimodal_concat"   '["data/embeddings/images", "data/embeddings/titles"]'   '"concat"'

train_head "multimodal_sum"   '["data/embeddings/images", "data/embeddings/titles"]'   '"sum"'

echo "========================================"
echo " Done — all 4 metric heads trained"
echo "========================================"

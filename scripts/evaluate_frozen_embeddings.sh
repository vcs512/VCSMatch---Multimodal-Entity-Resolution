#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
GEN_DIR="configs/_generated"

cd "$PROJECT_ROOT"


mkdir -p "$GEN_DIR"

echo "========================================"
echo " Pipeline: evaluate_frozen_embeddings"
echo "   Grid-search on val split for 4 variants"
echo "========================================"

run_grid_search() {
  local name="$1"
  shift
  local ret_config="$GEN_DIR/retrieval_${name}.json"
  local gs_config="$GEN_DIR/grid_search_${name}.json"

  python -c "
import json

retrieval = {
    'embeddings_dir': $1,
    'fusion_type': json.loads('$2'),
    'assignments_path': 'data/splits/assignments.csv',
    'output_dir': 'results/retrieval/${name}',
    'split': 'val',
    'k': 51,
    'threshold': 0.5,
    'projection_head_path': None,
    'projection_dim': 512
}
with open('${ret_config}', 'w') as f:
    json.dump(retrieval, f, indent=2)

grid = {
    'base_retrieval_config': '${ret_config}',
    'recall_ks': [5, 10, 50],
    'mlflow_tracking_uri': 'sqlite:///mlflow.db',
    'mlflow_experiment_name': 'frozen_grid_search',
    'grid_output_dir': 'results/grid_search/${name}'
}
with open('${gs_config}', 'w') as f:
    json.dump(grid, f, indent=2)
"

  echo "  [${name}] Running grid-search..."
  docker compose run --rm grid-search uv run python -m src.services.grid_search.grid_search --config "$gs_config"
  echo "  [${name}] Done → results/grid_search/${name}"
  echo ""
}

run_grid_search "images" '["data/embeddings/images"]' "null"

run_grid_search "titles"   '["data/embeddings/titles"]'   "null"

run_grid_search "multimodal_concat"   '["data/embeddings/images", "data/embeddings/titles"]'   '"concat"'

run_grid_search "multimodal_sum"   '["data/embeddings/images", "data/embeddings/titles"]'   '"sum"'

echo "========================================"
echo " Done — all 4 frozen grid-searches complete"
echo "========================================"

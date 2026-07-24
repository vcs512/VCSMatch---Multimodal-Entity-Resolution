#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
GEN_DIR="configs/_generated"

cd "$PROJECT_ROOT"


mkdir -p "$GEN_DIR"

echo "========================================"
echo " Pipeline: evaluate_metric_test"
echo "   Best metric head threshold -> retrieval + evaluation on test split"
echo "========================================"

run_best_metric() {
  local name="$1"
  local summary_path="results/grid_search/${name}_metric/results_summary.csv"
  local head_path="experiments/metric_learning/${name}/projection_head.pt"

  if [ ! -f "$summary_path" ]; then
    echo "  [${name}] SKIP -- ${summary_path} not found"
    echo ""
    return
  fi

  if [ ! -f "$head_path" ]; then
    echo "  [${name}] SKIP -- head not found at ${head_path}"
    echo ""
    return
  fi

  local threshold
  threshold=$(python3 -c "
import pandas as pd
df = pd.read_csv('${summary_path}')
best = df.loc[df['avg_f1'].idxmax()]
print(best['threshold'])
")

  local ret_config="$GEN_DIR/retrieval_${name}_metric_best.json"
  local eval_config="$GEN_DIR/eval_${name}_metric_best.json"

  python3 -c "
import json
retrieval = {
    'embeddings_dir': $2,
    'fusion_type': json.loads('$3'),
    'assignments_path': 'data/splits/assignments.csv',
    'output_dir': 'results/retrieval/${name}_metric_best',
    'split': 'test',
    'k': 51,
    'threshold': ${threshold},
    'projection_head_path': '${head_path}',
    'projection_dim': 512
}
with open('${ret_config}', 'w') as f:
    json.dump(retrieval, f, indent=2)
eval_cfg = {
    'retrieval_results_path': 'results/retrieval/${name}_metric_best/retrieval_results.csv',
    'output_dir': 'results/evaluation/${name}_metric_best',
    'recall_ks': [5, 10, 50]
}
with open('${eval_config}', 'w') as f:
    json.dump(eval_cfg, f, indent=2)
"

  echo "  [${name}] Running retrieval (threshold=${threshold})..."
  docker compose run --rm retrieval \
    uv run python -m src.services.retrieval.retrieval \
    --config "$ret_config"

  echo "  [${name}] Running evaluation..."
  docker compose run --rm evaluation \
    uv run python -m src.services.evaluation.evaluation \
    --config "$eval_config"

  echo "  [${name}] Done"
  echo ""
}

run_best_metric "images"   '["data/embeddings/images"]'   "null"

run_best_metric "titles"   '["data/embeddings/titles"]'   "null"

run_best_metric "multimodal_concat"   '["data/embeddings/images", "data/embeddings/titles"]'   '"concat"'

run_best_metric "multimodal_sum"   '["data/embeddings/images", "data/embeddings/titles"]'   '"sum"'

echo "========================================"
echo " Done -- metric test evaluations complete"
echo "========================================"
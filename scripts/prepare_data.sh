#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "========================================"
echo " Pipeline: prepare_data"
echo "========================================"

echo "[1/2] EDA (zero-shot + language detection)..."
docker compose run --rm eda

echo "[2/2] Dataset split (stratified 70/15/15)..."
docker compose run --rm dataset-split

echo "========================================"
echo " Done — augmented CSV in data/eda/, splits in data/splits/"
echo "========================================"
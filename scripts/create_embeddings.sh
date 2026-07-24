#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "========================================"
echo " Pipeline: create_embeddings"
echo "========================================"

echo "[1/3] Vision embeddings..."
docker compose run --rm embedding-vision

echo "[2/3] Title text embeddings..."
docker compose run --rm embedding-text-titles

echo "[3/3] Prompt text embeddings..."
docker compose run --rm embedding-text-prompts

echo "========================================"
echo " Done — embeddings saved to data/embeddings/"
echo "========================================"
#!/usr/bin/env bash
#SBATCH --job-name=pnsearch-mimo-reasoner
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/file_storage01/home/juanliu/25_ymj/search_data/outputs/mimo-reasoner-%j.log

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/file_storage01/home/juanliu/25_ymj/search}"
DATA_ROOT="${DATA_ROOT:-/file_storage01/home/juanliu/25_ymj/search_data}"
PYTHON="${PYTHON:-$DATA_ROOT/envs/pnsearch/bin/python}"
CONTROLLER_ENV="${CONTROLLER_ENV:-/file_storage01/home/juanliu/.config/cl-gism/controller.env}"

set -a
source "$CONTROLLER_ENV"
set +a
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src"

"$PYTHON" scripts/generate_reasoner_trajectories.py \
  --queries "$DATA_ROOT/processed/pasa/queries_train.jsonl" \
  --index "$DATA_ROOT/indexes/pasa.sqlite" \
  --output "$DATA_ROOT/processed/pasa/reasoner_mimo_v2_train.jsonl" \
  --preferences-output "$DATA_ROOT/processed/pasa/reasoner_mimo_v2_preferences.jsonl" \
  --teacher llm \
  --limit "${REASONER_LIMIT:-5}" \
  --max-rounds 3 \
  --max-calls 9 \
  --reranker-batch-size 8 \
  --resume

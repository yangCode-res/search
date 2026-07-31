#!/usr/bin/env bash
#SBATCH --job-name=pnsearch-pasa-mine
#SBATCH --partition=gre
#SBATCH --array=0-4%2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/file_storage01/home/juanliu/25_ymj/search_data/outputs/pasa-mine-%A_%a.log

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/file_storage01/home/juanliu/25_ymj/search}"
DATA_ROOT="${DATA_ROOT:-/file_storage01/home/juanliu/25_ymj/search_data}"
PYTHON="${PYTHON:-python3}"
SHARD_SIZE="${SHARD_SIZE:-1000}"
START=$((SLURM_ARRAY_TASK_ID * SHARD_SIZE))

cd "$PROJECT_ROOT"
mkdir -p "$DATA_ROOT/candidates/pasa_train_shards" "$DATA_ROOT/outputs"
export PYTHONPATH="$PROJECT_ROOT/src"

"$PYTHON" scripts/mine_pasa_candidates.py \
  --queries "$DATA_ROOT/processed/pasa/queries_train.jsonl" \
  --index "$DATA_ROOT/indexes/pasa.sqlite" \
  --output "$DATA_ROOT/candidates/pasa_train_shards/raw_${SLURM_ARRAY_TASK_ID}.jsonl" \
  --strategy broad \
  --results-per-query 48 \
  --start "$START" \
  --limit "$SHARD_SIZE" \
  --inject-gold

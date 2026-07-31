#!/usr/bin/env bash
#SBATCH --job-name=pnsearch-mimo-audit
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/file_storage01/home/juanliu/25_ymj/search_data/outputs/mimo-audit-%j.log

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/file_storage01/home/juanliu/25_ymj/search}"
DATA_ROOT="${DATA_ROOT:-/file_storage01/home/juanliu/25_ymj/search_data}"
PYTHON="${PYTHON:-$DATA_ROOT/envs/pnsearch/bin/python}"
MIMO_OUTPUT="${MIMO_OUTPUT:-$DATA_ROOT/candidates/pasa_train_mimo_v2.jsonl}"
AUDIT_OUTPUT="${AUDIT_OUTPUT:-$DATA_ROOT/outputs/mimo_teacher_v2_audit.json}"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src"
"$PYTHON" scripts/analyze_teacher_labels.py \
  --input "$MIMO_OUTPUT" \
  --output "$AUDIT_OUTPUT" \
  --max-normalized-gold-reject-rate "${MAX_GOLD_REJECT_RATE:-0.10}"

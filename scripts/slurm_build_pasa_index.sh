#!/usr/bin/env bash
#SBATCH --job-name=pnsearch-pasa-index
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/file_storage01/home/juanliu/25_ymj/search_data/outputs/pasa-index-%j.log

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/file_storage01/home/juanliu/25_ymj/search}"
DATA_ROOT="${DATA_ROOT:-/file_storage01/home/juanliu/25_ymj/search_data}"
PYTHON="${PYTHON:-python3}"

cd "$PROJECT_ROOT"
mkdir -p "$DATA_ROOT/indexes" "$DATA_ROOT/outputs"
export PYTHONPATH="$PROJECT_ROOT/src"

"$PYTHON" scripts/build_pasa_index.py \
  --paper-zip "$DATA_ROOT/raw/pasa/paper_database/cs_paper_2nd.zip" \
  --id-map "$DATA_ROOT/raw/pasa/paper_database/id2paper.json" \
  --output "$DATA_ROOT/indexes/pasa.sqlite" \
  --rebuild

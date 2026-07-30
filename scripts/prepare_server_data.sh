#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/file_storage01/home/juanliu/25_ymj/search}"
DATA_ROOT="${DATA_ROOT:-/file_storage01/home/juanliu/25_ymj/search_data}"
PYTHON="${PYTHON:-python3}"

cd "$PROJECT_ROOT"
mkdir -p "$DATA_ROOT/raw" "$DATA_ROOT/processed" "$DATA_ROOT/candidates" "$DATA_ROOT/models" "$DATA_ROOT/outputs"

"$PYTHON" -m pip install --user huggingface_hub
"$PYTHON" scripts/download_datasets.py --data-root "$DATA_ROOT"

mapfile -t ASTA_FILES < <(find "$DATA_ROOT/raw/asta-bench/tasks/paper_finder_bench" -type f -name '*.json' | sort)
ASTA_ARGS=()
for file in "${ASTA_FILES[@]}"; do
  ASTA_ARGS+=(--asta "$file")
done

PYTHONPATH=src "$PYTHON" scripts/prepare_benchmarks.py \
  --pasa-root "$DATA_ROOT/raw/pasa" \
  "${ASTA_ARGS[@]}" \
  --output "$DATA_ROOT/processed"

echo "Prepared datasets under $DATA_ROOT; no dataset files are stored in the Git repository."


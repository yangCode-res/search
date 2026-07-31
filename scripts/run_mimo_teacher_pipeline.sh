#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/file_storage01/home/juanliu/25_ymj/search}"
DATA_ROOT="${DATA_ROOT:-/file_storage01/home/juanliu/25_ymj/search_data}"
PYTHON="${PYTHON:-$DATA_ROOT/envs/pnsearch/bin/python}"
LABEL_LIMIT="${LABEL_LIMIT:-100}"
REASONER_LIMIT="${REASONER_LIMIT:-20}"
RUN_REASONER="${RUN_REASONER:-0}"
MIMO_OUTPUT="${MIMO_OUTPUT:-$DATA_ROOT/candidates/pasa_train_mimo.jsonl}"
MIMO_LISTWISE_OUTPUT="${MIMO_LISTWISE_OUTPUT:-$DATA_ROOT/processed/pasa/reranker_mimo_train.jsonl}"

cd "$PROJECT_ROOT"
if [[ -f "$PROJECT_ROOT/.env" ]]; then
  set -a
  source "$PROJECT_ROOT/.env"
  set +a
fi
export PYTHONPATH="$PROJECT_ROOT/src"

if [[ -z "${PNSEARCH_LLM_API_KEY:-${CL_GISM_CONTROLLER_API_KEY:-${LLM_API_KEY:-}}}" ]]; then
  echo "MiMo API key is not configured. Export PNSEARCH_LLM_API_KEY, CL_GISM_CONTROLLER_API_KEY, or LLM_API_KEY." >&2
  exit 2
fi

CANDIDATE_ARGS=()
shopt -s nullglob
for shard in "$DATA_ROOT"/candidates/pasa_train_shards/raw_*.jsonl; do
  CANDIDATE_ARGS+=(--candidates "$shard")
done
shopt -u nullglob
if [[ ${#CANDIDATE_ARGS[@]} -eq 0 ]]; then
  echo "No PaSa candidate shards found under $DATA_ROOT/candidates/pasa_train_shards" >&2
  exit 2
fi

"$PYTHON" scripts/label_candidates.py \
  --queries "$DATA_ROOT/processed/pasa/queries_train.jsonl" \
  "${CANDIDATE_ARGS[@]}" \
  --output "$MIMO_OUTPUT" \
  --limit "$LABEL_LIMIT" \
  --concurrency "${MIMO_CONCURRENCY:-2}" \
  --max-candidates-per-query "${MIMO_CANDIDATES_PER_QUERY:-48}" \
  --timeout "${MIMO_TIMEOUT:-180}" \
  --resume

"$PYTHON" scripts/build_listwise_dataset.py \
  --queries "$DATA_ROOT/processed/pasa/queries_train.jsonl" \
  --candidates "$MIMO_OUTPUT" \
  --output "$MIMO_LISTWISE_OUTPUT"

if [[ "$RUN_REASONER" == "1" ]]; then
  "$PYTHON" scripts/generate_reasoner_trajectories.py \
    --queries "$DATA_ROOT/processed/pasa/queries_train.jsonl" \
    --index "$DATA_ROOT/indexes/pasa.sqlite" \
    --output "$DATA_ROOT/processed/pasa/reasoner_mimo_train.jsonl" \
    --preferences-output "$DATA_ROOT/processed/pasa/reasoner_mimo_preferences.jsonl" \
    --teacher llm \
    --limit "$REASONER_LIMIT" \
    --resume
fi

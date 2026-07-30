#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-chaosuan}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/file_storage01/home/juanliu/25_ymj/search_data}"
DATASETS="${DATASETS:-pasa,asta,litsearch}"
TMP_ROOT="$(mktemp -d /tmp/pnsearch-data-relay.XXXXXX)"
trap 'rm -rf "$TMP_ROOT"' EXIT

if [[ ",$DATASETS," == *",pasa,"* ]]; then
  echo "Downloading PaSa dataset into temporary relay directory..."
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 \
    https://huggingface.co/datasets/CarlanLark/pasa-dataset \
    "$TMP_ROOT/pasa"
  git -C "$TMP_ROOT/pasa" lfs pull \
    --include="AutoScholarQuery/**,RealScholarQuery/**,sft_crawler/**,sft_selector/**"
fi

if [[ ",$DATASETS," == *",asta,"* ]]; then
  echo "Downloading AstaBench PaperFindingBench into temporary relay directory..."
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 \
    https://huggingface.co/datasets/allenai/asta-bench \
    "$TMP_ROOT/asta-bench"
  git -C "$TMP_ROOT/asta-bench" lfs pull --include="tasks/paper_finder_bench/**"
fi

if [[ ",$DATASETS," == *",litsearch,"* ]]; then
  echo "Downloading LitSearch query and clean corpus into temporary relay directory..."
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 \
    https://huggingface.co/datasets/princeton-nlp/LitSearch \
    "$TMP_ROOT/litsearch-data"
  git -C "$TMP_ROOT/litsearch-data" lfs pull --include="query/**,corpus_clean/**"
fi

ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_DATA_ROOT/raw/pasa' '$REMOTE_DATA_ROOT/raw/asta-bench'"
if [[ -d "$TMP_ROOT/pasa" ]]; then
  rsync -az --delete-excluded \
    --include='AutoScholarQuery/***' \
    --include='RealScholarQuery/***' \
    --include='sft_crawler/***' \
    --include='sft_selector/***' \
    --exclude='*' \
    "$TMP_ROOT/pasa/" "$REMOTE_HOST:$REMOTE_DATA_ROOT/raw/pasa/"
fi
if [[ -d "$TMP_ROOT/asta-bench" ]]; then
  rsync -az --delete-excluded \
    --include='tasks/' \
    --include='tasks/paper_finder_bench/***' \
    --exclude='*' \
    "$TMP_ROOT/asta-bench/" "$REMOTE_HOST:$REMOTE_DATA_ROOT/raw/asta-bench/"
fi
if [[ -d "$TMP_ROOT/litsearch-data" ]]; then
  ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_DATA_ROOT/raw/litsearch-data'"
  rsync -az \
    "$TMP_ROOT/litsearch-data/query" \
    "$TMP_ROOT/litsearch-data/corpus_clean" \
    "$REMOTE_HOST:$REMOTE_DATA_ROOT/raw/litsearch-data/"
fi

ssh "$REMOTE_HOST" "find '$REMOTE_DATA_ROOT/raw' -type f -not -path '*/.git/*' -print | sort"
echo "Relay complete. Temporary local dataset has been removed on exit."

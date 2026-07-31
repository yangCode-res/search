#!/usr/bin/env bash
#SBATCH --job-name=pnsearch-mimo-label
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=/file_storage01/home/juanliu/25_ymj/search_data/outputs/mimo-label-%j.log

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/file_storage01/home/juanliu/25_ymj/search}"
DATA_ROOT="${DATA_ROOT:-/file_storage01/home/juanliu/25_ymj/search_data}"
CONTROLLER_ENV="${CONTROLLER_ENV:-/file_storage01/home/juanliu/.config/cl-gism/controller.env}"

if [[ ! -f "$CONTROLLER_ENV" ]]; then
  echo "MiMo controller environment not found: $CONTROLLER_ENV" >&2
  exit 2
fi

set -a
source "$CONTROLLER_ENV"
set +a

cd "$PROJECT_ROOT"
LABEL_LIMIT="${LABEL_LIMIT:-100}" \
MIMO_CONCURRENCY="${MIMO_CONCURRENCY:-2}" \
RUN_REASONER=0 \
bash scripts/run_mimo_teacher_pipeline.sh

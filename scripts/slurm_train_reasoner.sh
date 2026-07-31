#!/usr/bin/env bash
#SBATCH --job-name=pnsearch-reasoner
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/file_storage01/home/juanliu/25_ymj/search_data/outputs/pnsearch-reasoner-%j.log

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/file_storage01/home/juanliu/25_ymj/search}"
DATA_ROOT="${DATA_ROOT:-/file_storage01/home/juanliu/25_ymj/search_data}"
ENV_ROOT="${ENV_ROOT:-$DATA_ROOT/envs/pnsearch}"
MODEL_PATH="${MODEL_PATH:-/file_storage01/home/juanliu/25_ymj/model/Qwen3-Coder-30B-A3B-Instruct}"
MAX_STEPS="${MAX_STEPS:--1}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_ROOT/models/pnsearch-reasoner-lora}"

cd "$PROJECT_ROOT"
source "$ENV_ROOT/bin/activate"
export PYTHONPATH="$PROJECT_ROOT/src"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export TRITON_CACHE_DIR="/tmp/pnsearch-triton-${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR"

srun torchrun --standalone --nproc_per_node=8 scripts/train_sft.py \
  --task reasoner \
  --model "$MODEL_PATH" \
  --train "$DATA_ROOT/processed/litsearch/reasoner_train.jsonl" \
  --validation "$DATA_ROOT/processed/litsearch/reasoner_validation.jsonl" \
  --output "$OUTPUT_DIR" \
  --deepspeed configs/deepspeed_zero3.json \
  --max-length 2048 \
  --epochs 2 \
  --learning-rate 1e-5 \
  --batch-size 1 \
  --gradient-accumulation 4 \
  --lora-r 16 \
  --max-steps "$MAX_STEPS"

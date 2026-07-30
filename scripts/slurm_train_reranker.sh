#!/usr/bin/env bash
#SBATCH --job-name=pnsearch-reranker
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=24:00:00
#SBATCH --output=/file_storage01/home/juanliu/25_ymj/search_data/outputs/pnsearch-reranker-%j.log

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/file_storage01/home/juanliu/25_ymj/search}"
DATA_ROOT="${DATA_ROOT:-/file_storage01/home/juanliu/25_ymj/search_data}"
ENV_ROOT="${ENV_ROOT:-$DATA_ROOT/envs/pnsearch}"
MODEL_PATH="${MODEL_PATH:-/file_storage01/home/juanliu/25_ymj/model/Qwen3-Coder-30B-A3B-Instruct}"

cd "$PROJECT_ROOT"
source "$ENV_ROOT/bin/activate"
export PYTHONPATH="$PROJECT_ROOT/src"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

srun torchrun --standalone --nproc_per_node=8 scripts/train_sft.py \
  --task reranker \
  --model "$MODEL_PATH" \
  --train "$DATA_ROOT/processed/litsearch/reranker_train.jsonl" \
  --validation "$DATA_ROOT/processed/litsearch/reranker_validation.jsonl" \
  --output "$DATA_ROOT/models/pnsearch-reranker-lora" \
  --deepspeed configs/deepspeed_zero3.json \
  --max-length 4096 \
  --epochs 3 \
  --learning-rate 1e-5 \
  --batch-size 1 \
  --gradient-accumulation 4 \
  --lora-r 16


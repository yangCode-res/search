#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/file_storage01/home/juanliu/25_ymj/search_data}"
ENV_ROOT="${ENV_ROOT:-$DATA_ROOT/envs/pnsearch}"
PROJECT_ROOT="${PROJECT_ROOT:-/file_storage01/home/juanliu/25_ymj/search}"

python3 -m venv "$ENV_ROOT"
source "$ENV_ROOT/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch transformers datasets accelerate peft deepspeed pyarrow
python -m pip install -e "$PROJECT_ROOT"

python -c 'import torch, transformers, datasets, accelerate, peft, deepspeed; print({"torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.cuda.is_available(), "gpu_count": torch.cuda.device_count()})'


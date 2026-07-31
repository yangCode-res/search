#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/file_storage01/home/juanliu/25_ymj/search_data}"
ENV_ROOT="${ENV_ROOT:-$DATA_ROOT/envs/pnsearch}"
PROJECT_ROOT="${PROJECT_ROOT:-/file_storage01/home/juanliu/25_ymj/search}"

if [[ ! -x "$ENV_ROOT/bin/python" ]]; then
  if ! python3 -m venv "$ENV_ROOT"; then
    python3 -m pip install --user virtualenv
    python3 -m virtualenv "$ENV_ROOT"
  fi
fi
source "$ENV_ROOT/bin/activate"
python -m pip install --upgrade pip setuptools wheel

if ! python -c 'import torch' >/dev/null 2>&1; then
  if [[ "${INSTALL_TORCH:-0}" != "1" ]]; then
    cat <<'EOF'
Training environment skeleton created, but PyTorch was not installed.
GPU access is required to inspect the cluster driver before choosing a CUDA wheel.
After the account GPU quota is available, rerun with for example:

  INSTALL_TORCH=1 TORCH_INDEX_URL=<cluster-compatible-index> bash scripts/setup_training_env.sh

Do not install the default mirror build blindly: it may pull an incompatible full CUDA runtime.
EOF
    exit 0
  fi
  if [[ -z "${TORCH_INDEX_URL:-}" ]]; then
    echo "TORCH_INDEX_URL is required when INSTALL_TORCH=1" >&2
    exit 2
  fi
  python -m pip install torch --index-url "$TORCH_INDEX_URL"
fi

python -m pip install transformers datasets accelerate peft deepspeed pyarrow
python -m pip install -e "$PROJECT_ROOT"

# Login nodes provide Python development headers that may be absent on GPU nodes. Triton compiles
# a tiny CUDA helper at the first training step, so keep a shared copy available to GCC.
PYTHON_HEADER_ROOT="${PYTHON_HEADER_ROOT:-$DATA_ROOT/envs/python-headers}"
PYTHON_INCLUDE_DIR="$(python -c 'import sysconfig; print(sysconfig.get_path("include"))')"
mkdir -p "$PYTHON_HEADER_ROOT"
if [[ -d "$PYTHON_INCLUDE_DIR" ]]; then
  cp -a "$PYTHON_INCLUDE_DIR/." "$PYTHON_HEADER_ROOT/"
fi
MULTIARCH_INCLUDE="/usr/include/$(gcc -dumpmachine)/python$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ -d "$MULTIARCH_INCLUDE" ]]; then
  cp -a "$MULTIARCH_INCLUDE/." "$PYTHON_HEADER_ROOT/"
fi

python -c 'import torch, transformers, datasets, accelerate, peft, deepspeed; print({"torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.cuda.is_available(), "gpu_count": torch.cuda.device_count()})'

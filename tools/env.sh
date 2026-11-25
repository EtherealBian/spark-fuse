#!/usr/bin/env bash
# Source this file from any working directory. Never modifies an external venv.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PATH="/usr/local/cuda/bin:$PATH" CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" SPARKFUSE_CACHE="${SPARKFUSE_CACHE:-$ROOT/.cache/kernels}" TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$ROOT/.cache/inductor}" TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$ROOT/.cache/triton}"
PY="${SPARKFUSE_PYTHON:-python3}"

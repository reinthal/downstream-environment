#!/usr/bin/env bash
# Rebuild the three manually-managed venvs from their frozen locks.
# (.venv itself is uv-managed: pyproject.toml + uv.lock, synced with
#  `uv sync --extra local --group dev` — do not use this script for it.)
#
# Box assumptions (not enforceable from a lockfile):
#   - NVIDIA driver 535 (CUDA 12.x): the vllm pin is the +cu129 wheel from
#     the GitHub release; PyPI's 0.21.0 wheel is CUDA-13-built and fails.
#   - system nvcc is 11.8: FlashInfer's sampler can't JIT; VLLMBackend sets
#     VLLM_USE_FLASHINFER_SAMPLER=0 at runtime.
#   - run vLLM with  PATH=$PWD/.venv-vllm/bin:$PATH .venv-vllm/bin/python
#     (spawned workers need ninja on PATH).
#
# Usage: requirements/setup.sh [vllm|api|sandbox|all]
set -euo pipefail
cd "$(dirname "$0")/.."
target="${1:-all}"

build() { # name, extra install args...
    local name="$1"; shift
    uv venv ".venv-$name" --python 3.12
    uv pip install --python ".venv-$name/bin/python" \
        -r "requirements/venv-$name.lock.txt" "$@"
}

if [[ "$target" == vllm || "$target" == all ]]; then
    build vllm --extra-index-url https://download.pytorch.org/whl/cu126 \
               --index-strategy unsafe-best-match
    # editable in-repo plugin (kept out of the lock: absolute-path -e line)
    uv pip install --python .venv-vllm/bin/python --no-deps -e serving/vllm_plugin
    .venv-vllm/bin/python -c "import vllm; print('vllm', vllm.__version__)"
fi
if [[ "$target" == api || "$target" == all ]]; then
    build api
    .venv-api/bin/python -c "import anthropic, numpy; print('anthropic', anthropic.__version__)"
fi
if [[ "$target" == sandbox || "$target" == all ]]; then
    build sandbox
    .venv-sandbox/bin/python -c "import numpy, pandas, scipy; print('sandbox ok')"
fi

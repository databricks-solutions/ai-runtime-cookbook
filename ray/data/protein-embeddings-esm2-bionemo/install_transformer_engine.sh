#!/bin/bash
# Install NVIDIA Transformer Engine (TE), which the BioNeMo ESM-2 checkpoint uses
# for its fused attention and linear layers.
#
# The databricks_ai_v6 environment provides PyTorch (CUDA 13) but not TE. TE's
# CUDA core library installs from a prebuilt wheel; only its PyTorch binding,
# transformer_engine_torch, compiles from source (a few minutes on an 8x H100 node).
#
# Optional: set TE_WHEEL_CACHE to a Unity Catalog volume directory. The first run
# saves the compiled binding there and later runs install it in seconds. Wheels
# are kept in a subdirectory keyed on the Python, PyTorch, and CUDA versions and
# NVTE_CUDA_ARCHS, so a changed environment builds a fresh wheel.
set -eo pipefail

TE_VERSION="${TE_VERSION:-2.17.1}"
# 90 = H100. Use "90;100" to also cover B200, or "86" for A10.
export NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS:-90}"

if [ -n "${TE_WHEEL_CACHE:-}" ]; then
  # For example .../py3.12-torch2.11.0+cu130-sm90 (the torch version includes CUDA).
  TE_WHEEL_CACHE="${TE_WHEEL_CACHE%/}/$(python -c 'import sys, torch; print(f"py{sys.version_info[0]}.{sys.version_info[1]}-torch{torch.__version__}")')-sm${NVTE_CUDA_ARCHS//;/_}"
fi

if python -c "import transformer_engine.pytorch" >/dev/null 2>&1; then
  echo "Transformer Engine is already installed."
  exit 0
fi

if [ -n "${TE_WHEEL_CACHE:-}" ] && ls "$TE_WHEEL_CACHE"/transformer_engine_torch-"$TE_VERSION"-*.whl >/dev/null 2>&1; then
  echo "Installing Transformer Engine ${TE_VERSION} with the cached binding from ${TE_WHEEL_CACHE}..."
  python -m pip install --find-links "$TE_WHEEL_CACHE" "transformer_engine[pytorch]==${TE_VERSION}"
  python -c "import importlib.metadata as md, transformer_engine.pytorch; print('Transformer Engine ready:', md.version('transformer_engine'))"
  exit 0
fi

echo "Building Transformer Engine ${TE_VERSION} for CUDA archs ${NVTE_CUDA_ARCHS}..."
# Build tools for the binding. The environment already has a recent setuptools.
python -m pip install cmake==4.4.3 ninja==1.13.0 pybind11==3.1.0

# The CUDA math-library headers (cusparse.h, nvtx3, ...) are not under
# /usr/local/cuda/include in this environment; they ship inside the nvidia-*
# pip wheels that PyTorch depends on. Put them on the compiler include path.
NV_INCLUDES=$(python -c 'import glob, nvidia; print(":".join(sorted({d for p in nvidia.__path__ for d in glob.glob(p + "/*/include")})))')
export CPATH="${NV_INCLUDES}:/usr/local/cuda/include:${CPATH:-}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"

python -m pip install --no-build-isolation "transformer_engine[pytorch]==${TE_VERSION}"

# Only node 0 writes the cache, so the nodes of a multi-node run don't write the same file at
# once. Caching is best-effort: TE is already installed, so a failed copy doesn't fail the run.
if [ -n "${TE_WHEEL_CACHE:-}" ] && [ "${NODE_RANK:-0}" = "0" ]; then
  # pip keeps the wheel it just built in its cache; copy it to the volume for reuse.
  mkdir -p "$TE_WHEEL_CACHE" || true
  find "$(python -m pip cache dir)/wheels" -name "transformer_engine_torch-${TE_VERSION}-*.whl" -exec cp {} "$TE_WHEEL_CACHE"/ \; || true
  ls "$TE_WHEEL_CACHE"/transformer_engine_torch-"$TE_VERSION"-*.whl >/dev/null 2>&1 \
    && echo "Cached the compiled binding in ${TE_WHEEL_CACHE}." \
    || echo "Could not find the compiled wheel in the pip cache; later runs will rebuild."
fi

python -c "import importlib.metadata as md, transformer_engine.pytorch; print('Transformer Engine ready:', md.version('transformer_engine'))"

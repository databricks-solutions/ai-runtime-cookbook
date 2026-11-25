#!/bin/bash
# Install NVIDIA Transformer Engine (TE), which the BioNeMo ESM-2 checkpoint uses
# for its fused attention and linear layers.
#
# The databricks_ai_v6 environment provides PyTorch (CUDA 13) but not TE. TE's
# CUDA core library installs from a prebuilt wheel; only its PyTorch binding,
# transformer_engine_torch, compiles from source (a few minutes on an 8x H100 node).
set -eo pipefail

TE_VERSION="${TE_VERSION:-2.17.1}"
# 90 = H100. Use "90;100" to also cover B200, or "86" for A10.
export NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS:-90}"

if python te_probe.py import >/dev/null 2>&1; then
  echo "Transformer Engine is already installed."
  exit 0
fi

echo "Building Transformer Engine ${TE_VERSION} for CUDA archs ${NVTE_CUDA_ARCHS}..."
# Build tools for the binding. The environment already has a recent setuptools.
python -m pip install cmake==4.4.3 ninja==1.13.0 pybind11==3.1.0

# The CUDA math-library headers (cusparse.h, nvtx3, ...) are not under
# /usr/local/cuda/include in this environment; they ship inside the nvidia-*
# pip wheels that PyTorch depends on. Put them on the compiler include path.
NV_INCLUDES=$(python te_probe.py include-dirs)
export CPATH="${NV_INCLUDES}:/usr/local/cuda/include:${CPATH:-}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"

python -m pip install --no-build-isolation "transformer_engine[pytorch]==${TE_VERSION}"

python te_probe.py version

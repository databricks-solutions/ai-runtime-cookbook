#!/bin/bash
# Builds llama-server and llama-quantize with CUDA for the A10 (sm_86).
#
# The llama.cpp source comes from the llama-cpp-python 0.3.37 sdist, which vendors llama.cpp
# at commit a956a852c. Fetching the sdist through the package index keeps the source pinned
# and avoids a GitHub clone. The Python wheel itself is not installed.
set -eo pipefail

LLAMA_CPP_PYTHON_VERSION=0.3.37
WORK=/tmp/llama-cpp-build
rm -rf "$WORK"
mkdir -p "$WORK/sdist"
python -m pip download --no-deps --no-binary :all: \
  "llama-cpp-python==$LLAMA_CPP_PYTHON_VERSION" -d "$WORK/sdist"
tar -xzf "$WORK/sdist/llama_cpp_python-$LLAMA_CPP_PYTHON_VERSION.tar.gz" -C "$WORK"
SRC="$WORK/llama_cpp_python-$LLAMA_CPP_PYTHON_VERSION/vendor/llama.cpp"

# The build needs nvcc with the CUDA headers for cuBLAS. Print them so a missing toolkit fails here.
command -v nvcc
nvcc --version | tail -n 2

# Build only the two tools this recipe runs. OpenSSL, the web UI and the CCCL fetch are disabled
# so that configure does not reach the network.
cmake -S "$SRC" -B "$WORK/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DGGML_CUDA_NCCL=OFF \
  -DLLAMA_OPENSSL=OFF \
  -DLLAMA_BUILD_UI=OFF \
  -DLLAMA_USE_PREBUILT_UI=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build "$WORK/build" --target llama-server llama-quantize -j "$(nproc)"
ls -la "$WORK/build/bin"

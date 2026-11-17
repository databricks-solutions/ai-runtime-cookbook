#!/bin/bash
# Quantize on GPU 0, then run the bf16 baseline and the quantized model side by side.
set -eo pipefail
cd "$CODE_SOURCE_PATH"
# FlashInfer JIT-compiles some kernels at startup. They need curand.h, which this environment
# ships only inside the nvidia-* pip wheels. Expose just those headers: the wheels' other CUDA
# headers don't match the system nvcc.
mkdir -p /tmp/curand-include
find "$(python -c 'import nvidia; print(nvidia.__path__[0])')" -name 'curand*.h' -exec cp {} /tmp/curand-include/ \;
export CPATH="/tmp/curand-include:${CPATH:-}"
python quantize.py

CUDA_VISIBLE_DEVICES=0 python batch_inference.py bf16 > /tmp/bf16.log 2>&1 &
p1=$!
CUDA_VISIBLE_DEVICES=1 python batch_inference.py quantized > /tmp/quantized.log 2>&1 &
p2=$!
status=0
wait $p1 || { echo "=== bf16 FAILED"; tail -60 /tmp/bf16.log; status=1; }
wait $p2 || { echo "=== quantized FAILED"; tail -60 /tmp/quantized.log; status=1; }
grep -h '^{' /tmp/bf16.log /tmp/quantized.log || true
[ "$status" -eq 0 ] || exit "$status"
python summarize.py

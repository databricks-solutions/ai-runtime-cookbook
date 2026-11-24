#!/bin/bash
# PTQ to NVFP4 -> evaluate bf16 and PTQ -> QAD on all GPUs -> evaluate QAD -> compare.
set -eo pipefail
cd "$CODE_SOURCE_PATH"

# FlashInfer compiles kernels for vLLM at startup and needs curand.h. This environment ships it only
# inside the nvidia-* pip wheels, so expose just those headers to the compiler.
mkdir -p /tmp/curand-include
find "$(python -c 'import nvidia; print(nvidia.__path__[0])')" -name 'curand*.h' -exec cp {} /tmp/curand-include/ \;
export CPATH="/tmp/curand-include:${CPATH:-}"

# The bf16 baseline runs on GPU 7 while the PTQ stages use GPU 0. QAD needs every GPU, so it waits.
CUDA_VISIBLE_DEVICES=7 python evaluate.py bf16 > /tmp/eval_bf16.log 2>&1 &
p_bf16=$!
trap 'kill "$p_bf16" 2>/dev/null || true' EXIT

CUDA_VISIBLE_DEVICES=0 python quantize.py
CUDA_VISIBLE_DEVICES=0 python export.py ptq
CUDA_VISIBLE_DEVICES=0 python evaluate.py ptq

wait "$p_bf16" || { echo "=== bf16 evaluation FAILED"; tail -60 /tmp/eval_bf16.log; exit 1; }
grep -h '^\[eval\]' /tmp/eval_bf16.log || true

accelerate launch --config_file "$CODE_SOURCE_PATH/accelerate_fsdp2.yaml" qad_train.py

CUDA_VISIBLE_DEVICES=0 python export.py qad
CUDA_VISIBLE_DEVICES=0 python evaluate.py qad
python summarize.py

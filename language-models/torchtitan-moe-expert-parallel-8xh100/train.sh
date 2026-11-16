#!/bin/bash
set -eo pipefail
# torchtitan's recipes package and asset script live in the repository, not the wheel.
git clone --depth 1 --branch v0.4.0 https://github.com/pytorch/torchtitan.git /tmp/torchtitan
export HF_ASSETS_PATH=/tmp/hf_assets/Qwen3-30B-A3B
python /tmp/torchtitan/scripts/download_hf_assets.py --repo_id Qwen/Qwen3-30B-A3B \
  --assets tokenizer --local_dir /tmp/hf_assets

export PYTHONPATH="$CODE_SOURCE_PATH:/tmp/torchtitan:${PYTHONPATH:-}"
NGPU=$(nvidia-smi -L | wc -l)
torchrun --nproc_per_node="$NGPU" --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
  --local-ranks-filter 0 --role rank --tee 3 \
  -m torchtitan.train --module moe_ep --config qwen3_moe_ep \
  --output-dir /tmp/torchtitan-out 2>&1 | tee /tmp/train.log

# Summarize throughput and loss from rank 0's metric lines.
python "$CODE_SOURCE_PATH/summarize.py" /tmp/train.log

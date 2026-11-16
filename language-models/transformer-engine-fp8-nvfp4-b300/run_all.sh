#!/bin/bash
# Tokenize once, then train one precision per GPU in parallel and summarize the results.
set -eo pipefail
cd "$CODE_SOURCE_PATH"
python bench.py prepare

read -ra PRECISIONS <<< "${PRECISIONS:-bf16 fp8_current mxfp8 nvfp4}"
pids=()
for i in "${!PRECISIONS[@]}"; do
  p=${PRECISIONS[$i]}
  CUDA_VISIBLE_DEVICES=$i python bench.py train "$p" > "/tmp/te-bench/log_$p.txt" 2>&1 &
  pids+=($!)
done
status=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "=== ${PRECISIONS[$i]} FAILED; last log lines:"
    tail -40 "/tmp/te-bench/log_${PRECISIONS[$i]}.txt"
    status=1
  fi
done
for p in "${PRECISIONS[@]}"; do grep -h '^\[' "/tmp/te-bench/log_$p.txt" | tail -3 || true; done
python summarize.py
exit $status

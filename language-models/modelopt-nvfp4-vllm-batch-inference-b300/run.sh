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

QDIR=$(python -c "import os, yaml; print(yaml.safe_load(open(os.environ['HYPERPARAMETERS_PATH']))['quantized_dir'])")
MODEL=$(python -c "import os, yaml; print(yaml.safe_load(open(os.environ['HYPERPARAMETERS_PATH']))['model_name'])")
CUDA_VISIBLE_DEVICES=0 python batch_inference.py bf16 "$MODEL" > /tmp/bf16.log 2>&1 &
p1=$!
CUDA_VISIBLE_DEVICES=1 python batch_inference.py quantized "$QDIR" > /tmp/quantized.log 2>&1 &
p2=$!
status=0
wait $p1 || { echo "=== bf16 FAILED"; tail -60 /tmp/bf16.log; status=1; }
wait $p2 || { echo "=== quantized FAILED"; tail -60 /tmp/quantized.log; status=1; }
grep -h '^{' /tmp/bf16.log /tmp/quantized.log || true
python - <<'PY'
import json, os
from pathlib import Path
r = {p.stem.split("_")[-1]: json.loads(p.read_text()) for p in Path("/tmp").glob("vllm_result_*.json")}
if {"bf16", "quantized"} <= r.keys():
    b, q = r["bf16"], r["quantized"]
    print(f"throughput speedup: {q['output_tokens_per_second'] / b['output_tokens_per_second']:.2f}x "
          f"({b['output_tokens_per_second']:,.0f} -> {q['output_tokens_per_second']:,.0f} output tok/s)")
    print(f"GSM8K accuracy: bf16 {b['gsm8k_accuracy']:.3f}, quantized {q['gsm8k_accuracy']:.3f}")
    try:
        import mlflow
        with mlflow.start_run(run_id=os.environ["MLFLOW_RUN_ID"]):
            for k, v in r.items():
                mlflow.log_metrics({f"{k}_output_tok_per_s": v["output_tokens_per_second"],
                                    f"{k}_gsm8k_accuracy": v["gsm8k_accuracy"]})
    except Exception as e:
        print(f"MLflow logging skipped: {e}")
PY
exit $status

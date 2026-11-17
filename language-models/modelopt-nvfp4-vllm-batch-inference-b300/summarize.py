"""Compare the bf16 and quantized batch-inference results and log them to MLflow."""

import json
import os
from pathlib import Path

import mlflow

results = {label: json.loads(Path(f"/tmp/vllm_result_{label}.json").read_text())
           for label in ("bf16", "quantized")}
bf16, quantized = results["bf16"], results["quantized"]
speedup = quantized["output_tokens_per_second"] / bf16["output_tokens_per_second"]
print(f"throughput speedup: {speedup:.2f}x "
      f"({bf16['output_tokens_per_second']:,.0f} -> {quantized['output_tokens_per_second']:,.0f} output tok/s)")
print(f"GSM8K accuracy: bf16 {bf16['gsm8k_accuracy']:.3f}, quantized {quantized['gsm8k_accuracy']:.3f}")

with mlflow.start_run(run_id=os.environ["MLFLOW_RUN_ID"]):
    mlflow.log_metric("quantized_speedup", speedup)
    for label, r in results.items():
        mlflow.log_metrics({f"{label}_output_tok_per_s": r["output_tokens_per_second"],
                            f"{label}_gsm8k_accuracy": r["gsm8k_accuracy"]})

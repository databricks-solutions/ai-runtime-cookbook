"""Start the rollout server for train.py, with the model named in the recipe's parameters.

workload.yaml runs this on GPU 0 in the background and waits for /health. os.execvp replaces this process
with `vllm serve`, so the background PID is the server's own.
"""

import os

import yaml

with open(os.environ["HYPERPARAMETERS_PATH"]) as f:
    P = yaml.safe_load(f)

# The trainer's weight sync and LoRA adapter loading need the runtime-update flags, and VLLM_SERVER_DEV_MODE
# (set in workload.yaml) exposes the weight-transfer routes. Keep the default single API server: vLLM 0.33 rejects
# VLLM_ALLOW_RUNTIME_LORA_UPDATING with --api-server-count > 1.
# --max-loras 8 covers the trainer's max_staleness + 2 adapter slots plus the final evaluation adapter.
os.execvp(
    "vllm",
    [
        "vllm",
        "serve",
        P["model_name"],
        "--port",
        "8000",
        "--gpu-memory-utilization",
        "0.9",
        "--max-model-len",
        "8192",
        "--weight-transfer-config",
        '{"backend": "nccl"}',
        "--logprobs-mode",
        "processed_logprobs",
        "--enable-lora",
        "--max-lora-rank",
        "32",
        "--max-loras",
        "8",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
    ],
)

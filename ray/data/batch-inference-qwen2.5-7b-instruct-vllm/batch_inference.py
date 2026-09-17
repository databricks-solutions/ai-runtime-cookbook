#!/usr/bin/env python3
"""Offline Qwen2.5-7B-Instruct batch inference with Ray Data and vLLM.

The workload starts a Ray head on a single 8x H100 node. The Ray Data LLM API
launches one vLLM replica per GPU, processes prompts from Alpaca, and writes the
generated text to a Unity Catalog volume as Parquet.

The model and dataset are public and do not require a Hugging Face token. Set
OUTPUT_PATH to an existing writable Unity Catalog volume before running.
"""

import os

import ray
from datasets import load_dataset
from ray.data.llm import build_processor, vLLMEngineProcessorConfig

MODEL_SOURCE = "Qwen/Qwen2.5-7B-Instruct"
NUM_PROMPTS = 1000
# Unity Catalog volume path where results land as Parquet. Set this in workload.yaml.
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/Volumes/main/default/air_examples/ray_batch_inference")


def build_prompts():
    """Build a Ray Dataset of prompts from a public instruction dataset."""
    raw = load_dataset("tatsu-lab/alpaca", split=f"train[:{NUM_PROMPTS}]")
    items = []
    for row in raw:
        instruction = row["instruction"]
        if row.get("input"):
            instruction = f"{instruction}\n\n{row['input']}"
        items.append({"instruction": instruction})
    return ray.data.from_items(items)


def main():
    ray.init(address="auto")
    # Derive replicas from the live cluster so the example scales when nodes are added.
    total_gpus = int(ray.cluster_resources().get("GPU", 0))
    print(f"Ray cluster ready: {total_gpus} GPU(s)", flush=True)

    ds = build_prompts()

    # vLLM engine config. concurrency = number of replicas Ray Data runs in parallel;
    # one per GPU in the cluster here. engine_kwargs are passed through to the vLLM engine.
    config = vLLMEngineProcessorConfig(
        model_source=MODEL_SOURCE,
        engine_kwargs={
            "max_model_len": 4096,
            "tensor_parallel_size": 1,
            "enable_chunked_prefill": True,
        },
        concurrency=total_gpus,
        batch_size=64,
    )

    # preprocess maps each input row to a chat request; postprocess keeps the columns
    # we want to persist. ray.data.llm adds a `generated_text` column.
    processor = build_processor(
        config,
        preprocess=lambda row: dict(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": row["instruction"]},
            ],
            sampling_params=dict(max_tokens=256, temperature=0.7),
        ),
        postprocess=lambda row: dict(
            instruction=row["instruction"],
            output=row["generated_text"],
        ),
    )

    # materialize once so the write and the sample print don't re-run inference.
    out = processor(ds).materialize()
    out.write_parquet(OUTPUT_PATH)
    print(f"Wrote {out.count()} rows to {OUTPUT_PATH}", flush=True)
    for row in out.take(2):
        print("INSTRUCTION:", row["instruction"][:120], flush=True)
        print("OUTPUT:", row["output"][:200], flush=True)

    ray.shutdown()


if __name__ == "__main__":
    main()

"""SGLang offline batch inference with and without EAGLE-3 speculative decoding.

Runs the same prompts through Qwen3-8B twice on one GPU: once plain, once with an EAGLE-3
draft model proposing tokens that the target model verifies. Reports output tokens/s and the
average accept length (tokens produced per target-model forward pass).
"""

import json
import os
import time
from pathlib import Path

import mlflow
import sglang as sgl
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer

with open(os.environ["HYPERPARAMETERS_PATH"]) as f:
    P = yaml.safe_load(f)


def build_prompts():
    tok = AutoTokenizer.from_pretrained(P["model_name"])
    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(P["num_prompts"]))
    return [
        tok.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        for q in ds["question"]
    ]


def run(label, prompts, **spec_kwargs):
    engine = sgl.Engine(
        model_path=P["model_name"],
        mem_fraction_static=0.75,
        max_running_requests=P["max_running_requests"],
        # The FlashInfer sampler JIT-compiles kernels that need extra CUDA headers.
        sampling_backend="pytorch",
        **spec_kwargs,
    )
    sampling = {"temperature": 0.0, "max_new_tokens": P["max_new_tokens"]}
    engine.generate(prompts[:4], sampling)  # warm-up: CUDA graphs, kernel autotuning
    t0 = time.time()
    outs = engine.generate(prompts, sampling)
    elapsed = time.time() - t0
    engine.shutdown()

    tokens = sum(o["meta_info"]["completion_tokens"] for o in outs)
    accept = [o["meta_info"]["spec_accept_length"] for o in outs if "spec_accept_length" in o["meta_info"]]
    result = {
        "label": label,
        "seconds": elapsed,
        "output_tokens_per_second": tokens / elapsed,
        "mean_accept_length": sum(accept) / len(accept) if accept else 1.0,
    }
    print(json.dumps(result), flush=True)
    return result, [o["text"] for o in outs]


def main():
    prompts = build_prompts()
    base, base_text = run("baseline", prompts)
    spec, spec_text = run(
        "eagle3",
        prompts,
        speculative_algorithm="EAGLE3",
        speculative_draft_model_path=P["draft_model_name"],
        speculative_num_steps=P["speculative_num_steps"],
        speculative_eagle_topk=P["speculative_eagle_topk"],
        speculative_num_draft_tokens=P["speculative_num_draft_tokens"],
    )
    same = sum(a == b for a, b in zip(base_text, spec_text)) / len(prompts)
    speedup = spec["output_tokens_per_second"] / base["output_tokens_per_second"]
    print(f"\nEAGLE-3 speedup: {speedup:.2f}x "
          f"({base['output_tokens_per_second']:,.0f} -> {spec['output_tokens_per_second']:,.0f} output tok/s), "
          f"mean accept length {spec['mean_accept_length']:.2f}, "
          f"identical greedy outputs: {same:.0%}")

    out = Path(P["output_root"]) / os.environ["MLFLOW_RUN_ID"]
    out.mkdir(parents=True, exist_ok=True)
    summary = {"baseline": base, "eagle3": spec, "speedup": speedup, "identical_outputs": same}
    (out / "results.json").write_text(json.dumps(summary, indent=2))
    with mlflow.start_run(run_id=os.environ["MLFLOW_RUN_ID"]):
        mlflow.log_metrics({"speedup": speedup, "mean_accept_length": spec["mean_accept_length"],
                            "baseline_tok_per_s": base["output_tokens_per_second"],
                            "eagle3_tok_per_s": spec["output_tokens_per_second"]})


# SGLang starts worker processes, so the entry point must be guarded.
if __name__ == "__main__":
    main()

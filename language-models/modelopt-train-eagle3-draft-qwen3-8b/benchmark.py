"""Serve Qwen3-8B with SGLang and measure speculative decoding on held-out UltraChat prompts.

Configurations run in turn on one GPU: no draft; a second no-draft pass as a control for
run-to-run numerics; the untrained draft head (before training); the EAGLE-3 draft trained by
train_draft.py; and optionally the public AngelSlim EAGLE-3 draft. All use the same prompts,
greedy decoding and the same SGLang build. Each configuration is timed once.
"""

import json
import math
import os
import shutil
import statistics
import time
from pathlib import Path

import mlflow
import nvidia
import sglang as sgl
import yaml

from common import load_split, tokenizer

with open(os.environ["HYPERPARAMETERS_PATH"]) as f:
    P = yaml.safe_load(f)

RUN_ID = os.environ["MLFLOW_RUN_ID"]
OUT = Path(P["output_root"]) / RUN_ID
DRAFT_DIR = OUT / "draft"
UNTRAINED_DIR = OUT / "draft_untrained"
SEED = int(P["seed"])


def expose_curand_headers():
    """Put curand.h on CPATH for SGLang's FlashInfer JIT.

    FlashInfer JIT-compiles some kernels that need curand.h. This environment ships it only inside
    the nvidia-* wheels, so copy just those headers to one include directory.
    """
    include = Path("/tmp/curand-include")
    include.mkdir(parents=True, exist_ok=True)
    for header in Path(nvidia.__path__[0]).rglob("curand*.h"):
        shutil.copy(header, include)
    os.environ["CPATH"] = f"{include}:{os.environ.get('CPATH', '')}"


def mean_se(xs):
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var / len(xs))


def first_divergence(a, b):
    """Index of the first token where two token lists differ, or None if they are identical.

    A list that is a strict prefix of the other diverges at the end of the shorter one.
    """
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def run(label, prompts, **spec_kwargs):
    engine = sgl.Engine(
        model_path=P["model_name"],
        mem_fraction_static=0.75,
        max_running_requests=int(P["max_running_requests"]),
        # Avoids the FlashInfer sampler JIT, which needs extra CUDA headers on this image.
        sampling_backend="pytorch",
        **spec_kwargs,
    )
    sampling = {"temperature": 0.0, "max_new_tokens": int(P["max_new_tokens"])}
    engine.generate(prompts[:4], sampling)  # warm-up: CUDA graphs and kernel selection
    t0 = time.time()
    outs = engine.generate(prompts, sampling)
    elapsed = time.time() - t0
    engine.shutdown()

    tokens = sum(o["meta_info"]["completion_tokens"] for o in outs)
    accept = [o["meta_info"]["spec_accept_length"] for o in outs if "spec_accept_length" in o["meta_info"]]
    result = {"label": label, "requests": len(outs), "completion_tokens": tokens,
              "seconds": elapsed, "output_tokens_per_second": tokens / elapsed}
    if accept:
        result["mean_accept_length"], result["accept_length_se"] = mean_se(accept)
    print(json.dumps(result), flush=True)
    return result, [o["text"] for o in outs]


def compare(result, texts, base_tokens, tok):
    """Greedy agreement with the first baseline, per prompt, in retokenized token positions."""
    tokens = [tok(t, add_special_tokens=False).input_ids for t in texts]
    div = [first_divergence(b, t) for b, t in zip(base_tokens, tokens)]
    diverged = [d for d in div if d is not None]
    result["identical_greedy_outputs"] = 1 - len(diverged) / len(div)
    result["first_divergence_tokens"] = div
    result["median_first_divergence_tokens"] = statistics.median(diverged) if diverged else None


def main():
    expose_curand_headers()
    tok = tokenizer(P["model_name"])
    pairs = load_split("test_sft", int(P["num_eval_prompts"]), SEED)
    prompts = [tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
               for u, _ in pairs]
    print(f"eval prompts: {len(prompts)} (UltraChat test_sft, seed {SEED}); "
          f"mean prompt chars {sum(map(len, prompts)) / len(prompts):.0f}", flush=True)

    spec_args = dict(
        speculative_algorithm="EAGLE3",
        speculative_num_steps=int(P["speculative_num_steps"]),
        speculative_eagle_topk=int(P["speculative_eagle_topk"]),
        speculative_num_draft_tokens=int(P["speculative_num_draft_tokens"]),
    )
    results, texts = {}, {}
    results["baseline"], texts["baseline"] = run("baseline", prompts)
    results["baseline_repeat"], texts["baseline_repeat"] = run("baseline_repeat", prompts)
    results["untrained_draft"], texts["untrained_draft"] = run(
        "untrained_draft", prompts, speculative_draft_model_path=str(UNTRAINED_DIR), **spec_args)
    results["trained_draft"], texts["trained_draft"] = run(
        "trained_draft", prompts, speculative_draft_model_path=str(DRAFT_DIR), **spec_args)
    if str(P["include_reference_draft"]).lower() == "true":
        results["reference_draft"], texts["reference_draft"] = run(
            "reference_draft", prompts, speculative_draft_model_path=P["reference_draft_name"], **spec_args)

    base_tokens = [tok(t, add_special_tokens=False).input_ids for t in texts["baseline"]]
    base_r = results["baseline"]
    for k in results:
        if k == "baseline":
            continue
        compare(results[k], texts[k], base_tokens, tok)
        results[k]["speedup_vs_baseline"] = (results[k]["output_tokens_per_second"]
                                             / base_r["output_tokens_per_second"])

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "benchmark.json").write_text(json.dumps(results, indent=2))

    metrics = {"baseline_output_tokens_per_s": base_r["output_tokens_per_second"]}
    for k, r in results.items():
        if k == "baseline":
            continue
        metrics[f"{k}_output_tokens_per_s"] = r["output_tokens_per_second"]
        metrics[f"{k}_speedup"] = r["speedup_vs_baseline"]
        metrics[f"{k}_identical_greedy_outputs"] = r["identical_greedy_outputs"]
        if r["median_first_divergence_tokens"] is not None:
            metrics[f"{k}_median_first_divergence_tokens"] = r["median_first_divergence_tokens"]
        if "mean_accept_length" in r:
            metrics[f"{k}_mean_accept_length"] = r["mean_accept_length"]
            metrics[f"{k}_accept_length_se"] = r["accept_length_se"]
    mlflow.start_run(run_id=RUN_ID)
    mlflow.log_metrics(metrics)
    mlflow.log_dict({k: r["first_divergence_tokens"] for k, r in results.items() if k != "baseline"},
                    "first_divergence_tokens_per_prompt.json")
    mlflow.end_run()


# SGLang starts worker processes, so the entry point must be guarded.
if __name__ == "__main__":
    main()

"""Offline batch generation with GLM-5.3-Flash under two vLLM parallel layouts on one 8x H100 node.

tp8     Tensor parallel 8. One engine; every GPU holds a shard of every layer.
dp8_ep  Data parallel 8 with expert parallel 8. Eight engines, one per GPU. Attention and other dense
        weights are replicated on each GPU, and the routed experts of each MoE layer are sharded across
        all eight GPUs.

Stages, run in order by workload.yaml: `download`, `generate --layout tp8`,
`generate --layout dp8_ep`, `summary`.
"""

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import yaml

with open(os.environ["HYPERPARAMETERS_PATH"]) as f:
    P = yaml.safe_load(f)

GIB = 1024**3
MODEL_NAME = P["model_name"]


def pick_scratch_dir() -> Path:
    """Return the local disk that holds the checkpoint.

    The repo does not document the disk layout of an AI Runtime 8x H100 node, so take the first
    local mount that exists. The download stage checks its free space before writing anything.
    """
    if P["local_scratch_dir"]:
        return Path(P["local_scratch_dir"])
    for candidate in ("/local_disk0", "/local_nvme0"):
        if os.path.isdir(candidate):
            return Path(candidate)
    return Path(tempfile.gettempdir())


SCRATCH = pick_scratch_dir()
MODEL_DIR = SCRATCH / "glm-5.3-flash"
# Keep the Hub and Xet caches on the same local disk as the checkpoint, not the small home volume.
os.environ.setdefault("HF_HOME", str(SCRATCH / "hf_home"))


def download() -> None:
    from huggingface_hub import HfApi, snapshot_download

    entries = [e for e in HfApi().list_repo_tree(MODEL_NAME, recursive=True) if getattr(e, "size", None)]
    total = sum(e.size for e in entries)
    free = shutil.disk_usage(SCRATCH).free
    need = int(total * P["disk_headroom"])
    print(
        f"Scratch disk {SCRATCH}: {free / GIB:,.0f} GiB free. {MODEL_NAME} main branch: "
        f"{total / GIB:,.1f} GiB in {len(entries)} files. Need {need / GIB:,.0f} GiB with headroom.",
        flush=True,
    )
    if free < need:
        raise SystemExit(
            f"Not enough free space on {SCRATCH}: need {need / GIB:,.0f} GiB, have {free / GIB:,.0f} GiB. "
            "Set the local_scratch_dir parameter to a larger local disk."
        )

    start = time.time()
    snapshot_download(MODEL_NAME, local_dir=str(MODEL_DIR), max_workers=P["download_workers"])
    elapsed = time.time() - start

    missing = [
        e.path
        for e in entries
        if not (MODEL_DIR / e.path).is_file() or (MODEL_DIR / e.path).stat().st_size != e.size
    ]
    if missing:
        raise SystemExit(f"Download incomplete; missing or wrong size: {missing[:5]}")
    print(
        f"Downloaded {total / GIB:,.1f} GiB in {elapsed / 60:.1f} min "
        f"({total / elapsed / 1e6:,.0f} MB/s) to {MODEL_DIR}",
        flush=True,
    )


def load_prompts(tokenizer):
    """Build chat-templated prompt token IDs from a public instruction dataset."""
    from datasets import load_dataset

    rows = load_dataset(P["dataset_name"], split=f"train[:{P['num_prompts']}]")
    texts, token_ids = [], []
    for row in rows:
        text = row["instruction"] + (f"\n\n{row['input']}" if row["input"] else "")
        templated = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort=P["reasoning_effort"],
        )
        # The chat template already writes the special tokens, so do not add them again.
        token_ids.append(tokenizer(templated, add_special_tokens=False)["input_ids"])
        texts.append(text)
    return texts, token_ids


def engine_kwargs(num_engines: int) -> dict:
    return dict(
        model=str(MODEL_DIR),
        max_model_len=P["max_model_len"],
        gpu_memory_utilization=P["gpu_memory_utilization"],
        # Text-only batch job: skip the vision encoder's profiling and memory.
        language_model_only=True,
        max_num_seqs=P["concurrent_seqs_total"] // num_engines,
        # Warm-up batch size. vLLM's default (16384) sizes MoE activation buffers, which the DP+EP warm-up
        # pass overflows on 80 GB GPUs.
        max_num_batched_tokens=P["max_num_batched_tokens"],
        # Same MoE kernel in both layouts. The default FlashInfer CUTLASS FP8 MoE path overflowed in DP+EP.
        moe_backend=P["moe_backend"],
        seed=0,
    )


def run_tp8(token_ids: list[list[int]]) -> dict:
    from vllm import LLM, SamplingParams

    start = time.time()
    llm = LLM(**engine_kwargs(num_engines=1), tensor_parallel_size=8, enable_expert_parallel=False)
    startup_s = time.time() - start
    sampling = SamplingParams(temperature=0.0, max_tokens=P["max_new_tokens"])
    prompts = [{"prompt_token_ids": ids} for ids in token_ids]

    # Warm up so Triton autotuning and first-call allocation do not count toward throughput.
    llm.generate(prompts[:8], sampling, use_tqdm=False)
    start = time.time()
    outputs = llm.generate(prompts, sampling, use_tqdm=False)
    generate_s = time.time() - start

    rows = [
        {
            "prompt_index": i,
            "prompt_tokens": len(o.prompt_token_ids),
            "output_tokens": len(o.outputs[0].token_ids),
            "output_token_ids": list(o.outputs[0].token_ids),
            "output_text": o.outputs[0].text,
            "finish_reason": str(o.outputs[0].finish_reason),
        }
        for i, o in enumerate(outputs)
    ]
    return {
        "rows": rows,
        "generate_s": generate_s,
        "startup_s": startup_s,
        "engines": 1,
    }


def _dp_worker(rank: int, dp_size: int, port: int, indices, token_ids, barrier, out_dir: str) -> None:
    # Same environment that vLLM's data_parallel_offline.py example sets for each rank.
    os.environ.update(
        VLLM_DP_RANK=str(rank),
        VLLM_DP_RANK_LOCAL=str(rank),
        VLLM_DP_SIZE=str(dp_size),
        VLLM_DP_MASTER_IP="127.0.0.1",
        VLLM_DP_MASTER_PORT=str(port),
    )
    from vllm import LLM, SamplingParams

    result = {"rank": rank}
    try:
        start = time.time()
        llm = LLM(
            **engine_kwargs(num_engines=dp_size),
            tensor_parallel_size=1,
            enable_expert_parallel=True,
            enable_ep_weight_filter=P["ep_weight_filter"],
        )
        result["startup_s"] = time.time() - start
        sampling = SamplingParams(temperature=0.0, max_tokens=P["max_new_tokens"])
        prompts = [{"prompt_token_ids": ids} for ids in token_ids]
        llm.generate(prompts[:4], sampling, use_tqdm=False)
        # All ranks start the timed phase together; the parent waits on the same barrier.
        barrier.wait(timeout=P["startup_timeout_s"])
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        result["rows"] = [
            {
                "prompt_index": int(idx),
                "prompt_tokens": len(o.prompt_token_ids),
                "output_tokens": len(o.outputs[0].token_ids),
                "output_token_ids": list(o.outputs[0].token_ids),
                "output_text": o.outputs[0].text,
                "finish_reason": str(o.outputs[0].finish_reason),
            }
            for idx, o in zip(indices, outputs)
        ]
        result["ok"] = True
    except Exception as e:  # Report the failure to the parent instead of hanging at the barrier.
        result["ok"] = False
        result["error"] = repr(e)
        barrier.abort()
    finally:
        Path(out_dir, f"rank_{rank}.json").write_text(json.dumps(result))


def run_dp8_ep(token_ids: list[list[int]], out_dir: Path) -> dict:
    import multiprocessing as mp

    from vllm.utils.network_utils import get_open_port

    dp_size = P["data_parallel_size"]
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(dp_size + 1)
    port = get_open_port()
    # Round-robin the prompts so every rank gets a similar load.
    shards = [list(range(r, len(token_ids), dp_size)) for r in range(dp_size)]
    procs = [
        ctx.Process(
            target=_dp_worker,
            args=(r, dp_size, port, shards[r], [token_ids[i] for i in shards[r]], barrier, str(out_dir)),
        )
        for r in range(dp_size)
    ]
    start_wall = time.time()
    for p in procs:
        p.start()
    try:
        barrier.wait(timeout=P["startup_timeout_s"])
    except Exception as e:
        for p in procs:
            p.terminate()
        raise SystemExit(f"DP ranks failed before the timed phase: {e!r}") from e
    generate_start = time.time()
    for p in procs:
        p.join()
    generate_s = time.time() - generate_start

    rows, startups = [], []
    for r in range(dp_size):
        shard = json.loads((out_dir / f"rank_{r}.json").read_text())
        if not shard.get("ok"):
            raise SystemExit(f"DP rank {r} failed: {shard.get('error')}")
        rows.extend(shard["rows"])
        startups.append(shard["startup_s"])
        (out_dir / f"rank_{r}.json").unlink()
    print(
        f"DP+EP wall time from barrier to last rank: {generate_s:.1f}s "
        f"(start to end {time.time() - start_wall:.0f}s)"
    )
    rows.sort(key=lambda row: row["prompt_index"])
    return {
        "rows": rows,
        "generate_s": generate_s,
        "startup_s": max(startups),
        "engines": dp_size,
    }


def _log_mlflow(metrics: dict) -> None:
    # Log to the AIR-provided run. A logging failure fails the workload.
    import mlflow

    with mlflow.start_run(run_id=os.environ["MLFLOW_RUN_ID"]):
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})


def generate(layout: str) -> None:
    from transformers import AutoTokenizer

    from datasets import Dataset

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    texts, token_ids = load_prompts(tokenizer)
    print(f"{layout}: {len(token_ids)} prompts, max {P['max_new_tokens']} new tokens", flush=True)

    out_dir = Path(P["output_root"]) / os.environ.get("MLFLOW_RUN_ID", "local") / layout
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_tp8(token_ids) if layout == "tp8" else run_dp8_ep(token_ids, out_dir)

    rows = result["rows"]
    for row in rows:
        row["layout"] = layout
        row["instruction"] = texts[row["prompt_index"]]
    Dataset.from_list(rows).to_parquet(str(out_dir / "generations.parquet"))

    output_tokens = sum(r["output_tokens"] for r in rows)
    metrics = {
        "num_prompts": len(rows),
        "output_tokens": output_tokens,
        "generate_seconds": result["generate_s"],
        "output_tokens_per_second": output_tokens / result["generate_s"],
        "startup_seconds": result["startup_s"],
        "engines": result["engines"],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"{layout}: {json.dumps(metrics, indent=2)}", flush=True)
    _log_mlflow({f"{layout}/{k}": v for k, v in metrics.items()})


def _divergence_stats(tp_ids: list[list[int]], ep_ids: list[list[int]]) -> dict:
    """Token-level agreement between the two layouts' greedy outputs.

    For each prompt, count how many leading tokens the two outputs share. Bit-identical
    outputs are not expected across layouts: each layout splits every GEMM differently,
    so floating-point reduction orders differ, and small logit differences flip greedy
    token choices that then compound.
    """
    prefix_fracs = []
    identical = 0
    for x, y in zip(tp_ids, ep_ids):
        n = min(len(x), len(y))
        p = 0
        while p < n and x[p] == y[p]:
            p += 1
        if p == n and len(x) == len(y):
            identical += 1
            prefix_fracs.append(1.0)
        else:
            prefix_fracs.append(p / n if n else 0.0)
    diverging = [f for f in prefix_fracs if f < 1.0]
    prefix_fracs.sort()
    n = len(prefix_fracs)
    return {
        "prompts": n,
        "identical_fraction": identical / n if n else 1.0,
        "mean_prefix_fraction": sum(prefix_fracs) / n if n else 1.0,
        "median_prefix_fraction": prefix_fracs[n // 2] if n else 1.0,
        "diverging_prompts": len(diverging),
        "mean_prefix_fraction_diverging": sum(diverging) / len(diverging) if diverging else 1.0,
        "median_prefix_fraction_diverging": sorted(diverging)[len(diverging) // 2] if diverging else 1.0,
    }


def summary() -> None:
    run_dir = Path(P["output_root"]) / os.environ.get("MLFLOW_RUN_ID", "local")
    metrics = {layout: json.loads((run_dir / layout / "metrics.json").read_text()) for layout in ("tp8", "dp8_ep")}
    print("\nlayout   output tok/s   generate s   startup s   engines")
    for layout, m in metrics.items():
        print(
            f"{layout:8s} {m['output_tokens_per_second']:12,.0f} {m['generate_seconds']:12,.1f} "
            f"{m['startup_seconds']:11,.0f} {m['engines']:9d}"
        )
    print("KV-cache size: read the 'GPU KV cache size' line of each engine in the run log.")

    # Greedy decoding: compare the two layouts' outputs token by token.
    from datasets import Dataset

    a = Dataset.from_parquet(str(run_dir / "tp8" / "generations.parquet")).sort("prompt_index")
    b = Dataset.from_parquet(str(run_dir / "dp8_ep" / "generations.parquet")).sort("prompt_index")
    stats = _divergence_stats(a["output_token_ids"], b["output_token_ids"])
    print(
        f"Greedy outputs identical across layouts: {stats['identical_fraction']:.1%} "
        f"({stats['prompts']} prompts)."
    )
    print(
        f"The other {stats['diverging_prompts']} prompts diverge at a median of "
        f"{stats['median_prefix_fraction_diverging']:.0%} of the shorter output "
        f"(mean {stats['mean_prefix_fraction_diverging']:.0%})."
    )
    print(
        f"Output token totals: tp8 {metrics['tp8']['output_tokens']:,}, "
        f"dp8_ep {metrics['dp8_ep']['output_tokens']:,} "
        f"(difference {abs(metrics['tp8']['output_tokens'] - metrics['dp8_ep']['output_tokens']) / metrics['tp8']['output_tokens']:.1%})."
    )
    _log_mlflow({f"divergence/{k}": v for k, v in stats.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("download")
    gen = sub.add_parser("generate")
    gen.add_argument("--layout", choices=["tp8", "dp8_ep"], required=True)
    sub.add_parser("summary")
    args = parser.parse_args()

    if args.stage == "download":
        download()
    elif args.stage == "generate":
        generate(args.layout)
    else:
        summary()

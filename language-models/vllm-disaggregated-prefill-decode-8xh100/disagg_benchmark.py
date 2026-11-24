"""Compare colocated and disaggregated vLLM serving on one 8x H100 node.

Both layouts use the same 8 GPUs and the same open-loop request stream: Poisson
arrivals, a fixed share of long prompts, and fixed-length greedy outputs. For
each layout this script starts the vLLM instances, puts disagg_router.py in
front of them, and records per-request TTFT and inter-token latency (ITL). Raw
per-request results go to the output volume, and the summaries go to the AI
Runtime MLflow run.
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import aiohttp
import mlflow
import numpy as np
import yaml

NUM_GPUS = 8
ROUTER_PORT = 8000
COMPLETIONS_PATH = "/v1/completions"
BOOTSTRAP_REPS = 200
HERE = Path(__file__).resolve().parent
ROUTER_SCRIPT = HERE / "disagg_router.py"
VLLM_BIN = os.path.join(os.path.dirname(sys.executable), "vllm")

with open(os.environ["HYPERPARAMETERS_PATH"]) as f:
    RAW = yaml.safe_load(f)


def load_config():
    c = {
        "model": str(RAW["model_name"]),
        "layouts": [s.strip() for s in str(RAW["layouts"]).split(",") if s.strip()],
        "colocated_instances": int(RAW["colocated_instances"]),
        "prefill_instances": int(RAW["prefill_instances"]),
        "decode_instances": int(RAW["decode_instances"]),
        "arrival_rate": float(RAW["arrival_rate"]),
        "duration_s": float(RAW["duration_s"]),
        "warmup_s": float(RAW["warmup_s"]),
        "drain_timeout_s": float(RAW["drain_timeout_s"]),
        "long_fraction": float(RAW["long_fraction"]),
        "short_prompt_tokens": int(RAW["short_prompt_tokens"]),
        "long_prompt_tokens": int(RAW["long_prompt_tokens"]),
        "max_output_tokens": int(RAW["max_output_tokens"]),
        "max_model_len": int(RAW["max_model_len"]),
        "gpu_memory_utilization": float(RAW["gpu_memory_utilization"]),
        "max_num_batched_tokens": int(RAW["max_num_batched_tokens"]),
        "max_num_seqs": int(RAW["max_num_seqs"]),
        "startup_timeout_s": float(RAW["startup_timeout_s"]),
        "seed": int(RAW["seed"]),
        "output_root": str(RAW["output_root"]),
    }
    if not c["layouts"] or any(x not in ("colocated", "disagg") for x in c["layouts"]):
        raise ValueError(f"layouts must be a subset of colocated,disagg: {c['layouts']}")
    if c["colocated_instances"] != NUM_GPUS:
        raise ValueError("colocated_instances must equal the 8 GPUs on the node (one TP=1 instance per GPU)")
    if c["prefill_instances"] < 1 or c["decode_instances"] < 1:
        raise ValueError("disagg needs at least one prefill and one decode instance")
    if c["prefill_instances"] + c["decode_instances"] != NUM_GPUS:
        raise ValueError("prefill_instances + decode_instances must equal 8")
    if not 0.0 <= c["long_fraction"] <= 1.0:
        raise ValueError("long_fraction must be in [0, 1]")
    if c["long_prompt_tokens"] + c["max_output_tokens"] > c["max_model_len"]:
        raise ValueError("long prompt plus output must fit in max_model_len")
    if not 0.0 <= c["warmup_s"] < c["duration_s"]:
        raise ValueError("warmup_s must be in [0, duration_s)")
    return c


def instance_specs(layout, c):
    if layout == "colocated":
        return [
            {"name": f"colocated-{i}", "role": "colocated", "gpu": i, "port": 8100 + i, "side_channel": None, "kv": None}
            for i in range(c["colocated_instances"])
        ]
    specs = []
    for j in range(c["prefill_instances"]):
        specs.append(
            {
                "name": f"prefill-{j}",
                "role": "prefill",
                "gpu": j,
                "port": 8100 + j,
                "side_channel": 5600 + j,
                "kv": {"kv_connector": "NixlConnector", "kv_role": "kv_producer", "kv_load_failure_policy": "fail"},
            }
        )
    for k in range(c["decode_instances"]):
        specs.append(
            {
                "name": f"decode-{k}",
                "role": "decode",
                "gpu": c["prefill_instances"] + k,
                "port": 8200 + k,
                "side_channel": 5610 + k,
                "kv": {"kv_connector": "NixlConnector", "kv_role": "kv_consumer", "kv_load_failure_policy": "fail"},
            }
        )
    return specs


def vllm_cmd(spec, c):
    cmd = [
        VLLM_BIN if os.path.exists(VLLM_BIN) else "vllm",
        "serve",
        c["model"],
        "--served-model-name",
        c["model"],
        "--host",
        "127.0.0.1",
        "--port",
        str(spec["port"]),
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        str(c["max_model_len"]),
        "--gpu-memory-utilization",
        str(c["gpu_memory_utilization"]),
        "--max-num-batched-tokens",
        str(c["max_num_batched_tokens"]),
        "--max-num-seqs",
        str(c["max_num_seqs"]),
    ]
    if spec["kv"] is not None:
        cmd += ["--kv-transfer-config", json.dumps(spec["kv"])]
    return cmd


def spawn(cmd, log_path, env):
    log = open(log_path, "wb")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    return proc, log


def _signal_group(proc, sig):
    # Signal the whole group even when the parent has exited, because vLLM's engine-core children can outlive it
    # and keep GPU memory. The group may already be gone, which is fine.
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        pass


def stop(procs):
    for proc, _ in procs:
        _signal_group(proc, signal.SIGTERM)
    deadline = time.monotonic() + 60
    for proc, _ in procs:
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(1)
    for proc, _ in procs:
        _signal_group(proc, signal.SIGKILL)
        proc.wait()
    for _, log in procs:
        log.close()
    wait_gpus_free()


def wait_gpus_free(timeout_s=180):
    # The next layout needs all 8 GPUs, and leftover memory from this one would make it fail for a confusing reason.
    deadline = time.monotonic() + timeout_s
    while True:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        used = [int(x) for x in out.split()]
        if max(used) < 500:
            return
        if time.monotonic() > deadline:
            raise RuntimeError(f"GPU memory still in use after teardown: {used} MiB")
        time.sleep(5)


async def wait_ready(targets, timeout_s, label):
    """targets maps a health URL to (name, process). Fails fast if a process exits."""
    deadline = time.monotonic() + timeout_s
    pending = dict(targets)
    async with aiohttp.ClientSession() as session:
        while pending:
            for url, (name, proc) in list(pending.items()):
                if proc.poll() is not None:
                    raise RuntimeError(f"{label}: {name} exited with code {proc.returncode}; see its log in the run directory")
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        if resp.status == 200:
                            del pending[url]
                            print(f"{label}: {name} ready", flush=True)
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    pass
            if pending and time.monotonic() > deadline:
                names = sorted(name for name, _ in pending.values())
                raise TimeoutError(f"{label}: not ready after {timeout_s} s: {names}")
            if pending:
                await asyncio.sleep(5)


def make_schedule(c):
    """Poisson arrivals with a seeded prompt-length mix. Both layouts get the same list."""
    rng = np.random.default_rng(c["seed"])
    items, t = [], 0.0
    while True:
        t += rng.exponential(1.0 / c["arrival_rate"])
        if t >= c["duration_s"]:
            break
        is_long = bool(rng.random() < c["long_fraction"])
        items.append(
            {
                "index": len(items),
                "arrival_s": float(t),
                "class": "long" if is_long else "short",
                "prompt_tokens": c["long_prompt_tokens"] if is_long else c["short_prompt_tokens"],
            }
        )
    return items


def warmup_items(c):
    # Not measured. The first request through the NIXL path also sets up the UCX connections.
    return [
        {
            "index": 10**6 + k,
            "arrival_s": None,
            "class": cls,
            "prompt_tokens": c["long_prompt_tokens"] if cls == "long" else c["short_prompt_tokens"],
        }
        for k, cls in enumerate(["short", "long"] * 4)
    ]


def prompt_token_ids(item, c):
    # Seeded per request, so both layouts see identical prompts. Random ids avoid prefix-cache hits.
    rng = np.random.default_rng([c["seed"], item["index"]])
    return rng.integers(1000, 150000, size=item["prompt_tokens"]).tolist()


async def send_one(session, item, c, t0):
    body = {
        "model": c["model"],
        "prompt": prompt_token_ids(item, c),
        "max_tokens": c["max_output_tokens"],
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    base = {"index": item["index"], "class": item["class"], "arrival_s": item["arrival_s"]}
    t_send = time.perf_counter()
    token_times, usage_tokens = [], None
    try:
        async with session.post(f"http://127.0.0.1:{ROUTER_PORT}{COMPLETIONS_PATH}", json=body) as resp:
            if resp.status != 200:
                text = (await resp.text())[:300]
                return {**base, "error": f"HTTP {resp.status}: {text}"}
            async for line in resp.content:
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    break
                event = json.loads(payload)
                if event.get("usage"):
                    usage_tokens = event["usage"]["completion_tokens"]
                if event.get("choices"):
                    token_times.append(time.perf_counter())
    except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as exc:
        return {**base, "error": f"{type(exc).__name__}: {exc}"}

    n_tokens = usage_tokens if usage_tokens is not None else len(token_times)
    if not token_times:
        return {**base, "error": "stream had no tokens"}
    if n_tokens != c["max_output_tokens"]:
        return {**base, "error": f"expected {c['max_output_tokens']} output tokens, got {n_tokens}"}
    itl_ms = np.diff(np.asarray(token_times)) * 1e3
    return {
        **base,
        "start_s": t_send - t0,
        "end_s": token_times[-1] - t0,
        "ttft_ms": (token_times[0] - t_send) * 1e3,
        "e2e_ms": (token_times[-1] - t_send) * 1e3,
        "n_tokens": n_tokens,
        "itl_ms": itl_ms.tolist(),
        "error": None,
    }


async def drive(schedule, warm, c):
    connector = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=600)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        warm_results = await asyncio.gather(*[send_one(session, w, c, time.perf_counter()) for w in warm])
        failed = [r["error"] for r in warm_results if r["error"]]
        if failed:
            raise RuntimeError(f"warm-up requests failed: {failed[:3]}")

        t0 = time.perf_counter()
        tasks = []
        for item in schedule:
            wait = t0 + item["arrival_s"] - time.perf_counter()
            if wait > 0:
                await asyncio.sleep(wait)
            tasks.append(asyncio.create_task(send_one(session, item, c, t0)))

        _, pending = await asyncio.wait(tasks, timeout=c["drain_timeout_s"])
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    results = []
    for item, task in zip(schedule, tasks):
        if task.done() and not task.cancelled() and task.exception() is None:
            results.append(task.result())
        else:
            results.append(
                {
                    "index": item["index"],
                    "class": item["class"],
                    "arrival_s": item["arrival_s"],
                    "error": "not finished within drain_timeout_s",
                }
            )
    return results


def _stats(ttft_ms, itl_arrays):
    ttft = np.asarray(ttft_ms, dtype=float)
    itl = np.concatenate(itl_arrays)
    return {
        "ttft_ms_p50": float(np.percentile(ttft, 50)),
        "ttft_ms_p99": float(np.percentile(ttft, 99)),
        "itl_ms_p50": float(np.percentile(itl, 50)),
        "itl_ms_p99": float(np.percentile(itl, 99)),
    }


def bootstrap_se(ttft_ms, itl_arrays, seed):
    """Standard error of each statistic, resampling requests (the unit of independence)."""
    rng = np.random.default_rng(seed)
    n = len(ttft_ms)
    draws = []
    for _ in range(BOOTSTRAP_REPS):
        idx = rng.integers(0, n, n)
        draws.append(_stats([ttft_ms[i] for i in idx], [itl_arrays[i] for i in idx]))
    return {f"{k}_se": float(np.std([d[k] for d in draws], ddof=1)) for k in draws[0]}


def summarize(records, c, router_stats):
    measured = [r for r in records if r["arrival_s"] >= c["warmup_s"]]
    ok = [r for r in measured if r["error"] is None]
    failed = [r["error"] for r in measured if r["error"] is not None]
    if not ok:
        raise RuntimeError(f"no measured request succeeded; first error: {failed[:1]}")
    # Output throughput runs from the first measured arrival to the last completion. The request rate uses the
    # arrival window, so a slow tail does not make a layout look overloaded.
    window_s = max(r["end_s"] for r in ok) - c["warmup_s"]
    arrival_window_s = c["duration_s"] - c["warmup_s"]
    ttft = [r["ttft_ms"] for r in ok]
    itl = [np.asarray(r["itl_ms"], dtype=float) for r in ok]
    achieved_rate = len(ok) / arrival_window_s
    summary = {
        "requests_measured": len(measured),
        "requests_ok": len(ok),
        "requests_failed": len(failed),
        "first_error": failed[0] if failed else None,
        "offered_rate_rps": c["arrival_rate"],
        "achieved_rate_rps": achieved_rate,
        "measured_window_s": window_s,
        "output_tokens_per_s": sum(r["n_tokens"] for r in ok) / window_s,
        "router_missing_kv_params": router_stats["missing_kv_params"],
    }
    summary["keeps_up"] = bool(achieved_rate >= 0.95 * c["arrival_rate"])
    summary.update(_stats(ttft, itl))
    summary.update(bootstrap_se(ttft, itl, c["seed"]))

    for cls in ("short", "long"):
        idx = [i for i, r in enumerate(ok) if r["class"] == cls]
        if idx:
            summary[f"{cls}_ttft_ms_p99"] = float(np.percentile([ttft[i] for i in idx], 99))
            summary[f"{cls}_itl_ms_p99"] = float(np.percentile(np.concatenate([itl[i] for i in idx]), 99))
    summary["e2e_ms_p50"] = float(np.percentile([r["e2e_ms"] for r in ok], 50))
    summary["e2e_ms_p99"] = float(np.percentile([r["e2e_ms"] for r in ok], 99))
    return summary


async def run_layout(layout, c, run_dir, schedule, warm):
    layout_dir = run_dir / layout
    logs_dir = layout_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    specs = instance_specs(layout, c)
    mode = "colocated" if layout == "colocated" else "disagg"
    procs = []
    try:
        targets = {}
        for spec in specs:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(spec["gpu"]))
            if spec["side_channel"] is not None:
                env["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(spec["side_channel"])
            proc, log = spawn(vllm_cmd(spec, c), logs_dir / f"{spec['name']}.log", env)
            procs.append((proc, log))
            targets[f"http://127.0.0.1:{spec['port']}/health"] = (spec["name"], proc)
        await wait_ready(targets, c["startup_timeout_s"], layout)

        if mode == "colocated":
            router_args = ["--instances"] + [f"127.0.0.1:{s['port']}" for s in specs]
        else:
            router_args = ["--prefill"] + [f"127.0.0.1:{s['port']}" for s in specs if s["role"] == "prefill"]
            router_args += ["--decode"] + [f"127.0.0.1:{s['port']}" for s in specs if s["role"] == "decode"]
        router_cmd = [sys.executable, str(ROUTER_SCRIPT), "--mode", mode, "--port", str(ROUTER_PORT), *router_args]
        router_proc, router_log = spawn(router_cmd, logs_dir / "router.log", dict(os.environ))
        procs.append((router_proc, router_log))
        await wait_ready({f"http://127.0.0.1:{ROUTER_PORT}/healthz": ("router", router_proc)}, 120, layout)

        records = await drive(schedule, warm, c)
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://127.0.0.1:{ROUTER_PORT}/healthz") as resp:
                router_stats = await resp.json()
    finally:
        stop(procs)

    # Failures in the warm-up window are excluded from the metrics, but they still mean the layout is broken.
    all_errors = [r["error"] for r in records if r["error"] is not None]
    if all_errors:
        raise RuntimeError(f"{layout}: {len(all_errors)} requests failed; first: {all_errors[0]}")
    summary = summarize(records, c, router_stats)
    with open(layout_dir / "requests.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    with open(layout_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"{layout}: " + json.dumps({k: v for k, v in summary.items() if not k.endswith("_se")}, indent=2), flush=True)
    return summary


def build_comparison(summaries):
    if "colocated" not in summaries or "disagg" not in summaries:
        return {}
    a, b = summaries["colocated"], summaries["disagg"]
    out = {}
    for key in ("ttft_ms_p99", "itl_ms_p99", "ttft_ms_p50", "itl_ms_p50", "output_tokens_per_s"):
        out[f"disagg_minus_colocated/{key}"] = b[key] - a[key]
        if f"{key}_se" in a:
            out[f"disagg_minus_colocated/{key}_se"] = float(np.hypot(a[f"{key}_se"], b[f"{key}_se"]))
    return out


def log_to_mlflow(run_id, c, summaries, comparison):
    params = {
        "model": c["model"],
        "layouts": ",".join(c["layouts"]),
        "colocated_instances": c["colocated_instances"],
        "prefill_instances": c["prefill_instances"],
        "decode_instances": c["decode_instances"],
        "arrival_rate_rps": c["arrival_rate"],
        "duration_s": c["duration_s"],
        "warmup_s": c["warmup_s"],
        "long_fraction": c["long_fraction"],
        "short_prompt_tokens": c["short_prompt_tokens"],
        "long_prompt_tokens": c["long_prompt_tokens"],
        "max_output_tokens": c["max_output_tokens"],
        "seed": c["seed"],
    }
    metrics = {}
    for layout, summary in summaries.items():
        for key, value in summary.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            metrics[f"{layout}/{key}"] = float(value)
    metrics.update({k: float(v) for k, v in comparison.items()})
    with mlflow.start_run(run_id=run_id):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)


async def main():
    c = load_config()
    run_id = os.environ["MLFLOW_RUN_ID"]
    run_dir = Path(c["output_root"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    schedule = make_schedule(c)
    mix = Counter(item["class"] for item in schedule)
    print(
        f"Schedule: {len(schedule)} requests over {c['duration_s']} s at {c['arrival_rate']} req/s; "
        f"classes: {dict(mix)}; prompt tokens short={c['short_prompt_tokens']} long={c['long_prompt_tokens']}; "
        f"output tokens={c['max_output_tokens']}",
        flush=True,
    )
    with open(run_dir / "config.json", "w") as f:
        json.dump({**c, "schedule_class_counts": dict(mix)}, f, indent=2)

    warm = warmup_items(c)
    summaries = {}
    for layout in c["layouts"]:
        print(f"=== layout: {layout}", flush=True)
        summaries[layout] = await run_layout(layout, c, run_dir, schedule, warm)

    comparison = build_comparison(summaries)
    with open(run_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    log_to_mlflow(run_id, c, summaries, comparison)
    print("Logged to MLflow run", run_id, "; results in", run_dir, flush=True)


if __name__ == "__main__":
    asyncio.run(main())

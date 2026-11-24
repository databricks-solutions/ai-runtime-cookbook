"""Compare prefix caching and CPU KV offload in vLLM 0.34 on one H100.

One seeded request stream runs through four vLLM servers, each started in a fresh process:

  prefix_off          prefix caching disabled
  prefix_gpu          prefix caching on; cached blocks live only in the GPU KV cache
  prefix_cpu_offload  prefix caching on, plus the OffloadingConnector CPU tier
  prefix_off_repeat   same settings as prefix_off, run last as a control for numerical noise

The GPU KV budget is fixed and holds only part of the document prefixes, so the GPU-only cache cannot keep them
all. Decoding is greedy. The run reports each configuration's output match rate against prefix_off; the control's
match rate shows how much prefix_off disagrees with itself.
"""

import asyncio
import csv
import json
import os
import re
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

PORT = 8100
BASE_URL = f"http://127.0.0.1:{PORT}"
CONFIG_NAMES = ["prefix_off", "prefix_gpu", "prefix_cpu_offload", "prefix_off_repeat"]
BOOTSTRAP_REPS = 200
VOCAB_LO, VOCAB_HI = 1000, 150000
GIB = 2**30
VLLM_BIN = os.path.join(os.path.dirname(sys.executable), "vllm")
COUNTER_NAMES = [
    "vllm:prefix_cache_queries",
    "vllm:prefix_cache_hits",
    "vllm:external_prefix_cache_queries",
    "vllm:external_prefix_cache_hits",
]

with open(os.environ["HYPERPARAMETERS_PATH"]) as f:
    RAW = yaml.safe_load(f)


def load_config():
    c = {
        "model": str(RAW["model_name"]),
        "num_documents": int(RAW["num_documents"]),
        "doc_tokens": int(RAW["doc_tokens"]),
        "question_tokens": int(RAW["question_tokens"]),
        "num_requests": int(RAW["num_requests"]),
        "max_output_tokens": int(RAW["max_output_tokens"]),
        "max_model_len": int(RAW["max_model_len"]),
        "gpu_kv_gib": float(RAW["gpu_kv_gib"]),
        "cpu_offload_cap_gib": float(RAW["cpu_offload_cap_gib"]),
        "cpu_offload_host_fraction": float(RAW["cpu_offload_host_fraction"]),
        "concurrency": int(RAW["concurrency"]),
        "max_num_seqs": int(RAW["max_num_seqs"]),
        "startup_timeout_s": float(RAW["startup_timeout_s"]),
        "seed": int(RAW["seed"]),
        "output_root": str(RAW["output_root"]),
    }
    if c["doc_tokens"] % 64 != 0:
        raise ValueError("doc_tokens must be a multiple of the 64-token KV block size")
    if c["doc_tokens"] + c["question_tokens"] + c["max_output_tokens"] > c["max_model_len"]:
        raise ValueError("doc_tokens + question_tokens + max_output_tokens must fit in max_model_len")
    if c["num_documents"] < 2 or c["num_requests"] < c["num_documents"]:
        raise ValueError("need at least 2 documents and at least one request per document on average")
    if not 0.0 < c["cpu_offload_host_fraction"] <= 1.0:
        raise ValueError("cpu_offload_host_fraction must be in (0, 1]")
    return c


def read_int(path):
    try:
        text = Path(path).read_text().strip()
    except OSError:
        return None
    return int(text) if text.isdigit() else None


def host_memory_gib():
    """Usable host RAM. /proc/meminfo shows the whole host, so the container's cgroup limit also applies."""
    fields = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in ("MemTotal", "MemAvailable"):
            fields[key] = int(value.split()[0]) / 2**20
    cg_limit = read_int("/sys/fs/cgroup/memory.max")
    cg_current = read_int("/sys/fs/cgroup/memory.current")
    cgroup_free = None
    if cg_limit is not None and cg_current is not None:
        cgroup_free = (cg_limit - cg_current) / GIB
    usable = fields["MemAvailable"] if cgroup_free is None else min(fields["MemAvailable"], cgroup_free)
    return {
        "mem_total_gib": fields["MemTotal"],
        "mem_available_gib": fields["MemAvailable"],
        "cgroup_free_gib": cgroup_free,
        "usable_gib": usable,
    }


def make_requests(c):
    """Seeded documents and request order. Documents repeat, so the same prefix comes back after other traffic."""
    rng = np.random.default_rng(c["seed"])
    doc_ids = rng.integers(0, c["num_documents"], size=c["num_requests"]).tolist()
    docs = [
        np.random.default_rng([c["seed"], 1, d]).integers(VOCAB_LO, VOCAB_HI, size=c["doc_tokens"]).tolist()
        for d in range(c["num_documents"])
    ]
    requests = []
    for i, d in enumerate(doc_ids):
        question = np.random.default_rng([c["seed"], 2, i]).integers(VOCAB_LO, VOCAB_HI, size=c["question_tokens"])
        requests.append(
            {"index": i, "doc": d, "prompt": docs[d] + question.tolist(), "max_tokens": c["max_output_tokens"]}
        )
    counts = Counter(doc_ids)
    per_doc = [counts.get(d, 0) for d in range(c["num_documents"])]
    print(
        f"Request stream: {len(requests)} requests over {c['num_documents']} documents; visits per document "
        f"min={min(per_doc)} median={int(np.median(per_doc))} max={max(per_doc)}; "
        f"prompt tokens={c['doc_tokens'] + c['question_tokens']}, output tokens={c['max_output_tokens']}",
        flush=True,
    )
    return requests


def server_cmd(c, name, cpu_bytes):
    cmd = [
        VLLM_BIN if os.path.exists(VLLM_BIN) else "vllm",
        "serve",
        c["model"],
        "--served-model-name",
        c["model"],
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        str(c["max_model_len"]),
        # An absolute KV budget. It is fixed across configurations, and it overrides gpu_memory_utilization.
        "--kv-cache-memory-bytes",
        str(int(c["gpu_kv_gib"] * GIB)),
        "--max-num-seqs",
        str(c["max_num_seqs"]),
        "--no-enable-prefix-caching" if name.startswith("prefix_off") else "--enable-prefix-caching",
    ]
    if name == "prefix_cpu_offload":
        kv_transfer = {
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {"cpu_bytes_to_use": cpu_bytes},
        }
        cmd += ["--kv-transfer-config", json.dumps(kv_transfer)]
    return cmd


def start_server(cmd, log_path):
    log = open(log_path, "wb")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy(), start_new_session=True)
    return proc, log


def _signal_group(proc, sig):
    # vLLM's engine-core child can outlive the parent and keep GPU memory, so signal the whole group.
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        pass


def stop_server(proc, log):
    _signal_group(proc, signal.SIGTERM)
    deadline = time.monotonic() + 60
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(1)
    _signal_group(proc, signal.SIGKILL)
    proc.wait()
    log.close()
    wait_gpu_free()


def wait_gpu_free(timeout_s=180):
    # The next server needs the GPU, and leftover memory would make it fail for a confusing reason.
    deadline = time.monotonic() + timeout_s
    while True:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        used = int(out.split()[0])
        if used < 500:
            return
        if time.monotonic() > deadline:
            raise RuntimeError(f"GPU memory still in use after teardown: {used} MiB")
        time.sleep(5)


async def wait_ready(proc, log_path, timeout_s):
    deadline = time.monotonic() + timeout_s
    async with aiohttp.ClientSession() as session:
        while True:
            if proc.poll() is not None:
                tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-60:])
                raise RuntimeError(f"vLLM exited with code {proc.returncode}. Last log lines:\n{tail}")
            try:
                async with session.get(f"{BASE_URL}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        return
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass
            if time.monotonic() > deadline:
                raise TimeoutError(f"vLLM not ready after {timeout_s} s")
            await asyncio.sleep(5)


async def fetch_counters(session):
    async with session.get(f"{BASE_URL}/metrics") as resp:
        resp.raise_for_status()
        text = await resp.text()
    totals = {}
    for name in COUNTER_NAMES:
        # prometheus_client writes counters with a _total suffix; the labelled series are summed over engines.
        pattern = re.compile(rf"^{re.escape(name)}(?:_total)?(?:\{{[^}}]*\}})?\s+(\S+)$", re.MULTILINE)
        values = [float(v) for v in pattern.findall(text)]
        totals[name] = sum(values) if values else None
    offload = [line for line in text.splitlines() if line.startswith("vllm:kv_offload_config_info")]
    totals["kv_offload_config_info"] = offload[0] if offload else None
    return totals


async def send(session, sem, c, req, t0):
    body = {
        "model": c["model"],
        "prompt": req["prompt"],
        "max_tokens": req["max_tokens"],
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    base = {"index": req["index"], "doc": req.get("doc")}
    async with sem:
        t_send = time.perf_counter()
        first_token = None
        pieces, usage_tokens, n_events = [], None, 0
        try:
            async with session.post(f"{BASE_URL}/v1/completions", json=body) as resp:
                if resp.status != 200:
                    return {**base, "error": f"HTTP {resp.status}: {(await resp.text())[:300]}"}
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
                        n_events += 1
                        if first_token is None:
                            first_token = time.perf_counter()
                        pieces.append(event["choices"][0].get("text") or "")
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as exc:
            return {**base, "error": f"{type(exc).__name__}: {exc}"}
    t_end = time.perf_counter()
    n_tokens = usage_tokens if usage_tokens is not None else n_events
    if first_token is None:
        return {**base, "error": "stream had no tokens"}
    if n_tokens != req["max_tokens"]:
        return {**base, "error": f"expected {req['max_tokens']} output tokens, got {n_tokens}"}
    return {
        **base,
        "ttft_s": first_token - t_send,
        "end_s": t_end - t0,
        "e2e_s": t_end - t_send,
        "n_tokens": n_tokens,
        "text": "".join(pieces),
        "error": None,
    }


async def run_stream(session, c, requests):
    sem = asyncio.Semaphore(c["concurrency"])
    t0 = time.perf_counter()
    results = await asyncio.gather(*[send(session, sem, c, r, t0) for r in requests])
    return results, time.perf_counter() - t0


def bootstrap_se(values, stat, seed):
    """Standard error of a statistic, resampling requests (the unit of independence)."""
    rng = np.random.default_rng(seed)
    n = len(values)
    draws = [stat(values[rng.integers(0, n, n)]) for _ in range(BOOTSTRAP_REPS)]
    return float(np.std(draws, ddof=1))


def summarize(name, results, wall_s, before, after, seed):
    errors = [r["error"] for r in results if r["error"]]
    if errors:
        raise RuntimeError(f"{name}: {len(errors)} requests failed; first error: {errors[0]}")
    ttft_ms = np.asarray([r["ttft_s"] * 1e3 for r in results])
    e2e_ms = np.asarray([r["e2e_s"] * 1e3 for r in results])
    out_tokens = sum(r["n_tokens"] for r in results)

    def delta(counter):
        if before[counter] is None or after[counter] is None:
            return None
        return after[counter] - before[counter]

    gpu_q = delta("vllm:prefix_cache_queries")
    gpu_h = delta("vllm:prefix_cache_hits")
    ext_q = delta("vllm:external_prefix_cache_queries")
    ext_h = delta("vllm:external_prefix_cache_hits")
    if not name.startswith("prefix_off") and not gpu_q:
        raise RuntimeError(f"{name}: prefix-cache query counter missing or zero; metrics are not usable")
    return {
        "requests": len(results),
        "output_tokens": out_tokens,
        "wall_s": wall_s,
        "output_tokens_per_s": out_tokens / wall_s,
        "requests_per_s": len(results) / wall_s,
        "ttft_mean_ms": float(ttft_ms.mean()),
        "ttft_mean_ms_se": bootstrap_se(ttft_ms, np.mean, seed),
        "ttft_median_ms": float(np.median(ttft_ms)),
        "ttft_median_ms_se": bootstrap_se(ttft_ms, np.median, seed),
        "e2e_median_ms": float(np.median(e2e_ms)),
        "gpu_prefix_hit_rate": (gpu_h / gpu_q) if gpu_q else None,
        "external_prefix_hit_rate": (ext_h / ext_q) if ext_q else None,
        "prefix_cache_queries": gpu_q,
        "prefix_cache_hits": gpu_h,
        "external_prefix_cache_queries": ext_q,
        "external_prefix_cache_hits": ext_h,
        "kv_offload_config_info": after["kv_offload_config_info"],
    }


def gpu_kv_tokens_from_log(log_path):
    match = re.search(r"GPU KV cache size: ([\d,]+) tokens", log_path.read_text(errors="replace"))
    return int(match.group(1).replace(",", "")) if match else None


async def run_config(c, name, cpu_bytes, run_dir):
    log_path = run_dir / f"{name}_server.log"
    proc, log = start_server(server_cmd(c, name, cpu_bytes), log_path)
    try:
        await wait_ready(proc, log_path, c["startup_timeout_s"])
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=0),
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=600),
        ) as session:
            # Unrelated random prompts warm up the engine and the kernels. They are not timed, and they don't use
            # document prefixes.
            rng = np.random.default_rng([c["seed"], 3])
            warm = [
                {"index": -1 - k, "prompt": rng.integers(VOCAB_LO, VOCAB_HI, size=64).tolist(), "max_tokens": 8}
                for k in range(4)
            ]
            warm_results = await asyncio.gather(*[send(session, asyncio.Semaphore(4), c, w, 0.0) for w in warm])
            failed = [r["error"] for r in warm_results if r["error"]]
            if failed:
                raise RuntimeError(f"warm-up failed: {failed[:2]}")

            requests = make_requests(c)
            before = await fetch_counters(session)
            results, wall_s = await run_stream(session, c, requests)
            after = await fetch_counters(session)
    finally:
        stop_server(proc, log)

    summary = summarize(name, results, wall_s, before, after, c["seed"])
    summary["gpu_kv_cache_tokens"] = gpu_kv_tokens_from_log(log_path)
    outputs = [r["text"] for r in sorted(results, key=lambda r: r["index"])]
    print(
        f"{name}: {summary['output_tokens_per_s']:.1f} output tok/s, wall {wall_s:.1f} s, "
        f"TTFT median {summary['ttft_median_ms']:.0f} ms, GPU prefix hit rate {summary['gpu_prefix_hit_rate']}, "
        f"external hit rate {summary['external_prefix_hit_rate']}",
        flush=True,
    )
    return summary, outputs


def write_csv(path, summaries):
    keys = [
        "output_tokens_per_s",
        "wall_s",
        "ttft_mean_ms",
        "ttft_median_ms",
        "e2e_median_ms",
        "gpu_prefix_hit_rate",
        "external_prefix_hit_rate",
        "gpu_kv_cache_tokens",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", *keys])
        for name in CONFIG_NAMES:
            writer.writerow([name, *[summaries[name][k] for k in keys]])


def log_to_mlflow(run_id, c, cpu_gib, mem, summaries, match):
    params = {
        "model": c["model"],
        "num_documents": c["num_documents"],
        "doc_tokens": c["doc_tokens"],
        "num_requests": c["num_requests"],
        "gpu_kv_gib": c["gpu_kv_gib"],
        "cpu_offload_gib": cpu_gib,
        "concurrency": c["concurrency"],
        "max_output_tokens": c["max_output_tokens"],
        "seed": c["seed"],
    }
    metrics = {"host_usable_ram_gib": mem["usable_gib"], "cpu_offload_gib": cpu_gib}
    for name, s in summaries.items():
        for key, value in s.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[f"{name}/{key}"] = float(value)
    for name, frac in match.items():
        metrics[f"{name}/output_match_vs_prefix_off"] = frac
    metrics = {k: v for k, v in metrics.items() if v is not None}
    with mlflow.start_run(run_id=run_id):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)


async def main():
    c = load_config()
    run_id = os.environ["MLFLOW_RUN_ID"]
    run_dir = Path(c["output_root"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    mem = host_memory_gib()
    cpu_gib = round(min(c["cpu_offload_cap_gib"], c["cpu_offload_host_fraction"] * mem["usable_gib"]), 1)
    if cpu_gib < 1.0:
        raise RuntimeError(f"not enough host RAM for a CPU tier: usable {mem['usable_gib']:.1f} GiB")
    cpu_bytes = int(cpu_gib * GIB)
    distinct_prefix_tokens = c["num_documents"] * c["doc_tokens"]
    print(
        f"Host RAM: total {mem['mem_total_gib']:.1f} GiB, available {mem['mem_available_gib']:.1f} GiB, "
        f"cgroup free {mem['cgroup_free_gib']}; CPU offload tier {cpu_gib} GiB; GPU KV budget {c['gpu_kv_gib']} GiB; "
        f"distinct document prefixes {distinct_prefix_tokens} tokens",
        flush=True,
    )
    with open(run_dir / "config.json", "w") as f:
        json.dump({**c, "host_memory": mem, "cpu_offload_gib": cpu_gib}, f, indent=2)

    summaries, outputs = {}, {}
    for name in CONFIG_NAMES:
        print(f"=== config: {name}", flush=True)
        summaries[name], outputs[name] = await run_config(c, name, cpu_bytes, run_dir)

    # Document prefixes as a share of the GPU KV cache, using the token count vLLM reported.
    gpu_tokens = summaries["prefix_off"]["gpu_kv_cache_tokens"]
    for name in CONFIG_NAMES:
        summaries[name]["distinct_prefix_tokens"] = distinct_prefix_tokens
        summaries[name]["gpu_kv_fraction_of_prefixes"] = (
            gpu_tokens / distinct_prefix_tokens if gpu_tokens else None
        )

    base = outputs["prefix_off"]
    match = {}
    for name in CONFIG_NAMES[1:]:
        same = sum(a == b for a, b in zip(base, outputs[name]))
        match[name] = same / len(base)
        print(f"{name}: greedy outputs identical to prefix_off for {same}/{len(base)} requests", flush=True)
        summaries[name]["output_match_vs_prefix_off"] = match[name]

    with open(run_dir / "results.json", "w") as f:
        json.dump({"config": c, "cpu_offload_gib": cpu_gib, "summaries": summaries}, f, indent=2)
    write_csv(run_dir / "results.csv", summaries)
    with open(run_dir / "outputs_prefix_off.jsonl", "w") as f:
        for i, text in enumerate(base):
            f.write(json.dumps({"index": i, "text": text}) + "\n")
    log_to_mlflow(run_id, c, cpu_gib, mem, summaries, match)
    print("Logged to MLflow run", run_id, "; results in", run_dir, flush=True)


if __name__ == "__main__":
    asyncio.run(main())

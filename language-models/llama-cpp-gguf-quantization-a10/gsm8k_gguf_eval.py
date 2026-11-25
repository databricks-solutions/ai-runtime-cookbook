"""GSM8K accuracy and serving throughput for Qwen3-8B on one A10.

Compares bf16 served by vLLM with GGUF quantizations served by llama.cpp's llama-server.
Every configuration uses the same questions, chat template, system prompt, greedy decoding,
max tokens and number of requests in flight. Each configuration runs in its own server process.
"""

import csv
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlflow
import nvidia
import yaml
from datasets import load_dataset
from huggingface_hub import hf_hub_download, snapshot_download

with open(os.environ["HYPERPARAMETERS_PATH"]) as f:
    P = yaml.safe_load(f)

NUM_QUESTIONS = int(P["num_questions"])
MAX_TOKENS = int(P["max_tokens"])
CONCURRENCY = int(P["concurrency"])
MAX_MODEL_LEN = int(P["max_model_len"])
LLAMA_CTX = int(P["llama_ctx"])
GPU_MEMORY_UTILIZATION = float(P["gpu_memory_utilization"])
Q2_TYPE = P["q2_type"]
PORT = 8000

# Built by build_llama_server.sh.
LLAMA_BIN = Path("/tmp/llama-cpp-build/build/bin")
# Scratch space for the 3 GB Q2_K file, kept off the output volume.
MODEL_DIR = Path("/tmp/gguf-models")

SYSTEM = "Solve the math problem step by step. End with the final answer as: #### <number>"

OUT = Path(P["output_root"]) / os.environ["MLFLOW_RUN_ID"]


def number(text):
    """Final answer after '####', else the last number in the text."""
    match = re.search(r"####\s*\$?(-?[\d,]*\.?\d+)", text)
    found = match.group(1) if match else (re.findall(r"-?[\d,]*\.?\d+", text) or [None])[-1]
    if found is None:
        return None
    return found.replace(",", "").rstrip(".")


def is_correct(pred, gold):
    if pred is None:
        return False
    return math.isclose(float(pred), float(gold), abs_tol=1e-6)


class GpuMemorySampler:
    """Records the peak memory.used of GPU 0, sampled every 250 ms by nvidia-smi."""

    def __init__(self):
        self.peak_mib = 0
        self.proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-lms", "250"],
            stdout=subprocess.PIPE,
            text=True,
        )
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        for line in self.proc.stdout:
            self.peak_mib = max(self.peak_mib, int(line.strip()))

    def stop(self):
        self.proc.terminate()
        self.proc.wait()
        self.thread.join()
        return self.peak_mib


def gpu_used_mib():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    return int(out.splitlines()[0])


def start_server(cfg):
    log_path = OUT / "servers" / f"{cfg['name']}.log"
    log = open(log_path, "w")
    if cfg["engine"] == "vllm":
        cmd = [
            "vllm", "serve", cfg["model"],
            "--served-model-name", "model",
            "--dtype", "bfloat16",
            "--max-model-len", str(MAX_MODEL_LEN),
            "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
            "--max-num-seqs", str(CONCURRENCY),
            "--port", str(PORT),
        ]
    else:
        cmd = [
            str(LLAMA_BIN / "llama-server"),
            "-m", cfg["model"],
            "-ngl", "99",
            "-c", str(LLAMA_CTX),
            "-np", str(CONCURRENCY),
            "-fa", "on",
            "--jinja",
            "--reasoning-format", "none",
            "--no-webui",
            "--host", "127.0.0.1",
            "--port", str(PORT),
        ]
    env = {**os.environ, "LD_LIBRARY_PATH": f"{LLAMA_BIN}:{os.environ.get('LD_LIBRARY_PATH', '')}"}
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)

    deadline = time.time() + 900
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"{cfg['name']} server exited with code {proc.returncode}; see {log_path}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5) as r:
                if r.status == 200:
                    print(f"{cfg['name']}: server ready", flush=True)
                    return proc
        except OSError:
            pass
        time.sleep(5)
    raise RuntimeError(f"{cfg['name']} server not ready after 900 s; see {log_path}")


def stop_server(proc):
    # start_new_session makes the server's pid its process-group id, which also covers vLLM workers.
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    proc.wait(timeout=300)
    # Wait until the GPU is released so the next configuration starts from an empty card.
    deadline = time.time() + 300
    while gpu_used_mib() > 1000 and time.time() < deadline:
        time.sleep(2)


def ask(question):
    body = {
        "model": "model",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
        ],
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as r:
        resp = json.load(r)
    choice = resp["choices"][0]
    return choice["message"]["content"] or "", choice["finish_reason"], resp["usage"]["completion_tokens"]


def evaluate(cfg, questions, gold, test_index):
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        replies = list(pool.map(ask, questions))
    wall = time.time() - t0

    empty = sum(1 for text, _, _ in replies if not text.strip())
    if empty > 0.05 * len(replies):
        raise RuntimeError(f"{cfg['name']}: {empty}/{len(replies)} empty replies; check the server log")

    rows = []
    for i, (text, finish, tokens) in enumerate(replies):
        pred = number(text)
        rows.append({
            "test_index": test_index[i],
            "question": questions[i],
            "gold": gold[i],
            "prediction": pred,
            "correct": is_correct(pred, gold[i]),
            "finish_reason": finish,
            "completion_tokens": tokens,
            "text": text,
        })
    with open(OUT / "predictions" / f"{cfg['name']}.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    acc = correct / n
    print(f"{cfg['name']}: accuracy {acc:.4f} ({correct}/{n}), wall {wall:.1f} s", flush=True)
    return {
        "config": cfg["name"],
        "weights_gb": cfg["weights_bytes"] / 1e9,
        "n": n,
        "accuracy": acc,
        "accuracy_se": math.sqrt(acc * (1 - acc) / n),
        "truncated": sum(r["finish_reason"] == "length" for r in rows),
        "output_tokens_per_s": sum(r["completion_tokens"] for r in rows) / wall,
        "wall_seconds": wall,
        "correct": [r["correct"] for r in rows],
    }


def quantize_q2(q8_path):
    q2_path = MODEL_DIR / f"Qwen3-8B-{Q2_TYPE}.gguf"
    env = {**os.environ, "LD_LIBRARY_PATH": f"{LLAMA_BIN}:{os.environ.get('LD_LIBRARY_PATH', '')}"}
    # The input is already Q8_0, so requantizing needs --allow-requantize. Q8_0 is close to
    # lossless, so this is close to quantizing from bf16, but it is not identical. No
    # importance matrix is used, which makes Q2_K worse than a calibrated Q2 would be.
    subprocess.run(
        [str(LLAMA_BIN / "llama-quantize"), "--allow-requantize", q8_path, str(q2_path), Q2_TYPE, str(os.cpu_count())],
        check=True, env=env,
    )
    return q2_path


def expose_curand_headers():
    """Put curand.h on CPATH for vLLM's FlashInfer JIT.

    FlashInfer JIT-compiles vLLM's sampler at startup. In this environment curand.h ships only
    inside the nvidia-* pip wheels, so copy just those headers to one include directory. Runs
    after the llama.cpp build, so the headers cannot affect its nvcc.
    """
    include = Path("/tmp/curand-include")
    include.mkdir(parents=True, exist_ok=True)
    for header in Path(nvidia.__path__[0]).rglob("curand*.h"):
        shutil.copy(header, include)
    os.environ["CPATH"] = f"{include}:{os.environ.get('CPATH', '')}"


def main():
    expose_curand_headers()
    print(subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout.strip(), flush=True)
    for sub in ("servers", "predictions"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # GSM8K test split, same seeded sample for every configuration. GSM8K has no categories to stratify.
    full = load_dataset("openai/gsm8k", "main", split="test")
    full = full.add_column("test_index", list(range(len(full))))
    ds = full.shuffle(seed=0).select(range(NUM_QUESTIONS))
    questions = ds["question"]
    gold = [number(a) for a in ds["answer"]]
    test_index = ds["test_index"]
    print(f"sampled {len(ds)} of {len(full)} test questions (seed 0)", flush=True)

    bf16_dir = Path(snapshot_download("Qwen/Qwen3-8B"))
    bf16_bytes = sum(os.path.getsize(p) for p in bf16_dir.glob("*.safetensors"))
    q8 = hf_hub_download(P["gguf_repo"], P["gguf_q8_file"])
    q4 = hf_hub_download(P["gguf_repo"], P["gguf_q4_file"])
    q2 = quantize_q2(q8)

    configs = [
        {"name": "bf16", "engine": "vllm", "model": str(bf16_dir), "weights_bytes": bf16_bytes},
        {"name": "Q8_0", "engine": "llama", "model": q8, "weights_bytes": os.path.getsize(q8)},
        {"name": "Q4_K_M", "engine": "llama", "model": q4, "weights_bytes": os.path.getsize(q4)},
        {"name": Q2_TYPE, "engine": "llama", "model": str(q2), "weights_bytes": os.path.getsize(q2)},
    ]

    rows = []
    for cfg in configs:
        sampler = GpuMemorySampler()
        proc = start_server(cfg)
        try:
            row = evaluate(cfg, questions, gold, test_index)
        finally:
            stop_server(proc)
            peak_mib = sampler.stop()
        row["peak_gpu_mib"] = peak_mib
        rows.append(row)

    # Paired comparison against bf16 on the same questions. The standard error is of the mean
    # per-question difference, so it reflects the pairing.
    bf16_correct = rows[0]["correct"]
    for row in rows:
        diffs = [int(c) - int(b) for c, b in zip(row["correct"], bf16_correct)]
        row["diff_vs_bf16"] = statistics.mean(diffs)
        row["diff_vs_bf16_se"] = statistics.stdev(diffs) / math.sqrt(len(diffs))
        row["bf16_only_correct"] = sum(b and not c for b, c in zip(bf16_correct, row["correct"]))
        row["config_only_correct"] = sum(c and not b for b, c in zip(bf16_correct, row["correct"]))

    columns = [
        "config", "weights_gb", "n", "accuracy", "accuracy_se", "diff_vs_bf16", "diff_vs_bf16_se",
        "bf16_only_correct", "config_only_correct", "truncated", "output_tokens_per_s",
        "wall_seconds", "peak_gpu_mib",
    ]
    with open(OUT / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in columns})
    print((OUT / "results.csv").read_text(), flush=True)

    with mlflow.start_run(run_id=os.environ["MLFLOW_RUN_ID"]):
        mlflow.log_params({
            "num_questions": NUM_QUESTIONS,
            "max_tokens": MAX_TOKENS,
            "concurrency": CONCURRENCY,
            "llama_cpp_source": "llama-cpp-python 0.3.37 sdist (vendored llama.cpp a956a852c)",
        })
        for row in rows:
            k = row["config"].lower()
            mlflow.log_metric(f"gsm8k_accuracy_{k}", row["accuracy"])
            mlflow.log_metric(f"gsm8k_accuracy_se_{k}", row["accuracy_se"])
            mlflow.log_metric(f"output_tokens_per_second_{k}", row["output_tokens_per_s"])
            mlflow.log_metric(f"peak_gpu_memory_mib_{k}", row["peak_gpu_mib"])
            mlflow.log_metric(f"weights_gb_{k}", row["weights_gb"])
            mlflow.log_metric(f"truncated_{k}", row["truncated"])
            mlflow.log_metric(f"accuracy_diff_vs_bf16_{k}", row["diff_vs_bf16"])
            mlflow.log_metric(f"accuracy_diff_vs_bf16_se_{k}", row["diff_vs_bf16_se"])
        mlflow.log_artifact(str(OUT / "results.csv"))


if __name__ == "__main__":
    main()

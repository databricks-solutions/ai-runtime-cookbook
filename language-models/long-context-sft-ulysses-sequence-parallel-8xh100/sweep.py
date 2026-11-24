#!/usr/bin/env python3
"""Sequence-length sweep for DeepSpeed Ulysses sequence parallelism on one 8xH100 node.

Each point runs train_sp.py in its own `accelerate launch`, so an out-of-memory crash
or a hang affects only that point. Results go to the AIR MLflow run and to a CSV and
JSON table under output_root. The sweep exits non-zero only if an SP=8 point fails,
because SP=1 running out of memory at the longest length is the expected baseline.
"""

import csv
import json
import os
import signal
import subprocess
import sys
import time

import mlflow
import yaml
from huggingface_hub import hf_hub_download, snapshot_download

HERE = os.path.dirname(os.path.abspath(__file__))
LAUNCH_CONFIG = os.path.join(HERE, "accelerate_zero3.yaml")
TRAIN_SCRIPT = os.path.join(HERE, "train_sp.py")
TABLE_FIELDS = [
    "tag", "sp_size", "dp_size", "seq_len", "status", "tokens_per_s", "sec_per_iter",
    "peak_allocated_gb_per_gpu", "peak_reserved_gb_per_gpu", "final_loss", "returncode",
]


def load_params() -> dict:
    """Read the hyperparameters block that `air` materializes from `parameters:`."""
    path = os.environ.get("HYPERPARAMETERS_PATH")
    if path and os.path.exists(path):
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def parse_lens(value) -> list[int]:
    return [int(x) for x in str(value).split(",") if x.strip()]


def run_point(sp_size, seq_len, p, world, data_path, out_dir, port):
    tag = f"sp{sp_size}_seq{seq_len}"
    result_json = os.path.join(out_dir, f"{tag}.json")
    log_path = os.path.join(out_dir, f"{tag}.log")
    if os.path.exists(result_json):
        os.remove(result_json)
    cmd = [
        sys.executable, "-u", "-m", "accelerate.commands.launch",
        "--config_file", LAUNCH_CONFIG,
        "--num_processes", str(world),
        "--main_process_port", str(port),
        TRAIN_SCRIPT,
        "--model-name", p["model_name"],
        "--data-path", data_path,
        "--sp-size", str(sp_size),
        "--seq-len", str(seq_len),
        "--steps", str(p["steps"]),
        "--learning-rate", str(float(p["learning_rate"])),
        "--result-json", result_json,
    ]
    print(f"[sweep] {tag}: starting ({world} GPUs)", flush=True)
    started = time.time()
    timeout_s = float(p["point_timeout_minutes"]) * 60
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            returncode = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            # Kill the whole process group, since torch workers can outlive their launcher.
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            returncode = None

    record = {"tag": tag, "sp_size": sp_size, "dp_size": world // sp_size, "seq_len": seq_len,
              "returncode": returncode}
    if returncode == 0 and os.path.exists(result_json):
        with open(result_json) as f:
            res = json.load(f)
        record.update(
            status="ok",
            tokens_per_s=res["tokens_per_s"],
            sec_per_iter=res["sec_per_iter"],
            peak_allocated_gb_per_gpu=res["peak_allocated_gb_per_gpu"],
            peak_reserved_gb_per_gpu=res["peak_reserved_gb_per_gpu"],
            final_loss=res["rank0_losses"][-1],
        )
    else:
        with open(log_path) as f:
            text = f.read()
        if returncode is None:
            status = "timeout"
        elif "OutOfMemoryError" in text or "CUDA out of memory" in text:
            status = "oom"
        else:
            status = "error"
        record["status"] = status
    print(f"[sweep] {tag}: {record['status']} after {time.time() - started:.0f}s", flush=True)
    return record


def metrics_for(record):
    """Numeric metrics for one point. Missing values are left out, not logged as zero."""
    tag = record["tag"]
    out = {f"{tag}_ok": 1.0 if record["status"] == "ok" else 0.0}
    for key in ["tokens_per_s", "sec_per_iter", "peak_allocated_gb_per_gpu",
                "peak_reserved_gb_per_gpu", "final_loss"]:
        if record.get(key) is not None:
            out[f"{tag}_{key}"] = float(record[key])
    return out


def main():
    p = load_params()
    run_id = os.environ["MLFLOW_RUN_ID"]  # Fail if absent; there is no local fallback run.
    world = int(os.environ.get("LOCAL_WORLD_SIZE", "8"))
    out_dir = os.path.join(p["output_root"], run_id)
    os.makedirs(out_dir, exist_ok=True)

    # Download once here, so the per-point training processes read from the local cache.
    snapshot_download(repo_id=p["model_name"])
    data_path = hf_hub_download(repo_id=p["dataset_name"], filename="long.jsonl", repo_type="dataset")

    points = [(8, s) for s in parse_lens(p["sp8_seq_lens"])] + [(1, s) for s in parse_lens(p["sp1_seq_lens"])]
    for sp_size, seq_len in points:
        if sp_size > world or world % sp_size or seq_len % sp_size:
            raise ValueError(f"point sp={sp_size} seq={seq_len} does not fit {world} GPUs")

    records = []
    with mlflow.start_run(run_id=run_id):
        mlflow.log_params({
            "model_name": p["model_name"],
            "dataset_name": p["dataset_name"],
            "sp8_seq_lens": str(p["sp8_seq_lens"]),
            "sp1_seq_lens": str(p["sp1_seq_lens"]),
            "timed_steps": str(p["steps"]),
            "gpus": str(world),
        })
        for idx, (sp_size, seq_len) in enumerate(points):
            record = run_point(sp_size, seq_len, p, world, data_path, out_dir, port=29600 + idx)
            records.append(record)
            mlflow.log_metrics(metrics_for(record))

        table_json = os.path.join(out_dir, "sweep_results.json")
        table_csv = os.path.join(out_dir, "sweep_table.csv")
        with open(table_json, "w") as f:
            json.dump(records, f, indent=2)
        with open(table_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TABLE_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
        mlflow.log_artifact(table_csv, artifact_path="sweep")
        mlflow.log_artifact(table_json, artifact_path="sweep")

    print("\n[sweep] results", flush=True)
    print(f"{'point':<22}{'status':<9}{'tok/s':>12}{'s/iter':>9}{'peak GB':>10}{'loss':>9}", flush=True)
    for r in records:
        print(
            f"{r['tag']:<22}{r['status']:<9}"
            f"{r.get('tokens_per_s') or float('nan'):>12.0f}"
            f"{r.get('sec_per_iter') or float('nan'):>9.2f}"
            f"{r.get('peak_allocated_gb_per_gpu') or float('nan'):>10.2f}"
            f"{r.get('final_loss') or float('nan'):>9.3f}",
            flush=True,
        )

    sp8_failed = [r["tag"] for r in records if r["sp_size"] == 8 and r["status"] != "ok"]
    if sp8_failed:
        print(f"[sweep] SP=8 points failed: {sp8_failed}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

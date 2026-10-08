#!/usr/bin/env python3
"""Protein embeddings at scale with BioNeMo ESM-2 (Transformer Engine) and Ray Data.

The workload starts a Ray head on a single 8x H100 node. Ray Data runs one
embedding actor per GPU; each loads NVIDIA BioNeMo's Transformer Engine (TE)
build of ESM-2 650M (`nvidia/esm2_t33_650M_UR50D`) once and embeds batches of
protein sequences. Each batch is split into length-sorted micro-batches under
a token budget, so little compute goes to padding.

Input is DeepLoc 2.0 (`bloyal/deeploc`, CC-BY-4.0): about 28,000 UniProt
sequences labeled with subcellular localization. The script writes one row per
protein (accession, split, labels, sequence, and a 1,280-dimension mean-pooled
embedding) to a Unity Catalog volume as Parquet.

To check that the embeddings carry biological signal, the script then fits a
linear probe (logistic regression) that predicts whether a protein is
membrane-bound from its embedding alone. It compares the probe with an
amino-acid-composition baseline on the dataset's test split. That split is a
random 10% sample rather than a homology partition, so close homologs of test
proteins can appear in training and both scores are optimistic.

The model and dataset are public and do not require a Hugging Face token.
OUTPUT_PATH must be on an existing writable Unity Catalog volume; each run
writes to its own subdirectory, named after the run's MLflow run ID.
"""

import json
import os
import time

import numpy as np
import pandas as pd
import ray
from huggingface_hub import hf_hub_download

MODEL_SOURCE = "nvidia/esm2_t33_650M_UR50D"
DATASET_REPO = "bloyal/deeploc"
DATASET_FILES = {
    "train": "deeploc-train.parquet",
    "validation": "deploc-val.parquet",
    "test": "deeploc-test.parquet",
}
# ESM-2 was trained on up to 1,022 residues plus the <cls> and <eos> tokens.
MAX_LENGTH = 1024
# Padded tokens per forward pass. Short sequences get large micro-batches and
# long ones get small micro-batches, so GPU memory use stays roughly flat.
TOKENS_PER_BATCH = 65536
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
# 0 embeds every sequence. Set a small value for a quick smoke test.
NUM_SEQUENCES = int(os.environ.get("NUM_SEQUENCES", "0"))
# Unity Catalog volume path where results land as Parquet. Set this in workload.yaml.
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/Volumes/main/default/air_examples/protein-embeddings-esm2-bionemo")
# AI Runtime creates an MLflow run per attempt, so each attempt gets its own output directory.
RUN_OUTPUT_PATH = os.path.join(OUTPUT_PATH, os.environ.get("MLFLOW_RUN_ID") or "local")


def load_sequences():
    """Download DeepLoc 2.0 and return one row per protein with its split and labels."""
    frames = []
    for split, filename in DATASET_FILES.items():
        path = hf_hub_download(repo_id=DATASET_REPO, filename=filename, repo_type="dataset")
        frame = pd.read_parquet(path, columns=["ACC", "Kingdom", "Membrane", "Sequence"])
        frame["split"] = split
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df.columns = ["accession", "kingdom", "membrane", "sequence", "split"]
    if NUM_SEQUENCES > 0:
        df = df.sample(n=min(NUM_SEQUENCES, len(df)), random_state=0)
    df["membrane"] = df["membrane"].astype(int)
    df["length"] = df["sequence"].str.len()
    return df.reset_index(drop=True)


class ESM2Embedder:
    """GPU actor: load the TE-accelerated ESM-2 once, then mean-pool residue embeddings."""

    def __init__(self):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_SOURCE, trust_remote_code=True)
        # trust_remote_code loads BioNeMo's TE model definition (esm_nv.py) from the model repo.
        self.model = (
            AutoModel.from_pretrained(MODEL_SOURCE, trust_remote_code=True, dtype=torch.bfloat16)
            .to("cuda")
            .eval()
        )
        self.hidden_size = self.model.config.hidden_size

    def embed(self, sequences):
        torch = self.torch
        enc = self.tokenizer(
            sequences, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        ).to("cuda")
        with torch.inference_mode():
            hidden = self.model(**enc).last_hidden_state.float()
        # Average over residues only: drop <cls> (first token) and <eos> (last real token).
        mask = enc["attention_mask"].clone()
        mask[:, 0] = 0
        mask[torch.arange(mask.size(0), device=mask.device), enc["attention_mask"].sum(1) - 1] = 0
        mask = mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return pooled.cpu().numpy()

    def __call__(self, batch):
        started = time.perf_counter()
        sequences = batch["sequence"]
        order = np.argsort([len(s) for s in sequences])
        embeddings = np.empty((len(sequences), self.hidden_size), dtype=np.float32)
        start = 0
        while start < len(order):
            # Grow the micro-batch while its padded size (count x longest) fits the token budget.
            end = start + 1
            while end < len(order):
                longest = min(len(sequences[order[end]]) + 2, MAX_LENGTH)
                if (end + 1 - start) * longest > TOKENS_PER_BATCH:
                    break
                end += 1
            idx = order[start:end]
            embeddings[idx] = self.embed([sequences[i] for i in idx])
            start = end
        batch["embedding"] = embeddings
        # GPU-actor time per protein, used to report steady-state throughput without startup.
        batch["embed_seconds"] = np.full(len(sequences), (time.perf_counter() - started) / len(sequences))
        return batch


def composition(sequence):
    """Amino-acid frequencies: a sequence-only baseline feature set."""
    counts = np.array([sequence.count(a) for a in AMINO_ACIDS], dtype=np.float32)
    return counts / max(len(sequence), 1)


def fit_probe(x_train, y_train, x_test, y_test):
    """Fit a logistic-regression probe and score it on the held-out split."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    probe.fit(x_train, y_train)
    prob = probe.predict_proba(x_test)[:, 1]
    return {
        "accuracy": round(float(accuracy_score(y_test, prob > 0.5)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, prob)), 4),
    }


def check_output_volume(path):
    """Fail fast, before loading any model, if OUTPUT_PATH's Unity Catalog volume is missing."""
    parts = path.rstrip("/").split("/")
    if path.startswith("/Volumes/") and len(parts) >= 5 and not os.path.isdir("/".join(parts[:5])):
        raise SystemExit(
            f"OUTPUT_PATH is on volume {'.'.join(parts[2:5])}, which does not exist or is not "
            "accessible. Create it (CREATE VOLUME) or set OUTPUT_PATH in workload.yaml to a "
            "writable Unity Catalog volume path."
        )


def log_to_mlflow(params, metrics, tables):
    """Attach params, metrics, and result tables to the MLflow run AI Runtime creates for the workload."""
    run_id = os.environ.get("MLFLOW_RUN_ID")
    if not run_id:
        return
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    for key, value in params.items():
        client.log_param(run_id, key, value)
    for key, value in metrics.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                client.log_metric(run_id, f"{key}.{sub_key}", sub_value)
        else:
            client.log_metric(run_id, key, value)
    for artifact_file, table in tables.items():
        client.log_table(run_id, data=table, artifact_file=artifact_file)


def main():
    check_output_volume(OUTPUT_PATH)
    # Creating the run directory now also fails fast if the volume isn't writable.
    os.makedirs(RUN_OUTPUT_PATH, exist_ok=True)
    ray.init(address="auto")
    total_gpus = int(ray.cluster_resources().get("GPU", 0))
    if total_gpus < 1:
        raise SystemExit("No GPUs found in the Ray cluster; cannot run inference.")
    print(f"Ray cluster ready: {total_gpus} GPU(s)", flush=True)

    df = load_sequences()
    print(f"Loaded {len(df)} sequences ({df['length'].sum():,} residues)", flush=True)

    # Split the table into many blocks so every GPU actor gets work, but no more blocks than rows.
    ds = ray.data.from_pandas(df).repartition(min(total_gpus * 8, len(df)))

    # Fetch the config and BioNeMo's model code (esm_nv.py) once, so the GPU actors on this
    # node read it from the cache instead of all copying it into place at the same time.
    from transformers import AutoConfig

    AutoConfig.from_pretrained(MODEL_SOURCE, trust_remote_code=True)

    # One actor per GPU; Ray Data's queue hands each free actor the next batch.
    start = time.time()
    out = ds.map_batches(
        ESM2Embedder,
        concurrency=total_gpus,
        num_gpus=1,
        batch_size=512,
    ).materialize()
    elapsed = time.time() - start
    # Overwrite so a re-run into the same directory replaces it instead of appending to it.
    out.drop_columns(["embed_seconds"]).write_parquet(RUN_OUTPUT_PATH, mode="overwrite")
    print(f"Wrote {out.count()} embeddings to {RUN_OUTPUT_PATH}", flush=True)

    # Linear probe: membrane-bound vs. soluble, trained on train + validation, scored on test
    # (a random split, not a homology partition, so scores are optimistic).
    result = out.select_columns(["split", "membrane", "sequence", "embedding", "embed_seconds"]).to_pandas()
    is_test = (result["split"] == "test").to_numpy()
    y = result["membrane"].to_numpy()
    x_esm = np.stack(result["embedding"].to_numpy())
    x_aac = np.stack([composition(s) for s in result["sequence"]])
    esm_probe = fit_probe(x_esm[~is_test], y[~is_test], x_esm[is_test], y[is_test])
    aac_probe = fit_probe(x_aac[~is_test], y[~is_test], x_aac[is_test], y[is_test])

    # Wall time includes model load; per-GPU throughput counts only time spent embedding.
    gpu_seconds = float(result["embed_seconds"].sum())
    metrics = {
        "proteins": len(result),
        "wall_time_s": round(elapsed, 1),
        "proteins_per_gpu_sec": round(len(result) / gpu_seconds, 1),
        # Count only embedded residues: longer sequences are truncated to MAX_LENGTH - 2.
        "residues_per_gpu_sec": round(float(df["length"].clip(upper=MAX_LENGTH - 2).sum()) / gpu_seconds),
        "embedding_dim": int(x_esm.shape[1]),
        "membrane_probe_esm2": esm_probe,
        "membrane_probe_aa_composition": aac_probe,
    }
    print("METRICS " + json.dumps(metrics), flush=True)
    probe_table = pd.DataFrame(
        [{"features": "esm2_embedding", **esm_probe}, {"features": "aa_composition", **aac_probe}]
    )
    log_to_mlflow(
        params={"model": MODEL_SOURCE, "dataset": DATASET_REPO, "num_gpus": total_gpus, "output_path": RUN_OUTPUT_PATH},
        metrics=metrics,
        tables={"membrane_probe.json": probe_table},
    )

    ray.shutdown()


if __name__ == "__main__":
    main()

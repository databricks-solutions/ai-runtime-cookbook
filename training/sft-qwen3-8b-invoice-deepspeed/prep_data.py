# Databricks notebook source
# DBTITLE 1,Introduction
# MAGIC %md
# MAGIC # Data Prep — Public HF dataset -> HF datasets on a UC Volume
# MAGIC
# MAGIC One-time prerequisite for `train.py`. The `air` GPU worker has no Spark, so this
# MAGIC notebook stages the training and eval data as Hugging Face `Dataset` objects on a
# MAGIC Unity Catalog Volume that `train.py` then reads with `load_from_disk`.
# MAGIC
# MAGIC Self-contained: it pulls the public **Winuim/invoice-sft-dataset-v2** dataset
# MAGIC (OCR text paired with ground-truth extraction JSON in chat-message format),
# MAGIC builds a conversational `messages` column, and writes a train/eval split.
# MAGIC
# MAGIC **Run once on Databricks (Serverless CPU)** before launching `databricks air run -f workload.yaml`.
# MAGIC Set the `catalog` / `schema` / `volume` widgets to match `workload.yaml`.

# COMMAND ----------

# MAGIC %pip install --quiet datasets huggingface_hub
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Configuration
# These must match the `catalog` / `schema` / `volume` parameters in workload.yaml.
# The catalog, schema, and volume must already exist and be writable.
dbutils.widgets.text("catalog", "main", "Catalog")
dbutils.widgets.text("schema", "default", "Schema")
dbutils.widgets.text("volume", "invoice_sft", "Volume")
dbutils.widgets.text("hf_dataset", "Winuim/invoice-sft-dataset-v2", "HF dataset")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VOLUME = dbutils.widgets.get("volume")
HF_DATASET = dbutils.widgets.get("hf_dataset")

VOLUME_BASE = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
TRAIN_DATASET_PATH = f"{VOLUME_BASE}/invoice_train_dataset"
EVAL_DATASET_PATH = f"{VOLUME_BASE}/invoice_eval_dataset"

print(f"HF dataset: {HF_DATASET}")
print(f"Output:     {TRAIN_DATASET_PATH}, {EVAL_DATASET_PATH}")

# COMMAND ----------

# DBTITLE 1,Load + parse the chat-message dataset
import json
import os

# On Serverless the HF `datasets` cache defaults to /root/.cache, which is not
# writable. Point it at a writable local path before importing datasets.
os.environ["HF_HOME"] = "/tmp/hf_home"
os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_home/datasets"

from datasets import load_dataset

# Each row is a list of {role, content} chat messages: system + user (OCR text)
# + assistant (ground-truth extraction JSON).
raw = load_dataset(HF_DATASET, split="train", cache_dir="/tmp/hf_home/datasets")


def to_example(row):
    """Fold system+user into one user turn and keep the assistant JSON as the target."""
    system_prompt, ocr_text, extraction_json = "", None, None
    for msg in row["messages"]:
        if msg["role"] == "system":
            system_prompt = msg["content"]
        elif msg["role"] == "user":
            ocr_text = msg["content"]
        elif msg["role"] == "assistant":
            extraction_json = msg["content"]
    return {
        "messages": [
            {"role": "user", "content": (system_prompt + ocr_text).strip()},
            {"role": "assistant", "content": extraction_json.strip()},
        ],
        "_valid": ocr_text is not None and extraction_json is not None,
    }


parsed = raw.map(to_example, remove_columns=raw.column_names)


def _is_valid_json(row):
    if not row["_valid"]:
        return False
    try:
        json.loads(row["messages"][1]["content"])
        return True
    except Exception:
        return False


before = len(parsed)
parsed = parsed.filter(_is_valid_json).remove_columns(["_valid"])
print(f"Kept {len(parsed)}/{before} rows (dropped empty / invalid-JSON rows).")

# COMMAND ----------

# DBTITLE 1,Split and save to the UC Volume
# 95/5 train/eval split (seed fixed for reproducible re-runs). The public
# dataset is a single `train` split, so we carve the eval set out here.
split = parsed.train_test_split(test_size=0.05, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

train_dataset.save_to_disk(TRAIN_DATASET_PATH)
eval_dataset.save_to_disk(EVAL_DATASET_PATH)

print(f"✓ Train: {len(train_dataset)} samples -> {TRAIN_DATASET_PATH}")
print(f"✓ Eval:  {len(eval_dataset)} samples -> {EVAL_DATASET_PATH}")

# COMMAND ----------

# DBTITLE 1,Verify staged data
from datasets import load_from_disk

train_check = load_from_disk(TRAIN_DATASET_PATH)
eval_check = load_from_disk(EVAL_DATASET_PATH)
print(f"Train: {len(train_check)} samples, columns={train_check.column_names}")
print(f"Eval:  {len(eval_check)} samples, columns={eval_check.column_names}")
print("\nSample (first train example):")
for msg in train_check[0]["messages"]:
    print(f"  [{msg['role']}]: {msg['content'][:200]}...")
print("\nDone. Data is staged and ready for `databricks air run -f workload.yaml`.")

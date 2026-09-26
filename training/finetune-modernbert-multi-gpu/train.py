#!/usr/bin/env python3
"""Fine-tune ModernBERT on AG News across multiple GPUs with PyTorch DDP.

Launched by ``torchrun`` (see workload.yaml) — one process per GPU. The Hugging Face ``Trainer``
reads the torchrun env (``RANK`` / ``WORLD_SIZE`` / ``LOCAL_RANK``) and runs DDP automatically.
Rank 0 owns the MLflow run: the Trainer's MLflow callback streams the loss/eval curves into it,
then rank 0 logs the final metrics and registers the model (as a ``text-classification`` pipeline)
to Unity Catalog under the ``@champion`` alias. Config comes from the workload ``parameters:`` block
(``$HYPERPARAMETERS_PATH``).

Run from this recipe's directory:

    databricks air run -f workload.yaml
"""

import logging
import os
import time

import mlflow
import numpy as np
import torch
import yaml
from datasets import load_dataset
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    pipeline,
    set_seed,
)

# Quiet benign serverless MLflow/py4j chatter emitted during logging (these are not errors).
for _name in ("mlflow.tracking.context.registry", "pyspark.sql.connect", "py4j"):
    logging.getLogger(_name).setLevel(logging.ERROR)

LABELS = ["World", "Sports", "Business", "Sci/Tech"]
ID2LABEL = dict(enumerate(LABELS))
LABEL2ID = {label: i for i, label in enumerate(LABELS)}


def load_params() -> dict:
    """Read the ``parameters:`` block that ``air`` materializes from the workload YAML."""
    path = os.environ.get("HYPERPARAMETERS_PATH")
    if path and os.path.exists(path):
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def ensure_uc_schema(catalog: str, schema: str) -> None:
    """Create the target UC schema if missing, so a fresh catalog works with no manual setup."""
    from databricks.sdk import WorkspaceClient

    try:
        WorkspaceClient().schemas.create(name=schema, catalog_name=catalog)
        print(f"Created schema {catalog}.{schema}")
    except Exception as e:  # noqa: BLE001
        msg = str(e).lower()
        if "already exists" in msg:
            return
        if any(t in msg for t in ("permission", "denied", "does not have", "unauthorized")):
            raise RuntimeError(
                f"Cannot create {catalog}.{schema}: grant CREATE on catalog '{catalog}', or set "
                f"uc_catalog/uc_schema to an existing schema you can write to."
            ) from e
        raise


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def main() -> None:
    p = load_params()
    model_name = p.get("model_name", "answerdotai/ModernBERT-base")
    dataset_name = p.get("dataset_name", "fancyzhx/ag_news")
    max_length = int(p.get("max_length", 256))
    max_train_samples = int(p.get("max_train_samples", -1))
    max_eval_samples = int(p.get("max_eval_samples", -1))
    seed = int(p.get("seed", 42))
    uc_catalog = p.get("uc_catalog", "main")
    uc_schema = p.get("uc_schema", "default")
    model_fqn = f"{uc_catalog}.{uc_schema}.{p.get('registered_model_name', 'modernbert_agnews')}"

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_main = rank == 0
    set_seed(seed)
    print(f"[rank {rank}/{world_size}] visible GPUs: {torch.cuda.device_count()}")

    # Load AG News (all ranks need the data for DDP), optionally subsample, and tokenize.
    ds = load_dataset(dataset_name)
    if max_train_samples > 0:
        ds["train"] = ds["train"].shuffle(seed=seed).select(range(min(max_train_samples, ds["train"].num_rows)))
    if max_eval_samples > 0:
        ds["test"] = ds["test"].shuffle(seed=seed).select(range(min(max_eval_samples, ds["test"].num_rows)))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized = ds.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
        batched=True, remove_columns=["text"],
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID,
        attn_implementation=p.get("attn_implementation", "sdpa"),
    )
    args = TrainingArguments(
        output_dir="/tmp/modernbert-agnews-ddp",
        num_train_epochs=float(p.get("epochs", 2)),
        per_device_train_batch_size=int(p.get("train_batch_size", 64)),
        per_device_eval_batch_size=int(p.get("eval_batch_size", 128)),
        learning_rate=float(p.get("learning_rate", 5e-5)),
        weight_decay=float(p.get("weight_decay", 0.01)),
        warmup_steps=float(p.get("warmup_steps", 0.1)),  # float in [0,1) = ratio of total steps
        bf16=True,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        # Only rank 0 reports to MLflow; other ranks skip it so 8 processes don't fight over the run.
        report_to=["mlflow"] if is_main else [],
        ddp_find_unused_parameters=False,
        seed=seed,
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=tokenized["train"], eval_dataset=tokenized["test"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    # Rank 0 owns the MLflow run. AIR sets MLFLOW_RUN_ID (the run the CLI shows); we POP it and open
    # that run ourselves BEFORE training, so the Trainer's MLflow callback streams its curves into
    # our run instead of starting — and auto-ending — its own (which on transformers 5.x would close
    # the run before we attach the model). Other ranks never touch MLflow.
    if is_main:
        mlflow.set_registry_uri("databricks-uc")
        ensure_uc_schema(uc_catalog, uc_schema)
        run_id = os.environ.pop("MLFLOW_RUN_ID", None)
        if run_id:
            mlflow.start_run(run_id=run_id)
        else:
            mlflow.start_run(run_name="modernbert-agnews-ddp")

    t0 = time.time()
    trainer.train()
    train_seconds = time.time() - t0
    metrics = trainer.evaluate()
    print(f"[rank {rank}/{world_size}] train_seconds={train_seconds:.1f} eval={metrics}")

    if is_main:
        # Extra config under distinct keys (HF already logged the Trainer args + model config).
        for k, v in {
            "base_model": model_name, "dataset": dataset_name, "tokenizer_max_length": max_length,
            "ddp_world_size": world_size,
            "effective_batch_size": int(p.get("train_batch_size", 64)) * world_size,
            "training_mode": f"ddp-{world_size}gpu",
        }.items():
            try:
                mlflow.log_param(k, v)
            except Exception:  # noqa: BLE001
                pass
        mlflow.log_metric("train_seconds", train_seconds)
        # Final eval metrics under distinct final_* keys so they don't overwrite the per-step
        # training `loss` curve the callback already streamed.
        mlflow.log_metrics({
            f"final_{k.replace('eval_', '')}": float(v)
            for k, v in metrics.items() if isinstance(v, (int, float))
        })

        # Register as a text-classification pipeline, packaged on CPU so it loads on any hardware and
        # MLflow's signature inference (which runs the pipeline once) avoids a device mismatch.
        clf = pipeline("text-classification", model=trainer.model.to("cpu"), tokenizer=tokenizer)
        info = mlflow.transformers.log_model(
            transformers_model=clf, name="model", task="text-classification",
            input_example=["Wall Street stocks rallied as tech earnings beat expectations."],
            registered_model_name=model_fqn,
            # Pin serving requirements explicitly: MLflow's default inference imports every candidate
            # framework (incl. tensorflow) and errors when it isn't installed.
            pip_requirements=["torch", "transformers==5.16.1", "accelerate==1.15.0"],
        )
        MlflowClient().set_registered_model_alias(model_fqn, "champion", info.registered_model_version)
        print(f"Registered {model_fqn} v{info.registered_model_version} and set @champion "
              f"(DDP across {world_size} GPUs, {train_seconds:.1f}s).")
        mlflow.end_run()


if __name__ == "__main__":
    main()

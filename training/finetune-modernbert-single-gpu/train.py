#!/usr/bin/env python3
"""Fine-tune ModernBERT for topic classification on AG News (single GPU_1xA10).

Fine-tunes ``answerdotai/ModernBERT-base`` (a modern BERT-class encoder) into a 4-way AG News topic
classifier. The Hugging Face ``Trainer``'s MLflow callback streams the training loss and per-epoch
eval metrics to the run live; this script then logs the final metrics and registers the model (as a
``text-classification`` pipeline) to Unity Catalog under the ``@champion`` alias. Hyperparameters
come from the workload YAML ``parameters:`` block, which ``air`` materializes to
``$HYPERPARAMETERS_PATH``.

Run from this recipe's directory:

    databricks air run -f workload.yaml
"""

import logging
import os

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

# AG News labels; the index order matches the dataset's integer labels.
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
    max_train_samples = int(p.get("max_train_samples", 20000))
    max_eval_samples = int(p.get("max_eval_samples", 2000))
    seed = int(p.get("seed", 42))
    uc_catalog = p.get("uc_catalog", "main")
    uc_schema = p.get("uc_schema", "default")
    model_fqn = f"{uc_catalog}.{uc_schema}.{p.get('registered_model_name', 'modernbert_agnews')}"

    set_seed(seed)
    print(f"CUDA available: {torch.cuda.is_available()}"
          + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))

    # Load AG News, optionally subsample for a fast demo, and tokenize.
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
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    args = TrainingArguments(
        output_dir="/tmp/modernbert-agnews",
        num_train_epochs=float(p.get("epochs", 1)),
        per_device_train_batch_size=int(p.get("train_batch_size", 32)),
        per_device_eval_batch_size=int(p.get("eval_batch_size", 64)),
        learning_rate=float(p.get("learning_rate", 5e-5)),
        weight_decay=float(p.get("weight_decay", 0.01)),
        warmup_steps=float(p.get("warmup_steps", 0.1)),  # float in [0,1) = ratio of total steps
        bf16=use_bf16,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        report_to=["mlflow"],  # HF's MLflow callback streams loss/eval curves into the active run
        seed=seed,
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=tokenized["train"], eval_dataset=tokenized["test"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    mlflow.set_registry_uri("databricks-uc")
    ensure_uc_schema(uc_catalog, uc_schema)

    # Own the MLflow run ourselves. AIR sets MLFLOW_RUN_ID (the run the CLI shows); we POP it and
    # resume that run explicitly BEFORE training. With the env var gone and a run already active,
    # the HF Trainer's MLflow callback logs its loss/eval curves into our run instead of starting —
    # and auto-ending — its own. (On transformers 5.x the callback otherwise closes the run at
    # train-end, so the later model logging would hang the job.)
    run_id = os.environ.pop("MLFLOW_RUN_ID", None)
    run = mlflow.start_run(run_id=run_id) if run_id else mlflow.start_run(run_name="modernbert-agnews")
    with run:
        trainer.train()   # callback streams loss/eval curves into this run (no run of its own)
        metrics = trainer.evaluate()
        print("Eval:", metrics)

        # Add our extra config under distinct keys (HF already logged the Trainer args + model
        # config); skip any that would collide with an already-logged key.
        for k, v in {
            "base_model": model_name, "dataset": dataset_name, "tokenizer_max_length": max_length,
            "max_train_samples": max_train_samples, "training_mode": "single-gpu",
        }.items():
            try:
                mlflow.log_param(k, v)
            except Exception:  # noqa: BLE001
                pass
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
            # Pin the serving requirements explicitly: MLflow's default inference imports every
            # candidate framework (incl. tensorflow) and errors when it isn't installed.
            pip_requirements=["torch", "transformers==5.16.1", "accelerate==1.15.0"],
        )
        MlflowClient().set_registered_model_alias(model_fqn, "champion", info.registered_model_version)
        print(f"Registered {model_fqn} v{info.registered_model_version} and set @champion.")


if __name__ == "__main__":
    main()

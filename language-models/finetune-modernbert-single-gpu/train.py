#!/usr/bin/env python3
"""Fine-tune ModernBERT for topic classification on AG News (single GPU_1xA10).

Fine-tunes ``answerdotai/ModernBERT-base`` (a modern BERT-class encoder) into a 4-way AG News topic
classifier. The Hugging Face ``Trainer``'s MLflow callback streams the training loss and per-epoch
eval metrics to the run live; this script then logs the final metrics and registers the model (as a
``text-classification`` pipeline) to Unity Catalog (a new version per run). Hyperparameters
come from the workload YAML ``parameters:`` block, which ``air`` materializes to
``$HYPERPARAMETERS_PATH``.

Run from this recipe's directory:

    databricks air run -f workload.yaml --watch
"""

import logging
import os

import mlflow
import numpy as np
import torch
import yaml
from datasets import load_dataset
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
    with open(os.environ["HYPERPARAMETERS_PATH"]) as f:
        return yaml.safe_load(f)


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
    model_name = p["model_name"]
    dataset_name = p["dataset_name"]
    max_length = int(p["max_length"])
    max_train_samples = int(p["max_train_samples"])
    max_eval_samples = int(p["max_eval_samples"])
    seed = int(p["seed"])
    uc_catalog = p["uc_catalog"]
    uc_schema = p["uc_schema"]
    model_fqn = f"{uc_catalog}.{uc_schema}.{p['registered_model_name']}"

    set_seed(seed)
    print(f"CUDA available: {torch.cuda.is_available()}"
          + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))

    # Load AG News, optionally subsample for a fast demo, and tokenize.
    ds = load_dataset(dataset_name)
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
        attn_implementation=p["attn_implementation"],
    )
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    args = TrainingArguments(
        output_dir="/tmp/modernbert-agnews",
        num_train_epochs=float(p["epochs"]),
        per_device_train_batch_size=int(p["train_batch_size"]),
        per_device_eval_batch_size=int(p["eval_batch_size"]),
        learning_rate=float(p["learning_rate"]),
        weight_decay=float(p["weight_decay"]),
        warmup_steps=float(p["warmup_steps"]),  # float in [0,1) = ratio of total steps
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
    # and auto-ending — its own. Otherwise the callback ends the run at train end, and the model
    # logging below goes to a new run.
    run_id = os.environ.pop("MLFLOW_RUN_ID", None)
    if not run_id:
        raise RuntimeError("MLFLOW_RUN_ID is not set; AIR provides it for each run.")
    run = mlflow.start_run(run_id=run_id)
    with run:
        trainer.train()   # callback streams loss/eval curves into this run (no run of its own)
        metrics = trainer.evaluate()
        print("Eval:", metrics)

        # Extra config under distinct keys; HF already logs the Trainer args and model config.
        mlflow.log_params({
            "base_model": model_name, "dataset": dataset_name, "tokenizer_max_length": max_length,
            "max_train_samples": max_train_samples, "training_mode": "single-gpu",
        })
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
            # Set pip_requirements explicitly so MLflow skips its default inference (which imports
            # every candidate framework, incl. tensorflow, and errors when it isn't installed).
            # transformers/accelerate are pinned to the recipe's versions; torch is inherited from
            # the AIR environment.
            pip_requirements=["torch", "transformers==5.16.1", "accelerate==1.15.0"],
        )
        print(f"Registered {model_fqn} version {info.registered_model_version}.")


if __name__ == "__main__":
    main()

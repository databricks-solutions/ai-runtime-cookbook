#!/usr/bin/env python3
"""LoRA fine-tuning of Longformer for long-document sequence classification.

Runs on a single GPU (launched from workload.yaml). Fine-tunes
``allenai/longformer-base-4096`` with a PEFT LoRA adapter on the public IMDB
sentiment dataset — Longformer supports sequences up to 4096 tokens, which suits
long reviews. Metrics go to the MLflow run that ``air`` injects; the trained
adapter (plus the classification head) and tokenizer are saved to a Unity Catalog
volume.

LoRA targets Longformer's local + global attention projections, and the
randomly-initialized classifier head is kept trainable via ``modules_to_save``
(without it, the head would stay frozen and the model could not learn the task).

Hyperparameters are read from the YAML block passed by ``air`` via HYPERPARAMETERS_PATH.
"""

import os

# HF caches must live on a writable path (the default /root/.cache is read-only
# on serverless/AIR workers). Set before importing datasets/transformers.
os.environ.setdefault("HF_HOME", "/tmp/hf_home")
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_home/datasets")

import numpy as np
import yaml
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    LongformerForSequenceClassification,
    Trainer,
    TrainingArguments,
)


def load_params() -> dict:
    """Read the hyperparameters block that `air` materializes from `parameters:`."""
    path = os.environ.get("HYPERPARAMETERS_PATH")
    if path and os.path.exists(path):
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def main():
    p = load_params()
    output_root = p.get(
        "output_root",
        "/Volumes/main/default/air_examples/lora-longformer-base-4096-classification",
    )
    model_name = p.get("model_name", "allenai/longformer-base-4096")
    dataset_name = p.get("dataset_name", "stanfordnlp/imdb")
    num_labels = int(p.get("num_labels", 2))
    max_length = int(p.get("max_length", 4096))
    train_samples = int(p.get("train_samples", 2000))
    eval_samples = int(p.get("eval_samples", 500))
    learning_rate = float(p.get("learning_rate", 1e-4))
    num_epochs = int(p.get("num_epochs", 1))
    per_device_batch_size = int(p.get("per_device_batch_size", 4))
    gradient_accumulation_steps = int(p.get("gradient_accumulation_steps", 1))
    lora_r = int(p.get("lora_r", 16))
    lora_alpha = int(p.get("lora_alpha", 32))
    lora_dropout = float(p.get("lora_dropout", 0.05))
    max_steps = int(p.get("max_steps", -1))

    # Per-attempt output dir keyed by the AIR MLflow run, so reruns and retries
    # don't overwrite a previous run's adapter.
    run_id = os.environ.get("MLFLOW_RUN_ID")
    output_dir = f"{output_root}/{run_id}" if run_id else output_root

    # --- Data (public IMDB) ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    raw = load_dataset(dataset_name)  # cache dir comes from HF_DATASETS_CACHE
    train_ds = raw["train"].shuffle(seed=42).select(range(min(train_samples, len(raw["train"]))))
    eval_ds = raw["test"].shuffle(seed=42).select(range(min(eval_samples, len(raw["test"]))))

    def tokenize(batch):
        # No padding here — the collator pads each batch dynamically.
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"]).rename_column("label", "labels")
    eval_ds = eval_ds.map(tokenize, batched=True, remove_columns=["text"]).rename_column("label", "labels")
    print(f"Train: {len(train_ds)}  |  Eval: {len(eval_ds)}", flush=True)

    # --- Model + LoRA ---
    # LongformerForSequenceClassification auto-sets global attention on the CLS
    # token when global_attention_mask is not provided, so we don't set it here.
    model = LongformerForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["query", "value", "key", "query_global", "value_global", "key_global"],
        modules_to_save=["classifier"],  # train the classification head, not just LoRA
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="weighted"),
        }

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        max_steps=max_steps,  # -1 = unbounded (use num_train_epochs)
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_strategy="no",  # a single explicit evaluate() at the end (smoke-friendly)
        save_strategy="no",
        logging_steps=10,
        bf16=True,
        report_to="mlflow",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        # Longformer needs lengths that are a multiple of its attention window,
        # so pad each batch up to the next multiple of 512 (not a fixed 512) to
        # avoid re-padding (and a warning) on every forward pass.
        data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=512),
        compute_metrics=compute_metrics,
    )

    # report_to="mlflow" lets Trainer's MLflowCallback manage the run. AIR sets
    # MLFLOW_RUN_ID in the environment, which the callback resumes, so metrics log
    # into the run AIR injected rather than a new one. We must NOT open the run
    # ourselves here — a manual mlflow.start_run() leaves a run active and the
    # callback's own start_run() then raises "Run ... is already active".
    trainer.train()
    eval_metrics = trainer.evaluate()
    print(f"\nEval accuracy: {eval_metrics['eval_accuracy']:.4f}  |  f1: {eval_metrics['eval_f1']:.4f}", flush=True)

    # Save the LoRA adapter (+ the classifier head via modules_to_save) and tokenizer.
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Adapter saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Full-weight supervised fine-tuning of Qwen3-8B for invoice/receipt extraction.

Launched via ``torchrun`` from ``workload.yaml`` on one 8xH100 node. The model is
trained with TRL's ``SFTTrainer`` and DeepSpeed ZeRO Stage 3 (embedded config, so
this script is self-contained), then the consolidated checkpoint is written to a
Unity Catalog Volume. Metrics are logged to the MLflow run that ``air`` injects.

Hyperparameters are read from the YAML block passed by ``air`` via HYPERPARAMETERS_PATH.

Prerequisite: run ``prep_data.py`` on Databricks once to stage the training and
eval data as Hugging Face datasets on the UC Volume this script reads from.
"""

import json
import os
import tempfile

import mlflow
import torch
import yaml
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


# ---------------------------------------------------------------------------
# DeepSpeed ZeRO-3 config (embedded so train.py is fully self-contained).
# "auto" values are filled by HF Trainer from TrainingArguments at runtime.
# ---------------------------------------------------------------------------
DS_CONFIG = {
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_gather_16bit_weights_on_model_save": True,
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": 1.0,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False,
}


def load_params() -> dict:
    """Read the hyperparameters block that `air` materializes from `parameters:`."""
    path = os.environ.get("HYPERPARAMETERS_PATH")
    if path and os.path.exists(path):
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def main():
    p = load_params()
    catalog = p.get("catalog", "main")
    schema = p.get("schema", "default")
    volume = p.get("volume", "invoice_sft")
    volume_model = p.get("volume_model", "invoice_sft_checkpoints")
    model_name = p.get("model_name", "Qwen/Qwen3-8B")
    learning_rate = float(p.get("learning_rate", 1e-5))
    num_epochs = int(p.get("num_epochs", 3))
    max_seq_length = int(p.get("max_seq_length", 4096))
    per_device_batch_size = int(p.get("per_device_batch_size", 2))
    gradient_accumulation_steps = int(p.get("gradient_accumulation_steps", 2))
    warmup_steps = int(p.get("warmup_steps", 20))
    max_steps = int(p.get("max_steps", -1))

    volume_base = f"/Volumes/{catalog}/{schema}/{volume}"
    volume_model_base = f"/Volumes/{catalog}/{schema}/{volume_model}"
    run_tag = f"lr{learning_rate}_ep{num_epochs}"
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Write embedded DeepSpeed config to a temp file (HF Trainer needs a path).
    ds_path = os.path.join(tempfile.mkdtemp(), "ds_config.json")
    with open(ds_path, "w") as f:
        json.dump(DS_CONFIG, f)

    # --- Load datasets from the UC Volume (staged by prep_data.py) ---
    print(f"Loading datasets from {volume_base} ...", flush=True)
    train_dataset = load_from_disk(f"{volume_base}/invoice_train_dataset")
    eval_dataset = load_from_disk(f"{volume_base}/invoice_eval_dataset")

    # Quick end-to-end validation mode: shrink the data so a full
    # train -> eval -> checkpoint-save pipeline completes in minutes.
    step_capped = max_steps is not None and max_steps > 0
    if step_capped:
        train_dataset = train_dataset.select(range(min(256, len(train_dataset))))
        eval_dataset = eval_dataset.select(range(min(64, len(eval_dataset))))
        print(f"[validation mode] max_steps={max_steps}, "
              f"capped to {len(train_dataset)} train / {len(eval_dataset)} eval samples", flush=True)

    # Suppress Qwen3 <think> blocks during chat template rendering.
    def _add_template_kwargs(example):
        example["chat_template_kwargs"] = {"enable_thinking": False}
        return example

    train_dataset = train_dataset.map(_add_template_kwargs)
    eval_dataset = eval_dataset.map(_add_template_kwargs)
    print(f"  Train: {len(train_dataset)}  |  Eval: {len(eval_dataset)}", flush=True)

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    # --- Model (full precision bf16, no quantization) ---
    # NOTE: flash_attention_2 needs the flash-attn C++ extension, which can't
    # compile under air's uv build isolation. SDPA is the built-in fallback
    # (same correctness, slightly slower).
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        use_cache=False,
    )

    # --- SFT config ---
    training_args = SFTConfig(
        output_dir=f"{volume_model_base}/invoice-ft-output-{run_tag}",
        run_name="qwen3-8b-fullweight-sft-invoice",
        num_train_epochs=num_epochs,
        max_steps=max_steps,                 # -1 = unbounded (use num_train_epochs)
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=warmup_steps,           # transformers 5.x removed warmup_ratio
        lr_scheduler_type="cosine",
        logging_steps=1 if step_capped else 10,
        # When step-capped for validation, eval/save on a step cadence so the
        # save + load_best_model_at_end path still runs within the cap.
        eval_strategy="steps" if step_capped else "epoch",
        save_strategy="steps" if step_capped else "epoch",
        eval_steps=max_steps if step_capped else None,
        save_steps=max_steps if step_capped else None,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=max_seq_length,
        packing=False,
        completion_only_loss=True,
        deepspeed=ds_path,
        report_to="mlflow",
    )

    # --- Train ---
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)

    if local_rank == 0:
        mlflow.log_params({
            "base_model": model_name,
            "training_method": "full_weight_sft",
            "deepspeed_stage": 3,
            "max_seq_length": max_seq_length,
            "train_samples": len(train_dataset),
            "eval_samples": len(eval_dataset),
        })

    print("\nTraining complete!", flush=True)
    print(f"  Train loss: {metrics['train_loss']:.4f}", flush=True)
    print(f"  Eval loss:  {eval_metrics['eval_loss']:.4f}", flush=True)

    final_path = f"{volume_model_base}/invoice-ft-final-{run_tag}"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"  Model saved to: {final_path}", flush=True)


if __name__ == "__main__":
    main()

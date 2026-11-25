"""QLoRA fine-tuning of the gpt-oss-20b mixture-of-experts model on one A10 with Unsloth.

Loads gpt-oss-20b in 4-bit, attaches LoRA adapters, runs a short SFT pass on a reasoning
dataset, reports peak GPU memory and time per step, then generates a sample answer.
"""

# Unsloth patches transformers, TRL and PEFT on import, so import it first.
import unsloth  # noqa: F401  # isort: skip
from unsloth import FastLanguageModel  # isort: skip

import json
import os
import time
from pathlib import Path

import mlflow
import torch
import yaml
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

with open(os.environ["HYPERPARAMETERS_PATH"]) as f:
    P = yaml.safe_load(f)

RUN_ID = os.environ["MLFLOW_RUN_ID"]
OUT = Path(P["output_root"]) / RUN_ID


def main():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=P["model_name"],
        max_seq_length=P["max_seq_length"],
        load_in_4bit=True,
        dtype=None,
    )
    after_load_gb = torch.cuda.max_memory_reserved() / 1e9
    model = FastLanguageModel.get_peft_model(
        model,
        r=P["lora_r"],
        lora_alpha=2 * P["lora_r"],
        lora_dropout=0,
        target_modules=P["target_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    model.print_trainable_parameters()

    data = load_dataset(P["dataset_name"], split="train")
    data = data.map(
        lambda batch: {
            "text": [
                tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                for m in batch["messages"]
            ]
        },
        batched=True,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=data,
        args=SFTConfig(
            dataset_text_field="text",
            max_length=P["max_seq_length"],
            per_device_train_batch_size=P["batch_size"],
            gradient_accumulation_steps=P["grad_accum"],
            warmup_steps=5,
            max_steps=P["max_steps"],
            learning_rate=float(P["learning_rate"]),
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            logging_steps=5,
            seed=3407,
            output_dir="/tmp/unsloth-outputs",
            save_strategy="no",
            # Logs the loss curve to the AIR run (MLFLOW_RUN_ID); the run is reopened below for final metrics.
            report_to="mlflow",
        ),
    )
    t0 = time.time()
    result = trainer.train()
    train_s = time.time() - t0

    metrics = {
        "train_loss": result.training_loss,
        "train_seconds": train_s,
        "seconds_per_step": train_s / P["max_steps"],
        "peak_gpu_mem_gb_after_load": after_load_gb,
        "peak_gpu_mem_gb_training": torch.cuda.max_memory_reserved() / 1e9,
        "gpu_total_mem_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
    }
    losses = [round(h["loss"], 4) for h in trainer.state.log_history if "loss" in h]
    print(f"[train] loss log: {losses}")

    FastLanguageModel.for_inference(model)
    messages = [{"role": "user", "content": P["sample_prompt"]}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        reasoning_effort="low",
    ).to("cuda")
    out = model.generate(**inputs, max_new_tokens=256)
    sample = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"[sample] {sample[:1500]}")

    print(json.dumps(metrics, indent=2))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (OUT / "sample.txt").write_text(sample)
    if P["save_adapter"]:
        model.save_pretrained(str(OUT / "lora_adapter"))
        tokenizer.save_pretrained(str(OUT / "lora_adapter"))
    with mlflow.start_run(run_id=RUN_ID):
        mlflow.log_metrics(metrics)
    print(f"Wrote outputs to {OUT}")


if __name__ == "__main__":
    main()

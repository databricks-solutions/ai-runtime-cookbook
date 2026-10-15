"""Single-A10 LoRA SFT; save merged Hugging Face weights for vLLM serving."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import mlflow
import yaml
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer


class LossToMlflow(TrainerCallback):
    """Record Trainer step metrics in AIR's MLflow run."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs:
            values = {
                key: float(value)
                for key, value in logs.items()
                if isinstance(value, (int, float)) and math.isfinite(value)
            }
            if values:
                mlflow.log_metrics(values, step=state.global_step)


class TrainTimeBudget(TrainerCallback):
    """Stop optimization at the budget; evaluation and saving follow normally."""

    def __init__(self, minutes: float):
        self.seconds = minutes * 60
        self.started = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.started = time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):
        if self.started is not None and time.monotonic() - self.started >= self.seconds:
            control.should_training_stop = True


def main(params: dict | None = None, run_id: str | None = None) -> None:
    if params is None:
        with open(os.environ["HYPERPARAMETERS_PATH"], encoding="utf-8") as handle:
            params = yaml.safe_load(handle)

    model_name = params["model_name"]
    output_root = Path(params["output_root"])
    # AIR creates the run and exposes its ID. Resume it explicitly rather than
    # creating a second training run with disconnected metrics and artifacts.
    if run_id is None:
        run_id = os.environ.pop("MLFLOW_RUN_ID")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"MLflow run ID: {run_id}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    total = int(params["train_examples"]) + int(params["eval_examples"])
    source = load_dataset(params["dataset_name"], split="train")
    source = source.shuffle(seed=42).select(range(total))
    split = source.train_test_split(test_size=int(params["eval_examples"]), seed=42)

    def format_example(row: dict) -> dict:
        request = row["instruction"]
        if row["input"]:
            request += "\n\n" + row["input"]
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": request}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return {"prompt": prompt, "completion": row["output"] + tokenizer.eos_token}

    train_data = split["train"].map(format_example, remove_columns=source.column_names)
    eval_data = split["test"].map(format_example, remove_columns=source.column_names)

    model = AutoModelForCausalLM.from_pretrained(model_name)
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    config = SFTConfig(
        output_dir=str(run_dir / "checkpoints"),
        run_name="qwen2-5-0-5b-lora-sft",
        max_steps=int(params["max_steps"]),
        learning_rate=float(params["learning_rate"]),
        per_device_train_batch_size=int(params["per_device_train_batch_size"]),
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=int(params["gradient_accumulation_steps"]),
        max_length=int(params["max_length"]),
        completion_only_loss=True,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=2,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
        peft_config=lora,
        callbacks=[LossToMlflow(), TrainTimeBudget(float(params["max_train_minutes"]))],
    )
    with mlflow.start_run(run_id=run_id):
        mlflow.set_tags({
            "recipe": "sft-qwen2.5-0.5b-lora-serving",
            "training_method": "LoRA SFT",
            "serving_artifact": "merged Hugging Face weights in Unity Catalog Volume",
        })
        mlflow.log_params({
            "base_model": model_name,
            "dataset": params["dataset_name"],
            "train_examples": len(train_data),
            "eval_examples": len(eval_data),
            "data_seed": 42,
            "max_steps": config.max_steps,
            "max_train_minutes": params["max_train_minutes"],
            "learning_rate": config.learning_rate,
            "per_device_train_batch_size": config.per_device_train_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "lora_rank": lora.r,
            "lora_alpha": lora.lora_alpha,
            "max_length": config.max_length,
        })
        mlflow.log_artifact(__file__, artifact_path="source")

        # LoRA leaves the base weights unchanged. Disable the adapter to score
        # the untouched model on exactly the same held-out examples.
        with trainer.model.disable_adapter():
            baseline = trainer.evaluate(metric_key_prefix="base")
        baseline_loss = baseline["base_loss"]
        mlflow.log_metric("base_eval_loss", baseline_loss, step=0)

        training = trainer.train()
        metrics = trainer.evaluate()
        eval_loss = metrics["eval_loss"]
        mlflow.log_metrics({
            "final_train_loss": training.metrics["train_loss"],
            "final_eval_loss": eval_loss,
            "eval_loss_improvement": baseline_loss - eval_loss,
            "train_runtime_seconds": training.metrics["train_runtime"],
        }, step=trainer.state.global_step)
        print(f"Base evaluation loss: {baseline_loss:.4f}", flush=True)
        print(f"Fine-tuned evaluation loss: {eval_loss:.4f}", flush=True)

        # The adapter is small enough to keep as an MLflow artifact. The
        # full merged weights remain on the volume for custom vLLM serving.
        adapter_dir = run_dir / "lora_adapter"
        trainer.model.save_pretrained(adapter_dir, safe_serialization=True)
        mlflow.log_artifacts(str(adapter_dir), artifact_path="lora_adapter")

        merged = trainer.model.merge_and_unload()
        final_dir = run_dir / "merged"
        merged.save_pretrained(final_dir, safe_serialization=True)
        tokenizer.save_pretrained(final_dir)
        result = {
            "mlflow_run_id": run_id,
            "model_name": model_name,
            "dataset_name": params["dataset_name"],
            "train_steps": trainer.state.global_step,
            "train_loss": training.metrics["train_loss"],
            "base_eval_loss": baseline_loss,
            "eval_loss": eval_loss,
            "eval_loss_improvement": baseline_loss - eval_loss,
            "model_path": str(final_dir),
        }
        result_path = run_dir / "result.json"
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(result_path))
        mlflow.set_tag("merged_model_path", str(final_dir))
        print(f"Merged model: {final_dir}", flush=True)


if __name__ == "__main__":
    main()

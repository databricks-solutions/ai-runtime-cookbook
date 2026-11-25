"""Supervised fine-tuning of the student on the teacher's correct solutions (TRL SFTTrainer)."""

import json
import math
import os
import time
from pathlib import Path

import mlflow
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

from common import P, WORK


class StopOnNonFiniteGradients(TrainerCallback):
    """Fail fast instead of finishing an epoch with NaN weights and saving a broken student."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        grad_norm = (logs or {}).get("grad_norm")
        if grad_norm is not None and not math.isfinite(grad_norm):
            raise RuntimeError(f"non-finite gradient norm at step {state.global_step}: {logs}")


def main():
    # With cuDNN fused attention, a validation run on H100 hit NaN gradients mid-epoch;
    # the FlashAttention and memory-efficient SDPA kernels are stable.
    torch.backends.cuda.enable_cudnn_sdp(False)
    data = load_dataset("json", data_files=f"{WORK}/teacher_sft.jsonl", split="train")
    tok = AutoTokenizer.from_pretrained(P["student_model"])
    args = SFTConfig(
        output_dir=f"{WORK}/student-ckpt",
        num_train_epochs=P["epochs"],
        per_device_train_batch_size=P["batch_size"],
        gradient_accumulation_steps=1,
        learning_rate=float(P["learning_rate"]),
        lr_scheduler_type="cosine",
        warmup_steps=0.05,
        max_length=1024,
        bf16=True,
        logging_steps=10,
        logging_nan_inf_filter=False,
        save_strategy="no",
        report_to="mlflow",
    )
    # AIR sets MLFLOW_RUN_ID. Open that run before training and pop the variable, so the
    # trainer's MLflow callback logs into this run instead of starting and ending its own.
    run_id = os.environ.pop("MLFLOW_RUN_ID", None)
    if not run_id:
        raise RuntimeError("MLFLOW_RUN_ID is not set; AIR provides it for each run.")
    with mlflow.start_run(run_id=run_id):
        t0 = time.time()
        trainer = SFTTrainer(model=P["student_model"], args=args, train_dataset=data, processing_class=tok,
                             callbacks=[StopOnNonFiniteGradients()])
        result = trainer.train()
        trainer.save_model(f"{WORK}/student")
        tok.save_pretrained(f"{WORK}/student")
    stats = {"student_train_loss": result.training_loss, "student_train_seconds": time.time() - t0,
             "student_train_examples": len(data)}
    print(json.dumps(stats), flush=True)
    Path(f"{WORK}/student_stats.json").write_text(json.dumps(stats))


if __name__ == "__main__":
    main()

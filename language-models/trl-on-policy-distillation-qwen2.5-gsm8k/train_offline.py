"""Offline arm, step 2: SFT on the teacher's fixed solutions for the same number of steps as the on-policy run."""

import json
import os
import time
from pathlib import Path

import mlflow
import torch
from datasets import load_dataset
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

from common import (
    BATCH_SIZE,
    LEARNING_RATE,
    STUDENT_MODEL,
    TRAIN_STEPS,
    WORK,
    StopOnNonFiniteGradients,
)


class MLflowLossCallback(TrainerCallback):
    """Logs the SFT loss to the AIR run under offline/, so it does not overwrite the on-policy curve keys."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        for key in ("loss", "grad_norm", "learning_rate"):
            if logs and key in logs:
                mlflow.log_metric(f"offline/{key}", float(logs[key]), step=state.global_step)


def main():
    torch.backends.cuda.enable_cudnn_sdp(False)  # same cuDNN workaround as train_opd.py
    data = load_dataset("json", data_files=f"{WORK}/offline_sft.jsonl", split="train")
    args = SFTConfig(
        output_dir=f"{WORK}/offline-ckpt",
        # max_steps, not epochs: the fixed-text set is slightly smaller than the question set, so this keeps the
        # optimizer-step count equal to the on-policy run.
        max_steps=TRAIN_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=LEARNING_RATE,
        max_length=1024,
        gradient_checkpointing=False,
        bf16=True,
        logging_steps=10,
        logging_nan_inf_filter=False,
        save_strategy="no",
        # report_to="none" keeps the trainer's own MLflow run out. MLflowLossCallback logs under offline/ instead.
        report_to="none",
        seed=0,
    )
    # The run stays open from before training until the final stats are written.
    mlflow.start_run(run_id=os.environ["MLFLOW_RUN_ID"])
    t0 = time.time()
    trainer = SFTTrainer(
        model=STUDENT_MODEL,
        args=args,
        train_dataset=data,
        callbacks=[StopOnNonFiniteGradients(), MLflowLossCallback()],
    )
    # Control for the save path: the untrained student saved and reloaded through the same call as the trained one.
    trainer.save_model(f"{WORK}/control-offline")
    trainer.processing_class.save_pretrained(f"{WORK}/control-offline")
    # The SFT target should end with the chat template's end-of-turn token (<|im_end|> for Qwen2.5).
    tokenizer = trainer.processing_class
    example = data[0]
    rendered = tokenizer.apply_chat_template(example["prompt"] + example["completion"], tokenize=False)
    ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    print(f"offline SFT example last token ids {ids[-4:]} decoded {tokenizer.decode(ids[-4:])!r}", flush=True)
    result = trainer.train()
    trainer.save_model(f"{WORK}/offline-student")
    trainer.processing_class.save_pretrained(f"{WORK}/offline-student")
    stats = {
        "offline_train_steps": int(result.global_step),
        "offline_train_seconds": time.time() - t0,
        "offline_training_loss": float(result.training_loss),
    }
    print(json.dumps(stats), flush=True)
    Path(f"{WORK}/offline_train_stats.json").write_text(json.dumps(stats))
    mlflow.end_run()


if __name__ == "__main__":
    main()

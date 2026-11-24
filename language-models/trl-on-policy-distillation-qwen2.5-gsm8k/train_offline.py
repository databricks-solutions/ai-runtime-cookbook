"""Offline arm, step 2: SFT on the teacher's fixed solutions for the same number of steps as the on-policy run."""

import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

from common import (
    BATCH_SIZE,
    LEARNING_RATE,
    STUDENT_MODEL,
    TRAIN_STEPS,
    WORK,
    StopOnNonFiniteGradients,
)


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
        # report_to="none": the on-policy run owns the MLflow curve keys, and this run's loss would overwrite them.
        report_to="none",
        seed=0,
    )
    t0 = time.time()
    trainer = SFTTrainer(
        model=STUDENT_MODEL,
        args=args,
        train_dataset=data,
        callbacks=[StopOnNonFiniteGradients()],
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


if __name__ == "__main__":
    main()

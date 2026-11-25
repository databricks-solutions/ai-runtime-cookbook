"""On-policy distillation with TRL's DistillationTrainer.

Each step the student samples completions for a batch of GSM8K questions. The frozen teacher then scores every
token of those completions, and the loss pulls the student's distribution toward the teacher's on the student's
own text. With beta=1.0 the loss is the reverse KL, KL(student || teacher).
"""

import json
import os
import time
from pathlib import Path

import torch
from trl import DistillationConfig, DistillationTrainer

from common import (
    BATCH_SIZE,
    BETA,
    LEARNING_RATE,
    MAX_COMPLETION_LENGTH,
    STUDENT_MODEL,
    TEACHER_MODEL,
    TRAIN_STEPS,
    WORK,
    StopOnNonFiniteGradients,
    messages,
    train_subset,
)


def main():
    # cuDNN fused attention produced NaN gradients for this student on H100 in a validation run.
    torch.backends.cuda.enable_cudnn_sdp(False)
    # Keep only the prompt column: the student generates its own completions, and no reference answer is used.
    questions = train_subset()
    prompts = questions.map(
        lambda ex: {"prompt": messages(ex["question"])},
        remove_columns=questions.column_names,
    )
    args = DistillationConfig(
        output_dir=f"{WORK}/opd-ckpt",
        num_train_epochs=1,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=LEARNING_RATE,
        beta=BETA,
        temperature=1.0,
        max_completion_length=MAX_COMPLETION_LENGTH,
        use_vllm=True,
        vllm_mode="colocate",
        # The teacher is loaded first and shares the GPU, so the colocated student engine gets a fixed slice.
        vllm_gpu_memory_utilization=0.3,
        vllm_max_model_length=2048,
        teacher_model_init_kwargs={"dtype": "bfloat16"},
        gradient_checkpointing=False,
        bf16=True,
        logging_steps=5,
        logging_nan_inf_filter=False,
        save_strategy="no",
        # Loss and entropy curves go to the AIR run (MLFLOW_RUN_ID).
        report_to="mlflow",
        seed=0,
    )
    trainer = DistillationTrainer(
        model=STUDENT_MODEL,
        teacher_model=TEACHER_MODEL,
        args=args,
        train_dataset=prompts,
        callbacks=[StopOnNonFiniteGradients()],
    )
    # Control for the save path: the untrained student saved and reloaded through the same call as the trained one.
    trainer.save_model(f"{WORK}/control-opd")
    trainer.processing_class.save_pretrained(f"{WORK}/control-opd")
    t0 = time.time()
    result = trainer.train()
    train_seconds = time.time() - t0
    trainer.save_model(f"{WORK}/opd-student")
    trainer.processing_class.save_pretrained(f"{WORK}/opd-student")
    stats = {
        "opd_train_steps": int(result.global_step),
        "opd_expected_steps": TRAIN_STEPS,
        "opd_train_seconds": train_seconds,
        "opd_training_loss": float(result.training_loss),
    }
    if stats["opd_train_steps"] != TRAIN_STEPS:
        raise RuntimeError(f"expected {TRAIN_STEPS} steps, ran {stats['opd_train_steps']}")
    print(json.dumps(stats), flush=True)
    Path(f"{WORK}/opd_stats.json").write_text(json.dumps(stats))


if __name__ == "__main__":
    Path(WORK).mkdir(parents=True, exist_ok=True)
    main()

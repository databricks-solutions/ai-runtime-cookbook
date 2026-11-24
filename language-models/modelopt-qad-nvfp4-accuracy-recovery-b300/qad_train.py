"""Stage 3: quantization-aware distillation (QAD).

The student is the NVFP4 fake-quantized model from stage 1 (weights and activations quantized in
the forward pass). The teacher is the original bf16 model. The loss is the forward KL between their
logits on GSM8K train answer tokens, so the student learns the teacher's next-token distribution
rather than only the hard answer tokens.

Launched with accelerate + FSDP2 on all GPUs (see run.sh). Accelerate's FSDP2 upcasts the bf16
student to fp32 master weights under bf16 mixed precision (accelerate/utils/fsdp_utils.py); plain
DDP would apply lr=1e-5 updates to bf16 weights, where many of them round away.
"""

import os

import mlflow
import modelopt.torch.opt as mto
import torch
import transformers
from common import P, collate_fn, paths, train_examples
from modelopt.torch.quantization.plugins.transformers_trainer import QADTrainer

mto.enable_huggingface_checkpointing()


def main():
    dirs = paths()  # read MLFLOW_RUN_ID before rank 0 removes it below
    if int(os.environ.get("RANK", "0")) == 0:
        # Open the AIR run before training, so the Trainer's MLflow callback logs into it.
        # The callback then sees an active run and does not start or end one.
        mlflow.start_run(run_id=os.environ.pop("MLFLOW_RUN_ID"))

    tok = transformers.AutoTokenizer.from_pretrained(P["model_name"])
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    student = transformers.AutoModelForCausalLM.from_pretrained(str(dirs["ptq_ckpt"]),
                                                                dtype=torch.bfloat16)
    teacher = transformers.AutoModelForCausalLM.from_pretrained(P["model_name"],
                                                                dtype=torch.bfloat16)
    student.config.use_cache = False
    train_ds = train_examples(tok, limit=P["train_examples"])

    # QADTrainer writes modelopt_state_train.pth into output_dir during construction.
    dirs["qad_ckpt"].mkdir(parents=True, exist_ok=True)
    args = transformers.TrainingArguments(
        output_dir=str(dirs["qad_ckpt"]),
        num_train_epochs=int(P["epochs"]),
        learning_rate=float(P["learning_rate"]),
        lr_scheduler_type="cosine",
        warmup_steps=0.05,  # a float in [0, 1) is a ratio of total steps in transformers 5.x
        per_device_train_batch_size=int(P["per_device_batch_size"]),
        gradient_accumulation_steps=1,
        bf16=True,
        logging_steps=1,
        report_to=["mlflow"],
        save_strategy="no",
        do_eval=False,
        seed=42,
        dataloader_drop_last=True,
    )
    trainer = QADTrainer(
        model=student,
        processing_class=tok,
        args=args,
        distill_args={"teacher_model": teacher, "temperature": 1.0, "criterion": "logits_loss",
                      "liger_jsd_beta": 0.0},
        train_dataset=train_ds,
        data_collator=collate_fn(tok.pad_token_id),
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(str(dirs["qad_ckpt"]))

    if int(os.environ.get("RANK", "0")) == 0:
        mlflow.log_params({"student": "NVFP4 PTQ of " + P["model_name"],
                           "teacher": P["model_name"], "loss": "forward KL on logits",
                           "train_examples": len(train_ds)})
        mlflow.end_run()
    print(f"[qad] saved {dirs['qad_ckpt']}", flush=True)


if __name__ == "__main__":
    main()

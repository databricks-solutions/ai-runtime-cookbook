"""Preference tuning (DPO) then reinforcement learning (GRPO) with Hugging Face TRL.

Stage 1: DPO with a LoRA adapter on UltraFeedback preference pairs, then merge the adapter.
Stage 2: GRPO with a fresh LoRA adapter on GSM8K math questions, rewarded for a correct
         final answer in the requested format.
Eval:    GSM8K test accuracy (greedy) for the base model, after DPO, and after GRPO.
"""

import gc
import json
import os
import re
import time
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer, GRPOConfig, GRPOTrainer

with open(os.environ["HYPERPARAMETERS_PATH"]) as f:
    P = yaml.safe_load(f)

RUN_ID = os.environ.get("MLFLOW_RUN_ID", "local")
WORK = Path("/tmp/trl-recipe")
OUT = Path(P["output_root"]) / RUN_ID
SYSTEM = (
    "Solve the math problem. Think step by step, then give the final answer "
    "on the last line as: #### <number>"
)


def gsm8k_answer(text):
    return normalize(text.split("####")[-1].strip())


def extract_number(text):
    """Final answer after '####', else the last number in the text (small models often skip the marker)."""
    match = re.search(r"####\s*\$?(-?[\d,]*\.?\d+)", text)
    if match:
        return normalize(match.group(1))
    numbers = re.findall(r"-?[\d,]*\.?\d+", text)
    return normalize(numbers[-1]) if numbers else None


def normalize(number):
    number = number.replace(",", "").rstrip(".")
    return number[:-2] if number.endswith(".0") else number


def to_prompt(question):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]


def lora():
    return LoraConfig(
        r=P["lora_r"],
        lora_alpha=2 * P["lora_r"],
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )


@torch.no_grad()
def gsm8k_accuracy(model_path, questions, answers):
    tok = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).cuda().eval()
    correct = 0
    bs = P["eval_batch_size"]
    for i in range(0, len(questions), bs):
        texts = [
            tok.apply_chat_template(to_prompt(q), tokenize=False, add_generation_prompt=True)
            for q in questions[i : i + bs]
        ]
        enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
        out = model.generate(**enc, max_new_tokens=P["max_completion_length"], do_sample=False)
        decoded = tok.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        for pred, gold in zip(decoded, answers[i : i + bs]):
            correct += extract_number(pred) == gold
    del model
    release_gpu_memory()
    return correct / len(questions)


def release_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    metrics = {}

    test = load_dataset("openai/gsm8k", "main", split="test").select(range(P["eval_samples"]))
    test_q = test["question"]
    test_a = [gsm8k_answer(a) for a in test["answer"]]

    t0 = time.time()
    metrics["gsm8k_acc_base"] = gsm8k_accuracy(P["model_name"], test_q, test_a)
    print(f"[eval] base accuracy: {metrics['gsm8k_acc_base']:.3f} ({time.time() - t0:.0f}s)")

    # ---- Stage 1: DPO ----
    tok = AutoTokenizer.from_pretrained(P["model_name"])
    prefs = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
    prefs = prefs.shuffle(seed=0).select(range(P["dpo_samples"]))
    dpo_args = DPOConfig(
        output_dir=str(WORK / "dpo"),
        per_device_train_batch_size=P["dpo_batch_size"],
        gradient_accumulation_steps=P["dpo_grad_accum"],
        learning_rate=float(P["dpo_learning_rate"]),
        beta=float(P["dpo_beta"]),
        max_length=P["dpo_max_length"],
        num_train_epochs=1,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
    )
    t0 = time.time()
    trainer = DPOTrainer(
        model=P["model_name"],
        args=dpo_args,
        train_dataset=prefs,
        processing_class=tok,
        peft_config=lora(),
    )
    result = trainer.train()
    metrics["dpo_train_loss"] = result.training_loss
    metrics["dpo_seconds"] = time.time() - t0
    print(f"[dpo] loss {result.training_loss:.4f} in {metrics['dpo_seconds']:.0f}s")
    dpo_dir = WORK / "dpo-merged"
    trainer.model.merge_and_unload().save_pretrained(dpo_dir)
    tok.save_pretrained(dpo_dir)
    del trainer
    release_gpu_memory()

    metrics["gsm8k_acc_after_dpo"] = gsm8k_accuracy(str(dpo_dir), test_q, test_a)
    print(f"[eval] after DPO accuracy: {metrics['gsm8k_acc_after_dpo']:.3f}")

    # ---- Stage 2: GRPO ----
    train = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=0)
    train = train.select(range(P["grpo_samples"])).map(
        lambda x: {"prompt": to_prompt(x["question"]), "solution": gsm8k_answer(x["answer"])},
        remove_columns=["question", "answer"],
    )

    def correctness_reward(completions, solution, **kwargs):
        texts = [c[0]["content"] for c in completions]
        return [1.0 if extract_number(t) == s else 0.0 for t, s in zip(texts, solution)]

    def format_reward(completions, **kwargs):
        return [0.2 if "####" in c[0]["content"] else 0.0 for c in completions]

    grpo_args = GRPOConfig(
        output_dir=str(WORK / "grpo"),
        per_device_train_batch_size=P["grpo_batch_size"],
        gradient_accumulation_steps=1,
        num_generations=P["grpo_num_generations"],
        max_completion_length=P["max_completion_length"],
        learning_rate=float(P["grpo_learning_rate"]),
        loss_type=P["grpo_loss_type"],
        max_steps=P["grpo_steps"],
        temperature=1.0,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
    )
    t0 = time.time()
    trainer = GRPOTrainer(
        model=str(dpo_dir),
        reward_funcs=[correctness_reward, format_reward],
        args=grpo_args,
        train_dataset=train,
        processing_class=AutoTokenizer.from_pretrained(dpo_dir),
        peft_config=lora(),
    )
    trainer.train()
    metrics["grpo_seconds"] = time.time() - t0
    history = [h for h in trainer.state.log_history if "reward" in h]
    if history:
        metrics["grpo_reward_first"] = history[0]["reward"]
        metrics["grpo_reward_last"] = history[-1]["reward"]
    print(f"[grpo] {P['grpo_steps']} steps in {metrics['grpo_seconds']:.0f}s; reward log: "
          f"{[round(h['reward'], 3) for h in history]}")
    grpo_dir = WORK / "grpo-merged"
    trainer.model.merge_and_unload().save_pretrained(grpo_dir)
    AutoTokenizer.from_pretrained(dpo_dir).save_pretrained(grpo_dir)
    del trainer
    release_gpu_memory()

    metrics["gsm8k_acc_after_grpo"] = gsm8k_accuracy(str(grpo_dir), test_q, test_a)
    print(f"[eval] after GRPO accuracy: {metrics['gsm8k_acc_after_grpo']:.3f}")

    print(json.dumps(metrics, indent=2))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    try:
        import mlflow

        with mlflow.start_run(run_id=RUN_ID):
            mlflow.log_metrics(metrics)
    except Exception as e:  # MLflow logging is optional for the recipe.
        print(f"MLflow logging skipped: {e}")
    print(f"Wrote metrics to {OUT}")


if __name__ == "__main__":
    main()

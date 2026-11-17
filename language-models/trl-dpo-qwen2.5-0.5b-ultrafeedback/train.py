"""Preference tuning (DPO) of Qwen2.5-0.5B-Instruct with Hugging Face TRL, with a reward-model eval.

1. Train a LoRA adapter with DPO on UltraFeedback preference pairs.
2. Measure held-out preference accuracy: how often the tuned model ranks the chosen answer
   above the rejected one, relative to the base model.
3. Generate answers to held-out prompts with and without the adapter, score both with a
   reward model, and report the DPO model's win rate over the base model.
"""

import json
import os
import time
from pathlib import Path

import mlflow
import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import DPOConfig, DPOTrainer

with open(os.environ["HYPERPARAMETERS_PATH"]) as f:
    P = yaml.safe_load(f)

OUT = Path(P["output_root"]) / os.environ["MLFLOW_RUN_ID"]


@torch.no_grad()
def generate(model, tok, prompts):
    """Greedy answers for a list of chat prompts, in batches."""
    model.eval()
    answers = []
    for i in range(0, len(prompts), P["generation_batch_size"]):
        texts = [tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
                 for p in prompts[i : i + P["generation_batch_size"]]]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        out = model.generate(**enc, max_new_tokens=P["max_new_tokens"], do_sample=False,
                             use_cache=True, pad_token_id=tok.pad_token_id)
        answers += tok.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
    return answers


@torch.no_grad()
def reward_scores(prompts, answers):
    """Score each (prompt, answer) pair with the reward model."""
    tok = AutoTokenizer.from_pretrained(P["reward_model"])
    rm = AutoModelForSequenceClassification.from_pretrained(
        P["reward_model"], dtype=torch.bfloat16, num_labels=1
    ).cuda().eval()
    scores = []
    for prompt, answer in zip(prompts, answers):
        text = tok.apply_chat_template(prompt + [{"role": "assistant", "content": answer}], tokenize=False)
        if tok.bos_token and text.startswith(tok.bos_token):
            text = text[len(tok.bos_token) :]
        enc = tok(text, return_tensors="pt", truncation=True, max_length=4096).to("cuda")
        scores.append(rm(**enc).logits[0][0].item())
    return scores


def main():
    tok = AutoTokenizer.from_pretrained(P["model_name"])
    train = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
    train = train.shuffle(seed=0).select(range(P["train_pairs"]))
    test = load_dataset("trl-lib/ultrafeedback_binarized", split="test").shuffle(seed=0)
    eval_pairs = test.select(range(P["eval_pairs"]))

    trainer = DPOTrainer(
        model=P["model_name"],
        args=DPOConfig(
            output_dir="/tmp/dpo",
            per_device_train_batch_size=P["batch_size"],
            gradient_accumulation_steps=P["grad_accum"],
            per_device_eval_batch_size=P["batch_size"],
            # AIR passes small floats in parameters as strings such as "5e-05".
            learning_rate=float(P["learning_rate"]),
            beta=float(P["beta"]),
            max_length=P["max_length"],
            num_train_epochs=1,
            bf16=True,
            gradient_checkpointing=True,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
        ),
        train_dataset=train,
        eval_dataset=eval_pairs,
        processing_class=tok,
        peft_config=LoraConfig(r=P["lora_r"], lora_alpha=2 * P["lora_r"],
                               target_modules="all-linear", task_type="CAUSAL_LM"),
    )
    t0 = time.time()
    result = trainer.train()
    metrics = {"train_loss": result.training_loss, "train_seconds": time.time() - t0}

    # With LoRA, the reference model is the base model (adapter disabled), so preference
    # accuracy is 50% before training by construction.
    held_out = trainer.evaluate()
    held_out = {**trainer.state.log_history[-1], **held_out}  # TRL adds its reward metrics when logging
    metrics["eval_preference_accuracy"] = held_out["eval_rewards/accuracies"]
    metrics["eval_reward_margin"] = held_out["eval_rewards/margins"]
    print(f"[eval] held-out preference accuracy {metrics['eval_preference_accuracy']:.3f} "
          f"on {len(eval_pairs)} pairs, mean margin {metrics['eval_reward_margin']:.3f}")

    # Win rate on prompts that were not used for training or the preference eval.
    start = P["eval_pairs"]
    prompts = [row["chosen"][:-1] for row in test.select(range(start, start + P["win_rate_prompts"]))]
    gen_tok = AutoTokenizer.from_pretrained(P["model_name"], padding_side="left")
    model = trainer.model
    with model.disable_adapter():
        base_answers = generate(model, gen_tok, prompts)
    dpo_answers = generate(model, gen_tok, prompts)
    base_scores = reward_scores(prompts, base_answers)
    dpo_scores = reward_scores(prompts, dpo_answers)
    # A tie counts as half a win.
    wins = sum(1.0 if d > b else 0.5 if d == b else 0.0 for d, b in zip(dpo_scores, base_scores))
    metrics.update({
        "win_rate_vs_base": wins / len(prompts),
        "mean_reward_base": sum(base_scores) / len(prompts),
        "mean_reward_dpo": sum(dpo_scores) / len(prompts),
        "mean_answer_chars_base": sum(map(len, base_answers)) / len(prompts),
        "mean_answer_chars_dpo": sum(map(len, dpo_answers)) / len(prompts),
    })
    print(f"[eval] DPO win rate vs base {metrics['win_rate_vs_base']:.3f} on {len(prompts)} prompts; "
          f"mean reward {metrics['mean_reward_base']:.2f} -> {metrics['mean_reward_dpo']:.2f}")
    print(json.dumps(metrics, indent=2))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with open(OUT / "generations.jsonl", "w") as f:
        for p, b, d, bs, ds in zip(prompts, base_answers, dpo_answers, base_scores, dpo_scores):
            f.write(json.dumps({"prompt": p[-1]["content"], "base": b, "dpo": d,
                                "base_reward": bs, "dpo_reward": ds}) + "\n")
    model.save_pretrained(OUT / "dpo_adapter")
    tok.save_pretrained(OUT / "dpo_adapter")
    with mlflow.start_run(run_id=os.environ["MLFLOW_RUN_ID"]):
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(OUT / "metrics.json"))
        mlflow.log_artifact(str(OUT / "generations.jsonl"))
    print(f"Wrote metrics, generations and the LoRA adapter to {OUT}")


if __name__ == "__main__":
    main()

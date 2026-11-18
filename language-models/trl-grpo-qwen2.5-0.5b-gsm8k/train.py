"""GRPO reinforcement learning on GSM8K math with Hugging Face TRL.

Trains a LoRA adapter on Qwen2.5-0.5B-Instruct with GRPO (Group Relative Policy Optimization).
The reward is 1 for a correct final answer, plus 0.2 if the answer contains the `####` marker.
The recipe measures greedy GSM8K test accuracy before and after training.
"""

import json
import os
import re
from pathlib import Path

import mlflow
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

with open(os.environ["HYPERPARAMETERS_PATH"]) as f:
    P = yaml.safe_load(f)

# Read the AIR run ID now: MLflow clears MLFLOW_RUN_ID when the trainer ends the run.
RUN_ID = os.environ["MLFLOW_RUN_ID"]
OUT = Path(P["output_root"]) / RUN_ID
SYSTEM = (
    "Solve the math problem. Think step by step, then give the final answer "
    "on the last line as: #### <number>"
)


def normalize(number):
    # Compare as numbers, so 18 and 18.00, or 1,000 and 1000, match.
    return float(number.replace(",", ""))


def gsm8k_answer(text):
    return normalize(text.split("####")[-1].strip())


def extract_number(text):
    """Final answer after '####', else the last number in the text (small models often skip the marker)."""
    match = re.search(r"####\s*\$?(-?[\d,]*\.?\d+)", text)
    if match:
        return normalize(match.group(1))
    numbers = re.findall(r"-?[\d,]*\.?\d+", text)
    return normalize(numbers[-1]) if numbers else None


def to_prompt(question):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]


def gsm8k_accuracy(model, tok, questions, answers):
    """Greedy accuracy on GSM8K questions."""
    model.eval()
    correct = 0
    bs = P["eval_batch_size"]
    for i in range(0, len(questions), bs):
        texts = [tok.apply_chat_template(to_prompt(q), tokenize=False, add_generation_prompt=True)
                 for q in questions[i : i + bs]]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        out = model.generate(**enc, max_new_tokens=P["max_completion_length"], do_sample=False,
                             use_cache=True, pad_token_id=tok.pad_token_id)
        decoded = tok.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        correct += sum(extract_number(pred) == gold for pred, gold in zip(decoded, answers[i : i + bs]))
    return correct / len(questions)


def correctness_reward(completions, solution, **kwargs):
    return [1.0 if extract_number(c[0]["content"]) == s else 0.0 for c, s in zip(completions, solution)]


def format_reward(completions, **kwargs):
    return [0.2 if "####" in c[0]["content"] else 0.0 for c in completions]


def main():
    test = load_dataset("openai/gsm8k", "main", split="test")
    if P["eval_samples"] > 0:
        test = test.select(range(P["eval_samples"]))
    test_q = test["question"]
    test_a = [gsm8k_answer(a) for a in test["answer"]]

    # GRPOTrainer's sampler shuffles the prompts. The run uses steps × batch_size ÷ num_generations of them.
    train = load_dataset("openai/gsm8k", "main", split="train").map(
        lambda x: {"prompt": to_prompt(x["question"]), "solution": gsm8k_answer(x["answer"])},
        remove_columns=["question", "answer"],
    )

    # Left padding, for batched generation.
    tok = AutoTokenizer.from_pretrained(P["model_name"], padding_side="left")
    trainer = GRPOTrainer(
        model=P["model_name"],
        reward_funcs=[correctness_reward, format_reward],
        args=GRPOConfig(
            output_dir="/tmp/grpo",
            per_device_train_batch_size=P["batch_size"],
            num_generations=P["num_generations"],
            max_completion_length=P["max_completion_length"],
            # AIR passes small floats in parameters as strings such as "2e-05".
            learning_rate=float(P["learning_rate"]),
            loss_type=P["loss_type"],
            max_steps=P["steps"],
            temperature=1.0,
            bf16=True,
            gradient_checkpointing=True,
            logging_steps=10,
            save_strategy="no",
            # Logs the loss and reward curves to the AIR run (MLFLOW_RUN_ID).
            report_to="mlflow",
        ),
        train_dataset=train,
        processing_class=tok,
        peft_config=LoraConfig(r=P["lora_r"], lora_alpha=2 * P["lora_r"],
                               target_modules="all-linear", task_type="CAUSAL_LM"),
    )
    model = trainer.model

    metrics = {}
    with model.disable_adapter():
        metrics["gsm8k_acc_base"] = gsm8k_accuracy(model, tok, test_q, test_a)
    print(f"[eval] base accuracy {metrics['gsm8k_acc_base']:.3f} on {len(test_q)} questions")

    metrics["train_seconds"] = trainer.train().metrics["train_runtime"]
    rewards = [h["reward"] for h in trainer.state.log_history if "reward" in h]
    metrics["reward_first_logged"], metrics["reward_last_logged"] = rewards[0], rewards[-1]
    print(f"[grpo] {P['steps']} steps in {metrics['train_seconds']:.0f}s; mean reward per logging "
          f"interval: {[round(r, 3) for r in rewards]}")

    metrics["gsm8k_acc_grpo"] = gsm8k_accuracy(model, tok, test_q, test_a)
    print(f"[eval] after GRPO accuracy {metrics['gsm8k_acc_grpo']:.3f}")
    print(json.dumps(metrics, indent=2))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    model.save_pretrained(OUT / "grpo_adapter")
    # The MLflow callback ended the AIR run when training finished; reopen it by ID.
    with mlflow.start_run(run_id=RUN_ID):
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(OUT / "metrics.json"))
    print(f"Wrote metrics and the LoRA adapter to {OUT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""LoRA hyperparameter search for Qwen2.5-0.5B with Ray Tune and ASHA.

The workload starts a four-node Ray cluster with one A10 GPU per node. Ray Tune
runs one trial per GPU. ASHA concentrates GPU time on promising configurations
by stopping trials that fall behind at each rung.

The model and dataset are public and do not require a Hugging Face token.
"""

import os

import mlflow
import ray
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATASET_NAME = "tatsu-lab/alpaca"
MAX_SEQ_LEN = 512

# Trials report every EVAL_STEPS optimizer steps, so ASHA sees at most MAX_ITERATIONS
# reports per trial and can start pruning once a trial has sent GRACE_PERIOD of them.
EVAL_STEPS = 25
MAX_ITERATIONS = 12
GRACE_PERIOD = 3

NUM_SAMPLES = 8
TRAIN_EXAMPLES = 2000
EVAL_EXAMPLES = 200


def build_datasets():
    """Tokenizes the SFT data once on the driver.

    Returns TensorDatasets so the tokenized splits serialize by value, which is what lets
    tune.with_parameters hand them to trials on any node in the cluster.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = load_dataset(DATASET_NAME, split=f"train[:{TRAIN_EXAMPLES + EVAL_EXAMPLES}]")

    def format_example(row):
        prompt = f"### Instruction:\n{row['instruction']}\n\n"
        if row.get("input"):
            prompt += f"### Input:\n{row['input']}\n\n"
        text = f"{prompt}### Response:\n{row['output']}{tokenizer.eos_token}"
        out = tokenizer(text, truncation=True, max_length=MAX_SEQ_LEN, padding="max_length")
        # -100 is cross-entropy's ignore_index, so the loss covers only real tokens and
        # eval_loss stays a meaningful signal for ASHA to rank trials by.
        out["labels"] = [token if mask == 1 else -100 for token, mask in zip(out["input_ids"], out["attention_mask"])]
        return out

    tokenized = raw.map(format_example, remove_columns=raw.column_names)
    split = tokenized.train_test_split(test_size=EVAL_EXAMPLES, shuffle=True, seed=0)

    def to_tensors(ds):
        return TensorDataset(
            torch.tensor(ds["input_ids"], dtype=torch.long),
            torch.tensor(ds["attention_mask"], dtype=torch.long),
            torch.tensor(ds["labels"], dtype=torch.long),
        )

    return to_tensors(split["train"]), to_tensors(split["test"])


def evaluate(model, loader, device):
    """Mean cross-entropy over the held-out split. This is the metric ASHA prunes on."""
    model.eval()
    total, batches = 0.0, 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            out = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                labels=labels.to(device),
            )
            total += out.loss.item()
            batches += 1
    model.train()
    return total / max(batches, 1)


def train_fn(config, train_data=None, eval_data=None):
    """One trial: LoRA fine-tunes Qwen on a single GPU and reports eval_loss to ASHA."""
    # Ray Tune pins one GPU per trial via CUDA_VISIBLE_DEVICES, so cuda:0 is this trial's.
    device = torch.device("cuda")

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
    model.config.use_cache = False
    lora = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_r"] * config["lora_alpha_ratio"],
        lora_dropout=config["lora_dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora).to(device)

    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_data, batch_size=config["batch_size"])

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    model.train()
    step = 0
    max_steps = EVAL_STEPS * MAX_ITERATIONS
    # Cycle the loader over multiple epochs until the step budget is spent.
    while step < max_steps:
        for input_ids, attention_mask, labels in train_loader:
            out = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                labels=labels.to(device),
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            if step % EVAL_STEPS == 0:
                # ASHA stops or continues the trial based on this report.
                tune.report(
                    {
                        "eval_loss": evaluate(model, eval_loader, device),
                        "train_loss": out.loss.item(),
                        "step": step,
                    }
                )
            if step >= max_steps:
                break


def main():
    ray.init(address="auto")

    num_nodes = int(os.environ.get("NUM_NODES", 1))
    total_gpus = int(ray.cluster_resources().get("GPU", 0))
    if total_gpus < 1:
        raise SystemExit("No GPUs registered with Ray; check GPU discovery on the cluster.")
    print(f"Cluster ready: {num_nodes} node(s), {total_gpus} GPU(s)", flush=True)
    print(f"Running {NUM_SAMPLES} trials, up to {total_gpus} concurrently\n", flush=True)

    train_data, eval_data = build_datasets()

    param_space = {
        "lr": tune.loguniform(1e-5, 1e-3),
        "lora_r": tune.choice([8, 16, 32]),
        "lora_alpha_ratio": tune.choice([1, 2]),
        "lora_dropout": tune.uniform(0.0, 0.1),
        "weight_decay": tune.choice([0.0, 0.01]),
        "batch_size": tune.choice([4, 8]),
    }

    tuner = tune.Tuner(
        # with_resources gives each trial a whole GPU so trials never share a device.
        tune.with_resources(
            tune.with_parameters(train_fn, train_data=train_data, eval_data=eval_data),
            resources={"gpu": 1},
        ),
        param_space=param_space,
        tune_config=tune.TuneConfig(
            metric="eval_loss",
            mode="min",
            scheduler=ASHAScheduler(
                max_t=MAX_ITERATIONS,
                grace_period=GRACE_PERIOD,
                reduction_factor=2,
            ),
            num_samples=NUM_SAMPLES,
        ),
    )

    results = tuner.fit()

    # Surface trial failures: a best result is only meaningful when the whole sweep ran.
    if results.num_errors:
        raise RuntimeError(
            f"{results.num_errors} of {len(results)} trials errored; see the per-trial error files above."
        )

    best = results.get_best_result("eval_loss", "min")
    print(f"\nBest config:    {best.config}", flush=True)
    print(f"Best eval_loss: {best.metrics['eval_loss']:.4f}", flush=True)

    # AI Runtime injects MLFLOW_RUN_ID and configures the databricks tracking URI on the
    # node, so logging needs no credentials here. Gating on the variable keeps the script
    # runnable off-platform, where it is unset.
    if os.environ.get("MLFLOW_RUN_ID"):
        with mlflow.start_run(run_id=os.environ["MLFLOW_RUN_ID"]):
            mlflow.log_params(
                {
                    "model": MODEL_NAME,
                    "dataset": DATASET_NAME,
                    "num_samples": NUM_SAMPLES,
                    "scheduler": "ASHA",
                    "asha_max_t": MAX_ITERATIONS,
                    "asha_grace_period": GRACE_PERIOD,
                    **{f"best_{k}": v for k, v in best.config.items()},
                }
            )
            mlflow.log_metric("best_eval_loss", best.metrics["eval_loss"])
            for i, result in enumerate(results):
                if result.metrics and "eval_loss" in result.metrics:
                    mlflow.log_metric("trial_eval_loss", result.metrics["eval_loss"], step=i)

    ray.shutdown()


if __name__ == "__main__":
    main()

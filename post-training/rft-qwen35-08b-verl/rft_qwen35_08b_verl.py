"""CLI example of GRPO with Qwen3.5-0.8B, verl, and vLLM on one 8xH100 node.

Run with databricks air run -f workload.yaml. AIR supplies HYPERPARAMETERS_PATH
and MLFLOW_RUN_ID; this script logs to that attempt's run.
"""

import json
import os
import sys
from importlib.metadata import distribution, version
from pathlib import Path

import mlflow
import pandas as pd
import ray
import torch
import yaml


def check_opencv_wheel():
    # Check metadata before importing cv2: the other wheel can abort Python.
    package = distribution("opencv-python-headless")
    tags = (package.read_text("WHEEL") or "").splitlines()
    if package.version != "4.13.0.92" or "Tag: cp37-abi3-manylinux2014_x86_64" not in tags:
        raise RuntimeError("OpenCV must use the 4.13.0.92 manylinux2014 wheel pinned in workload.yaml.")


def check_environment():
    check_opencv_wheel()
    expected_versions = {
        "verl": "0.9.1", "torch": "2.11.0", "vllm": "0.24.0",
        "transformers": "5.9.0", "xgrammar": "0.2.3",
    }
    for package, expected in expected_versions.items():
        installed = version(package)
        if installed.split("+")[0] != expected:
            raise RuntimeError(f"Expected {package} {expected}, found {installed}. Check workload.yaml.")
    if torch.version.cuda != "13.0":
        raise RuntimeError(f"Expected PyTorch built for CUDA 13.0, found {torch.version.cuda}.")
    if int(os.environ.get("NUM_NODES", "1")) != 1 or torch.cuda.device_count() != 8:
        raise RuntimeError("This recipe requires one 8xH100 node. Check compute in workload.yaml.")


def prepare_data(params, run_dir, model_path):
    from datasets import Dataset, load_dataset
    from transformers import AutoTokenizer
    from verl.utils.reward_score.gsm8k import extract_solution

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Reserve GSM8K's test split for final evaluation, outside this smoke test.
    raw_data = load_dataset(params["dataset_id"], "main", split="train")
    splits = raw_data.train_test_split(test_size=params["validation_examples"], seed=params["seed"])
    instruction = 'Solve the problem step by step. End with "#### " followed by only the final numeric answer.'

    def prepare_rows(data, split, limit):
        if limit > len(data):
            raise ValueError(f"Requested {limit} {split} examples, but only {len(data)} are available.")
        rows = []
        for index, example in enumerate(data.shuffle(seed=params["seed"]).select(range(limit))):
            reference = extract_solution(example["answer"], method="strict")
            if reference is None:
                raise ValueError(f"Missing reference answer in {split} example {index}.")
            prompt = [{"role": "user", "content": f"{example['question']}\n\n{instruction}"}]
            prompt_ids = tokenizer.apply_chat_template(
                prompt, tokenize=True, add_generation_prompt=True, return_dict=False,
            )
            if len(prompt_ids) > params["max_prompt_tokens"]:
                continue
            rows.append({
                "data_source": params["dataset_id"],
                "prompt": prompt,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": reference},
                "extra_info": {"split": split, "index": index},
            })
        return rows

    train_rows = prepare_rows(splits["train"], "train", params["train_examples"])
    val_rows = prepare_rows(splits["test"], "validation", params["validation_examples"])
    if len(train_rows) // params["prompts_per_step"] < params["training_steps"]:
        raise ValueError("Increase train_examples to cover every training step after prompt-length filtering.")
    if not val_rows:
        raise ValueError("No validation prompts remain after prompt-length filtering.")
    train_prompts = {row["prompt"][0]["content"] for row in train_rows}
    val_prompts = {row["prompt"][0]["content"] for row in val_rows}
    if train_prompts & val_prompts:
        raise ValueError("Training and validation prompts overlap.")

    train_path, val_path = run_dir / "train.parquet", run_dir / "validation.parquet"
    Dataset.from_list(train_rows).to_parquet(str(train_path))
    Dataset.from_list(val_rows).to_parquet(str(val_path))
    print(f"Prepared {len(train_rows)} training prompts and {len(val_rows)} validation prompts.", flush=True)
    return train_path, val_path, len(train_rows), len(val_rows)


def check_reward():
    from verl.utils.reward_score.gsm8k import compute_score

    examples = [
        ("There are 6 x 7 = 42 marbles.\n#### 42", "42", 1),
        ("There are 7 marbles.\n#### 7", "42", 0),
        ("The answer is 42.", "42", 0),
        ("#### 1,000", "1000", 1),
    ]
    for response, reference, expected in examples:
        if compute_score(response, reference) != expected:
            raise RuntimeError(f"Unexpected GSM8K reward for {response!r}.")


def build_config(params, run_dir, model_path, train_path, val_path):
    import verl
    from hydra import compose, initialize_config_dir
    from verl.utils.config import validate_config

    config_dir = Path(verl.__file__).parent / "trainer" / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="ppo_trainer", overrides=[
            "algorithm.adv_estimator=grpo",
            "actor_rollout_ref.actor.use_kl_loss=False",  # No reference model.
            f"data.train_files=[{json.dumps(str(train_path))}]",
            f"data.val_files=[{json.dumps(str(val_path))}]",
            f"data.train_batch_size={params['prompts_per_step']}",
            f"data.max_prompt_length={params['max_prompt_tokens']}",
            f"data.max_response_length={params['max_response_tokens']}",
            f"data.seed={params['seed']}",
            f"actor_rollout_ref.model.path={json.dumps(model_path)}",
            "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",
            "actor_rollout_ref.model.use_remove_padding=False",
            "actor_rollout_ref.actor.strategy=fsdp2",
            "actor_rollout_ref.actor.fsdp_config.use_torch_compile=False",
            f"actor_rollout_ref.actor.optim.lr={params['learning_rate']}",
            f"actor_rollout_ref.actor.ppo_mini_batch_size={params['prompts_per_step']}",
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4",
            "actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_model]",
            "actor_rollout_ref.rollout.name=vllm",
            f"actor_rollout_ref.rollout.n={params['rollouts_per_prompt']}",
            "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
            # One worker per engine avoids nested multiprocessing rendezvous.
            "+actor_rollout_ref.rollout.engine_kwargs.vllm.distributed_executor_backend=uni",
            f"actor_rollout_ref.rollout.max_model_len={params['max_prompt_tokens'] + params['max_response_tokens']}",
            "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8",
            "actor_rollout_ref.rollout.enforce_eager=True",
            "actor_rollout_ref.rollout.val_kwargs.do_sample=False",
            "actor_rollout_ref.rollout.val_kwargs.n=1",
            "trainer.use_v1=True",
            "trainer.v1.trainer_mode=sync",
            "trainer.nnodes=1",
            "trainer.n_gpus_per_node=8",
            "trainer.total_epochs=1",
            f"trainer.total_training_steps={params['training_steps']}",
            f"trainer.test_freq={params['training_steps']}",
            f"trainer.save_freq={params['training_steps']}",
            "trainer.val_before_train=True",
            "trainer.logger=[console,mlflow]",
            "trainer.experiment_name=rft-qwen35-08b-verl",
            f"trainer.default_local_dir={json.dumps(str(run_dir / 'checkpoints'))}",
            f"trainer.validation_data_dir={json.dumps(str(run_dir / 'validation'))}",
            f"trainer.rollout_data_dir={json.dumps(str(run_dir / 'rollouts'))}",
        ])
    validate_config(config, use_reference_policy=False, use_critic=False)
    return config


@ray.remote(num_gpus=1, max_calls=1, max_retries=0)
def check_worker_environment(run_id):
    check_opencv_wheel()
    import cv2
    import flashinfer
    import tvm_ffi
    from vllm import LLM
    from verl.workers.rollout.base import get_rollout_class

    rollout_class = get_rollout_class("vllm", "async")
    mlflow.set_tracking_uri("databricks")
    client = mlflow.MlflowClient()
    client.get_run(run_id)
    client.log_metric(run_id, "integration/ray_mlflow_ready", 1.0, step=0)
    torch.ones(1, device="cuda").add_(1)
    torch.cuda.synchronize()
    return {
        "torch": torch.__version__, "vllm": version("vllm"), "verl": version("verl"),
        "opencv": cv2.__version__, "flashinfer": flashinfer.__version__, "tvm_ffi": tvm_ffi.__version__,
        "xgrammar": version("xgrammar"), "vllm_entrypoint": LLM.__name__, "rollout_backend": rollout_class.__name__,
    }


def train(config, run_id, training_steps):
    from mlflow.utils.databricks_utils import get_databricks_env_vars
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
    from verl.trainer.main_ppo import TaskRunnerV1, run_ppo

    runtime_env = get_ppo_ray_runtime_env(config)
    runtime_env["py_executable"] = sys.executable
    # Pass session credentials only to workers; never persist them in config.yaml.
    runtime_env["env_vars"].update(get_databricks_env_vars("databricks"))
    runtime_env["env_vars"].update({
        "MLFLOW_TRACKING_URI": "databricks",
        "MLFLOW_RUN_ID": run_id,
        "MLFLOW_ENABLE_ASYNC_LOGGING": "false",
        "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING": "false",
    })
    try:
        # A single CLI process starts local Ray; verl allocates its own GPU workers.
        ray.init(num_gpus=8, runtime_env=runtime_env)
        gpu_count = ray.cluster_resources().get("GPU", 0)
        if gpu_count != 8:
            raise RuntimeError(f"Ray sees {gpu_count} GPUs; expected 8.")
        print("Ray worker check:", ray.get(check_worker_environment.remote(run_id)), flush=True)
        run_ppo(config, task_runner_class=TaskRunnerV1)
        history = mlflow.MlflowClient().get_metric_history(run_id, "critic/score/mean")
        if not set(range(1, training_steps + 1)).issubset({metric.step for metric in history}):
            raise RuntimeError("Training returned without logging every expected step to MLflow.")
    finally:
        ray.shutdown()


def compare_responses(run_dir, training_steps, validation_count):
    from verl.utils.reward_score.gsm8k import compute_score

    validation_dir = run_dir / "validation"
    before = pd.read_json(validation_dir / "0.jsonl", lines=True, dtype={"gts": str})
    after = pd.read_json(validation_dir / f"{training_steps}.jsonl", lines=True, dtype={"gts": str})
    comparison = before[["input", "gts", "output", "score"]].merge(
        after[["input", "gts", "output", "score"]],
        on=["input", "gts"], suffixes=("_before", "_after"), validate="one_to_one",
    )
    if not len(comparison) == len(before) == len(after) == validation_count:
        raise RuntimeError("Validation runs must cover the same prompts.")
    for frame in (before, after):
        if any(compute_score(row.output, row.gts) != row.score for row in frame.itertuples()):
            raise RuntimeError("Saved validation scores do not match the reward function.")

    summary = [
        {"model": "Before training", "accuracy": float(before["score"].mean())},
        {"model": f"After {training_steps} steps", "accuracy": float(after["score"].mean())},
    ]
    summary_path = run_dir / "validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    comparison.to_json(run_dir / "validation_comparison.jsonl", orient="records", lines=True)
    export_dir = run_dir / "checkpoints" / f"global_step_{training_steps}" / "actor" / "huggingface"
    # Direct stat avoids stale FUSE directory listings for this single-file export.
    if not (export_dir / "model.safetensors").is_file():
        raise RuntimeError(f"Expected exported model weights in {export_dir}.")
    mlflow.log_metrics({"accuracy_before": summary[0]["accuracy"], "accuracy_after": summary[1]["accuracy"]})
    mlflow.log_artifact(str(summary_path))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Hugging Face model export: {export_dir}", flush=True)


def main():
    from huggingface_hub import snapshot_download
    from omegaconf import OmegaConf

    with open(os.environ["HYPERPARAMETERS_PATH"]) as handle:
        params = yaml.safe_load(handle)
    for name in (
        "train_examples", "validation_examples", "prompts_per_step", "training_steps",
        "max_prompt_tokens", "max_response_tokens",
    ):
        if params[name] < 1:
            raise ValueError(f"{name} must be positive.")
    if params["rollouts_per_prompt"] < 2:
        raise ValueError("GRPO needs at least two rollouts per prompt.")
    check_environment()
    output_root = Path(params["output_root"])
    if (
        len(output_root.parts) < 5 or output_root.parts[:2] != ("/", "Volumes")
        or ".." in output_root.parts or any(char in str(output_root) for char in "<>")
    ):
        raise ValueError("Set output_root to a folder in an existing Unity Catalog volume.")
    volume_root = Path(*output_root.parts[:5])
    if not volume_root.is_dir():
        raise ValueError(f"Volume not accessible: {volume_root}. Check the path and your permissions.")

    run_id = os.environ["MLFLOW_RUN_ID"]
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"Outputs: {run_dir}", flush=True)
    mlflow.set_tracking_uri("databricks")
    with mlflow.start_run(run_id=run_id):
        # All eight workers share this node's local model cache.
        model_path = snapshot_download(params["model_id"], ignore_patterns=["*.md", ".gitattributes"])
        train_path, val_path, train_count, val_count = prepare_data(params, run_dir, model_path)
        check_reward()
        config = build_config(params, run_dir, model_path, train_path, val_path)
        config_path = run_dir / "config.yaml"
        OmegaConf.save(config, str(config_path), resolve=True)
        mlflow.log_params({
            "train_prompts": train_count, "validation_prompts": val_count,
            "gpus": 8, "verl_version": version("verl"), "outputs": str(run_dir),
        })
        mlflow.log_artifact(str(config_path))
        try:
            train(config, run_id, params["training_steps"])
            compare_responses(run_dir, params["training_steps"], val_count)
        finally:
            if (run_dir / "validation").is_dir():
                try:
                    mlflow.log_artifacts(str(run_dir / "validation"), artifact_path="validation")
                except Exception as error:
                    print(f"Could not upload validation artifacts; they remain in {run_dir}: {error}", flush=True)


if __name__ == "__main__":
    main()

# Databricks notebook source
# MAGIC %md
# MAGIC # Reinforcement learning with verl on AI Runtime
# MAGIC
# MAGIC Fine-tune a small language model with reinforcement learning on one 8xH100 node using [verl](https://github.com/verl-project/verl). The task is grade-school math: the model must solve a word problem and end its response with `#### <number>`. A programmatic reward scores each response `1` if the final number is correct and `0` otherwise, and Group Relative Policy Optimization (GRPO) uses those rewards to update the model.
# MAGIC
# MAGIC In this notebook, you:
# MAGIC
# MAGIC 1. Prepare prompts and reference answers from a small subset of GSM8K.
# MAGIC 2. Inspect the reward function.
# MAGIC 3. Configure GRPO rollouts and training for one node.
# MAGIC 4. Train Qwen2.5-0.5B-Instruct for five steps.
# MAGIC 5. Compare validation responses before and after training.
# MAGIC
# MAGIC This is a minimal, end-to-end example of the training loop. It is not a benchmark, and five steps are not expected to produce a large quality improvement. It uses no LLM judge, tools, or custom container.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Connect to compute
# MAGIC
# MAGIC This notebook requires the Databricks **AI v6** environment and an **8xH100** accelerator. AI v6 includes PyTorch 2.11.0 with CUDA 13 and vLLM 0.24.0, matching the backend versions used by verl 0.9.1.
# MAGIC
# MAGIC 1. Select **Connect** at the top of the notebook.
# MAGIC 2. Select **Serverless GPU**.
# MAGIC 3. In the **Environment** side panel, set **Accelerator** to **8xH100**.
# MAGIC 4. Select **AI v6** as the base environment.
# MAGIC 5. Select **Apply**, then select **Confirm**.
# MAGIC
# MAGIC If you previously ran this notebook with another dependency stack, use **Reset to defaults** in the **Apply** drop-down to clear the cached environment, then reselect **AI v6** and apply. Restarting Python alone does not clear installed libraries.
# MAGIC
# MAGIC You also need write access to a Unity Catalog volume and the notebook's MLflow experiment, and network access to PyPI and Hugging Face. See [AI Runtime environments](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/environment).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install dependencies
# MAGIC
# MAGIC Install verl 0.9.1 with its vLLM dependencies and FSDP runtime helpers. The vLLM extra requires the same PyTorch and vLLM versions as AI v6 and updates Transformers to 5.9.0. This small example uses PyTorch SDPA for policy attention and does not install or compile another FlashAttention build.
# MAGIC
# MAGIC `--exclude-newer` bounds newly installed transitive dependencies. The OpenCV dependency selects the SHA-256-pinned **manylinux2014** wheel for version 4.13.0.92. The same release's **manylinux_2_28** wheel bundles a different OpenSSL build that can abort on AI Runtime hosts. Selecting only the version is not sufficient. This workaround does not change the host's FIPS settings.
# MAGIC
# MAGIC OpenCV, TVM FFI, and xgrammar are reinstalled so that their native libraries and package metadata are present together in the notebook environment. xgrammar stays at the version supplied by AI v6.
# MAGIC
# MAGIC Installation takes several minutes. Restart Python afterwards so that the notebook uses the installed versions.

# COMMAND ----------

# MAGIC %uv pip install --exclude-newer 2026-10-01 \
# MAGIC   --reinstall-package apache-tvm-ffi --reinstall-package opencv-python-headless --reinstall-package xgrammar \
# MAGIC   "verl[vllm]==0.9.1" "TransferQueue==0.1.10" "xgrammar==0.2.3" \
# MAGIC   "cupy-cuda13x==14.0.1" "liger-kernel==0.8.2" "torchcodec==0.11.0" \
# MAGIC   "ray[default]==2.56.0" "mlflow-skinny==3.13.0" \
# MAGIC   "datasets==4.8.5" "peft==0.19.1" "numpy==2.3.4" "pandas==2.3.3" \
# MAGIC   "hydra-core==1.3.2" \
# MAGIC   "opencv-python-headless @ https://files.pythonhosted.org/packages/92/ce/bd17ff5772938267fd49716e94ca24f616ff4cb1ff4c6be13085108037be/opencv_python_headless-4.13.0.92-cp37-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl#sha256=0525a3d2c0b46c611e2130b5fdebc94cf404845d8fa64d2f3a3b679572a5bd22"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure the run
# MAGIC
# MAGIC Set the `volume_path` widget to a folder in a Unity Catalog volume that you can write to. Each run writes its data, responses, and checkpoints to a new subfolder. Training metrics are logged directly to the notebook's MLflow experiment. Keep the small training settings for your first run.

# COMMAND ----------

import json
import sys
import uuid
from importlib.metadata import version
from pathlib import Path

import mlflow
import pandas as pd
import torch

# Requires write access to this existing volume; replace it with another writable volume if needed.
dbutils.widgets.text("volume_path", "/Volumes/main/default/air_examples/rft-qwen25-05b-verl")

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
DATASET_ID = "openai/gsm8k"
DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"

TRAIN_EXAMPLES = 256  # Prompts sampled from the GSM8K train split
VAL_EXAMPLES = 32  # Prompts sampled from the GSM8K test split
PROMPTS_PER_STEP = 32  # Prompts in each training step
ROLLOUTS_PER_PROMPT = 4  # Sampled responses per prompt, compared by GRPO
TRAINING_STEPS = 5
MAX_PROMPT_TOKENS = 512
MAX_RESPONSE_TOKENS = 512
SEED = 42
GPU_COUNT = 8

def check_opencv_wheel():
    # Read metadata before importing cv2: the wrong native wheel can abort Python, not raise an exception.
    from importlib.metadata import distribution

    package = distribution("opencv-python-headless")
    tags = (package.read_text("WHEEL") or "").splitlines()
    if package.version != "4.13.0.92" or "Tag: cp37-abi3-manylinux2014_x86_64" not in tags:
        raise RuntimeError("OpenCV must use the pinned 4.13.0.92 manylinux2014 wheel. Rerun the installation cell.")


check_opencv_wheel()
expected_versions = {"verl": "0.9.1", "torch": "2.11.0", "vllm": "0.24.0", "transformers": "5.9.0", "xgrammar": "0.2.3"}
for package, expected in expected_versions.items():
    installed = version(package)
    if installed.split("+")[0] != expected:
        raise RuntimeError(f"Expected {package} {expected}, found {installed}. Reset to AI v6 and rerun installation.")
if torch.version.cuda != "13.0":
    raise RuntimeError(f"Expected PyTorch built for CUDA 13.0, found {torch.version.cuda}. Select AI v6.")
if torch.cuda.device_count() != GPU_COUNT:
    raise RuntimeError(f"Found {torch.cuda.device_count()} GPUs. Connect this notebook to one 8xH100 node.")

volume_path = Path(dbutils.widgets.get("volume_path").strip())
if (
    len(volume_path.parts) < 5 or volume_path.parts[:2] != ("/", "Volumes")
    or ".." in volume_path.parts or any(c in str(volume_path) for c in "<>")
):
    raise ValueError("Set volume_path to /Volumes/<catalog>/<schema>/<volume>, optionally followed by subfolders.")
volume_root = Path(*volume_path.parts[:5])
if not volume_root.is_dir():
    raise ValueError(f"Volume not accessible: {volume_root}. Check the path and your Unity Catalog permissions.")

run_name = f"grpo-hello-world-{uuid.uuid4().hex[:8]}"
run_dir = volume_path / "verl-hello-world" / run_name
run_dir.mkdir(parents=True, exist_ok=False)
print(f"Outputs: {run_dir}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare prompts and reference answers
# MAGIC
# MAGIC Each training example contains the prompt the model sees and the reference answer used only for scoring. The GSM8K worked solutions are not included in the prompt. The `data_source` field selects verl's built-in GSM8K scorer, and `reward_model.ground_truth` holds the reference answer.
# MAGIC
# MAGIC Training prompts come from the GSM8K train split and validation prompts from the test split. For a real experiment, create your own validation set and reserve the test set for final evaluation.

# COMMAND ----------

from datasets import Dataset, load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from verl.utils.reward_score.gsm8k import compute_score, extract_solution

# Download the model to local disk. All GPU workers run on this node and read the same path.
model_path = snapshot_download(MODEL_ID, revision=MODEL_REVISION, ignore_patterns=["*.md", ".gitattributes"])
tokenizer = AutoTokenizer.from_pretrained(model_path)
raw_data = load_dataset(DATASET_ID, "main", revision=DATASET_REVISION)
INSTRUCTION = 'Solve the problem step by step. End with "#### " followed by only the final numeric answer.'


def prepare_rows(split, limit):
    rows = []
    for index, example in enumerate(raw_data[split].shuffle(seed=SEED).select(range(limit))):
        reference = extract_solution(example["answer"], method="strict")
        if reference is None:
            raise ValueError(f"Missing reference answer in {split} example {index}.")
        prompt = [{"role": "user", "content": f"{example['question']}\n\n{INSTRUCTION}"}]
        prompt_ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_dict=False)
        if len(prompt_ids) > MAX_PROMPT_TOKENS:
            continue
        rows.append({
            "data_source": DATASET_ID,
            "prompt": prompt,
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": reference},
            "extra_info": {"split": split, "index": index},
        })
    return rows


train_rows = prepare_rows("train", TRAIN_EXAMPLES)
val_rows = prepare_rows("test", VAL_EXAMPLES)
assert len(train_rows) // PROMPTS_PER_STEP >= TRAINING_STEPS, "Increase TRAIN_EXAMPLES to cover all training steps."
assert val_rows, "No validation prompts remain after length filtering."
train_prompts = {r["prompt"][0]["content"] for r in train_rows}
val_prompts = {r["prompt"][0]["content"] for r in val_rows}
assert not train_prompts & val_prompts, "Training and validation prompts overlap."

train_path, val_path = run_dir / "train.parquet", run_dir / "validation.parquet"
Dataset.from_list(train_rows).to_parquet(str(train_path))
Dataset.from_list(val_rows).to_parquet(str(val_path))
print(f"Prepared {len(train_rows)} training prompts and {len(val_rows)} validation prompts.\n")
print("Example prompt:\n" + train_rows[0]["prompt"][0]["content"])
print("\nReference answer:", train_rows[0]["reward_model"]["ground_truth"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inspect the reward
# MAGIC
# MAGIC The reward function reads the last `#### <number>` near the end of a response, removes thousands separators, and compares the number with the reference answer. A correct number without the `#### ` marker scores `0`, so the reward teaches the output format as well as correctness. Training and validation use this same function.

# COMMAND ----------

reward_examples = pd.DataFrame([
    {"response": "There are 6 x 7 = 42 marbles.\n#### 42", "reference": "42", "expected": 1},
    {"response": "There are 7 marbles.\n#### 7", "reference": "42", "expected": 0},
    {"response": "The answer is 42.", "reference": "42", "expected": 0},
    {"response": "#### 1,000", "reference": "1000", "expected": 1},
])
reward_examples["reward"] = [compute_score(r.response, r.reference) for r in reward_examples.itertuples()]
assert reward_examples["reward"].tolist() == reward_examples["expected"].tolist()
display(reward_examples)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure rollouts and training
# MAGIC
# MAGIC Each training step takes 32 prompts and samples 4 rollouts per prompt, so GRPO compares 128 responses. The V1 trainer alternates vLLM generation and FSDP2 policy updates on the same eight GPUs. Its rollout servers use asynchronous requests, but training steps remain synchronous and on-policy. This example applies no KL penalty, so no reference model is loaded.
# MAGIC
# MAGIC Validation generates one greedy response per question before training and after the last step, using identical settings both times. The final checkpoint contains the training state and a Hugging Face export of the model.

# COMMAND ----------

import verl
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from verl.utils.config import validate_config

config_dir = Path(verl.__file__).parent / "trainer" / "config"
with initialize_config_dir(config_dir=str(config_dir), version_base=None):
    config = compose(config_name="ppo_trainer", overrides=[
        # Algorithm
        "algorithm.adv_estimator=grpo",
        "actor_rollout_ref.actor.use_kl_loss=False",  # No KL penalty, so no reference model
        # Data
        f"data.train_files=[{json.dumps(str(train_path))}]",
        f"data.val_files=[{json.dumps(str(val_path))}]",
        f"data.train_batch_size={PROMPTS_PER_STEP}",
        f"data.max_prompt_length={MAX_PROMPT_TOKENS}",
        f"data.max_response_length={MAX_RESPONSE_TOKENS}",
        f"data.seed={SEED}",
        # Policy model and FSDP2 training
        f"actor_rollout_ref.model.path={json.dumps(model_path)}",
        "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",  # Use PyTorch attention for this small model
        "actor_rollout_ref.model.use_remove_padding=False",
        "actor_rollout_ref.actor.strategy=fsdp2",
        "actor_rollout_ref.actor.fsdp_config.use_torch_compile=False",  # Skip compilation for a short run
        "actor_rollout_ref.actor.optim.lr=1e-6",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={PROMPTS_PER_STEP}",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4",
        "actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_model]",
        # Rollouts with vLLM
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.n={ROLLOUTS_PER_PROMPT}",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.distributed_executor_backend=uni",  # One worker per engine; avoid nested MP rendezvous
        f"actor_rollout_ref.rollout.max_model_len={MAX_PROMPT_TOKENS + MAX_RESPONSE_TOKENS}",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8",
        "actor_rollout_ref.rollout.enforce_eager=True",  # Skip CUDA graph capture for a short run
        # V1 synchronous training loop and outputs
        "trainer.use_v1=True",
        "trainer.v1.trainer_mode=sync",
        f"trainer.n_gpus_per_node={GPU_COUNT}",
        "trainer.total_epochs=1",
        f"trainer.total_training_steps={TRAINING_STEPS}",
        f"trainer.test_freq={TRAINING_STEPS}",
        f"trainer.save_freq={TRAINING_STEPS}",
        "trainer.logger=[console,mlflow]",
        f"trainer.experiment_name={run_name}",
        f"trainer.default_local_dir={json.dumps(str(run_dir / 'checkpoints'))}",
        f"trainer.validation_data_dir={json.dumps(str(run_dir / 'validation'))}",
        f"trainer.rollout_data_dir={json.dumps(str(run_dir / 'rollouts'))}",
    ])
validate_config(config, use_reference_policy=False, use_critic=False)
config_path = run_dir / "config.yaml"
OmegaConf.save(config, str(config_path), resolve=True)
print(f"Saved verl configuration to {config_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train
# MAGIC
# MAGIC verl runs generation and training in Ray workers. Start Ray with `ray_init()` from the `serverless_gpu` package, which also prints a link to the Ray dashboard, and then launch training from this notebook. Do not wrap the training call in the `@distributed` decorator: verl starts its own GPU workers, and only one process should launch training.
# MAGIC
# MAGIC Training prints metrics for each step. The most useful ones are:
# MAGIC
# MAGIC - `critic/score/mean`: The mean reward of the training rollouts in the step.
# MAGIC - `response_length/mean`: The mean response length in tokens.
# MAGIC - `val-core/openai/gsm8k/acc/mean@1`: Validation accuracy, reported before training and after the last step.
# MAGIC
# MAGIC verl's native MLflow backend logs directly to the same run as this notebook, so metrics appear during training and remain available if a later step fails. In MLflow, `@1` is displayed as `_at_1` because `@` is not permitted in metric names.
# MAGIC
# MAGIC Before training, a short Ray task imports the complete vLLM/verl rollout backend and checks MLflow access. It uses the current session's credentials; no personal access token is needed. Credentials are passed only in Ray's runtime environment, not saved in the training configuration. The cell takes several minutes.

# COMMAND ----------

import ray
from mlflow.utils.databricks_utils import get_databricks_env_vars
from serverless_gpu import ray_init
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.main_ppo import TaskRunnerV1, run_ppo

if ray.is_initialized():
    raise RuntimeError("Ray is already running. Finish or stop the previous workload before rerunning this cell.")


@ray.remote(num_gpus=1, max_calls=1, max_retries=0)
def check_worker_environment(run_id):
    # Fail before loading cv2 if Ray resolves a different wheel. Never retry a native import crash.
    check_opencv_wheel()
    # Import inside the task so this checks the worker environment, not the notebook kernel.
    import cv2
    import flashinfer
    import mlflow
    import torch
    import tvm_ffi
    from importlib.metadata import version
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


mlflow.set_tracking_uri("databricks")
with mlflow.start_run(run_name=run_name) as mlflow_run:
    mlflow_run_id = mlflow_run.info.run_id
    print(f"MLflow run ID: {mlflow_run_id}")
    mlflow.log_params({
        "model": MODEL_ID, "model_revision": MODEL_REVISION,
        "dataset": DATASET_ID, "dataset_revision": DATASET_REVISION,
        "train_prompts": len(train_rows), "validation_prompts": len(val_rows),
        "prompts_per_step": PROMPTS_PER_STEP, "rollouts_per_prompt": ROLLOUTS_PER_PROMPT,
        "training_steps": TRAINING_STEPS, "gpus": GPU_COUNT,
        "verl_version": verl.__version__, "outputs": str(run_dir),
    })
    mlflow.log_artifact(str(config_path))

    runtime_env = get_ppo_ray_runtime_env(config)
    runtime_env["py_executable"] = sys.executable  # Use the notebook's installed packages in every Ray process.
    runtime_env["env_vars"].update(get_databricks_env_vars("databricks"))
    runtime_env["env_vars"].update({
        "MLFLOW_TRACKING_URI": "databricks",
        "MLFLOW_RUN_ID": mlflow_run_id,  # verl resumes this run instead of creating another one.
        "MLFLOW_ENABLE_ASYNC_LOGGING": "false",
        "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING": "false",  # The notebook owns system-metric collection.
    })
    try:
        ray_init(runtime_env=runtime_env)
        if ray.cluster_resources().get("GPU", 0) != GPU_COUNT:
            raise RuntimeError(f"Ray sees {ray.cluster_resources().get('GPU', 0)} GPUs; expected {GPU_COUNT}.")
        print("Ray worker check:", ray.get(check_worker_environment.remote(mlflow_run_id)))
        run_ppo(config, task_runner_class=TaskRunnerV1)
        history = mlflow.MlflowClient().get_metric_history(mlflow_run_id, "critic/score/mean")
        if not set(range(1, TRAINING_STEPS + 1)).issubset({metric.step for metric in history}):
            raise RuntimeError("Training returned without logging every expected step to MLflow. Check the Ray logs.")
    finally:
        ray.shutdown()  # Stop workers before the notebook sets the final MLflow run status.
        validation_dir = run_dir / "validation"
        if validation_dir.is_dir():
            try:
                mlflow.log_artifacts(str(validation_dir), artifact_path="validation")
            except Exception as error:
                print(f"Warning: could not upload validation artifacts; they remain in {validation_dir}: {error}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compare responses before and after training
# MAGIC
# MAGIC verl saved the validation responses from before training (`0.jsonl`) and after the last step (`5.jsonl`). Compare the accuracy and read a few responses side by side. After five steps, accuracy can go up, stay the same, or go down; inspect the responses and the reward variation before increasing the training budget.

# COMMAND ----------

validation_dir = run_dir / "validation"
before = pd.read_json(validation_dir / "0.jsonl", lines=True, dtype={"gts": str})
after = pd.read_json(validation_dir / f"{TRAINING_STEPS}.jsonl", lines=True, dtype={"gts": str})
comparison = before[["input", "gts", "output", "score"]].merge(
    after[["input", "gts", "output", "score"]],
    on=["input", "gts"], suffixes=("_before", "_after"), validate="one_to_one",
)
assert len(comparison) == len(before) == len(after) == len(val_rows), "Validation runs must cover the same prompts."
for frame in (before, after):
    assert all(compute_score(r.output, r.gts) == r.score for r in frame.itertuples()), "Saved scores do not match the reward."

summary = pd.DataFrame([
    {"model": "Before training", "accuracy": before["score"].mean()},
    {"model": f"After {TRAINING_STEPS} steps", "accuracy": after["score"].mean()},
])
display(summary)
display(comparison.rename(columns={"input": "prompt", "gts": "reference"}).head(3))

export_dir = run_dir / "checkpoints" / f"global_step_{TRAINING_STEPS}" / "actor" / "huggingface"
# This small model exports one weights file. Direct stat avoids stale FUSE directory listings.
assert (export_dir / "model.safetensors").is_file(), f"Expected exported model weights in {export_dir}."
summary_path = run_dir / "validation_summary.json"
summary.to_json(summary_path, orient="records", indent=2)
with mlflow.start_run(run_id=mlflow_run_id):
    mlflow.log_metrics({"accuracy_before": before["score"].mean(), "accuracy_after": after["score"].mean()})
    mlflow.log_artifact(str(summary_path))
print(f"Hugging Face model export: {export_dir}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean up
# MAGIC
# MAGIC The training cell shut down Ray. Disconnect the notebook from GPU compute when you finish. Delete the run folder in the volume if you do not need the checkpoint, which is several gigabytes.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next steps
# MAGIC
# MAGIC - Open `rollouts/<step>.jsonl` in the run folder to read the sampled responses and their rewards. If every rollout for a prompt receives the same reward, that prompt gives GRPO no learning signal in that step.
# MAGIC - Load the exported model with `AutoModelForCausalLM.from_pretrained(export_dir)` to test it, or keep the `actor` checkpoint directory to resume training.
# MAGIC - After this smoke test succeeds, increase `TRAINING_STEPS` and `TRAIN_EXAMPLES`, and use a larger validation set for model selection.
# MAGIC - For more fine-tuning examples, see [Large language models (LLMs)](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/examples/gpu-llms).

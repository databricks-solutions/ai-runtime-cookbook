# Reinforcement fine-tuning Qwen3.5-0.8B with verl

Fine-tune [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) on GSM8K with verl's Group Relative Policy Optimization (GRPO). This CLI example runs five training steps on one 8xH100 node with FSDP2 and vLLM, then compares greedy validation responses before and after training; five steps demonstrate the training loop and do not establish a benchmark result.

## Before you run

- Set `parameters.output_root` in [workload.yaml](workload.yaml) to a folder in an existing writable Unity Catalog volume. The default is `/Volumes/main/default/air_examples/rft-qwen35-08b-verl`.
- Allow network access to PyPI and Hugging Face. The public model and dataset do not require a Hugging Face token.
- Keep the AI v6 environment and pinned dependencies in `workload.yaml`. The OpenCV wheel URL selects the manylinux2014 build: the other wheel of the same release can abort on AIR hosts.

## Run

From this recipe's directory:

```bash
databricks air run -f workload.yaml
```

Add `--watch` to stream logs. Training settings live in the YAML `parameters` block. The defaults use 256 training prompts, 32 validation prompts, 32 prompts per step, four rollouts per prompt, and 512-token prompt and response limits. Each step scores 32 × 4 = 128 responses. Increase `train_examples` along with `training_steps` when extending the run.

For task design, rollouts, rewards, and evaluation, see [How to set up a reinforcement learning task](#how-to-set-up-a-reinforcement-learning-task-on-ai-runtime).

## Results

Metrics are logged to AIR's MLflow run, including `critic/score/mean`, `response_length/mean`, and the final `accuracy_before` and `accuracy_after`. verl also logs validation accuracy under `val-core/openai/gsm8k/acc/mean@1` (`@1` appears as `_at_1` in MLflow).

Each attempt writes to `<output_root>/<MLFLOW_RUN_ID>/`:

- `train.parquet`, `validation.parquet`, and the resolved verl `config.yaml`.
- `rollouts/<step>.jsonl` and `validation/0.jsonl` / `validation/5.jsonl`, containing sampled and greedy responses with rewards.
- `validation_summary.json` and `validation_comparison.jsonl`, containing accuracy and paired responses before and after training.
- `checkpoints/global_step_5/actor/huggingface/`, containing the model export, alongside training-state checkpoints. The step number follows `training_steps`.

Validation responses and the summary are also logged as MLflow artifacts. The script checks that every training step reached MLflow, recomputes the validation rewards, and verifies that the exported weights exist. Accuracy can rise, stay flat, or fall after five steps. Load the export with `AutoModelForImageTextToText.from_pretrained(export_dir)` to try it, and remove the attempt's volume folder when you no longer need its checkpoint.

## How to set up a reinforcement learning task on AI Runtime

Reinforcement learning can help a language model answer questions and use tools more reliably by training on feedback from its own attempts. AI Runtime lets you run that training loop on managed GPUs alongside your data and MLflow experiments.

This guide explains how to prepare a reinforcement learning (RL) task for a language model on AI Runtime: what RL is, what you need to define before training, and how the [CLI example](#cli-example-grpo-on-gsm8k) puts the pieces together with verl.

## What is reinforcement learning?

Reinforcement learning (RL) is learning from attempts and feedback. The learner tries something, receives a numeric score called a **reward**, and adjusts so that higher-reward behavior becomes more likely.

Here, the learner is a language model that has already been trained. RL trains it further by updating its weights, not just its prompt. Five terms describe the setup:

| Term | Meaning |
| --- | --- |
| Task | What the model must do: the input it receives, the actions it may take, the output it must return, and what counts as success. Each prompt is one instance of the task. |
| Policy | The rule for choosing what to do next. Here, the language model is the policy: it generates text or tool calls from the conversation so far. |
| Environment | What the model interacts with during an attempt, such as a document index and search tools. A plain question-answering task needs no tools. |
| Rollout | One attempt at a task, from the prompt until the model finishes or hits a limit. It can be a single response or a sequence of messages, tool calls, and tool results, also called a **trajectory**. |
| Reward function | The scoring logic that turns a rollout into a number, for example `1` if the final answer matches a reference and `0` otherwise. |

The task defines success, the reward function measures it, and the RL algorithm uses those measurements to update the policy. For background, see [Key concepts in RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html).

## How training works

Training repeats one loop:

```text
task prompt -> rollout -> reward -> policy update -> next rollout
```

1. **Select prompts.** Take a batch of task prompts.
2. **Generate rollouts.** Let the policy attempt each prompt, executing any tool calls it makes.
3. **Compute rewards.** Score every rollout with the reward function.
4. **Update the policy.** Adjust the model's weights so that higher-reward behavior becomes more likely.
5. **Repeat** with new prompts and the updated model. Scoring responses without updating the policy is evaluation, not training.

**Group Relative Policy Optimization (GRPO)**, introduced in [DeepSeekMath](https://arxiv.org/abs/2402.03300), is the RL algorithm used in the CLI example. It implements step 4: for each prompt it samples several rollouts and compares each reward with the group's average. Rollouts above the average are reinforced, rollouts below it are discouraged. For example, rewards `[1, 0, 1, 0]` average `0.5`, so the two correct attempts are reinforced and the two incorrect ones discouraged. If all rewards in a group are equal, there is nothing to compare, and that prompt contributes no learning signal in that step.

You define the task and the reward function; GRPO decides how rewards change the model. See [verl's GRPO guide](https://verl.readthedocs.io/en/latest/algo/grpo.html) for details.

## When to use RL

With **supervised fine-tuning (SFT)**, you provide the responses the model should produce. With RL, you provide tasks and a way to score the model's own attempts. RL is a good fit when:

- You can define success and measure it consistently.
- The starting model sometimes succeeds, but not reliably.
- You can afford to generate and score several attempts per prompt.

If the model can't follow the output format or use the tools at all, fix the prompt or use SFT first. If it lacks the information needed to answer, fix retrieval or data access before training.

## Set up the task

Six steps take you from an idea to a validated training setup: define the task, prepare the data, configure rollouts, choose a reward, validate the reward, then train and evaluate. The running example is a **retrieval agent** that searches internal documents to answer questions such as "Which team owns the database used by the checkout service?"

### 1. Define the task

Pick a specific behavior, not a general goal such as "improve reasoning." Write down:

- **Input:** The question, instructions, and context the model receives.
- **Output:** What the model must return and how you identify its final answer.
- **Actions:** Which tools, if any, the model can call.
- **Success:** What makes a rollout correct. Is a correct final answer enough, or is supporting evidence required?
- **Stopping:** When a rollout ends, including turn and token limits.

Retrieval example: the agent can search a documentation index and read articles, and must return the owning team inside `<answer>...</answer>` tags. The tags are a convention for this task, not an AI Runtime requirement.

Choose a starting model that can already follow these instructions and produce tool calls your application can execute. Test this before training.

### 2. Prepare the task data

Keep what the model sees separate from what the scorer uses:

- **Model input:** The prompt, system instructions, and permitted context.
- **Scoring information:** Reference answers, test cases, or grading criteria. Never put these in the prompt or in tool responses.
- **Metadata:** A stable ID, task category, difficulty, and data version.

```json
{
  "id": "support-001",
  "prompt": "Which team owns the database used by the checkout service?",
  "reference_answers": ["Storage Platform"],
  "metadata": {"corpus_version": "support-docs-v1"}
}
```

Your framework defines the exact format. verl reads Parquet files with a `prompt` field of chat messages and a `reward_model.ground_truth` field for the reference, as shown in the [CLI example](#cli-example-grpo-on-gsm8k).

Create separate training, validation, and test sets with no duplicates across them. Use validation data to tune the reward and pick checkpoints, and reserve the test set for the final comparison. Store data and models in Unity Catalog volumes, with separate paths for versions you need to reproduce. See [Load data on AI Runtime](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/dataloading).

### 3. Configure rollouts

In the retrieval example, a rollout includes the tool interaction, not just the final answer. The documentation is fictional and the format is conceptual:

```text
User: Which team owns the database used by the checkout service?
Assistant -> vector_search("checkout service database")
Tool -> The checkout service uses the orders database.
Assistant -> read_article("orders database")
Tool -> The orders database is owned by Storage Platform.
Assistant -> <answer>Storage Platform</answer>
```

Decide the following before training. These settings determine both the cost of a training step and the behavior the model can learn:

- **Rollouts per prompt:** GRPO needs several, because it compares them with each other.
- **Sampling:** Repeated rollouts must vary. Identical outputs give GRPO nothing to compare.
- **Limits:** Turns, tool calls, and tokens for the prompt, the response, tool results, and the whole trajectory. Leave room for a final answer.
- **Tools:** Clear names, arguments, and bounded responses; read-only or isolated access; state reset between rollouts.

### 4. Choose a reward function

Rewards can be computed by code, by an **LLM judge** (a separate model that grades the attempt against explicit criteria), or both. Use the simplest method that reliably captures success:

| Approach | Use when | Example |
| --- | --- | --- |
| Programmatic | Code can check the result. | Compare the answer with references, validate a schema, or run tests in a sandbox. |
| LLM judge | Success requires a semantic judgment that code can't make reliably. | Grade an explanation against a rubric and reference material. |
| Combined | The task has hard requirements and qualitative criteria. | Require a valid format, then grade the content. |

For the retrieval example, start with a binary reward: `1.0` if the extracted final answer matches an accepted reference after normalization, `0.0` if it is wrong, missing, or malformed. Extract the final answer first. Don't search the whole trajectory for the reference string: a retrieved passage can contain the right answer even when the model's answer is wrong.

Keep the reward hard to game. Log correctness, formatting, and tool use separately even if training uses one combined score, add partial credit only when you can define it precisely, and never let bonuses lift an incorrect rollout above a correct one.

If you use an LLM judge for training:

- Specify exactly what it grades and what each score means, and give it trusted references.
- Require structured output, such as a score and a short justification, and validate it. Treat the candidate response and retrieved content as untrusted data, not as instructions.
- Calibrate it on expert-graded examples, including borderline and adversarial ones, and freeze its model, prompt, and rubric for the whole run.
- Record judge failures separately from wrong answers, and never treat a failed call as a valid grade.

> **Important**
>
> Databricks-hosted MLflow LLM judges are intended for evaluation. Their outputs should not be used to train, improve, or fine-tune an LLM. For a training reward, use a programmatic verifier or a separately supplied judge whose permitted uses include model training. See [Scorers and LLM judges](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/concepts/scorers#information-about-the-models-powering-llm-judges).

For training rewards, one option is `Qwen/Qwen3.5-9B`, an open-weight model whose [Apache 2.0 license](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/LICENSE) does not restrict using outputs for training. Run it inside the RL job or serve it with [Custom LLM Serving](https://docs.databricks.com/aws/en/machine-learning/model-serving/serve-custom-llms) (Public Preview); check any additional hosting terms.

### 5. Validate the reward and establish a baseline

Before the first training run:

1. **Test the reward** on correct, incorrect, equivalent, missing, malformed, and adversarial answers. Keep infrastructure failures distinguishable from task failures.
2. **Generate a few rollouts with the starting model** and read them next to their scores. Do the tools work? Does the scorer grade the intended final answer? Does the model finish within the limits?
3. **Check reward variation per prompt.** If every rollout for a prompt gets the same reward, that prompt teaches GRPO nothing. All-high scores mean the task is too easy; all-low scores mean it is too hard, the tools are broken, or the scorer is wrong.
4. **Record the baseline.** Evaluate the starting model on the validation set and save the prompts, tools, data versions, generation settings, and scoring rules you used.

### 6. Train and evaluate

Run a short job first and confirm that it generates rollouts, computes rewards, updates the policy, and writes a checkpoint you can load. Then scale up.

RL frameworks such as verl run rollout generation and policy updates in GPU workers coordinated by Ray. On AI Runtime you can:

- **Use a notebook.** Connect to an 8xH100 node, start Ray with `ray_init()` from the `serverless_gpu` package, and launch the trainer from the notebook. Don't wrap it in the `@distributed` decorator: the framework starts its own workers. See [Ray on AI Runtime](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/ray).
- **Use the Databricks CLI.** Define a workload YAML that starts Ray across one or more nodes and runs the trainer. Use this for multi-node training and scheduled runs. See [Databricks CLI quickstart for AI Runtime](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/cli/quickstart).

This CLI example uses one node and starts local Ray from Python. A multi-node job also needs to start Ray workers on the other nodes.

Write checkpoints to Unity Catalog volumes so they outlive the run. See [Improve training performance and resiliency on AI Runtime](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/guides/performance-and-resiliency).

Track the run in MLflow: the training reward and its components, validation success, response length, truncated rollouts, tool and judge errors, and the model, data, reward, and judge versions. Save sample trajectories as well. A rising reward curve alone can't distinguish real improvement from *reward hacking*, where the model exploits a weakness in the scorer. See [Experiment tracking and observability](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/tracking-observability).

To evaluate, compare checkpoints with the starting model under identical conditions: the same data, prompts, tools, scoring rules, sampling settings, and limits. Select a checkpoint on validation data, then confirm it on the test set; the last checkpoint isn't always the best. Measure task success separately from the training reward, and check for regressions in important task categories and for changes in response length, latency, and tool use.

## CLI example: GRPO on GSM8K

[rft_qwen35_08b_verl.py](rft_qwen35_08b_verl.py) runs the complete loop on one 8xH100 node. It fine-tunes `Qwen/Qwen3.5-0.8B` on grade-school math problems (GSM8K) with GRPO and a programmatic reward: the model must solve a word problem and end its response with `#### <number>`. It is a smoke test of the training loop, not a benchmark, and uses no tools or LLM judge. The walkthrough follows the script's functions; the code excerpts show the relevant parts of the configuration and training loop.

Qwen3.5-0.8B includes a vision encoder and hybrid attention. This recipe uses text-only prompts in the model's default non-thinking mode; its Hugging Face export loads with `AutoModelForImageTextToText`.

### Connect to compute and install dependencies

[workload.yaml](workload.yaml) selects the **AI v6** environment and one **8xH100** accelerator. AIR installs the pinned versions of `verl[vllm]`, Ray, MLflow, and supporting packages before starting the Python entry point. The OpenCV wheel URL selects the compatible manylinux2014 build.

vLLM compiles Qwen3.5 attention kernels on first use, so allow extra startup time. With these dependency pins, the Transformers policy uses a PyTorch fallback for its optional hybrid-attention kernels.

For notebook execution, attach to Serverless GPU with the same AI v6 environment and 8xH100 accelerator, install the same pinned packages, and restart Python. If you previously used another dependency stack, reset the environment to defaults first.

### Configure the run

Set `parameters.output_root` in `workload.yaml` to a folder in an existing writable Unity Catalog volume. The default is `/Volumes/main/default/air_examples/rft-qwen35-08b-verl`; replace it if you use another volume. Each AIR attempt gets its own `<MLFLOW_RUN_ID>` subfolder for data, responses, and checkpoints. In a notebook adaptation, the equivalent `volume_path` widget should point to this existing writable volume.

The script checks that the installed PyTorch, vLLM, verl, and Transformers versions match AI v6, that PyTorch uses CUDA 13.0, and that eight GPUs are visible. The rollout settings from step 3 live in the YAML `parameters` block:

```yaml
model_id: Qwen/Qwen3.5-0.8B
train_examples: 256  # Prompts sampled from the GSM8K training split
validation_examples: 32  # Held out from that training split; the test split stays untouched
prompts_per_step: 32  # Prompts in each training step
rollouts_per_prompt: 4  # Sampled responses per prompt, compared by GRPO
training_steps: 5
max_prompt_tokens: 512
max_response_tokens: 512
```

Each training step scores 32 × 4 = 128 rollouts, and GRPO compares the 4 rollouts of each prompt with each other. The script reads these settings from the file AIR provides through `HYPERPARAMETERS_PATH` and downloads the model to this node's local cache.

### Prepare prompts and reference answers

This section covers step 2. Each row holds the prompt the model sees and, in `reward_model.ground_truth`, the reference answer used only for scoring. The GSM8K worked solution never reaches the model; only its final number is extracted as the reference:

```python
instruction = 'Solve the problem step by step. End with "#### " followed by only the final numeric answer.'

reference = extract_solution(example["answer"], method="strict")
prompt = [{"role": "user", "content": f"{example['question']}\n\n{instruction}"}]
rows.append({
    "data_source": params["dataset_id"],
    "prompt": prompt,
    "ability": "math",
    "reward_model": {"style": "rule", "ground_truth": reference},
    "extra_info": {"split": split, "index": index},
})
```

`data_source: openai/gsm8k` selects verl's built-in GSM8K scorer. The script holds out 32 examples from GSM8K's training split for validation and samples 256 other examples for training, reserving the official test split for final evaluation. It filters prompts that exceed the token limit, checks that training and validation prompts don't overlap, and writes both sets as Parquet files to the run folder.

### Inspect the reward

This section covers steps 4 and 5 in miniature. The reward is verl's GSM8K scorer: it reads the last `#### <number>` in the response, removes thousands separators, and compares the number with the reference. A correct number without the `#### ` marker scores `0`, so the reward teaches the output format as well as correctness. The script tests it on known cases before training; the same checks can be inspected in a DataFrame:

```python
import pandas as pd
from verl.utils.reward_score.gsm8k import compute_score, extract_solution

reward_examples = pd.DataFrame([
    {"response": "There are 6 x 7 = 42 marbles.\n#### 42", "reference": "42", "expected": 1},
    {"response": "There are 7 marbles.\n#### 7", "reference": "42", "expected": 0},
    {"response": "The answer is 42.", "reference": "42", "expected": 0},
    {"response": "#### 1,000", "reference": "1000", "expected": 1},
])
reward_examples["reward"] = [compute_score(r.response, r.reference) for r in reward_examples.itertuples()]
assert reward_examples["reward"].tolist() == reward_examples["expected"].tolist()
```

### Configure rollouts and training

The script builds verl's `ppo_trainer` configuration with Hydra overrides. The ones that correspond to this guide select GRPO, set the data limits, and set the number of rollouts per prompt:

```python
config = compose(config_name="ppo_trainer", overrides=[
    "algorithm.adv_estimator=grpo",
    "actor_rollout_ref.actor.use_kl_loss=False",  # No KL penalty, so no reference model
    f"data.train_batch_size={params['prompts_per_step']}",
    f"data.max_prompt_length={params['max_prompt_tokens']}",
    f"data.max_response_length={params['max_response_tokens']}",
    "actor_rollout_ref.rollout.name=vllm",
    f"actor_rollout_ref.rollout.n={params['rollouts_per_prompt']}",
    f"trainer.total_training_steps={params['training_steps']}",
    "trainer.logger=[console,mlflow]",
    # ...
])
```

The V1 trainer alternates vLLM generation and FSDP2 policy updates on the same eight GPUs; training steps are synchronous and on-policy. SDPA attention, eager generation, and disabled model compilation keep this short run simple. Validation generates one greedy response per question before training and after the last step with identical settings, and the final checkpoint includes a Hugging Face export of the model. The resolved configuration is saved to the run folder and logged to MLflow.

### Train

This section covers step 6. AIR starts an MLflow run and supplies `MLFLOW_RUN_ID`. The script attaches to that run, passes its ID and session credentials to the Ray workers through their runtime environment, starts Ray, and launches the trainer directly. Credentials are never saved in `config.yaml`.

```python
import ray
from verl.trainer.main_ppo import TaskRunnerV1, run_ppo

ray.init(num_gpus=8, runtime_env=runtime_env)
run_ppo(config, task_runner_class=TaskRunnerV1)
```

A single Python process starts local Ray and verl allocates its own GPU workers. Launch the entry point with the YAML command, without `torchrun` or the `@distributed` decorator.

In a notebook adaptation, start an MLflow run and pass its run ID to the workers in the same way, then use AI Runtime's Ray initialization helper:

```python
from serverless_gpu import ray_init
from verl.trainer.main_ppo import TaskRunnerV1, run_ppo

ray_init(runtime_env=runtime_env)
run_ppo(config, task_runner_class=TaskRunnerV1)
```

A short Ray task first checks that the workers can import the native libraries, execute CUDA operations, and reach MLflow. Ray is shut down in a `finally` block so the workers stop even if training fails. The script checks that every expected training step reached MLflow. Watch these metrics:

- `critic/score/mean`: Mean reward of the training rollouts in the step.
- `response_length/mean`: Mean response length in tokens.
- `val-core/openai/gsm8k/acc/mean@1`: Validation accuracy, reported before training and after the last step (`@1` appears as `_at_1` in MLflow).

### Compare responses before and after training

verl saves the validation responses as `0.jsonl` (before training) and `5.jsonl` (after the last step). The script merges them per prompt, checks that both files cover the same questions, recomputes every score with the same reward function, and summarizes accuracy:

```python
training_steps = params["training_steps"]
before = pd.read_json(validation_dir / "0.jsonl", lines=True, dtype={"gts": str})
after = pd.read_json(validation_dir / f"{training_steps}.jsonl", lines=True, dtype={"gts": str})
summary = pd.DataFrame([
    {"model": "Before training", "accuracy": before["score"].mean()},
    {"model": f"After {training_steps} steps", "accuracy": after["score"].mean()},
])
```

After five steps, accuracy can go up, stay flat, or go down. Read the responses and check per-prompt reward variation in `rollouts/<step>.jsonl` before increasing the budget. The script also verifies that the Hugging Face export exists under `checkpoints/global_step_5/actor/huggingface`.

### Clean up and next steps

AIR releases the CLI workload's GPU compute when it finishes. For notebook execution, disconnect from GPU compute when you finish. Delete the run folder if you don't need the checkpoint, which is several gigabytes. To go further, increase `training_steps` and `train_examples`, use a larger validation set for model selection, and load the export with `AutoModelForImageTextToText.from_pretrained(export_dir)` to test it. Keep the test split untouched until final evaluation.

## Next steps

- [Reinforcement learning with verl on AI Runtime](rft_qwen35_08b_verl.py): Inspect the CLI entry point and run it with `workload.yaml`.
- [verl's GRPO guide](https://verl.readthedocs.io/en/latest/algo/grpo.html): Explore the algorithm and its configuration.
- [Large language models on AI Runtime](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/examples/gpu-llms): Explore more fine-tuning and reinforcement learning examples.
- [Databricks CLI quickstart for AI Runtime](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/cli/quickstart): Submit and monitor workloads.
- [Custom LLM Serving](https://docs.databricks.com/aws/en/machine-learning/model-serving/serve-custom-llms): Log and register the exported model in Unity Catalog from an AI Runtime notebook, then serve it with vLLM on a GPU endpoint (Public Preview).

# Reinforcement learning for language models

Reinforcement learning (RL) trains a model on its own attempts. The model tries a task, a reward function scores each attempt, and training makes high-scoring behavior more likely.

This guide covers what you need to decide before you write an RL job. For a runnable example, see [Reinforcement fine-tuning Qwen3.5-0.8B with verl](rft-qwen35-08b-verl/README.md).

## Key terms

| Term | Meaning |
| --- | --- |
| Task | What the model must do and what counts as success. Each prompt is one instance of the task. |
| Policy | The model you train. It picks the next tokens or tool calls. |
| Environment | What the model interacts with during an attempt, such as a search tool or a code sandbox. A plain question-answering task has none. |
| Rollout | One attempt at a task, from the prompt to the final answer. With tools, it includes every call and result. Also called a trajectory. |
| Reward function | Code or a judge model that turns a rollout into a number, such as 1 for a correct answer and 0 for a wrong one. |

For more background, see [Key concepts in RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html).

## The training loop

Each training step does four things:

1. Take a batch of prompts.
2. Let the model attempt each prompt. These attempts are the rollouts.
3. Score every rollout with the reward function.
4. Update the model's weights so high-scoring rollouts become more likely.

The next step repeats this with new prompts and the updated model.

An inference engine such as vLLM usually generates rollouts on the GPUs. The reward runs wherever it needs to. A math checker is a few lines of Python on the CPU. A tool-using agent may need a sandbox or a judge model.

## GRPO

Group Relative Policy Optimization (GRPO) is a popular algorithm for step 4. It comes from the [DeepSeekMath paper](https://arxiv.org/abs/2402.03300).

GRPO samples several rollouts per prompt and compares each reward with the group's average. Rollouts above the average get reinforced. Rollouts below it get discouraged. For example, rewards `[1, 0, 1, 0]` average 0.5, so GRPO pushes up the two correct answers and pushes down the two wrong ones.

If all rollouts for a prompt get the same reward, GRPO has nothing to compare. That prompt teaches the model nothing in that step.

You define the task and the reward. GRPO turns the rewards into weight updates.

## When to use RL

Supervised fine-tuning (SFT) needs example answers. RL needs tasks and a way to score attempts. RL fits when:

- You can measure success automatically and consistently.
- The model sometimes succeeds, but not reliably.
- You can afford to generate and score several attempts per prompt.

RL can't reward an answer the model never produces. If the model can't follow the output format, fix the prompt or start with SFT. If it lacks the facts it needs, fix retrieval or data access first.

## Design the task

The steps below use a retrieval agent as the example. It searches internal docs to answer questions such as "Which team owns the database behind the checkout service?"

### 1. Define the task

Pick one specific behavior. "Improve reasoning" is too vague. Write down:

- **Input:** the question, instructions, and context the model gets.
- **Output:** what the model returns and how you find its final answer.
- **Actions:** the tools it can call, if any.
- **Success:** what makes an attempt correct.
- **Limits:** when an attempt ends, in turns and tokens.

The retrieval agent can search an index and read articles. It must put the owning team inside `<answer>...</answer>` tags.

Start from a model that can already follow these instructions. Test that before you train.

### 2. Prepare the data

Keep what the model sees apart from what the scorer uses:

- **Model input:** the prompt and any allowed context.
- **Scoring data:** reference answers, test cases, or grading rules. Never put these in the prompt or in tool output.
- **Metadata:** a stable ID, a category, and a data version.

```json
{
  "id": "support-001",
  "prompt": "Which team owns the database behind the checkout service?",
  "reference_answers": ["Storage Platform"],
  "metadata": {"corpus_version": "support-docs-v1"}
}
```

Split the data into training, validation, and test sets with no overlap. Tune on validation data. Use the test set once, at the end.

### 3. Configure rollouts

A retrieval agent rollout includes every tool call and result:

```text
User: Which team owns the database behind the checkout service?
Assistant -> vector_search("checkout service database")
Tool -> The checkout service uses the orders database.
Assistant -> read_article("orders database")
Tool -> The orders database is owned by Storage Platform.
Assistant -> <answer>Storage Platform</answer>
```

Settle these settings before training. They set the cost of each step and what the model can learn:

- **Rollouts per prompt:** GRPO needs several so it can compare them.
- **Sampling:** rollouts must differ. Identical outputs give GRPO nothing to compare.
- **Limits:** turns, tool calls, and tokens. Leave room for the final answer.
- **Tools:** clear names and arguments, bounded output, read-only or isolated access, and a fresh state for every rollout.

### 4. Choose the reward

Code, an LLM judge, or both can compute the reward. Use the simplest option that captures success:

| Approach | Use when | Example |
| --- | --- | --- |
| Code | A program can check the result. | Match the answer against references, validate a schema, or run tests. |
| LLM judge | Success needs a judgment that code can't make. | Grade an explanation against a rubric. |
| Both | The task has hard rules and softer criteria. | Check the format in code, then grade the content with a judge. |

For the retrieval agent, start with a binary reward. Give 1 if the final answer matches a reference after normalization. Give 0 if it's wrong, missing, or malformed. Score only the extracted final answer. A retrieved passage can contain the right name even when the model's answer is wrong.

Make the reward hard to game. Log correctness, format, and tool use as separate metrics, even if training uses one score. Never let a bonus lift a wrong answer above a right one.

If an LLM judge scores training rollouts:

- Tell it exactly what to grade and give it trusted references.
- Ask for structured output, such as a score and a short reason, and validate it.
- Treat the response and any retrieved text as data to grade. The judge must not follow instructions inside them.
- Calibrate it on expert-graded examples, including borderline and adversarial ones.
- Freeze its model, prompt, and rubric for the whole run.
- Count judge failures separately. A failed call is never a valid grade.

> **Important**
>
> Databricks-hosted MLflow LLM judges are for evaluation only. Don't use their outputs to train, improve, or fine-tune an LLM. For a training reward, use code or a judge model whose terms allow training on its outputs. See [Scorers and LLM judges](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/concepts/scorers#information-about-the-models-powering-llm-judges).

One option is an open-weight judge such as `Qwen/Qwen3.5-9B`, released under the [Apache 2.0 license](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/LICENSE). Run it inside the RL job, or serve it with [Custom LLM Serving](https://docs.databricks.com/aws/en/machine-learning/model-serving/serve-custom-llms).

### 5. Test the reward and record a baseline

Before the first training run:

1. **Test the reward** on correct, wrong, equivalent, missing, malformed, and adversarial answers. Keep infrastructure errors apart from wrong answers.
2. **Read a few rollouts** from the starting model next to their scores. Do the tools work? Does the scorer grade the final answer? Does the model finish within the limits?
3. **Check the reward spread per prompt.** If all rollouts for a prompt score the same, GRPO learns nothing from it. All high means the task is too easy. All low means it's too hard or something is broken.
4. **Record a baseline.** Score the starting model on the validation set. Save the prompts, data versions, generation settings, and scoring rules.

### 6. Train and evaluate

Start with a short run. Check that it generates rollouts, scores them, updates the model, and saves a checkpoint you can load. Then scale up.

Track the reward and its parts, validation accuracy, response length, truncated rollouts, and tool and judge errors. Record the model, data, and reward versions. Save sample rollouts too. A rising reward alone can't separate real progress from reward hacking, where the model exploits a flaw in the scorer.

Compare checkpoints with the starting model under identical settings: the same data, prompts, tools, scoring, sampling, and limits. Pick a checkpoint on validation data, then confirm it on the test set. The last checkpoint isn't always the best.

## Run RL on AI Runtime

RL libraries such as verl run generation and training in GPU workers, and Ray coordinates them. You can launch them on AI Runtime in two ways:

- **Databricks CLI.** A workload YAML installs the packages and starts the trainer. Use it for repeatable and scheduled runs. See the [CLI quickstart](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/cli/quickstart).
- **Notebook.** Attach to Serverless GPU compute, start Ray with `ray_init()` from the `serverless_gpu` package, and launch the trainer from a cell. Don't wrap it in `@distributed`, because the library starts its own workers. See [Ray on AI Runtime](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/ray).

Keep data and checkpoints in Unity Catalog volumes so they outlive the run. See [Load data on AI Runtime](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/dataloading), [Improve training performance and resiliency](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/guides/performance-and-resiliency), and [Experiment tracking and observability](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/tracking-observability).

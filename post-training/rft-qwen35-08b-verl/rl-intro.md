# Reinforcement learning for language models

Reinforcement learning (RL) post-training updates an already trained language model using feedback on its own outputs. The model attempts a task, receives a numeric reward, and adjusts its weights so higher-scoring behavior becomes more likely.

Think of it as practice with a grader. Supervised fine-tuning (SFT) teaches a model to imitate example answers. RL lets the model try different approaches and learn which ones work. You don't need a worked solution for every prompt, but you do need a reliable way to score each attempt.

This guide explains the training loop and the choices behind a useful RL task. For a runnable example, see [Reinforcement fine-tuning Qwen3.5-0.8B with verl](README.md).

## How the training loop works

An RL training step alternates between using the model and updating it:

1. Take a batch of prompts.
2. Have the model generate attempts at each task.
3. Score those attempts with a reward function.
4. Update the model's weights to favor higher-scoring behavior.

The next step generates new attempts with the updated model. This is different from generating several answers and picking the best one at inference time. RL changes the model itself.

In RL terminology, the model being trained is the **policy**. One attempt is a **rollout**, also called a trajectory. For a question-answering task, a rollout is the generated response. For an agent, it also includes the tool calls and their results. The **reward function** turns that attempt into a number the training algorithm can use.

For example, [GSM8K](https://huggingface.co/datasets/openai/gsm8k) is a collection of grade-school math word problems. The model is asked to end its response with `####` followed by the final numeric answer. A Python scorer extracts that answer and compares it with the reference: a match earns 1, anything else earns 0. The scorer doesn't need a demonstration of how to solve the problem. It checks the final answer, not whether every step of the reasoning is sound.

## How GRPO uses the rewards

Group Relative Policy Optimization (GRPO) is a common RL algorithm for language models. It compares several attempts at the same prompt.

Suppose the question is "There are 6 rows of 7 marbles. How many marbles are there?" Four attempts might end like this:

| Rollout | Final answer | Reward |
| --- | --- | --- |
| 1 | `#### 42` | 1 |
| 2 | `#### 13` | 0 |
| 3 | `#### 42` | 1 |
| 4 | `#### 49` | 0 |

The average reward is 0.5. GRPO uses that average as a baseline, encouraging the two correct attempts and discouraging the two wrong ones. Each question gets its own comparison, rather than using one fixed baseline for both easy and hard problems.

If every attempt gets the same reward, there is no relative reward signal for that prompt in that step. Sampling several responses helps only if some of them score differently. This is why both the difficulty of the prompts and the variety of the generated answers matter.

For the full algorithm, see the [DeepSeekMath paper](https://arxiv.org/abs/2402.03300).

## When RL is useful

RL is a good fit when you can check success reliably and the starting model sometimes succeeds, but not consistently. RL can't reward an answer the model never produces, so a model that always fails gets no signal. Math answers and code that must pass tests are useful examples. The reward gives the model feedback without prescribing one exact solution to imitate.

Budget for generating and grading attempts, not just updating the weights. Each training step may need several responses per prompt. Longer responses, tool calls, and judge-model requests all add to the cost.

Before choosing RL, check whether a simpler change would solve the problem. If the model can't follow the output format, improve the prompt or start with SFT. If it lacks the facts needed to answer, fix retrieval or data access first.

## Set up the task

### Decide what success looks like

Start with a specific behavior rather than a goal such as "improve reasoning." Define what the model sees, what it must return, which tools it may use, and how an attempt ends. The model needs enough room to finish, so set token and tool-call limits with that in mind.

For example, a retrieval agent could answer questions about service ownership. Its tools let it search documentation and read articles. A rollout might look like this:

```text
User: Which team owns the database behind the checkout service?
Assistant -> search("checkout service database")
Tool -> The checkout service uses the orders database.
Assistant -> read_article("orders database")
Tool -> The orders database is owned by Storage Platform.
Assistant -> <answer>Storage Platform</answer>
```

The tools and documents are part of the agent's **environment**. Keep tool access read-only or isolated, bound the size of tool results, and reset any mutable state between rollouts. This is an example of a possible agent task, not part of the math recipe.

### Separate inputs from grading data

Each example needs the prompt and allowed context, plus whatever the scorer needs to grade the response. These are different inputs: the model gets the question, while the scorer might also get reference answers or private test cases. Don't leak scorer-only data into the prompt or tool results. Retrieved documents can still contain the evidence needed to answer the question.

Keep stable example IDs and data versions so you can trace surprising scores back to their source. Split examples into training, validation, and test sets without overlap. Use validation data to choose settings and checkpoints, and reserve the test set for the final assessment.

### Choose a reward you can trust

Use code when a program can check success: compare an extracted answer with a reference, validate a schema, or run tests. Training with rewards based on checkable outcomes is often called **RL with verifiable rewards (RLVR)**.

For the retrieval example, a simple reward could give 1 when the final answer matches an accepted team name and 0 otherwise. Score the extracted answer, not the whole transcript. A retrieved passage might mention the correct team even when the model's final answer is wrong.

Other tasks need a judgment about quality. An LLM judge can score an explanation against a rubric. A learned reward model, such as one trained on human preferences, is another option. In either case, check that the scores agree with human judgment before using them for training.

With an LLM judge, use a clear rubric and trusted references. Test it on borderline and adversarial responses, and keep its model and grading instructions fixed during the run. Treat generated answers and retrieved text as content to grade, not instructions for the judge to follow. Validate the returned score and track failed calls separately from wrong answers.

> **Important**
>
> Databricks-hosted MLflow LLM judges are for evaluation, not training rewards. Don't use their outputs to train, improve, or fine-tune an LLM. Use code or a judge model whose terms permit that use. See [Scorers and LLM judges](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/concepts/scorers#information-about-the-models-powering-llm-judges).

Start with a simple reward. If you add bonuses for format or efficient tool use, keep them secondary to correctness. Log these components separately so you can tell what changed.

## Check progress, not just reward

Before training, test the scorer on correct, wrong, equivalent, missing, malformed, and adversarial answers. Then inspect a few rollouts from the starting model alongside their scores. If every attempt scores zero, check the prompt, answer extraction, and generation limits before assuming the task is too hard. If every attempt scores one, make sure the scorer isn't accepting wrong answers before trying harder prompts.

Record the starting model's performance on the validation set. Save the model and data versions, prompts, scoring rules, and generation settings so you can make a fair comparison later.

Start with a short training run to check that generation, scoring, weight updates, and checkpoint loading work. That first run is a check of the setup, not evidence that the model has improved.

As training continues, watch validation performance alongside training reward, response length, truncated responses, and tool or judge errors. Read sample answers too. A rising reward can mean the model has found a flaw in the scorer rather than learned the task. This is called **reward hacking**.

Compare checkpoints with the starting model using the same validation prompts, tools, scoring, and generation settings. Choose a checkpoint on validation data, then assess it on the held-out test set. The last checkpoint isn't necessarily the best one.

## Run RL on AI Runtime

RL libraries such as [verl](https://github.com/verl-project/verl) run the whole loop for you. They generate rollouts with an inference engine such as vLLM, score them, and update the model. Ray coordinates the GPU workers. On AI Runtime, you launch the job with the Databricks CLI, and a `workload.yaml` file defines the compute, dependencies, and command. Keep data and checkpoints in a Unity Catalog volume so they outlive the job.

The [recipe README](README.md) runs this loop with verl on one 8x H100 node. For CLI setup and your first workload, see the [AI Runtime quickstart](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/quickstart).

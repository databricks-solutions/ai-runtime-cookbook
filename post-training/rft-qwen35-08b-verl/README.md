# Reinforcement fine-tuning Qwen3.5-0.8B with verl

This recipe uses [verl](https://github.com/verl-project/verl) to fine-tune [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) with GRPO, a reinforcement learning (RL) algorithm. verl is an open-source RL library for language models. It generates responses with vLLM and trains with PyTorch FSDP2, and Ray runs both on the same GPUs.

The task is [GSM8K](https://huggingface.co/datasets/openai/gsm8k), a set of grade-school math word problems. Each problem has one numeric answer, so code can check every response exactly. A response scores 1 if the number after its last `####` matches the reference and 0 otherwise. vLLM generates on the GPUs, and the reward runs as plain Python on the CPU. No judge model or sandbox is needed, and the same check measures validation accuracy.

Training runs 5 steps on one 8x H100 node. Each step samples 4 responses for each of 32 prompts and scores all 128. The recipe measures greedy accuracy on 32 held-out problems before and after training. The workload takes about 11 minutes.

To use your own task, replace the GSM8K data in `rft_qwen35_08b_verl.py` and add your scoring function through verl's `reward.custom_reward_function` setting. The recipe doesn't set it because verl has a built-in GSM8K scorer. For the ideas behind RL tasks and rewards, see [Reinforcement learning for language models](rl-intro.md).

## Before you run

`parameters.output_root` in `workload.yaml` must point to an existing Unity Catalog volume that you can write to. The default is `/Volumes/main/default/air_examples/rft-qwen35-08b-verl`. Change it if your workspace doesn't have that volume.

## Run

```bash
databricks air run -f workload.yaml
```

Training settings live in the `parameters` block of `workload.yaml`. To train longer, raise `training_steps`. Keep `train_examples` at least `training_steps` × `prompts_per_step`, and raise `timeout_minutes` (60 by default) to match.

## Results

The AIR MLflow run records the mean training reward (`critic/score/mean`), the mean response length (`response_length/mean`), and validation accuracy before and after training (`accuracy_before` and `accuracy_after`).

Five steps check that the loop works, but they're too few to change accuracy reliably. In 4 test runs, the mean training reward rose from about 0.25 to about 0.4 over the 5 steps, and validation accuracy rose by 0 to 3 of 32 questions (model revision `2fc0636`).

Each run also writes to `<output_root>/<MLFLOW_RUN_ID>/`:

- `validation_comparison.jsonl`: each validation question with its answers and scores before and after training.
- `rollouts/<step>.jsonl`: every sampled training response with its reward.
- `checkpoints/global_step_<training_steps>/actor/huggingface/`: the trained model. Qwen3.5 has a vision encoder, so load it with `AutoModelForImageTextToText.from_pretrained(path)`.

The checkpoints take several GB. Delete the run folder when you no longer need them.

# Preference tuning and GRPO with Hugging Face TRL

This recipe post-trains `Qwen/Qwen2.5-0.5B-Instruct` in two stages on one A10 with [TRL](https://github.com/huggingface/trl). Stage 1 runs DPO (Direct Preference Optimization) with a LoRA adapter on 2,000 UltraFeedback preference pairs. Stage 2 runs 40 steps of GRPO (Group Relative Policy Optimization) on GSM8K math questions, with a reward for a correct final answer and a smaller reward for the requested `#### <number>` format.

The recipe checks GSM8K test accuracy on 200 questions for the base model, after DPO, and after GRPO. The whole run takes about 25 minutes.

## Run

```bash
databricks air run -f workload.yaml --watch
```

## Results

The run prints accuracy after each stage. It writes `metrics.json` to `<output_root>/<MLflow run ID>/` and logs the same metrics to the run's MLflow run.

Two validation runs on an A10 produced these results:

| Stage | GSM8K accuracy, run 1 | GSM8K accuracy, run 2 |
|---|---|---|
| Base model | 34.5% | 34.5% |
| After DPO | 36.5% | 35.5% |
| After GRPO | 39.0% | 37.5% |

The base model uses greedy decoding and scores the same in both runs. Training results vary between runs. With 200 questions, the standard error is about 3.4 points. Treat these gains as directional. Raise `eval_samples` (up to 1,319) for a tighter estimate.

To try other GRPO variants, change `grpo_loss_type` to `grpo`, `dr_grpo` or `bnpo`. For larger models, switch to `GPU_1xH100`.

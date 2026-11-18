# Preference tuning with DPO and Hugging Face TRL

This recipe preference-tunes `Qwen/Qwen2.5-0.5B-Instruct` with DPO (Direct Preference Optimization) on one A10, using [TRL](https://github.com/huggingface/trl) and a LoRA adapter. It trains one epoch on 4,000 UltraFeedback pairs, each with a chosen and a rejected answer to the same prompt.

The recipe evaluates the tuned model in two ways, both on UltraFeedback test data it didn't train on:

- **Preference accuracy:** on 500 test pairs, how often the tuned model ranks the chosen answer above the rejected one, relative to the base model. A random guess would score 50%.
- **Reward-model win rate:** on 300 more test prompts, the base model and the DPO model each answer greedily. The `Skywork/Skywork-Reward-V2-Qwen3-0.6B` reward model scores both answers, and the win rate is how often the DPO answer scores higher.

The run takes about 33 minutes in the container, plus provisioning and queueing.

## Run

```bash
databricks air run -f workload.yaml --watch
```

## Results

The run prints both evaluations. It writes `metrics.json`, `generations.jsonl` (each prompt with both answers and their reward scores) and the LoRA adapter (`dpo_adapter/`) to `<output_root>/<MLflow run ID>/`. It also logs the training loss and reward curves and the final metrics to the MLflow run, and uploads `metrics.json` and `generations.jsonl`.

Three validation runs on an A10 produced these results. Each cell gives the range across the runs:

| Metric | Value |
|---|---|
| Held-out preference accuracy (500 pairs) | 66.1% to 66.9% |
| Reward-model win rate vs the base model (300 prompts) | 58.8% to 60.0% |
| Mean reward score, base → DPO | 0.85 → 1.16 to 1.26 |
| Mean answer length, base → DPO | 1,247 → 1,406 to 1,410 characters |
| Training time (4,000 pairs) | 18 minutes |

The standard errors are about 2.1 points for the preference accuracy and 2.8 points for the win rate. The spread across runs is smaller than that. DPO answers were about 13% longer, and reward models often favor longer answers. Read `generations.jsonl` to judge the answers yourself.

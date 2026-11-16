# Fine-tune gpt-oss-20b (mixture of experts) on one A10 with Unsloth

This recipe runs QLoRA fine-tuning of `unsloth/gpt-oss-20b` on a single 24 GB A10 using [Unsloth](https://github.com/unslothai/unsloth). gpt-oss-20b is a 21B-parameter mixture-of-experts (MoE) model. Unsloth loads it in 4-bit and trains LoRA adapters (0.44% of parameters) on the `HuggingFaceH4/Multilingual-Thinking` reasoning dataset for 30 steps, then generates a sample answer.

The run takes about 10 minutes. About half of that is downloading and loading the model.

## Run

```bash
databricks air run -f workload.yaml --watch
```

## Results

The run prints the training loss, peak GPU memory, time per step and a sample answer. It writes `metrics.json` and `sample.txt` to `<output_root>/<MLflow run ID>/`. Set `save_adapter: true` to also save the LoRA adapter there.

A validation run on an A10 produced these results:

| Metric | Value |
|---|---|
| Peak GPU memory after loading | 12.6 GB |
| Peak GPU memory during training | 15.8 GB of 23.7 GB |
| Time per step (batch 1 × 4 accumulation, 1,024 tokens) | 9.7 s |
| Training loss, first → last logged | 5.74 → 4.80 |

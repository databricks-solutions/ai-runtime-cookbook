# LoRA fine-tuning of Qwen3-VL-2B for receipt field extraction on one A10

This recipe fine-tunes `Qwen/Qwen3-VL-2B-Instruct` with LoRA on the language model to extract receipt fields as JSON. It runs on one 1×A10 GPU (`GPU_1xA10`) and compares three rows on the same 100 test receipts, with the same greedy decoding and scorer: the plain base model, the base model given the field schema in its prompt, and the tuned model.

The vision tower and `lm_head` stay frozen. LoRA adapters go on the language model's attention and MLP projections. Images are capped at 768 visual tokens (`max_pixels`). The run completed on one 24 GB A10. The CUDA caching allocator logged out-of-memory retry warnings during training and generation, and the run recovered from each one.

Data is [CORD-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) (CC-BY-4.0): 800 train, 100 validation and 100 test receipts. The test split is scored in full. The validation split is used only for validation loss. The schema prompt's field list is built from the train split only.

The Nemotron VL LoRA recipe in this cookbook also fine-tunes on CORD-v2, but on 8×H100 and it logs training metrics only. This recipe adds a field-extraction evaluation on a single A10.

## Before you run

The default `output_root` in `workload.yaml`, `/Volumes/main/default/air_examples/finetune-qwen3-vl-2b-lora-receipts-a10`, must be an existing Unity Catalog volume path you can write to. To use a different volume, override it at submission:

```bash
databricks air run -f workload.yaml --watch --override parameters.output_root=/Volumes/<catalog>/<schema>/<volume>/<path>
```

## Run

```bash
databricks air run -f workload.yaml --watch
```

## Results

All rows use the same 100 test receipts, greedy decoding with at most 512 new tokens, and the same scorer. Field-level F1 is computed per receipt over the leaf (field path, value) pairs of the gold `gt_parse` object. Numeric values are compared as numbers, and list items are matched without regard to order. Standard errors cover test-receipt sampling only. The run used one training seed.

| Row | Prompt | Field F1 (SE) | Exact match (SE) | JSON parse rate | Validation loss |
|---|---|---|---|---|---|
| Base | Plain instruction | 0.000 (SE 0.000) | 0.00 (SE 0.000) | 0.97 | 1.034 |
| Base + schema | Plain instruction plus the field paths from the train split | 0.245 (SE 0.010) | 0.00 (SE 0.000) | 1.00 | n/a |
| Tuned (LoRA) | Plain instruction (the training prompt) | 0.861 (SE 0.017) | 0.40 (SE 0.049) | 1.00 | 0.034 |

Paired differences, with SE computed from the per-receipt differences:

| Comparison | Field F1 difference (SE) | Exact match difference (SE) |
|---|---|---|
| Tuned − base | 0.861 (SE 0.017) | 0.40 (SE 0.049) |
| Tuned − base + schema | 0.616 (SE 0.018) | 0.40 (SE 0.049) |

The plain base row scores 0 because the model does not know CORD's key names. It invents its own keys, so none of its (path, value) pairs match the gold. The base + schema row is the fair baseline, since it names the same fields the tuned model learns. Against that row, the tuned model's field F1 is 0.616 higher, and its exact-match gain is 0.40.

A second default run on the same settings, before the base + schema row was added, gave tuned field F1 0.857 (SE 0.019) and exact match 0.41 (SE 0.049), with the same base row. The run-to-run variation is about 0.004 in field F1 and 0.01 in exact match.

The run also writes these files to `/Volumes/main/default/air_examples/finetune-qwen3-vl-2b-lora-receipts-a10/<MLFLOW_RUN_ID>/`:

- `lora_adapter/`: the PEFT LoRA adapter. Its `adapter_config.json` names `Qwen/Qwen3-VL-2B-Instruct` as the base model. Loading the adapter was not part of this validation.
- `predictions_base.jsonl`, `predictions_base_schema.jsonl` and `predictions_tuned.jsonl`: the raw output and per-receipt scores for each row. These are also attached to the MLflow run.

Measured on one 1×A10 (`GPU_1xA10`) with environment version 5, the run takes about 23 minutes in the container, not counting queue time.

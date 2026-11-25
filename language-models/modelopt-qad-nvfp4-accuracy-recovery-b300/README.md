# Try to recover NVFP4 accuracy with quantization-aware distillation on B300

This recipe quantizes `Qwen/Qwen3-1.7B` to NVFP4 (4-bit floating point) with [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) and then tries to recover the accuracy that post-training quantization (PTQ) loses. It trains the NVFP4 student with quantization-aware distillation (QAD) against the original bf16 model as teacher, with a forward KL loss on logits. It then exports the student as an NVFP4 checkpoint and compares bf16, PTQ and QAD on all 1,319 GSM8K test questions with vLLM.

The whole run takes about 17 minutes in the container on one 8xB300 node. QAD training takes about 8 minutes (`train_runtime` 468.2 s, 466 steps). The QAD stage runs on all 8 GPUs with FSDP2. The run uses the full node because B300 is offered only as an 8-GPU shape, and FSDP2 shards the student across the 8 GPUs during QAD.

## Before you run

The run writes checkpoints and results to `/Volumes/main/default/air_examples/modelopt-qad-nvfp4-accuracy-recovery-b300/<MLflow run ID>/`. That volume must already exist and be writable. To write elsewhere, pass another existing writable volume directory:

```bash
databricks air run -f workload.yaml --override parameters.output_root=/Volumes/<catalog>/<schema>/<volume>/<dir> --watch
```

## Run

```bash
databricks air run -f workload.yaml --watch
```

## Results

The run prints a table of GSM8K accuracy with standard errors. It also prints paired differences between the three models with their paired standard errors. The same numbers are logged to the AIR MLflow run, and the training loss is logged there every step. The run directory contains these files:

- `results.json`: the table and the comparisons.
- `eval_bf16.jsonl`, `eval_ptq.jsonl`, `eval_qad.jsonl`: one row per test question (correct or not, prediction, output length), so you can compare the models question by question.
- `ptq_nvfp4/` and `qad_nvfp4/`: the exported NVFP4 checkpoints. `qad_ckpt/` holds the QAD ModelOpt checkpoint.

Three default runs were made on 8xB300, each on the full 1,319-question GSM8K test split with one QAD run and one seed. The latest run uses standard environment `"6"`; the earlier runs used `databricks_ai_v6` with the same recipe settings. The earlier runs also differed from each other only in a docstring and a code comment.

Standard v6 validation:

| Model | GSM8K accuracy | SE | Diff vs bf16 | Paired SE |
|---|---|---|---|---|
| Qwen3-1.7B bf16 | 0.8165 | 0.0107 | | |
| Qwen3-1.7B NVFP4 PTQ | 0.7240 | 0.0123 | -0.0925 | 0.0119 |
| Qwen3-1.7B NVFP4 QAD | 0.7619 | 0.0117 | -0.0546 | 0.0119 |

Earlier run 3 (`databricks_ai_v6`):

| Model | GSM8K accuracy | SE | Diff vs bf16 | Paired SE |
|---|---|---|---|---|
| Qwen3-1.7B bf16 | 0.8165 | 0.0107 | | |
| Qwen3-1.7B NVFP4 PTQ | 0.7240 | 0.0123 | -0.0925 | 0.0118 |
| Qwen3-1.7B NVFP4 QAD | 0.7779 | 0.0114 | -0.0387 | 0.0115 |

Earlier run 2 (`databricks_ai_v6`):

| Model | GSM8K accuracy | SE | Diff vs bf16 | Paired SE |
|---|---|---|---|---|
| Qwen3-1.7B bf16 | 0.8180 | 0.0106 | | |
| Qwen3-1.7B NVFP4 PTQ | 0.7240 | 0.0123 | -0.0940 | 0.0119 |
| Qwen3-1.7B NVFP4 QAD | 0.7483 | 0.0119 | -0.0697 | 0.0119 |

QAD minus PTQ is +0.0379 in the standard v6 run (paired SE 0.0128, 95% CI about [+0.013, +0.063]), +0.0538 in earlier run 3 (paired SE 0.0123, 95% CI about [+0.030, +0.078]), and +0.0243 in earlier run 2 (paired SE 0.0126, 95% CI about [-0.000, +0.049]). The point estimate is positive in all three runs, but run 2's 95% interval includes zero. The gain varied by about 0.03 across the three runs, and the environment changed for the latest run. These runs do not isolate the cause of the variation.

## How it works

1. **PTQ.** `quantize.py` calibrates NVFP4 with `mtq.NVFP4_DEFAULT_CFG` (weights and activations in NVFP4; the attention matmul and KV-cache quantizers stay disabled) on 512 GSM8K train questions. This is the same config the batch-inference recipe validated through vLLM on B300.
2. **Baseline and PTQ evaluation.** The bf16 model is evaluated on GPU 7 while PTQ runs on GPU 0. Both are then evaluated in vLLM with greedy decoding, 1,024 output tokens and the system prompt from the batch-inference recipe.
3. **QAD.** `qad_train.py` builds a `QADTrainer` from ModelOpt 0.48.0. The student is the PTQ checkpoint, with NVFP4 fake quantization in the forward pass. The teacher is the bf16 model. The loss is a forward KL between their logits on the answer tokens of GSM8K train. Training runs for 2 epochs at learning rate 1e-5 with cosine decay, under `accelerate` with FSDP2 on 8 GPUs.
4. **Export and evaluation.** `export.py` turns the PTQ and QAD checkpoints into NVFP4 checkpoints that vLLM loads. `evaluate.py` scores them, and `summarize.py` computes the comparison.

Before the earlier full runs, a smoke run with reduced settings (64 test questions, 64 training examples, 32 calibration samples, 1 epoch) tested the pipeline. Its numbers are not a result: bf16 0.7656, PTQ 0.6719, QAD 0.6562. The smoke run and the first full validation run used identical files; the second full validation run changed only comments. The later standard v6 run changed the environment and dependencies, not the recipe parameters.

Calibration and training use only the GSM8K train split. The test split is used only for the final evaluation, and no setting was chosen by looking at it.

## Caveats

- The training data is GSM8K train, the same task as the test set. The QAD gain therefore measures recovery on this task. It does not show that general capability is recovered. A general-knowledge check would need a separate held-out benchmark.
- Three default runs with one QAD seed each gave different QAD gains (+0.024 to +0.054 over PTQ). The cause of the difference is not isolated. More seeds would be needed to estimate the spread.
- The student is trained in the same answer format it is evaluated on (the system prompt and `####` final answer), so the gain partly reflects format.

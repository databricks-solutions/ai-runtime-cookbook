# Try to recover NVFP4 accuracy with quantization-aware distillation on B300

This recipe quantizes `Qwen/Qwen3-1.7B` to NVFP4 (4-bit floating point) with [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) and then tries to recover the accuracy that post-training quantization (PTQ) loses. It trains the NVFP4 student with quantization-aware distillation (QAD) against the original bf16 model as teacher, with a forward KL loss on logits. It then exports the student as an NVFP4 checkpoint and compares bf16, PTQ and QAD on all 1,319 GSM8K test questions with vLLM.

The whole run takes about 17 minutes in the container on one 8xB300 node (17m 35s in the final validation run). QAD training takes about 8 minutes (`train_runtime` 482.1 s, 466 steps). The QAD stage runs on all 8 GPUs with FSDP2. The run uses the full node because B300 is offered only as an 8-GPU shape, and FSDP2 shards both the student and the teacher across the 8 GPUs during QAD.

## Before you run

The run writes checkpoints and results to `/Volumes/main/default/air_examples/modelopt-qad-nvfp4-accuracy-recovery-b300/<MLflow run ID>/`. That volume must already exist and be writable. Replace the `output_root` parameter in `workload.yaml` if it is not.

## Run

```bash
databricks air run -f workload.yaml --watch
```

## Results

The run prints a table of GSM8K accuracy with standard errors. It also prints paired differences between the three models with their paired standard errors. The same numbers are logged to the AIR MLflow run, and the training loss is logged there every step. The run directory contains these files:

- `results.json`: the table and the comparisons.
- `eval_bf16.jsonl`, `eval_ptq.jsonl`, `eval_qad.jsonl`: one row per test question (correct or not, prediction, output length), so you can compare the models question by question.
- `ptq_nvfp4/` and `qad_nvfp4/`: the exported NVFP4 checkpoints. `qad_ckpt/` holds the QAD ModelOpt checkpoint.

Two default runs were made on 8xB300, each on the full 1,319-question GSM8K test split with one QAD run and one seed. Run 3 is the final-file run. Run 2 used the same settings and differed only in a docstring and a code comment.

Run 3 (final files):

| Model | GSM8K accuracy | SE | Diff vs bf16 | Paired SE |
|---|---|---|---|---|
| Qwen3-1.7B bf16 | 0.8165 | 0.0107 | | |
| Qwen3-1.7B NVFP4 PTQ | 0.7240 | 0.0123 | -0.0925 | 0.0118 |
| Qwen3-1.7B NVFP4 QAD | 0.7779 | 0.0114 | -0.0387 | 0.0115 |

Run 2:

| Model | GSM8K accuracy | SE | Diff vs bf16 | Paired SE |
|---|---|---|---|---|
| Qwen3-1.7B bf16 | 0.8180 | 0.0106 | | |
| Qwen3-1.7B NVFP4 PTQ | 0.7240 | 0.0123 | -0.0940 | 0.0119 |
| Qwen3-1.7B NVFP4 QAD | 0.7483 | 0.0119 | -0.0697 | 0.0119 |

QAD minus PTQ is +0.0538 in run 3 (paired SE 0.0123, 95% CI about [+0.030, +0.078]) and +0.0243 in run 2 (paired SE 0.0126, 95% CI about [-0.000, +0.049]). The point estimate is positive in both runs, and run 2's 95% interval includes zero. The two runs differ by about 0.03 with the same settings, which is about the size of the gain. The QAD gain's size is therefore not stable across these two runs.

## How it works

1. **PTQ.** `quantize.py` calibrates NVFP4 with `mtq.NVFP4_DEFAULT_CFG` (weights and activations in NVFP4; the attention matmul and KV-cache quantizers stay disabled) on 512 GSM8K train questions. This is the same config the batch-inference recipe validated through vLLM on B300.
2. **Baseline and PTQ evaluation.** The bf16 model is evaluated on GPU 7 while PTQ runs on GPU 0. Both are then evaluated in vLLM with greedy decoding, 1,024 output tokens and the system prompt from the batch-inference recipe.
3. **QAD.** `qad_train.py` builds a `QADTrainer` from ModelOpt 0.48.0. The student is the PTQ checkpoint, with NVFP4 fake quantization in the forward pass. The teacher is the bf16 model. The loss is a forward KL between their logits on the answer tokens of GSM8K train. Training runs for 2 epochs at learning rate 1e-5 with cosine decay, under `accelerate` with FSDP2 on 8 GPUs.
4. **Export and evaluation.** `export.py` turns the PTQ and QAD checkpoints into NVFP4 checkpoints that vLLM loads. `evaluate.py` scores them, and `summarize.py` computes the comparison.

Before the full runs, a smoke run with reduced settings (64 test questions, 64 training examples, 32 calibration samples, 1 epoch) tested the pipeline. Its numbers are not a result: bf16 0.7656, PTQ 0.6719, QAD 0.6562. No setting in `workload.yaml` changed after that run. The smoke run and the first validation run used identical files; the second validation run changed only comments.

Calibration and training use only the GSM8K train split. The test split is used only for the final evaluation, and no setting was chosen by looking at it.

## Caveats

- The training data is GSM8K train, the same task as the test set. The QAD gain therefore measures recovery on this task. It does not show that general capability is recovered. A general-knowledge check would need a separate held-out benchmark.
- Two default runs with one QAD seed each gave different QAD gains (+0.024 and +0.054 over PTQ). The cause of the difference is not isolated. More seeds would be needed to estimate the spread.
- The student is trained in the same answer format it is evaluated on (the system prompt and `####` final answer), so the gain partly reflects format.

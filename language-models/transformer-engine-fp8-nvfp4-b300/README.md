# bf16 vs FP8 vs NVFP4 training with Transformer Engine on B300

This recipe trains the same 2.2B-parameter Llama-style model from scratch in four precisions with [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine) (TE) on one 8xB300 node. The precisions are bf16, FP8 with per-tensor current scaling, MXFP8 (FP8 with per-block scales), and NVFP4 (4-bit). Each precision trains for 200 steps on WikiText-103 (16K tokens per step). The four precisions run in parallel, one per GPU.

Time in the container is 2 minutes 43 seconds when the binding comes from `TE_WHEEL_CACHE`. The first run also compiles TE's PyTorch binding and saves it to `TE_WHEEL_CACHE` in the output volume. That run took 4 minutes 12 seconds in the container.

The recipe pins Transformer Engine 2.17.1, because newer releases need a newer cuBLASLt than `databricks_ai_v6` provides. MXFP8 and NVFP4 need Blackwell GPUs, and the recipe was validated only on B300.

## Before you run

The default `output_root` and `TE_WHEEL_CACHE` are Unity Catalog volumes under `/Volumes/main/default/air_examples/transformer-engine-fp8-nvfp4-b300`. The volume must already exist and be writable. To write the outputs to another volume, run:

```bash
databricks air run -f workload.yaml --override parameters.output_root=/Volumes/<catalog>/<schema>/<volume>/<dir> --watch
```

## Run

```bash
databricks air run -f workload.yaml --watch
```

## Results

The run prints a comparison table. It also writes `results.json` to `<output_root>/<MLflow run ID>/` and logs tokens per second and final loss for each precision to MLflow.

Two validation runs on 8xB300 produced these results. Run 1 compiled and saved the TE binding. Run 2 installed the saved binding. Final loss is the mean of the last 20 steps.

| Precision | Run 1 tokens/s per GPU | Run 2 tokens/s per GPU | Run 1 speedup | Run 2 speedup | Run 1 final loss | Run 2 final loss | Peak memory |
|---|---|---|---|---|---|---|---|
| bf16 | 83,554 | 87,262 | 1.00× | 1.00× | 5.829 | 5.828 | 59.9 GB |
| FP8 (current scaling) | 111,967 | 113,113 | 1.34× | 1.30× | 5.834 | 5.831 | 55.8 GB |
| MXFP8 | 111,609 | 114,118 | 1.34× | 1.31× | 5.827 | 5.832 | 60.2 GB |
| NVFP4 | 137,667 | 139,096 | 1.65× | 1.59× | 5.847 | 5.840 | 55.8 GB |

Across the two runs, the speedup over bf16 was 1.30× to 1.34× for FP8, 1.31× to 1.34× for MXFP8, and 1.59× to 1.65× for NVFP4. Peak memory was the same in both runs.

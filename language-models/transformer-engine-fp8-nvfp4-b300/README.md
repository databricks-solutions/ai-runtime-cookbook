# bf16 vs FP8 vs NVFP4 training with Transformer Engine on B300

This recipe trains the same 2.2B-parameter Llama-style model from scratch in four precisions with [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine) (TE) on one 8xB300 node. The precisions are bf16, FP8 with per-tensor current scaling, MXFP8 (FP8 with per-block scales), and NVFP4 (4-bit). Each precision trains for 200 steps on WikiText-103 (16K tokens per step). The four precisions run in parallel, one per GPU.

The whole run takes about 5 minutes. The first run also compiles TE's PyTorch binding, which takes about 2 minutes. It saves the compiled binding to `TE_WHEEL_CACHE` in the output volume, and later runs install it from there.

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

A validation run on 8xB300 produced these results (final loss is the mean of the last 20 steps):

| Precision | Tokens/s per GPU | Speedup vs bf16 | Final loss | Peak memory |
|---|---|---|---|---|
| bf16 | 83,457 | 1.00× | 5.827 | 59.9 GB |
| FP8 (current scaling) | 111,831 | 1.34× | 5.840 | 55.8 GB |
| MXFP8 | 110,802 | 1.33× | 5.826 | 60.2 GB |
| NVFP4 | 136,477 | 1.64× | 5.851 | 55.8 GB |

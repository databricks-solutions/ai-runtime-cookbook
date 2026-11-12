# AI Runtime CLI quickstart

This is the runnable example corresponding to the [AI Runtime CLI quickstart](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/quickstart). It uploads `main.py`, runs it on one A10 GPU, and prints a greeting and the GPU name.

`workload.yaml` uses the Databricks AI environment version 6, which includes PyTorch.

## Run

[Select and verify your workspace profile](../../README.md#select-a-workspace-profile) before submitting. Keep that profile selected when [checking status or logs](../../README.md#submit-and-monitor).

```bash
databricks air run --file workload.yaml --watch
```

To bring your own stack, replace the `environment` block with Standard environment version 6 and explicit dependencies:

```yaml
environment:
  version: 6
  dependencies:
    - torch==2.11.0
```

CLI runs install these dependencies on every run, so this option adds PyTorch and CUDA wheel download time.

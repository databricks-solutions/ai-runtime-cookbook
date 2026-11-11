# AI Runtime CLI quickstart

This is the runnable example corresponding to the [AI Runtime CLI quickstart](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/quickstart). It uploads `main.py`, runs it on one H100 GPU, and prints a greeting and the GPU name.

## Run

[Select and verify your workspace profile](../../README.md#select-a-workspace-profile) before submitting. Keep that profile selected when [checking status or logs](../../README.md#submit-and-monitor).

```bash
databricks air run --file workload.yaml --watch
```

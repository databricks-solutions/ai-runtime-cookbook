# AI Runtime CLI quickstart

This is the runnable example corresponding to the [AI Runtime CLI quickstart](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/cli/quickstart). It uploads `main.py`, runs it on one H100 GPU, and prints a greeting and the GPU name.

## Run

```bash
databricks air run --file workload.yaml --watch
```

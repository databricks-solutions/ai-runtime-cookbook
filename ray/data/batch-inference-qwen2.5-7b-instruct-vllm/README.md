# Batch inference with Qwen2.5-7B-Instruct and vLLM

This recipe uses Ray Data and vLLM to generate responses for 1,000 Alpaca prompts with `Qwen/Qwen2.5-7B-Instruct`. It runs one vLLM replica per GPU on an 8x H100 node and writes the generated text as Parquet.

## Before you run

Set `OUTPUT_PATH` in `workload.yaml` to an existing writable Unity Catalog volume.

## Run

```bash
databricks air run -f workload.yaml
```

## Results

The recipe writes a Parquet dataset with `instruction` and `output` columns to `OUTPUT_PATH`.

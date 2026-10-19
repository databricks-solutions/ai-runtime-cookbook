# Batch image understanding with Qwen3-VL-8B and vLLM

This recipe uses Ray Data and vLLM to caption and classify the 3,669 photos in the [Oxford-IIIT Pet](https://huggingface.co/datasets/timm/oxford-iiit-pet) test split with `Qwen/Qwen3-VL-8B-Instruct`, running one vLLM replica per GPU on an 8x H100 node. vLLM structured outputs constrain every answer to a JSON schema with a one-sentence caption, the species, and one of the 37 breeds, so every response parses. To use your own images, change the prompt, the schema, and `load_images()` in `batch_inference.py`; for unlabeled images, also remove the accuracy columns and metrics, which compare against the dataset's labels.

## Before you run

`OUTPUT_PATH` in `workload.yaml` must be on an existing Unity Catalog volume that you can write to.

## Run

```bash
databricks air run -f workload.yaml
```

For a quick test on one GPU:

```bash
databricks air run -f workload.yaml \
  --override compute.accelerator_type=GPU_1xH100 --override compute.num_accelerators=1 \
  --override env_variables.NUM_IMAGES=64
```

## Results

Each run writes one Parquet row per image to `OUTPUT_PATH/<MLflow run ID>`, with the true and predicted species and breed, the caption, and per-image correctness flags. The workload's MLflow run records the accuracy and throughput metrics, a `sample_predictions.json` table with the first 50 predictions, and a `breed_confusions.json` table with the 20 most common mistakes.

On the full test split, every response is valid JSON, species accuracy is about 100%, zero-shot breed accuracy is about 88%, and steady-state throughput on one node is about 38 images per second (measured with model revision `0c351dd` and dataset revision `089695c`).

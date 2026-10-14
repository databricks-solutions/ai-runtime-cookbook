# LoRA fine-tuning of Nemotron 3.5 Lightning with NeMo AutoModel (multi-GPU)

This recipe fine-tunes `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` (a hybrid Mamba-attention mixture-of-experts model) on HellaSwag for 50 steps on one 8x H100 node, using FSDP2, eight-way expert parallelism, rank-8 LoRA, and the GMM/DeepEP backend from NVIDIA's `nemo-automodel:26.08.00` container run as a custom [AIR container image](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/cli/docker-images). It trains with a local batch of 32 and a global batch of 256 (no gradient accumulation), and runs validation and saves a checkpoint at the final step. The run completes in about 6 minutes (~355s of workload execution, excluding the one-time image push and cluster startup) and logs training metrics only — HellaSwag multiple-choice accuracy requires a separate evaluation.

## Before you run

Custom images must be enabled for your workspace, which a workspace admin does; see [Run AI Runtime workloads with custom Docker images](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/cli/docker-images) for how to enable them and the Unity Catalog Artifact Registry privileges the push requires. AIR runs custom images by their Unity Catalog coordinates.

Pushing the NGC image needs Docker installed and running locally, an [NGC account](https://org.ngc.nvidia.com/) with an [API key](https://org.ngc.nvidia.com/setup/api-key), and privileges to push an image into the target Unity Catalog schema. You can push to any schema you own as long as you update `environment.unity_catalog_image` in `workload.yaml` to match.

The `nvcr.io/nvidia/*` containers are typically gated, so first authenticate to NGC (skip if the image is public). Use `$oauthtoken` as the username and your NGC API key as the password:

```bash
docker login nvcr.io
```

Then push the image into your Unity Catalog. `--pull` fetches the `linux/amd64` image fresh from NGC even when another architecture is cached locally (for example an arm64 image pulled on an Apple Silicon machine); the container is large (several GB), so this takes a while:

```bash
databricks air images push -s nvcr.io/nvidia/nemo-automodel:26.08.00 --pull --catalog main --schema default
```

The image is referenced in `workload.yaml` via `environment.unity_catalog_image` (default `main.default.nemo-automodel:26.08.00`). Change `--catalog`/`--schema` above and that field together if you push to a different Unity Catalog location.

Set `env_variables.ADAPTER_OUTPUT_ROOT` in `workload.yaml` to an existing writable Unity Catalog volume if the default location (`/Volumes/main/default/air_examples/lora-nemotron-3.5-lightning-30b-a3b-automodel-multi-gpu`) is unsuitable; the volume must already exist.

## Run

```bash
databricks air run -f workload.yaml
```

## Results

AutoModel logs training loss, learning rate, gradient norm, and throughput every five steps to the AIR MLflow run, and validation loss at the final step. The final adapter and tokenizer files are uploaded to `final_adapter` in the run and copied to `ADAPTER_OUTPUT_ROOT/<MLFLOW_RUN_ID>/final_adapter` on the configured Unity Catalog volume; checkpoints stay on the node's local disk.

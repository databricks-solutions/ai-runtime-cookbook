# LoRA fine-tuning of Nemotron Lightning with NeMo AutoModel (multi-GPU)

This recipe fine-tunes `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` on HellaSwag for 50 steps on one 8x H100 node. It uses FSDP2, eight-way expert parallelism, rank-8 LoRA, and the GMM/DeepEP backend from NVIDIA's `nemo-automodel:26.08.00` container, run as a custom [AIR container image](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/cli/docker-images).

## Before you run

Enable [AIR custom images](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/cli/docker-images) in your workspace, then push the NGC image into your Unity Catalog Artifact Registry once (per catalog/schema). AIR runs custom images by their Unity Catalog coordinates.

The `nvcr.io/nvidia/*` containers are typically gated, so first authenticate to NGC (skip if the image is public). Use `$oauthtoken` as the username and your [NGC API key](https://org.ngc.nvidia.com/setup/api-key) as the password:

```bash
docker login nvcr.io
```

Then push the image into your Unity Catalog (pulls `linux/amd64` from NGC and pushes to `<catalog>.<schema>.nemo-automodel:26.08.00`; the ~large container makes this take a while):

```bash
databricks air images push -s nvcr.io/nvidia/nemo-automodel:26.08.00 --catalog main --schema default
```

The image is referenced in `workload.yaml` via `environment.unity_catalog_image` (default `main.default.nemo-automodel:26.08.00`). Change `--catalog`/`--schema` above and that field together if you push to a different Unity Catalog location.

Set `env_variables.ADAPTER_OUTPUT_ROOT` in `workload.yaml` to an existing writable Unity Catalog volume if the default location (`/Volumes/main/default/air_examples/lora-nemotron-lightning-30b-automodel-multi-gpu`) is unsuitable; the volume must already exist.

The HellaSwag dataset is public, but the Nemotron model repo may require accepting NVIDIA's license on Hugging Face. If the model (or your workspace) gates the download, add a `secrets:` block to `workload.yaml` mapping `HF_TOKEN` to the shared cookbook secret `my_scope/hf_token` (replace with your own scope/key if needed).

## Run

```bash
databricks air run -f workload.yaml
```

Adjust batch size and run length in `env_variables`. Defaults are local batch 32 and global batch 256, with no gradient accumulation. Set global batch to `8 * LOCAL_BATCH_SIZE` to retain that behavior. Validation and checkpoint saving occur at the final step.

## Results

The 50-step run completes in about 6 minutes (~355s of workload execution, excluding the one-time image push and cluster startup). AutoModel logs training loss, learning rate, gradient norm, and throughput every five steps to the AIR MLflow run; validation loss is logged at the final step. These are training metrics — HellaSwag multiple-choice accuracy requires separate evaluation. Checkpoints stay on local disk; the final adapter and tokenizer files are uploaded to `final_adapter` in the run and copied to `ADAPTER_OUTPUT_ROOT/<MLFLOW_RUN_ID>/final_adapter` on the configured Unity Catalog volume.

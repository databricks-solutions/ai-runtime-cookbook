# LoRA fine-tuning of Nemotron Super VL with NeMo AutoModel (multi-GPU)

This recipe fine-tunes `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16` on CORD-v2 receipt images for 50 steps on one 8x H100 node. It uses FSDP2, eight-way expert parallelism, rank-64 LoRA, and the GMM/DeepEP backend; the vision and audio towers stay frozen and MTP is disabled. Stock `nemo-automodel:26.08.00` does not yet include Super VL support, so the recipe builds a custom [AIR container image](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/cli/docker-images) from the bundled `Dockerfile`.

## Before you run

Enable [AIR custom images](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/cli/docker-images) in your workspace.

The base `nvcr.io/nvidia/*` container is gated, so authenticate to NGC first. Use `$oauthtoken` as the username and your [NGC API key](https://org.ngc.nvidia.com/setup/api-key) as the password:

```bash
docker login nvcr.io
```

Build the custom Super VL image for `linux/amd64` and push it into your Unity Catalog Artifact Registry once (per catalog/schema; the large container makes this take a while):

```bash
docker build --platform linux/amd64 -t nemo-automodel-vl:26.08.00 .
databricks air images push -s nemo-automodel-vl:26.08.00 --catalog main --schema default
```

The image is referenced in `workload.yaml` via `environment.unity_catalog_image` (default `main.default.nemo-automodel-vl:26.08.00`). Change `--catalog`/`--schema` above and that field together if you push to a different Unity Catalog location.

The Super VL model may require accepting NVIDIA's license on Hugging Face. The Hugging Face token behind the shared cookbook secret `my_scope/hf_token` (mapped to `HF_TOKEN` in `workload.yaml`) must belong to an account that has accepted the license (or otherwise been granted access); replace it with your own scope/key if needed.

Set `env_variables.ADAPTER_OUTPUT_ROOT` in `workload.yaml` to an existing writable Unity Catalog volume if the default location (`/Volumes/main/default/air_examples/lora-nemotron-super-vl-120b-a12b-automodel-multi-gpu`) is unsuitable; the volume must already exist.

## Run

```bash
databricks air run -f workload.yaml
```

Adjust batch size and run length in `env_variables`. Defaults are local batch 1 and global batch 8, with no gradient accumulation; set global batch to `8 * LOCAL_BATCH_SIZE` to retain that per-GPU behavior. Validation and checkpoint saving occur at the final step.

## Results

The 50-step run completes in about 12 minutes (~750s of workload execution, excluding the one-time image build/push and cluster startup). AutoModel logs training loss, learning rate, gradient norm, and throughput every five steps to the AIR MLflow run; validation loss is logged at the final step. These are training metrics — CORD-v2 field-extraction quality requires separate evaluation. Checkpoints stay on local disk; the final LoRA adapter and config are uploaded to `final_adapter` in the run and copied to `ADAPTER_OUTPUT_ROOT/<MLFLOW_RUN_ID>/final_adapter` on the configured Unity Catalog volume.

# LoRA fine-tuning of Nemotron 3.5 Super VL with NeMo AutoModel (multi-GPU)

This recipe uses NeMo AutoModel to fine-tune `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16` on the CORD-v2 receipt-image dataset with LoRA. It is a vision-language mixture-of-experts model; the vision and audio towers stay frozen and multi-token prediction (MTP) is disabled.

Training runs for 50 steps on one 8x H100 node. It uses FSDP2, eight-way expert parallelism, rank-64 LoRA, and the GMM/DeepEP backend. The local batch size is 1 per GPU for a global batch of 8 (no gradient accumulation). At the final step, it runs validation and saves a checkpoint. It logs training metrics only; CORD-v2 field-extraction quality requires a separate evaluation.

Environment configurations: stock `nemo-automodel:26.08.00` does not yet include Super VL support, so the recipe builds a custom [AIR container image](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/cli/docker-images) from the bundled `Dockerfile`, which adds Super VL support on top of NVIDIA's `nemo-automodel:26.08.00` image.

Workload execution takes about 12 minutes. For the first run, the one-time image build and upload may take additional time.

## Before you run

Enable the necessary previews in your workspace (a workspace admin does this) by following the custom Docker image guide: [AWS](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/cli/docker-images) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/ai-runtime/cli/docker-images).

### Custom image setup

Install Docker and start it on your machine. Choose a Unity Catalog schema where you have permission to push images; this example uses `main.default`. See the guide above for the required Artifact Registry privileges.

The Dockerfile's base image `nvcr.io/nvidia/nemo-automodel:26.08.00` is public and the AutoModel source is fetched from public GitHub, so you do not need an NGC account, API key, or Docker login to build it. If you switch to an NGC image that requires authentication, run `docker login nvcr.io` first, with `$oauthtoken` as the username and your NGC API key as the password.

Build the Super VL image for `linux/amd64` and push it into Unity Catalog:

```bash
docker build --platform linux/amd64 -t nemo-automodel-vl:26.08.00 .
databricks air images push -s nemo-automodel-vl:26.08.00 \
  --catalog main --schema default --artifact nemo-automodel-vl:26.08.00
```

`--artifact` names the destination so it matches `environment.unity_catalog_image` below. The container is large (several GB), so the build and push take a while.

This creates `main.default.nemo-automodel-vl:26.08.00`. To use a different catalog or schema, change `--catalog`/`--schema` above and update `environment.unity_catalog_image` in `workload.yaml` to match.

### Output location setup

Check `env_variables.ADAPTER_OUTPUT_ROOT` in `workload.yaml`. The recipe saves the final adapter under this path by default:

```text
/Volumes/main/default/air_examples/lora-nemotron-3.5-super-vl-120b-a12b-automodel-multi-gpu
```

The volume must already exist and be writable. To use another location, update `ADAPTER_OUTPUT_ROOT` to a directory in another existing writable volume.

## Run

```bash
databricks air run -f workload.yaml
```

## Results

AutoModel logs training loss, learning rate, gradient norm, and throughput every five steps to the AIR MLflow run, and validation loss at the final step. The final adapter and config are uploaded to `final_adapter` in the run and copied to `ADAPTER_OUTPUT_ROOT/<MLFLOW_RUN_ID>/final_adapter` on the configured Unity Catalog volume; checkpoints stay on the node's local disk.

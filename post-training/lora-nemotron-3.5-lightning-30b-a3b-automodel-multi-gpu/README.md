# LoRA fine-tuning of Nemotron 3.5 Lightning with NeMo AutoModel (multi-GPU)

This recipe uses NeMo AutoModel to fine-tune `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` on HellaSwag with LoRA. The model combines Mamba and attention layers in a mixture-of-experts architecture.

Training runs for 50 steps on one 8x H100 node. It uses FSDP2, eight-way expert parallelism, and rank-8 LoRA. The local batch size is 32 per GPU for a global batch of 256 (no gradient accumulation). At the final step, it runs validation and saves a checkpoint. It logs training metrics only; HellaSwag multiple-choice accuracy requires a separate evaluation.

Environment configurations: the workload runs NVIDIA's `nemo-automodel:26.08.00` container image as a custom [AIR container image](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/cli/docker-images). This container provides the GMM/DeepEP backend used for training.

Workload execution takes about 9 minutes. For the first run, image upload may take additional time.

## Before you run

Enable the necessary previews in your workspace (a workspace admin does this) by following the custom Docker image guide: [AWS](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/cli/docker-images) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/ai-runtime/cli/docker-images).

### Custom image setup

Install Docker and start it on your machine. Choose a Unity Catalog schema where you have permission to push images; this example uses `main.default`. See the guide above for the required Artifact Registry privileges.

The image `nvcr.io/nvidia/nemo-automodel:26.08.00` is public — you do not need an NGC account, API key, or Docker login to pull it. If you switch to an NGC image that requires authentication, run `docker login nvcr.io` first, with `$oauthtoken` as the username and your NGC API key as the password.

Push the NVIDIA NeMo AutoModel image into Unity Catalog:

```bash
databricks air images push -s nvcr.io/nvidia/nemo-automodel:26.08.00 \
  --pull --catalog main --schema default --artifact nemo-automodel:26.08.00
```

The CLI pulls a `linux/amd64` image, so an arm64 image (for example one pulled on Apple Silicon) is never pushed; `--pull` refreshes a cached image, and `--artifact` names the destination so it matches `environment.unity_catalog_image` below. The container is large (several GB), so this takes a while.

This command creates `main.default.nemo-automodel:26.08.00`. To use a different catalog or schema, change `--catalog`/`--schema` above and update `environment.unity_catalog_image` in `workload.yaml` to match.

### Output location setup

Check `env_variables.ADAPTER_OUTPUT_ROOT` in `workload.yaml`. The recipe saves the final adapter under this path by default:

```text
/Volumes/main/default/air_examples/lora-nemotron-3.5-lightning-30b-a3b-automodel-multi-gpu
```

The volume must already exist and be writable. To use another location, update `ADAPTER_OUTPUT_ROOT` to a directory in another existing writable volume.

## Run

```bash
databricks air run -f workload.yaml
```

## Results

AutoModel logs training loss, learning rate, gradient norm, and throughput every five steps to the AIR MLflow run, and validation loss at the final step. The final adapter and tokenizer files are uploaded to `final_adapter` in the run and copied to `ADAPTER_OUTPUT_ROOT/<MLFLOW_RUN_ID>/final_adapter` on the configured Unity Catalog volume; checkpoints stay on the node's local disk.

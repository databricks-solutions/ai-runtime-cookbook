# LoRA fine-tuning of Nemotron 3.5 Super VL with NeMo AutoModel (multi-GPU)

This recipe uses NeMo AutoModel to fine-tune `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16` on the CORD-v2 receipt-image dataset with LoRA. The model is a vision-language mixture of experts. Its vision and audio towers stay frozen, and multi-token prediction (MTP) is disabled.

Training runs for 50 steps on one 8x H100 node. It uses FSDP2, eight-way expert parallelism, rank-64 LoRA, and the GMM/DeepEP backend. The local batch size is 1 per GPU, for a global batch of 8 (no gradient accumulation). At the final step, it runs validation and saves a checkpoint. The recipe logs training metrics only. Measuring CORD-v2 field-extraction quality requires a separate evaluation.

Environment configurations: NVIDIA's stock `nemo-automodel:26.08.00` image does not yet support Super VL. To add it, the recipe builds a custom [AIR container image](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/cli/docker-images) from the bundled `Dockerfile`, which layers Super VL support on top of that image.

Workload execution takes about 12 minutes. For the first run, the one-time image build and upload may take additional time.

## Before you run

Complete these steps from this recipe's directory.

1. Configure and authenticate the CLI using the [cookbook setup instructions](../../README.md#run-a-recipe). Ask a workspace admin to enable the custom-image previews: [AWS](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/cli/docker-images) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/ai-runtime/cli/docker-images).

2. Install [Docker Engine on Linux](https://docs.docker.com/engine/install/) or [Docker Desktop on macOS](https://docs.docker.com/desktop/setup/install/mac-install/) and start it. Keep Docker running through the build and push. On macOS, choose the installer for your chip. AIR requires `linux/amd64`, including on Apple silicon.

3. Choose an existing Unity Catalog schema where you can push images; this example uses `main.default`. See the custom-image guide above for privileges. Set `ADAPTER_OUTPUT_ROOT` in `workload.yaml` to a directory in an existing writable volume. Its default is:

   ```text
   /Volumes/main/default/air_examples/lora-nemotron-3.5-super-vl-120b-a12b-automodel-multi-gpu
   ```

4. Check the CLI and Docker engine before building:

   ```bash
   databricks air images push --help
   docker info
   ```

   `docker info` should show a `Server` section without a connection error. If it cannot connect, start Docker and wait for the engine to be ready.

5. Build and push the image. Allow disk space and time for the several-GB image:

   ```bash
   docker build --platform linux/amd64 -t nemo-automodel-vl:26.08.00 .
   databricks air images push -s nemo-automodel-vl:26.08.00 \
     --catalog main --schema default --artifact nemo-automodel-vl:26.08.00
   ```

   The destination is `main.default.nemo-automodel-vl:26.08.00`. If you change the catalog or schema, update `environment.unity_catalog_image` in `workload.yaml` to match.

The base image and AutoModel source are public; no NGC login is required.

## Run

```bash
databricks air run -f workload.yaml --watch
```

## Results

AutoModel logs training loss, learning rate, gradient norm, and throughput every five steps to the AIR MLflow run. It logs validation loss at the final step. The final adapter and config are uploaded to `final_adapter` in the run, and copied to `ADAPTER_OUTPUT_ROOT/<MLFLOW_RUN_ID>/final_adapter` on the configured Unity Catalog volume. Checkpoints stay on the node's local disk.

# Chest X-ray DICOM batch inference with TorchXRayVision

This recipe reads chest X-rays from a Unity Catalog volume with `UCVolumeDataset` and writes 18 pathology scores per image. It runs [TorchXRayVision's `densenet121-res224-all` model][model] on one A10 GPU, using a custom Docker image.

You can adapt this recipe for research and evaluation. Validate it for your scanner, protocol, site, and patient population. Keep the uncalibrated scores out of clinical decisions such as diagnosis or triage.

## Before you run

### Build and push the image

Use Docker and Databricks CLI 1.20.0 or later. Enable the Databricks Artifact Registry preview in your workspace and choose a schema where you can push and read images. See [Custom Docker images on AI Runtime][docker].

The image includes the model weights and decoders for common compressed DICOM formats. For Azure, add `--build-arg BASE_IMAGE=databricksruntime/air:dcs-base-azure-runtime` to `docker build`.

From the recipe directory, set `AIR_PROFILE` to your CLI profile and run the build and push commands.

```bash
AIR_PROFILE=your-profile
docker build --platform linux/amd64 -t dicom-xray-inference:8 .
databricks air images push --profile "$AIR_PROFILE" \
  --source dicom-xray-inference:8 \
  --catalog main --schema default --artifact dicom-xray-inference:8
```

If you push outside `main.default`, update `environment.unity_catalog_image` to match in [workload.yaml](workload.yaml).

### Prepare the input

Set these paths in [workload.yaml](workload.yaml). Both default to subdirectories of the existing `main.default.air_examples` volume.

| Setting | Required access |
| --- | --- |
| `INPUT_PATH` | Read access to a directory containing the input DICOMs. |
| `OUTPUT_ROOT` | Write access to a directory for predictions. |

For your own images, upload single-frame grayscale DICOMs (`.dcm` or `.DCM`) from digital radiography (`DX`) or computed radiography (`CR`).

De-identify the tags and any identifying text burned into the pixels. The output omits patient names and IDs. Treat the retained UIDs and device metadata as sensitive.

To try synthetic images, use Python with NumPy and pydicom 3 installed. Generate eight files in a new local directory and upload them to the default `INPUT_PATH`.

```bash
NUM_SYNTHETIC_IMAGES=8 INPUT_PATH=./dicom-sample python assets/prepare_synthetic.py
databricks fs cp --recursive ./dicom-sample \
  dbfs:/Volumes/main/default/air_examples/dicom-batch-inference-torchxrayvision/input \
  --profile "$AIR_PROFILE"
```

## Run

```bash
databricks air run -f workload.yaml --watch --profile "$AIR_PROFILE"
```

For a smoke test, set `MAX_IMAGES` to `"8"` in `workload.yaml`. A positive limit uses a single loader process. The default `"0"` loads the full directory in parallel.

## Results

Each run writes `OUTPUT_ROOT/<MLFLOW_RUN_ID>/predictions.parquet`. Each row contains 18 `score_*` columns, DICOM UIDs, device metadata, original dimensions, and the model weights identifier. Join predictions to source images on `sop_instance_uid`.

The AIR MLflow run records `images`, `inference_wall_time_s`, and `inference_images_per_sec`. If you set `INFERENCE_REPEATS` above one, the file and inference metrics reflect the final pass.

## Throughput with cold and warm caches

The 932.690 images/s result uses a warm local cache. The initial Ray Data result is a first pass over the input files. Both pipelines used the same model, one A10, and 10,000 synthetic 512 × 512 DICOMs.

| Pipeline and pass | Work included in the timer | Images/s |
| --- | --- | ---: |
| Ray Data, first pass | File reads, CPU preprocessing, model loading, and inference. Parquet writing follows the timer. | 73.033 |
| `UCVolumeDataset`, cold warmup | Model loading, local caching, GPU preprocessing, and inference. Predictions are discarded. | 108.356 |
| `UCVolumeDataset`, warm pass | Cached file reads, GPU preprocessing, inference, and Parquet writing. | **932.690** |

With the default `WARMUP_PASS: "false"`, the run makes one cold pass and writes its predictions. Expect throughput closer to the cold result.

Set `WARMUP_PASS: "true"` to fill the cache before writing predictions. MLflow then records the cold rate as `warmup_images_per_sec` and the warm rate as `inference_images_per_sec`.

To generate 10,000 benchmark files, use `NUM_SYNTHETIC_IMAGES=10000` with the [sample generator](assets/prepare_synthetic.py).

<details>
<summary>View the throughput chart and GPU utilization</summary>

![Throughput on one A10 in images per second. Ray Data baseline 73.033, cold volume pass 108.356, warm volume pass 932.690.](assets/benchmark-throughput.svg)

Warm-pass GPU utilization was 99% at the median and 100% at the 95th percentile. It averaged 63.09% including idle samples. Peak GPU memory use was 6.37 GB.

</details>

### How the pipeline keeps the GPU busy

- `UCVolumeDataset` caches files locally. Twelve loader workers decode images and prefetch batches (`PREFETCH_FACTOR=32`). See [volume data loading][loader] and the [Databricks Engineering Blog][blog].
- The GPU applies linear windowing, normalization, and resizing. Mixed-size images follow the same path in groups of matching shapes. The CPU handles non-linear lookup tables.
- The recipe configures two CUDA streams for mixed-precision inference on batches of 256. A background writer saves predictions while the GPU processes the next batch.

To repeat the comparison, track throughput in MLflow after each change. For the image build, push, and workload validation steps, use the [Databricks AI Runtime skill][skill].

### Alternatives considered

This recipe feeds one GPU directly from a volume. To distribute inference across Ray actors, see [Ray Data batch inference on AI Runtime][ray].

[model]: https://github.com/mlmed/torchxrayvision/tree/1.5.7#models-demo-notebook
[docker]: https://docs.databricks.com/aws/en/machine-learning/ai-runtime/connecting/cli/docker-images
[loader]: https://docs.databricks.com/aws/en/machine-learning/ai-runtime/develop-workloads/dataloading#load-unstructured-data-with-ucvolumedataset
[ray]: https://docs.databricks.com/aws/en/machine-learning/ai-runtime/examples/gpu-batch-inference
[skill]: https://github.com/databricks/databricks-agent-skills/blob/main/skills/databricks-ai-runtime/SKILL.md
[blog]: https://www.databricks.com/blog/fast-fault-tolerant-pytorch-training-ai-runtime

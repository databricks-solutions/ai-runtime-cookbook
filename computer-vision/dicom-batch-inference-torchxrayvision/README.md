# Chest X-ray DICOM batch inference with TorchXRayVision

This recipe reads chest X-rays from a Unity Catalog volume with `UCVolumeDataset` and writes 18 pathology scores per image. On one A10 GPU, a custom Docker image runs [TorchXRayVision’s `densenet121-res224-all` model][model].

You can adapt this recipe for research and evaluation. Validate it for your scanner, protocol, site, and patient population. Keep the uncalibrated scores out of clinical decisions such as diagnosis or triage.

## Before you run

### Build and push the image

Use Docker and Databricks CLI 1.20.0 or later. Enable the Databricks Artifact Registry preview in your workspace and choose a schema where you can push and read images. See [Custom Docker images on AI Runtime][docker].

From the recipe directory, set `AIR_PROFILE` to your CLI profile. The image includes decoders for common DICOM compression formats and the model weights. For Azure, add `--build-arg BASE_IMAGE=databricksruntime/air:dcs-base-azure-runtime` to `docker build`.

```bash
AIR_PROFILE=your-profile
docker build --platform linux/amd64 -t dicom-xray-inference:8 .
databricks air images push --profile "$AIR_PROFILE" \
  --source dicom-xray-inference:8 \
  --catalog main --schema default --artifact dicom-xray-inference:8
```

If you push outside `main.default`, update `environment.unity_catalog_image` to match in [workload.yaml](workload.yaml).

### Prepare the input

In `workload.yaml`, set `INPUT_PATH` to a directory containing only input DICOMs and `OUTPUT_ROOT` to a writable directory. Both default to subdirectories of the existing `main.default.air_examples` volume.

**For your own images**, upload single-frame grayscale DICOMs (`.dcm` or `.DCM`) from digital radiography (`DX`) or computed radiography (`CR`). De-identify the tags and any identifying text burned into the pixels. The output omits patient names and IDs. Treat the retained UIDs and device metadata as sensitive.

**For synthetic images**, use Python with NumPy and pydicom 3 installed. Generate eight files in a new local directory and upload them to the default `INPUT_PATH`.

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

For a smoke test on a larger directory, set `MAX_IMAGES` to `"8"` in `workload.yaml`. A positive limit uses a single loader process. The default (`"0"`) loads the full directory in parallel.

## Results

- **Predictions** are in `OUTPUT_ROOT/<MLFLOW_RUN_ID>/predictions.parquet`. Join each row to its source image on `sop_instance_uid`. The `score_*` columns sit alongside study and series UIDs, modality, device metadata, original dimensions, and the model weights identifier.
- **MLflow** records `images`, `inference_wall_time_s`, and `inference_images_per_sec`. If you set `INFERENCE_REPEATS` above one, the file and inference metrics reflect the final pass.

## Improving DICOM throughput from 73 to 932 images/s

Both pipelines used the same model, one A10, and 10,000 synthetic 512 × 512 DICOMs. To generate a dataset this size, use `NUM_SYNTHETIC_IMAGES=10000` with the [sample generator](assets/prepare_synthetic.py).

| Pipeline and pass | What the pass did | Images/s |
| --- | --- | ---: |
| Ray Data baseline | Read small files and preprocessed images on the CPU. The GPU waited for batches. | 73.033 |
| `UCVolumeDataset`, cold pass | Loaded the model, cached the files from the volume, and ran inference without writing predictions. | 108.356 |
| `UCVolumeDataset`, warm pass | Reused the cached files, ran inference, and wrote predictions to Parquet. | **932.690** |

With `WARMUP_PASS` set to `"true"`, MLflow records the cold rate as `warmup_images_per_sec` and the warm rate as `inference_images_per_sec`. By default, the first pass writes predictions.

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

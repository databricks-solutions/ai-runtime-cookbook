# Custom Docker image for AI Runtime

This recipe adds [River](https://riverml.xyz/) 0.22.0 to a [Databricks AI Runtime base image](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/connecting/cli/docker-images) and runs a small online classifier on synthetic data. It demonstrates how to package a Python dependency in a custom image. AIR allocates one A10 GPU, although River trains this example on CPU; no dataset or model download is needed.

Adapted from the [Docker AIR support sample](https://github.com/RamVegiraju/docker-air-support).

## Before you run

Use an AWS workspace with the **AI Runtime Beta Features** and **Databricks Artifact Registry** previews enabled as described in the [custom Docker image guide](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/connecting/cli/docker-images). Install the [Databricks CLI](https://docs.databricks.com/aws/en/dev-tools/cli/install) version 1.20.0 or newer, configure an [OAuth profile](https://docs.databricks.com/aws/en/dev-tools/cli/authentication), and start Docker with [Buildx](https://github.com/docker/buildx#installing). Choose an existing Unity Catalog catalog and schema where you can push and read images. The [Artifact Registry setup guide](https://docs.databricks.com/aws/en/artifact-registry/get-started) explains the required `USE CATALOG`, `USE SCHEMA`, `CREATE VOLUME`, and `READ VOLUME` privileges. The image push creates an Artifact Registry object; it does not create a catalog or schema.

From this recipe directory, build the Linux x86-64 image and test its packaged entry point locally:

```bash
docker buildx build --platform linux/amd64 --provenance=false --load -t air-river-demo:v1 .
docker run --rm --platform linux/amd64 air-river-demo:v1 python /app/train.py
```

The build uses `--provenance=false` to produce a single-platform image manifest accepted by the registry. `--load` makes the result available to `docker run` and `databricks air images push`. On Apple Silicon, Docker runs the `linux/amd64` image through emulation for the local check. If PyPI is unreachable, set `UV_INDEX_URL` to a reachable package index and add `--secret id=uv_index_url,env=UV_INDEX_URL` to the build command; BuildKit keeps the index URL out of the image.

Set `PROFILE` to the CLI profile for your chosen workspace. Replace `your_catalog` and `your_schema` with the existing Unity Catalog location, then push the image:

```bash
PROFILE=your_profile
CATALOG=your_catalog
SCHEMA=your_schema
databricks air images push --profile "$PROFILE" \
  --source air-river-demo:v1 \
  --catalog "$CATALOG" --schema "$SCHEMA" \
  --artifact air_river_demo:v1
```

Edit `environment.unity_catalog_image` in `workload.yaml` to match `<catalog>.<schema>.air_river_demo:v1`. Use a new tag in the build, push, and workload configuration when you change the image. The push can take several minutes. For Azure, change the Dockerfile base to `databricksruntime/air:dcs-base-azure-runtime` and follow the [Azure custom Docker image guide](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/ai-runtime/cli/docker-images).

## Run

From this directory, validate the configuration against the selected workspace, then submit the workload and stream its logs:

```bash
databricks air run -f workload.yaml --dry-run --profile "$PROFILE"
databricks air run -f workload.yaml --watch --profile "$PROFILE"
```

The log prints a JSON record with the River version and accuracy on 100 held-out synthetic examples after training on 400 examples. This recipe writes no persistent model artifact. See the [AI Runtime CLI guide](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/cli/) for run management and the [Artifact Registry troubleshooting guide](https://docs.databricks.com/aws/en/artifact-registry/troubleshooting) if the image push fails.

#!/usr/bin/env python3
"""Offline image understanding with Qwen3-VL-8B-Instruct, Ray Data, and vLLM.

The workload starts a Ray head on a single 8x H100 node. The Ray Data LLM API
launches one vLLM replica per GPU and sends each replica pet photos from the
public Oxford-IIIT Pet test split (3,669 images, 37 breeds). For every image,
the vision-language model returns a JSON object with a one-sentence caption,
the species, and the breed. Structured outputs constrain the answer to the
schema, so every response parses and the breed is always one of the 37 labels.

Because the dataset has ground-truth labels, the script reports zero-shot
species and breed accuracy along with throughput, and writes one row per image
to a Unity Catalog volume as Parquet. To use your own images, replace
`load_images()`, the prompt, and the schema; for unlabeled images, also remove
the label and accuracy columns from `postprocess()` and the metrics in `main()`.

The model and dataset are public and do not require a Hugging Face token.
OUTPUT_PATH must be on an existing writable Unity Catalog volume; each run
writes to its own subdirectory, named after the run's MLflow run ID.
"""

import base64
import json
import os
import time

import pyarrow.parquet as pq
import ray
from huggingface_hub import hf_hub_download
from ray.data.llm import build_processor, vLLMEngineProcessorConfig

MODEL_SOURCE = "Qwen/Qwen3-VL-8B-Instruct"
DATASET_REPO = "timm/oxford-iiit-pet"
DATASET_FILE = "data/test-00000-of-00001.parquet"
# 0 processes the full test split. Set a small value for a quick smoke test.
NUM_IMAGES = int(os.environ.get("NUM_IMAGES", "0"))
# Unity Catalog volume path where results land as Parquet. Set this in workload.yaml.
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/Volumes/main/default/air_examples/batch-inference-qwen3-vl-8b-vllm")
# AI Runtime creates an MLflow run per attempt, so each attempt gets its own output directory.
RUN_OUTPUT_PATH = os.path.join(OUTPUT_PATH, os.environ.get("MLFLOW_RUN_ID") or "local")

# Oxford-IIIT Pet class names, in label-index order.
BREEDS = [
    "Abyssinian", "American Bulldog", "American Pit Bull Terrier", "Basset Hound", "Beagle",
    "Bengal", "Birman", "Bombay", "Boxer", "British Shorthair", "Chihuahua", "Egyptian Mau",
    "English Cocker Spaniel", "English Setter", "German Shorthaired", "Great Pyrenees",
    "Havanese", "Japanese Chin", "Keeshond", "Leonberger", "Maine Coon", "Miniature Pinscher",
    "Newfoundland", "Persian", "Pomeranian", "Pug", "Ragdoll", "Russian Blue", "Saint Bernard",
    "Samoyed", "Scottish Terrier", "Shiba Inu", "Siamese", "Sphynx", "Staffordshire Bull Terrier",
    "Wheaten Terrier", "Yorkshire Terrier",
]
CAT_BREEDS = {
    "Abyssinian", "Bengal", "Birman", "Bombay", "British Shorthair", "Egyptian Mau",
    "Maine Coon", "Persian", "Ragdoll", "Russian Blue", "Siamese", "Sphynx",
}

# vLLM structured outputs force the response to match this JSON schema. The
# caption comes first so the model describes the image before it classifies it.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "caption": {"type": "string", "maxLength": 300},
        "species": {"type": "string", "enum": ["cat", "dog"]},
        "breed": {"type": "string", "enum": BREEDS},
    },
    "required": ["caption", "species", "breed"],
    "additionalProperties": False,
}
# The schema already restricts the breed, but the model picks better when the prompt
# lists the allowed names as well.
PROMPT = (
    "Describe this photo in one sentence, then identify the animal's species and breed. "
    f"The breed must be one of: {', '.join(BREEDS)}. "
    "Respond with JSON that has the keys caption, species, and breed."
)


def load_images():
    """Build a Ray Dataset of images and labels from the public Oxford-IIIT Pet test split."""
    local_file = hf_hub_download(repo_id=DATASET_REPO, filename=DATASET_FILE, repo_type="dataset")
    table = pq.read_table(local_file, columns=["image", "label", "image_id"])
    if NUM_IMAGES > 0:
        table = table.slice(0, NUM_IMAGES)
    items = []
    for row in table.to_pylist():
        breed = BREEDS[row["label"]]
        items.append(
            {
                "image_id": row["image_id"],
                "image_bytes": row["image"]["bytes"],
                "true_breed": breed,
                "true_species": "cat" if breed in CAT_BREEDS else "dog",
            }
        )
    return ray.data.from_items(items)


def to_data_url(image_bytes):
    """Encode raw image bytes as a data URL that vLLM's chat parser accepts."""
    mime = "image/png" if image_bytes[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"


def preprocess(row):
    """Map each image to an OpenAI-style chat request with an image part."""
    return dict(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": to_data_url(row["image_bytes"])}},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
        sampling_params=dict(
            temperature=0.0,
            max_tokens=256,
            structured_outputs={"json": RESPONSE_SCHEMA},
        ),
    )


def postprocess(row):
    """Parse the JSON answer and keep the columns to persist (the image bytes are dropped)."""
    try:
        answer = json.loads(row["generated_text"])
    except (TypeError, json.JSONDecodeError):
        answer = {}
    return dict(
        image_id=row["image_id"],
        true_species=row["true_species"],
        true_breed=row["true_breed"],
        pred_species=answer.get("species"),
        pred_breed=answer.get("breed"),
        caption=answer.get("caption"),
        parsed=int(bool(answer)),
        species_correct=int(answer.get("species") == row["true_species"]),
        breed_correct=int(answer.get("breed") == row["true_breed"]),
        # Completion time, used to measure steady-state throughput separately from startup.
        completed_at=time.time(),
    )


def check_output_volume(path):
    """Fail fast, before loading any model, if OUTPUT_PATH's Unity Catalog volume is missing."""
    parts = path.rstrip("/").split("/")
    if path.startswith("/Volumes/") and len(parts) >= 5 and not os.path.isdir("/".join(parts[:5])):
        raise SystemExit(
            f"OUTPUT_PATH is on volume {'.'.join(parts[2:5])}, which does not exist or is not "
            "accessible. Create it (CREATE VOLUME) or set OUTPUT_PATH in workload.yaml to a "
            "writable Unity Catalog volume path."
        )


def log_to_mlflow(params, metrics, tables):
    """Attach params, metrics, and result tables to the MLflow run AI Runtime creates for the workload."""
    run_id = os.environ.get("MLFLOW_RUN_ID")
    if not run_id:
        return
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    for key, value in params.items():
        client.log_param(run_id, key, value)
    for key, value in metrics.items():
        client.log_metric(run_id, key, value)
    for artifact_file, table in tables.items():
        client.log_table(run_id, data=table, artifact_file=artifact_file)


def main():
    check_output_volume(OUTPUT_PATH)
    # Creating the run directory now also fails fast if the volume isn't writable.
    os.makedirs(RUN_OUTPUT_PATH, exist_ok=True)
    ray.init(address="auto")
    # Derive replicas from the live cluster so the example scales when nodes are added.
    total_gpus = int(ray.cluster_resources().get("GPU", 0))
    print(f"Ray cluster ready: {total_gpus} GPU(s)", flush=True)

    ds = load_images()

    # vLLM engine config. concurrency = number of replicas Ray Data runs in parallel;
    # one per GPU in the cluster here. engine_kwargs are passed through to the vLLM engine.
    config = vLLMEngineProcessorConfig(
        model_source=MODEL_SOURCE,
        engine_kwargs={
            "max_model_len": 8192,
            "tensor_parallel_size": 1,
            "limit_mm_per_prompt": {"image": 1},
            # Qwen3-VL turns each 32x32-pixel patch into one token. Downscale images above
            # about 6 MP (6,144 tokens) so one image plus the prompt fits in max_model_len.
            # The image processor applies max_pixels only together with min_pixels, so both
            # are set; min_pixels keeps the model's default of 64 tokens.
            "mm_processor_kwargs": {"min_pixels": 64 * 32 * 32, "max_pixels": 6144 * 32 * 32},
        },
        # Decodes the base64 images and builds the vision inputs for each request.
        prepare_multimodal_stage={"enabled": True},
        concurrency=total_gpus,
        batch_size=32,
    )
    processor = build_processor(config, preprocess=preprocess, postprocess=postprocess)

    # materialize once so the write, the metrics, and the sample print don't re-run inference.
    # Ray logs "Failed to convert column '__data' into pyarrow array" for the chat requests,
    # which mix nested types Arrow can't represent; it pickles them instead, which is harmless.
    start = time.time()
    out = processor(ds).materialize()
    elapsed = time.time() - start
    # Overwrite so a re-run into the same directory replaces it instead of appending to it.
    out.drop_columns(["completed_at"]).write_parquet(RUN_OUTPUT_PATH, mode="overwrite")

    results = out.to_pandas()
    num_rows = len(results)
    metrics = {
        "images": num_rows,
        # Includes model download and engine startup.
        "wall_time_s": round(elapsed, 1),
        "json_parse_rate": round(results["parsed"].mean(), 4),
        "species_accuracy": round(results["species_correct"].mean(), 4),
        "breed_accuracy": round(results["breed_correct"].mean(), 4),
    }
    # Steady-state throughput: images finish in batches, so count the images that finished after the
    # first batch (more than a second after the first image) over the time since the first image.
    done = results["completed_at"].sort_values().to_numpy()
    rest = done[done > done[0] + 1.0]
    if len(rest):
        metrics["inference_images_per_sec"] = round(len(rest) / (rest[-1] - done[0]), 1)
    print(f"Wrote {num_rows} rows to {RUN_OUTPUT_PATH}", flush=True)
    print("METRICS " + json.dumps(metrics), flush=True)

    errors = results[results["breed_correct"] == 0]
    confusions = errors.groupby(["true_breed", "pred_breed"]).size().reset_index(name="count")
    log_to_mlflow(
        params={"model": MODEL_SOURCE, "dataset": DATASET_REPO, "num_gpus": total_gpus, "output_path": RUN_OUTPUT_PATH},
        metrics=metrics,
        tables={
            "sample_predictions.json": results[["image_id", "true_breed", "pred_breed", "caption"]].head(50),
            "breed_confusions.json": confusions.sort_values("count", ascending=False).head(20),
        },
    )
    for row in results.head(3).itertuples():
        print(f"[{row.image_id}] true={row.true_breed} pred={row.pred_breed} | {row.caption}", flush=True)

    ray.shutdown()


if __name__ == "__main__":
    main()

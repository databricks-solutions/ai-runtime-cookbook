# LoRA Longformer for long-document classification

PEFT LoRA fine-tuning of `allenai/longformer-base-4096` for sequence
classification on a single GPU. Longformer supports sequences up to 4096 tokens,
which suits long documents; the example uses the public IMDB sentiment dataset.
LoRA adapts the local + global attention projections and trains the classification
head, so only a small fraction of parameters are updated. Metrics go to MLflow and
the adapter + tokenizer are saved to a Unity Catalog volume.

## Before you run

The adapter is written to the shared `/Volumes/main/default/air_examples/` volume,
which must already exist and be writable. Point `output_root` in the `parameters:`
block of `workload.yaml` at another writable volume if you cannot use that default.

## Run

```bash
databricks air run -f workload.yaml
```

Configure the model, dataset, LoRA rank/alpha, and training hyperparameters in the
`parameters:` block. For a quick end-to-end smoke test, set `max_steps` to a small
positive value. To run on your own data, publish it to the Hugging Face Hub with
`train` and `test` splits and `text` / `label` columns, then set `dataset_name` to
that Hub dataset ID and `num_labels` accordingly.

The LoRA adapter (with the classification head) and tokenizer are written to
`<output_root>/<MLFLOW_RUN_ID>` — by default under
`/Volumes/main/default/air_examples/lora-longformer-base-4096-classification/`. The
run id is shown on the AIR attempt's MLflow page.

## Using the adapter

Load the base model and apply the adapter for inference:

```python
from peft import PeftModel
from transformers import AutoTokenizer, LongformerForSequenceClassification

# num_labels must match what you trained with (2 for the default IMDB run).
adapter_path = "/Volumes/main/default/air_examples/lora-longformer-base-4096-classification/<MLFLOW_RUN_ID>"
base = LongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096", num_labels=2)
model = PeftModel.from_pretrained(base, adapter_path)
tokenizer = AutoTokenizer.from_pretrained(adapter_path)
```

Call `model.merge_and_unload()` to fold the adapter into the base weights for a
standalone model to register or deploy.

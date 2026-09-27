# LoRA Longformer for long-document classification

PEFT LoRA fine-tuning of `allenai/longformer-base-4096` for sequence
classification on a single GPU. Longformer's 4096-token attention window suits
long documents; the example uses the public IMDB sentiment dataset. LoRA adapts
the local + global attention projections and trains the classification head, so
only a small fraction of parameters are updated. Metrics go to MLflow and the
adapter + tokenizer are saved to a Unity Catalog volume.

## Prerequisites

A writable Unity Catalog volume for the adapter output (it must already exist).
Set `catalog` / `schema` / `volume` in the `parameters:` block of `workload.yaml`.

## Run

```bash
databricks air run -f workload.yaml
```

Configure the model, dataset, LoRA rank/alpha, and training hyperparameters in the
`parameters:` block. For a quick end-to-end smoke test, set `max_steps` to a small
positive value. To run on your own data, point `dataset_name` at a dataset with
`text` and `label` columns and set `num_labels` accordingly.

The LoRA adapter (with the classification head) and tokenizer are written to
`/Volumes/<catalog>/<schema>/<volume>/longformer-imdb-lora`.

## Using the adapter

Load the base model and apply the adapter for inference:

```python
from peft import PeftModel
from transformers import AutoTokenizer, LongformerForSequenceClassification

base = LongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096", num_labels=2)
model = PeftModel.from_pretrained(base, "/Volumes/<catalog>/<schema>/<volume>/longformer-imdb-lora")
tokenizer = AutoTokenizer.from_pretrained("/Volumes/<catalog>/<schema>/<volume>/longformer-imdb-lora")
```

Call `model.merge_and_unload()` to fold the adapter into the base weights for a
standalone model to register or deploy.

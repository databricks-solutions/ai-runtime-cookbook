# Longformer fine-tuning for long-document classification with LoRA

PEFT LoRA fine-tuning of `allenai/longformer-base-4096` for sequence
classification on a single GPU. Longformer supports sequences up to 4096 tokens,
which suits long documents; the example uses the public IMDB sentiment dataset.
LoRA adapts the local + global attention projections and trains the classification
head, so only a small fraction of parameters are updated. Metrics go to MLflow and
the adapter + tokenizer are saved to a Unity Catalog volume.

## Before you run

The adapter is written to the shared `/Volumes/main/default/air_examples/` volume,
which must already exist and be writable. To write elsewhere, pass another existing
writable volume directory:

```bash
databricks air run -f workload.yaml --override parameters.output_root=/Volumes/<catalog>/<schema>/<volume>/<dir> --watch
```

## Run

```bash
databricks air run -f workload.yaml --watch
```

Configure the model, dataset, LoRA rank/alpha, and training hyperparameters in the
`parameters:` block.

The LoRA adapter (with the classification head) and tokenizer are written to
`<output_root>/<MLFLOW_RUN_ID>` — by default under
`/Volumes/main/default/air_examples/finetune-longformer-base-4096-lora/`. The
run id is shown on the AIR attempt's MLflow page.

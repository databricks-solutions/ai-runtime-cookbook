# SFT Qwen3-8B for invoice extraction (DeepSpeed ZeRO-3)

Full-weight SFT of `Qwen/Qwen3-8B` for invoice/receipt entity extraction — TRL
`SFTTrainer` + DeepSpeed ZeRO-3 on one 8xH100 node. Hyperparameters live in the
`parameters:` block of `workload.yaml`.

## Before you run

1. A writable Unity Catalog volume for the staged data, checkpoints, and final
   model (it must already exist). The default is the shared
   `/Volumes/main/default/air_examples/sft-qwen3-8b-invoice-deepspeed`; override
   `output_root` in `workload.yaml` to use another writable volume.
2. **Stage the data once:** run `prep_data.py` from any environment with write
   access to that volume (for example a Databricks cluster web terminal), passing
   the same path. The AIR GPU worker has no Spark, so this pre-stages the public
   `Winuim/invoice-sft-dataset-v2` dataset as Hugging Face datasets under
   `<output_root>/data`, which `train.py` reads. Only answers that match the JSON
   schema in the prompt are kept (about 4.9k of 7k rows; the rest carry raw OCR
   amounts, non-ISO dates, or other item layouts), and each example is stored as a
   `prompt` / `completion` pair so the loss covers only the assistant's JSON.

   ```bash
   pip install datasets==5.1.0 huggingface_hub==1.33.0 fsspec==2024.9.0
   python prep_data.py --output-root /Volumes/main/default/air_examples/sft-qwen3-8b-invoice-deepspeed
   ```

## Run

```bash
databricks air run -f workload.yaml
```

Set `max_steps` to a small positive value for a quick end-to-end smoke test. Sweep by
editing `learning_rate` / `num_epochs` and running once per combination; each run writes
to its own `<output_root>/<MLFLOW_RUN_ID>` directory. The final model lands at
`<output_root>/<MLFLOW_RUN_ID>/final` (the run id is shown on the AIR attempt's MLflow page).

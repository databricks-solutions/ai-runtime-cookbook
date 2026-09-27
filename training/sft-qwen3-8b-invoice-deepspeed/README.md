# SFT Qwen3-8B for invoice extraction (DeepSpeed ZeRO-3)

Full-weight SFT of `Qwen/Qwen3-8B` for invoice/receipt entity extraction — TRL
`SFTTrainer` + DeepSpeed ZeRO-3 on one 8xH100 node. Hyperparameters live in the
`parameters:` block of `workload.yaml`.

## Prerequisites

1. A writable Unity Catalog volume for the data and one for checkpoints (both must
   already exist). Set them in the `parameters:` block of `workload.yaml`.
2. **Stage the data once:** run `prep_data.py` on Databricks (Serverless CPU) with its
   `catalog`/`schema`/`volume` widgets matching `workload.yaml`. The AIR GPU worker has
   no Spark, so this pre-stages the public `Winuim/invoice-sft-dataset-v2` dataset as
   Hugging Face datasets on the volume that `train.py` reads.

## Run

```bash
databricks air run -f workload.yaml
```

Set `max_steps` to a small positive value for a quick end-to-end smoke test. Sweep by
editing `learning_rate` / `num_epochs` and running once per combination. The checkpoint
lands at `/Volumes/<catalog>/<schema>/<volume_model>/invoice-ft-final-<run_tag>`.

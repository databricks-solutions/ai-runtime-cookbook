# DLRM recommendation

Train a [DLRM](https://arxiv.org/abs/1906.00091) click-through-rate model with TorchRec on about 26M rows from one day of [Criteo 1TB click logs](https://huggingface.co/datasets/criteo/CriteoClickLogs) across 4 A10 nodes with one GPU each, in about 6 minutes. TorchRec's planner shards the 26 categorical embedding tables across the GPUs.

How the data is used:

- Training uses the first `train_files` Parquet files (0.8M to 1.5M rows each) from `train_day`. Each GPU reads its own files, and all GPUs stop at the step count of the GPU with the fewest rows.
- Validation and test use separate files from `eval_day`, the following day, so evaluation is on later traffic than training.
- Integer features become `log(1 + x)`, with missing and negative values set to 0. Categorical features are hashed into `hash_size` rows per table, and missing values get their own row.

How the model is evaluated:

- Validation AUROC and log loss are logged every `eval_every_steps` steps. Test metrics are computed once at the end.
- A baseline that always predicts the training click rate gives the log loss to beat. Its AUROC is 0.5.

## Before you run

Review the CC-BY-NC-SA-4.0 license on the dataset page. Files download automatically without a token. To write outputs elsewhere, set `parameters.output_root` to an existing Unity Catalog volume you can write to.

The default tables (`hash_size: 1000000`, 64 dimensions) total about 7 GB, so sharding here spreads the work rather than being required. At 4M or more rows per table they no longer fit on one A10. To run on one 8xH100 node instead, set `compute.accelerator_type` to `GPU_8xH100` and `compute.num_accelerators` to `8`. This shape is not validated.

## Run

```bash
databricks air run -f workload.yaml
```

## Results

The AIR MLflow run contains training log loss, throughput, validation and test metrics, the baseline, and `sharding_plan.txt`, which shows the GPU that holds each table. These files are written to `<output_root>/<MLFLOW_RUN_ID>/`:

- `checkpoint/`: sharded model weights saved with `torch.distributed.checkpoint`, about 7 GB
- `model_config.json`: table sizes, layer sizes, feature columns, and world size
- `metrics.json`: final metrics

To reload the weights, build the same sharded model as `train.py` with the settings in `model_config.json`, then load the checkpoint into its state dict:

```python
import torch.distributed.checkpoint as dcp

state_dict = model.state_dict()  # model: DistributedModelParallel built as in train.py
dcp.load(state_dict, checkpoint_id="<output_root>/<MLFLOW_RUN_ID>/checkpoint")
model.load_state_dict(state_dict)
```

# Click-through-rate model on a Unity Catalog table

This recipe trains a click-through-rate model in PyTorch on two nodes with one A10 GPU each. UCDataset (Beta) reads the training data directly from a Unity Catalog table and splits the rows across the GPUs.

## Before you run

Create the sample table by running `create_table.sql` once in the SQL editor or a notebook. It writes 1 million synthetic rows to `main.default.air_examples_ctr_features`. To use another catalog or schema, change the table name in `create_table.sql` and `parameters.table` in `workload.yaml`.

Creating the table requires the `USE CATALOG`, `USE SCHEMA`, and `CREATE TABLE` privileges. Reading it requires `USE CATALOG`, `USE SCHEMA`, and `SELECT`. In the Beta, UCDataset reads Delta tables only. It can't read views, materialized views, streaming tables, or tables with row filters or column masks.

## Run

```bash
databricks air run -f workload.yaml --watch
```

To train on two nodes with eight H100 GPUs each, set `accelerator_type: GPU_8xH100` and `num_accelerators: 16`. `train.py` and the command don't change.

## Results

The AIR MLflow run logs these metrics for each epoch:

- `train_loss` and `test_loss`: mean log loss on the training and test splits.
- `train_rows`: training rows read across all GPUs. It equals the number of rows that match the training filter, because each row is read once per epoch.

The run also logs `test_loss_baseline`, the log loss of always predicting the test split's click rate. The recipe doesn't save a model or checkpoints.

With `databricks-uc-dataset` 0.0.1b1, `test_loss` drops below the 0.60 baseline after the first epoch and reaches about 0.55 after the third. Values vary slightly between runs. The workload takes about two minutes.

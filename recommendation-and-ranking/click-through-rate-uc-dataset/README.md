# Click-through-rate model on a Unity Catalog table

This recipe trains a click-through-rate model in PyTorch on two nodes with one A10 GPU each. UCDataset (Beta) reads the training data directly from a Unity Catalog table and splits the rows across the GPUs.

## Before you run

Create the sample table by running `create_table.sql` once. From the CLI, run the following command. It's experimental and can change between CLI releases (tested with Databricks CLI v1.26.0):

```bash
databricks experimental aitools tools query --file create_table.sql
```

The command needs a SQL warehouse that you can use, and it picks one unless you add `--warehouse <warehouse-id>`. You can also run the file in the SQL editor or a notebook. It writes 1 million synthetic rows to `main.default.air_examples_ctr_features` and leaves an existing table with that name unchanged. To use another catalog or schema, change the table name in `create_table.sql` and `parameters.table` in `workload.yaml`.

Creating the table requires the `USE CATALOG`, `USE SCHEMA`, and `CREATE TABLE` privileges. The workload reads the table as you, which requires `USE CATALOG`, `USE SCHEMA`, and `SELECT`. In the Beta, UCDataset reads Delta tables only. It can't read views, materialized views, streaming tables, or tables with row filters or column masks.

Before you submit, check the table in `parameters.table` with the same profile that you submit with, so that a wrong name or a missing privilege fails in seconds instead of after the GPUs start. If you renamed the table, use that name:

```bash
databricks experimental aitools tools query 'SELECT split, COUNT(*) FROM main.default.air_examples_ctr_features WHERE label IS NOT NULL GROUP BY split'
```

The query also fails if the `split` or `label` column is missing, and it shows how many rows each split has. It doesn't detect the table types that UCDataset can't read, listed above.

UCDataset can't read tables in workspaces whose serverless network policy limits outbound access, even when this check succeeds. In those workspaces, the run fails with `PERMISSION_DENIED: Serverless python and scala notebooks are prohibited from retrieving temporary storage credentials from Unity Catalog in network restricted workspaces`. An account admin controls this through the workspace's serverless network policy.

## Run

```bash
databricks air run -f workload.yaml --watch
```

## Results

The AIR MLflow run logs these metrics for each epoch:

- `train_loss` and `test_loss`: mean log loss on the training and test splits.
- `train_rows`: training rows read across all GPUs. It equals the number of rows that match the training filter, because each row is read once per epoch.

The run also logs `test_loss_baseline`, the log loss of always predicting the test split's click rate. The recipe doesn't save a model or checkpoints.

With `databricks-uc-dataset` 0.0.1b1, `test_loss` drops below the 0.60 baseline after the first epoch and reaches about 0.55 after the third. Values vary slightly between runs. The workload takes about two minutes.

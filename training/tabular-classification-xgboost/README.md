# Tabular classification with XGBoost

Train a GPU XGBoost binary classifier for credit card fraud detection on one A10 GPU. The example uses the public [credit card fraud dataset](https://www.openml.org/d/1597) (284,807 European card transactions over two days in September 2013, 492 of them fraudulent), with 28 anonymized PCA features plus the transaction amount.

## Before you run

Review the dataset's [OpenML page](https://www.openml.org/d/1597) for usage and citation information. It downloads automatically without credentials.

Transactions are split in time order: the earliest 60% train the model, the next 20% select the boosting round (early stopping on PR-AUC) and the alert threshold (maximum F1), and the latest 20% are scored once as the test set. Fraud is about 0.17% of transactions, so the recipe reports precision-recall metrics rather than accuracy.

## Run

```bash
databricks air run -f workload.yaml
```

## Results

The AIR MLflow run contains per-round train/validation PR-AUC, test average precision and ROC AUC, test precision/recall/F1 at the validation-selected threshold, and the trained model. `decision_threshold.json` records the threshold to apply to the model's fraud probabilities.

To adapt it, point `data_url` at a Parquet file with a `Time` ordering column, a binary `Class` label, and numeric feature columns.

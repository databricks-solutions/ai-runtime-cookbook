# GPU learning to rank with XGBoost

Train an XGBoost LambdaMART ranker on [LETOR MQ2008](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/letor-4-0/), using 46 query-document features and three relevance grades on one A10 GPU. The recipe uses the official Fold 1 partitions and up to 300 boosting rounds, selecting trees by validation NDCG@10 with early stopping.

Documents are grouped by query, and query IDs do not overlap across training, validation, and test partitions. Final test NDCG@10 averages queries equally, uses exponential relevance gains, and assigns zero to queries with no relevant documents. This compact dataset demonstrates the GPU ranking API; it is not a GPU speedup benchmark.

## Before you run

Review the dataset's linked usage terms. The archive downloads automatically from the public mirror used by XGBoost examples; no account or token is required.

## Run

```bash
databricks air run -f workload.yaml
```

## Results

The AIR MLflow run contains training/validation NDCG curves, final test NDCG@10, a seeded random-order baseline, gain-based feature importance, and a packaged XGBoost model with an input signature. Test predictions, relevance grades, and query IDs are logged in `test_predictions.csv`.

The model accepts 46 numeric feature columns in LETOR order and returns a relevance score per row. Group documents by query and sort scores descending to produce rankings; scores are not probabilities or comparable across unrelated queries.

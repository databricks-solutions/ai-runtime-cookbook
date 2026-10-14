# Two-tower recommendation

Train a PyTorch two-tower movie recommender on [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) for 15 epochs on one A10 GPU. User and item ID embeddings pass through separate small neural networks; Bayesian personalized ranking (BPR) learns to score a positive item above a uniformly sampled unobserved item.

Ratings of at least four are positive interactions. Global timestamp cutoffs divide all ratings into approximately 80% training, 10% validation, and 10% test; equal timestamps stay together. User and item vocabularies come only from training positives, and negative sampling excludes every item a user rated in training, including low ratings.

Validation Recall@20 selects the checkpoint. Final test Recall@20 and NDCG@20 average users equally and rank the entire training item vocabulary, excluding items that user already rated before the evaluation window. Cold-start users/items are excluded from this warm-start evaluation; positive-interaction coverage and evaluated user counts make that restriction visible. A popularity baseline uses the same candidates and training-only counts. The model stays fixed after the training cutoff; validation interactions only update the test exclusion list.

## Before you run

Review the MovieLens usage license on the linked dataset page, including its redistribution and commercial-use restrictions. The archive downloads automatically without a token. Set `parameters.output_root` to another existing writable Unity Catalog volume directory if needed.

## Run

```bash
databricks air run -f workload.yaml
```

## Results

The AIR MLflow run contains training loss, validation retrieval curves, final test metrics, a packaged pair-scoring model, and copies of the artifacts in `<output_root>/<MLFLOW_RUN_ID>/`: `best_model.pt`, `model_config.json`, `id_mappings.json`, `embeddings.npz`, and `metrics.json`.

Rows of `user_embeddings` and `item_embeddings` in `embeddings.npz` align with the original IDs in `id_mappings.json`. Dot products produce retrieval scores; exclude previously seen items before selecting the top K. The MLflow model accepts an int64 array of shape `(batch, 2)` containing **encoded user and item indices**, and returns one score per pair. Unknown IDs require a separate cold-start strategy.

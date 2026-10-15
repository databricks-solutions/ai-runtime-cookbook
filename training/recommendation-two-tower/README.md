# Two-tower recommendation

Train a two-tower movie recommender in PyTorch on [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) for 15 epochs on one A10 GPU. One tower embeds users and the other embeds movies. The model learns to score movies a user liked above random movies they haven't rated, using Bayesian personalized ranking (BPR) loss.

How the data is used:

- A rating of 4 or higher counts as a positive interaction.
- Ratings are split by time into about 80% training, 10% validation, and 10% test. Ratings with the same timestamp stay together.
- User and movie vocabularies come from training data only.

How the model is evaluated:

- Validation Recall@20 picks the best epoch.
- Test Recall@20 and NDCG@20 rank all training movies for each user, skipping movies the user rated before the test period. Each user counts equally.
- Users and movies not seen in training are left out. You can check the logged coverage and evaluated user counts.
- A popularity baseline uses the same candidates for comparison.

## Before you run

Review the MovieLens license on the dataset page. It restricts redistribution and commercial use. The archive downloads automatically without a token. To write outputs elsewhere, set `parameters.output_root` to an existing Unity Catalog volume you can write to.

## Run

```bash
databricks air run -f workload.yaml
```

## Results

The AIR MLflow run contains training loss, validation metrics, test metrics, and a packaged model that scores user-movie pairs. These files are written to `<output_root>/<MLFLOW_RUN_ID>/` and logged to the run:

- `best_model.pt`: model weights
- `model_config.json`: model dimensions
- `id_mappings.json`: original user and movie IDs
- `embeddings.npz`: user and movie embeddings
- `metrics.json`: final metrics

To use the outputs:

- Rows in `embeddings.npz` line up with the IDs in `id_mappings.json`.
- To recommend, take the dot product of a user embedding with all movie embeddings, remove movies the user has seen, and keep the top K.
- The MLflow model takes an int64 array of shape `(batch, 2)` with **encoded** user and movie indices, not raw IDs. It returns one score per pair.
- New users or movies need a separate cold-start approach.

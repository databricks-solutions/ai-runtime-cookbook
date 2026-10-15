# Learning to rank with XGBoost

Train an XGBoost ranking model (LambdaMART) for search results on one A10 GPU. Given a query and its candidate documents, the model scores each document so the most relevant ones come first.

The example uses [LETOR MQ2008](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/letor-4-0/), a web search dataset. Each query-document pair has 46 relevance features, such as BM25 and PageRank, and a relevance grade from 0 to 2.

- The recipe uses the official Fold 1 split. No query appears in more than one of training, validation, and test.
- Training runs up to 300 boosting rounds. Early stopping on validation NDCG@10 picks the number of trees.
- Test NDCG@10 is averaged over queries. Queries with no relevant documents score zero.

## Before you run

Review the dataset's usage terms on the linked page. The archive downloads automatically from a public mirror and needs no token.

## Run

```bash
databricks air run -f workload.yaml
```

## Results

The AIR MLflow run contains:

- Training and validation NDCG curves
- Test NDCG@10 and a random-order baseline
- Feature importance
- The XGBoost model with an input signature
- `test_predictions.csv` with scores, relevance grades, and query IDs

The model takes the 46 features in LETOR order and returns one score per document. To rank, group documents by query and sort by score, highest first. Scores are not probabilities and can't be compared across queries.

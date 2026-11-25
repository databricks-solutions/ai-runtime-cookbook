# Fine-tune an embedding model and a reranker for retrieval

This recipe fine-tunes [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) with Sentence Transformers 6 on the FiQA financial question-answering benchmark from [BEIR](https://huggingface.co/BeIR/fiqa) and evaluates the result on held-out test queries with nDCG@10 and Recall@100, on one A10 GPU. It also shows how to train and evaluate a [MiniLM cross-encoder](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L12-v2) that reranks the embedder's candidates. The cookbook's [text-embeddings-qwen3-search](../text-embeddings-qwen3-search/) recipe covers inference-only embeddings; this recipe covers the fine-tuning side.

The embedder is a bi-encoder: it encodes each question and each document separately, so the whole corpus can be encoded once and searched with vector similarity. The reranker is a cross-encoder: it reads a question and one candidate document together, which is slower but can be more accurate, so it only rescores the top candidates.

## Before you run

The recipe uses only public models and datasets, so it does not need a Hugging Face token. Check licenses before you reuse the results:

- EmbeddingGemma 2's model card lists Apache 2.0. Its license link points to Google's Gemma 4 license terms, so read those terms before you reuse the model.
- The reranker is Apache 2.0.
- The BEIR FiQA dataset and its relevance labels are CC BY-SA 4.0. This license requires attribution and share-alike terms. It is not a permissive license.

The run writes to `/Volumes/main/default/air_examples/finetune-embedding-reranker-sentence-transformers`. That volume must exist and be writable. To write to another volume, override `output_root`:

```bash
databricks air run -f workload.yaml --override parameters.output_root=/Volumes/<catalog>/<schema>/<volume>/<dir> --watch
```

## Run

From this recipe directory:

```bash
databricks air run -f workload.yaml --watch
```

The validated run took about 43 minutes in the container on one A10. Each stage is logged to the job output, so you can see progress before the run finishes.

## Results

Each run saves its files in `<output_root>/<MLFLOW_RUN_ID>/`:

| Path | What it contains |
| --- | --- |
| `embedder/` | The fine-tuned Sentence Transformers embedder. |
| `reranker/` | The fine-tuned cross-encoder reranker. |
| `metrics.json` | Dataset counts, stage timings, and dev and test metrics with standard errors for each system. It also includes the dev sweep over rerank depths and paired standard errors for the fine-tuning and reranking comparisons. |
| `parameters.json` | The settings used for the run. |

Results from the validated run (one A10, 57,638 documents, 648 test queries). Standard errors are over test queries. Test scores use the reranker depth that dev selected (20):

| System | Test nDCG@10 | Test Recall@100 |
| --- | ---: | ---: |
| Base embedder | 0.313 ± 0.013 | 0.646 ± 0.015 |
| Fine-tuned embedder | 0.423 ± 0.014 | 0.748 ± 0.014 |
| Fine-tuned embedder + fine-tuned reranker (top 20) | 0.388 ± 0.014 | 0.748 ± 0.014 |

Fine-tuning the embedder improves test nDCG@10 by about 11 points, about eleven times the paired standard error of the difference over the same test queries (0.010). In this run, the reranker did not beat the fine-tuned bi-encoder: it lowered test nDCG@10 from 0.423 to 0.388, a drop of about 3.4 points against a paired standard error of 0.009. On dev, reranking also lowered nDCG@10 at every depth tried, from 0.433 for the embedder alone to 0.389 at depth 20, 0.345 at depth 50, and 0.312 at depth 100, and dev selected the least aggressive depth. Whether reranking helps depends on that dev-selected depth: if reranking does not improve dev nDCG@10, do not add it to the pipeline. This run trained the reranker for one epoch on one random negative per positive, starting from an MS MARCO checkpoint; treat it as a starting point for your own tuning, not a demonstrated improvement.

The model revisions used were `google/embeddinggemma-2` at `f9972ac22e696a105c482202646979e33e4b5a1e`, `cross-encoder/ms-marco-MiniLM-L12-v2` at `7b0235231ca2674cb8ca8f022859a6eba2b1c968`, `BeIR/fiqa` at `979c07a7cb5ccc6ca009792241fa1250b98055dd`, and `BeIR/fiqa-qrels` at `252958f2d646e22cab6d0c72dd3f0d5de6d0655a`. Recording these revisions is optional provenance and is not needed to run the recipe.

The AIR MLflow run stores the parameters, the metrics, and the two JSON files. Test metrics are named `test_<system>_ndcg_at_10` and `test_<system>_recall_at_100`, with matching `_se` keys for the standard errors and `test_fine_tuned_minus_base_*` and `test_reranked_minus_fine_tuned_*` keys for the paired standard errors of the differences. The reranker depth (`selected_rerank_depth`) is chosen on the dev split, then reported once on the test split. Recall@100 for the reranked system is the same as the first-stage embedder when the reranker rescores the top 100 candidates. Rescoring fewer candidates keeps the rest in their original order, so Recall@100 is the same across depths.

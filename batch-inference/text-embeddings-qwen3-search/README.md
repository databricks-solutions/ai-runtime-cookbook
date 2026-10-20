# Batch text embeddings for technical-document search

Embed 2,000 sampled LangChain-related technical documents from [FreshStack](https://huggingface.co/datasets/freshstack/corpus-oct-2024) with [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) and Sentence Transformers 5 on one A10 GPU. The recipe saves reusable vectors and retrieves the five closest documents for each of three configurable natural-language queries, including document text and source URLs.

FreshStack was introduced in 2025 and uses an October 2024 corpus of chunked GitHub documentation and code. The recipe samples documents with a fixed seed, embeds up to 512 tokens per document or query in batches of 16, and performs exact cosine-similarity search. Qwen3's query instruction is applied only to queries. This is a search demonstration on a sampled corpus, not a FreshStack benchmark evaluation; it does not measure retrieval accuracy or generate answers.

## Before you run

Set `parameters.output_root` to an existing writable Unity Catalog volume directory if you cannot use the shared default. The dataset and model download automatically without a Hugging Face token. The model is Apache 2.0; FreshStack is CC BY-SA 4.0, and its [dataset card](https://huggingface.co/datasets/freshstack/corpus-oct-2024#license) also identifies licensing considerations for the original repositories. Source URLs are retained in the outputs for attribution.

## Run

```bash
databricks air run -f workload.yaml
```

Change `parameters.queries` to search for your own questions. Adjust `max_documents`, `batch_size`, `max_seq_length`, or `top_k` in `workload.yaml` to change the workload size and search depth. Loading the LangChain dataset downloads its full corpus (about 71 MB compressed) before sampling; increasing the sample or token limit increases inference time and memory use. Other FreshStack topic configurations use the same schema.

## Results

Artifacts are written under `<output_root>/<MLFLOW_RUN_ID>/`:

- `documents.parquet`: document IDs, original text, source URLs, and normalized float32 embeddings. Text is preserved in full even when the model's token limit truncates it during encoding.
- `search_results.parquet`: each query's ranked matches, cosine-similarity scores, document IDs, text, and source URLs.
- `parameters.json`: the configuration used to create the vectors and run the searches.
- `metrics.json`: document and query counts, embedding dimension, encoding duration, and documents per second. Timing covers document encoding only, including its first batch; it excludes downloads, model loading, query encoding, search, and output writes.

The AIR MLflow run also records the metrics, configuration, and search results. Document vectors remain in the configured volume. To search them again, encode new queries with the same model, token limit, query instruction, and normalization, then compare against the saved vectors.

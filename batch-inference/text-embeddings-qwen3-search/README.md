# Search technical documents with text embeddings

Find documents related to questions such as “How do I add chat history to a chatbot?” This recipe uses [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) and Sentence Transformers 5 on one A10 GPU to search 2,000 documents and return five matches per question.

The documents come from the LangChain collection in [FreshStack](https://huggingface.co/datasets/freshstack/corpus-oct-2024), a 2025 dataset built from GitHub documentation and code collected in October 2024. The model turns each document and question into an *embedding*: a list of numbers that represents its meaning. Similar embeddings help identify relevant documents.

## Before you run

Check `parameters.output_root` in [workload.yaml](workload.yaml). You need write access to the existing Unity Catalog volume; change the path if you cannot use the shared default.

The model and data download automatically without a Hugging Face token. The model uses Apache 2.0; FreshStack uses CC BY-SA 4.0 and includes documents subject to their [source repositories' licenses](https://huggingface.co/datasets/freshstack/corpus-oct-2024#license).

## Run

From this recipe directory:

```bash
databricks air run -f workload.yaml
```

Edit `parameters.queries` in `workload.yaml` to ask your own questions. Use `max_documents` to change the sample size and `top_k` to change the number of matches.

By default, the model reads up to 512 tokens from each document or question. Increase `max_seq_length` for longer text; this uses more GPU memory. The saved files always retain the full document text.

## Results

Each run saves its files in a separate directory: `<output_root>/<MLFLOW_RUN_ID>/`.

| File | What it contains |
| --- | --- |
| `search_results.parquet` | Ranked matches for each question, with scores, document text, IDs, and source URLs. |
| `documents.parquet` | All sampled documents and their embeddings, ready to reuse for search. |
| `parameters.json` | The settings used for this run. |
| `metrics.json` | Document counts, encoding time, and documents per second. |

The job logs print the ranked matches. The AIR MLflow run also stores the search results, settings, and metrics. Encoding time measures document processing only, including the first batch.

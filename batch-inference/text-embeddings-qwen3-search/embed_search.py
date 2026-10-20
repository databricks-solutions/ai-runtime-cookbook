#!/usr/bin/env python3
"""Batch-embed FreshStack technical documents and search them with Qwen3."""

import json
import os
import time
from pathlib import Path

import mlflow
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util


def main():
    with open(os.environ["HYPERPARAMETERS_PATH"]) as handle:
        p = yaml.safe_load(handle)
    for name in ("max_documents", "batch_size", "max_seq_length", "top_k"):
        if not isinstance(p[name], int) or p[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    queries = p["queries"]
    if not isinstance(queries, list) or not queries or any(
        not isinstance(query, str) or not query.strip() for query in queries
    ):
        raise ValueError("queries must be a nonempty list of nonempty strings")
    if not torch.cuda.is_available():
        raise RuntimeError("This recipe requires a CUDA GPU")

    # Sample independently of the search queries; no relevance labels enter inference.
    corpus = load_dataset(
        p["dataset_name"], p["dataset_config"], split=p["dataset_split"]
    )
    corpus = corpus.filter(lambda row: bool(row["text"].strip()))
    corpus = corpus.shuffle(seed=p["sample_seed"]).select(
        range(min(p["max_documents"], len(corpus)))
    )
    if not len(corpus):
        raise ValueError("The corpus contains no nonempty documents")
    texts = list(corpus["text"])
    document_ids = list(corpus["_id"])
    if len(set(document_ids)) != len(document_ids):
        raise ValueError("Document IDs must be unique")
    urls = [metadata["url"] for metadata in corpus["metadata"]]

    run_id = os.environ["MLFLOW_RUN_ID"]
    output_dir = Path(p["output_root"]) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_params({key: value for key, value in p.items() if key != "queries"})
        model = SentenceTransformer(
            p["model_name"],
            device="cuda",
            model_kwargs={"dtype": torch.bfloat16, "attn_implementation": "sdpa"},
            tokenizer_kwargs={"padding_side": "left"},
        )
        model.max_seq_length = p["max_seq_length"]
        encode_options = {
            "batch_size": p["batch_size"],
            "normalize_embeddings": True,
            "convert_to_numpy": True,
            "show_progress_bar": True,
        }
        torch.cuda.synchronize()
        started = time.perf_counter()
        embeddings = model.encode_document(texts, **encode_options).astype(np.float32)
        torch.cuda.synchronize()
        embedding_seconds = time.perf_counter() - started

        # Qwen3 expects an instruction on queries, but not on corpus documents.
        query_embeddings = model.encode_query(
            queries, prompt_name="query", **encode_options
        ).astype(np.float32)
        if not np.isfinite(embeddings).all() or not np.isfinite(query_embeddings).all():
            raise ValueError("The model returned non-finite embeddings")

        # Exact search over normalized vectors. Chunking bounds score-matrix memory.
        hits = util.semantic_search(
            query_embeddings,
            embeddings,
            top_k=min(p["top_k"], len(corpus)),
            score_function=util.dot_score,
        )
        results = []
        for query, query_hits in zip(queries, hits):
            for rank, hit in enumerate(query_hits, start=1):
                index = hit["corpus_id"]
                results.append({
                    "query": query,
                    "rank": rank,
                    "document_id": document_ids[index],
                    "score": hit["score"],
                    "url": urls[index],
                    "text": texts[index],
                })

        documents = pa.table({
            "document_id": document_ids,
            "text": texts,
            "url": urls,
            "embedding": pa.array(embeddings.tolist(), type=pa.list_(pa.float32())),
        })
        pq.write_table(documents, output_dir / "documents.parquet")
        pq.write_table(pa.Table.from_pylist(results), output_dir / "search_results.parquet")
        metrics = {
            "documents": len(corpus),
            "queries": len(queries),
            "embedding_dimensions": embeddings.shape[1],
            "embedding_seconds": embedding_seconds,
            "documents_per_second": len(corpus) / embedding_seconds,
        }
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        # Keep the exact settings alongside the reusable vectors.
        (output_dir / "parameters.json").write_text(json.dumps(p, indent=2) + "\n")
        mlflow.log_metrics(metrics)
        for name in ("metrics.json", "parameters.json", "search_results.parquet"):
            mlflow.log_artifact(str(output_dir / name), artifact_path="search")
        for result in results:
            print(
                f"{result['query']}\n  {result['rank']}. {result['score']:.4f} "
                f"{result['document_id']} {result['url']}",
                flush=True,
            )
        print(f"Metrics: {metrics}\nSearch artifacts: {output_dir}", flush=True)


if __name__ == "__main__":
    main()

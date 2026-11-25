"""Load the saved embedder and reranker in a fresh process, then score one query and one document.

workload.yaml runs this after finetune_retrieval.py, so it shares no state with training. The [artifact-load]
lines show that both models load from the output volume and produce finite scores.
"""

import os
from pathlib import Path

import numpy as np
import yaml
from sentence_transformers import CrossEncoder, SentenceTransformer

with open(os.environ["HYPERPARAMETERS_PATH"]) as handle:
    p = yaml.safe_load(handle)

output_dir = Path(p["output_root"]) / os.environ["MLFLOW_RUN_ID"]
embedder = SentenceTransformer(
    str(output_dir / "embedder"),
    config_kwargs={"vision_config": None, "audio_config": None},
)
reranker = CrossEncoder(str(output_dir / "reranker"))

question = "How long does it take to open a savings account?"
document = "Most banks let you open a savings account online in about ten minutes."
query_embedding, document_embedding = embedder.encode(
    ["task: search result | query: " + question, "title: none | text: " + document]
)
cosine = float(embedder.similarity(query_embedding, document_embedding))
score = float(reranker.predict([(question, document)])[0])
if not (np.isfinite(cosine) and np.isfinite(score)):
    raise RuntimeError(f"non-finite scores: cosine={cosine}, rerank={score}")
print(f"[artifact-load] embedder cosine={cosine:.4f}", flush=True)
print(f"[artifact-load] reranker score={score:.4f}", flush=True)

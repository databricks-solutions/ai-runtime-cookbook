"""Load the saved GraphSAGE checkpoint in a fresh process and predict one test paper's topic.

train.py writes the checkpoint and model_config.json before this script runs. The README snippet makes the same calls.
"""

import json
import os
from pathlib import Path

import torch
import yaml

from train import GraphSAGE, load_data, mean_adjacency


def main() -> None:
    with open(os.environ["HYPERPARAMETERS_PATH"], encoding="utf-8") as handle:
        p = yaml.safe_load(handle)
    output_dir = Path(p["output_root"]) / os.environ["MLFLOW_RUN_ID"]
    config = json.loads((output_dir / "model_config.json").read_text())
    model = GraphSAGE(**config)
    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location="cpu", weights_only=True))
    model.eval()

    features, _, edges, splits, class_labels, paper_ids = load_data(p)
    adjacency = mean_adjacency(edges, len(features), device="cpu")
    with torch.no_grad():
        logits = model(features, adjacency)
    node = int(splits["test"][0])
    topic = class_labels[str(int(logits[node].argmax()))]
    print(f"Paper {paper_ids[node]}: predicted topic {topic}", flush=True)


if __name__ == "__main__":
    main()

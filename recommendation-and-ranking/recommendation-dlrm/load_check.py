"""Load the saved sharded DLRM checkpoint and score one validation row.

Run under torchrun with the same number of ranks as training, because the checkpoint is sharded across them.
train.py saves the checkpoint before this script runs. The README snippet makes the same calls.
"""

import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import yaml

import data
from train import build_model, to_batch


def main() -> None:
    with open(os.environ["HYPERPARAMETERS_PATH"]) as handle:
        p = yaml.safe_load(handle)
    # The CLI can serialize small numbers as 1e-08, which PyYAML reads as a string.
    for key in ("learning_rate", "eps"):
        p[key] = float(p[key])

    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()

    model = build_model(p, device, world)
    state_dict = model.state_dict()
    dcp.load(state_dict, checkpoint_id=str(Path(p["output_root"]) / os.environ["MLFLOW_RUN_ID"] / "checkpoint"))
    model.load_state_dict(state_dict)

    # Every rank scores the same first validation row, so all ranks run the same forward pass.
    file_name = data.list_day_files(p["dataset_repo"], p["eval_day"])[0]
    path = data.download(p["dataset_repo"], file_name, p["data_root"])
    dense, sparse, labels = (torch.from_numpy(array) for array in data.load_split([path], p["hash_size"]))
    batch = to_batch(dense[:1], sparse[:1], labels[:1])
    model.eval()
    with torch.no_grad():
        _, (_, logits, _) = model(batch.to(device))
    if rank == 0:
        print(f"Predicted click probability for one validation row: {torch.sigmoid(logits).item():.4f}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

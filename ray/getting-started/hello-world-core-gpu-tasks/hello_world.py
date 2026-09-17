"""Ray Core GPU task scheduling on AI Runtime.

The workload submits one @ray.remote task per available GPU and prints where Ray
scheduled each task, including its node rank, Ray GPU ID, visible CUDA device,
and GPU model. ray_bootstrap.sh starts the Ray cluster before this script runs.
"""

import os
import subprocess
import time

import ray

ray.init(address="auto")

num_nodes = int(os.environ.get("NUM_NODES", 1))
gpus_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
expected_gpus = num_nodes * gpus_per_node

for _ in range(30):
    if len(ray.nodes()) >= num_nodes and ray.cluster_resources().get("GPU", 0) >= expected_gpus:
        break
    time.sleep(2)

total_gpus = int(ray.cluster_resources().get("GPU", 0))
if total_gpus < expected_gpus:
    raise SystemExit(
        f"Expected {expected_gpus} GPU(s) but Ray only sees {total_gpus}; " "check GPU discovery on all nodes."
    )

print(f"Ray cluster ready: {len(ray.nodes())} node(s), {total_gpus} GPU(s)")
print(f"Cluster resources: {ray.cluster_resources()}\n")


@ray.remote(num_gpus=1)
def hello_from_gpu():
    node_rank = os.environ.get("NODE_RANK", "?")
    # Ray sets CUDA_VISIBLE_DEVICES to the single assigned GPU, so
    # current_device() always returns 0. Report the physical GPU via
    # nvidia-smi and the Ray GPU ID instead.
    ray_gpu_ids = ray.get_gpu_ids()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu_name = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return f"Hello from node {node_rank} | Ray GPU id {ray_gpu_ids} | CUDA_VISIBLE_DEVICES={visible} | {gpu_name}"


print(f"Launching {total_gpus} task(s), one per GPU across the cluster...")
futures = [hello_from_gpu.remote() for _ in range(total_gpus)]
results = ray.get(futures)

for r in results:
    print(r)

ray.shutdown()

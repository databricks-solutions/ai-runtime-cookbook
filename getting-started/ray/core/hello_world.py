"""Ray Core remote-task example on AI Runtime.

Dispatches one @ray.remote task per GPU across the cluster. Each task prints
which node and physical GPU it was assigned to, confirming tasks reached every
node. Run after ray_bootstrap.sh has started the cluster.
"""

import os
import subprocess

import ray

ray.init(address="auto")

total_gpus = int(ray.cluster_resources().get("GPU", 0))

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

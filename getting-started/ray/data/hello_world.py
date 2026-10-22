"""Ray Data distributed preprocessing example on AI Runtime.

Builds a Ray Dataset and runs a distributed map / map_batches / filter
pipeline across CPU actors spread over the cluster. On AI Runtime, Ray Data
runs on CPU actors (heterogeneous CPU/GPU clusters are not supported yet), so
this example deliberately keeps the transforms on CPU. The common shape is Ray
Data preprocessing feeding into a Ray Train run.
"""

import os

import ray

# Connect to the cluster started by ray_bootstrap.sh.
ray.init(address="auto")

num_nodes = int(os.environ.get("NUM_NODES", 1))
num_cpus = int(ray.cluster_resources().get("CPU", 0))
print(f"Cluster ready: {num_nodes} node(s), {num_cpus} CPU(s) available")

# A simple synthetic dataset; range() produces a distributed Ray Dataset.
ds = ray.data.range(10_000)


def add_features(row):
    """Per-row transform, runs distributed across CPU tasks."""
    n = row["id"]
    return {"id": n, "squared": n * n, "is_even": n % 2 == 0}


def scale_batch(batch):
    """Vectorized per-batch transform (numpy), more efficient than per-row."""
    batch["scaled"] = batch["squared"] * 0.001
    return batch


# Distributed pipeline: map -> filter -> map_batches, then aggregate.
ds = ds.map(add_features)
ds = ds.filter(lambda row: row["is_even"])
ds = ds.map_batches(scale_batch, batch_format="numpy")

count = ds.count()
total = ds.sum("scaled")
print(f"\nPipeline produced {count} rows (even numbers only)")
print(f"Sum of scaled feature: {total:.2f}")
print("\nSample of 5 processed rows:")
for row in ds.take(5):
    print(f"  {row}")

ray.shutdown()

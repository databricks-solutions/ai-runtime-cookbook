# Protein embeddings with BioNeMo ESM-2 and Transformer Engine

This recipe uses Ray Data to embed the 28,303 UniProt proteins in [DeepLoc 2.0](https://huggingface.co/datasets/bloyal/deeploc) with NVIDIA BioNeMo's [Transformer Engine (TE) build of ESM-2 650M](https://huggingface.co/nvidia/esm2_t33_650M_UR50D), running one model copy per GPU on an 8x H100 node. It then fits a linear probe that predicts membrane-bound vs. soluble proteins from the embeddings, and compares it with an amino-acid-composition baseline on the dataset's test split.

[BioNeMo Recipes](https://github.com/NVIDIA-BioNeMo/bionemo-recipes) checkpoints, which succeed the archived BioNeMo Framework container, need TE. The `databricks_ai_v6` environment doesn't include it, so `install_transformer_engine.sh` compiles TE's PyTorch binding at the start of each run, which takes a few minutes. Other BioNeMo Recipes checkpoints, such as `nvidia/esm2_t36_3B_UR50D`, load the same way; lower `TOKENS_PER_BATCH` in `embed_proteins.py` if they run out of memory. To embed your own proteins, replace `load_sequences()` with a function that returns a DataFrame with a `sequence` column.

## Before you run

`OUTPUT_PATH` in `workload.yaml` must be on an existing Unity Catalog volume that you can write to.

To skip the TE build on later runs, set `TE_WHEEL_CACHE` to a volume directory. The first run saves the compiled binding there, in a subdirectory keyed on the Python, PyTorch, and CUDA versions and the GPU architectures in `NVTE_CUDA_ARCHS` (default `90`, H100), so a changed environment builds a fresh wheel.

## Run

```bash
databricks air run -f workload.yaml
```

For a quick test on one GPU:

```bash
databricks air run -f workload.yaml \
  --override compute.accelerator_type=GPU_1xH100 --override compute.num_accelerators=1 \
  --override env_variables.NUM_SEQUENCES=2000
```

## Results

Each run writes one Parquet row per protein to `OUTPUT_PATH/<MLflow run ID>`, with the accession, split, labels, sequence, and a 1,280-dimension mean-pooled embedding. The workload's MLflow run records throughput, the accuracy and ROC AUC of both probes, and a `membrane_probe.json` table comparing them.

On the full dataset, the ESM-2 probe reaches about 0.88 accuracy and 0.925 ROC AUC, against 0.77 and 0.76 for the amino-acid-composition baseline (measured with model revision `118f470` and dataset revision `51a44a7`). The dataset's test split is a random 10% sample rather than one of DeepLoc's homology partitions, so close homologs of test proteins can appear in training and both scores are optimistic.

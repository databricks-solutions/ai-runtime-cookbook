# Protein embeddings with BioNeMo ESM-2 and Transformer Engine

This recipe uses Ray Data to embed the 28,303 UniProt proteins in [DeepLoc 2.0](https://huggingface.co/datasets/bloyal/deeploc) with NVIDIA BioNeMo's [Transformer Engine (TE) build of ESM-2 650M](https://huggingface.co/nvidia/esm2_t33_650M_UR50D), running one model copy per GPU on an 8x H100 node. It then fits a linear probe that predicts membrane-bound vs. soluble proteins from the embeddings, and compares it with an amino-acid-composition baseline on the dataset's test split. That split is a random 10% sample rather than one of DeepLoc's homology partitions, so close homologs of test proteins can appear in training and both scores are optimistic.

[BioNeMo Recipes](https://github.com/NVIDIA-BioNeMo/bionemo-recipes) checkpoints, which succeed the archived BioNeMo Framework container, need TE. The `databricks_ai_v6` environment doesn't include it, so `install_transformer_engine.sh` compiles TE's PyTorch binding at the start of each run, which takes a few minutes. To embed your own proteins, replace `load_sequences()` with a function that returns a DataFrame with the columns `sequence`, `split` (`train`, `validation`, or `test`), `membrane` (0 or 1), and `length` (the residue count).

## Before you run

`OUTPUT_PATH` in `workload.yaml` must be on an existing Unity Catalog volume that you can write to. To write outputs to another volume, run:

```bash
databricks air run -f workload.yaml --override env_variables.OUTPUT_PATH=/Volumes/<catalog>/<schema>/<volume>/<dir> --watch
```

## Run

```bash
databricks air run -f workload.yaml --watch
```

## Results

Each run writes one Parquet row per protein to `OUTPUT_PATH/<MLflow run ID>`, with the accession, split, labels, sequence, and a 1,280-dimension mean-pooled embedding. The workload's MLflow run records throughput, the accuracy and ROC AUC of both probes, and a `membrane_probe.json` table comparing them.

`membrane_probe.json` reports the accuracy and ROC AUC of the ESM-2 probe and the amino-acid-composition baseline, each with a standard error. Accuracy uses the binomial standard error over the test proteins. ROC AUC uses a bootstrap over proteins with 1,000 resamples. The ESM-2 minus baseline difference has its own standard error: a paired one for accuracy, because both probes score the same proteins, and a bootstrap one for ROC AUC.

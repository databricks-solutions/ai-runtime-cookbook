# Reinforcement fine-tuning with verl

Fine-tune `Qwen/Qwen2.5-0.5B-Instruct` on GSM8K with verl's Group Relative Policy Optimization (GRPO). The notebook runs five training steps on one 8xH100 node and compares responses before and after training; it demonstrates the training loop rather than a benchmark result.

## Before you run

Use the AI v6 environment with an 8xH100 accelerator. You need access to PyPI and Hugging Face, write access to the notebook's MLflow experiment, and an existing writable Unity Catalog volume.

The `volume_path` widget defaults to `/Volumes/main/default/air_examples/rft-qwen25-05b-verl`. Replace it with a folder in another existing writable volume if needed.

## Run

Import [verl_hello_world.py](verl_hello_world.py) into your Databricks workspace as a notebook. Connect it to Serverless GPU, select AI v6 and 8xH100, and run the cells in order, including dependency installation and the Python restart.

See [How to set up a reinforcement learning task](rl-task-setup.md) for background on tasks, rollouts, rewards, and evaluation.

## Results

Training metrics and validation artifacts are logged to the notebook's MLflow run. Data, sampled responses, checkpoints, a Hugging Face model export, and `validation_summary.json` are saved under `<volume_path>/verl-hello-world/<run-name>/`.

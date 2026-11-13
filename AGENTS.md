# AGENTS.md

Follow [CONTRIBUTING.md](CONTRIBUTING.md) for recipe standards and validation. Keep changes scoped to the requested recipe or convention. Do not include credentials, private workspace details, or user-specific resource names. Before changing an existing recipe, preserve unrelated working-tree changes and validate the affected configuration where practical.

Before running a recipe that declares `secrets.HF_TOKEN`, follow the [agent secret checklist](docs/agents/hugging-face-secrets.md), then use the linked preflight helper to check every required model before allocating GPUs.

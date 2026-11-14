# AGENTS.md

Follow [CONTRIBUTING.md](CONTRIBUTING.md) for recipe standards and validation. Keep changes scoped to the requested recipe or convention. Do not include credentials, private workspace details, or user-specific resource names. Before changing an existing recipe, preserve unrelated working-tree changes and validate the affected configuration where practical.

Install the `databricks-ai-runtime` agent skill before writing, adapting, or submitting a workload. It covers the `databricks air` workload YAML, submission, monitoring, and debugging of failed runs:

```bash
databricks aitools install --skills-only --skills databricks-ai-runtime
```

Add `--scope project` to install it into the current project only. Source: [databricks-agent-skills/skills/databricks-ai-runtime](https://github.com/databricks/databricks-agent-skills/tree/main/skills/databricks-ai-runtime).

Before running a recipe that declares `secrets.HF_TOKEN`, follow the [agent secret checklist](docs/agents/hugging-face-secrets.md), then use the linked preflight helper to check every required model before allocating GPUs.

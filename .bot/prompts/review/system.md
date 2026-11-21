Repo-specific review guidance for `databricks-solutions/ai-runtime-cookbook`, a
public cookbook of runnable Databricks AI Runtime recipes. Each recipe pairs a
`workload.yaml` with the training/inference source it submits. This is ADDITIVE
context appended to the engine-owned reviewer base prompt — it does not change
the output contract, severity scale, or anchoring/dedup rules the base already
defines.

Recipes are examples meant to be copied and adapted, not production services, so
weigh findings accordingly:
- Judge correctness against what the recipe itself declares and runs — the
  `workload.yaml` parameters and the scripts/notebooks it references — not
  against hypothetical production hardening. A missing production-grade feature
  (retries, monitoring, config surface) is not a finding; name it in the summary
  as follow-up work if it matters.
- Verify the change's claims against the actual files: parameters the workload
  YAML passes, imports the source uses, and paths/dataset references that must
  resolve.
- Distinguish defects introduced by the change from pre-existing issues in
  untouched example code; the latter is not a finding.
- Secrets and privacy: flag any hardcoded token/credential, internal host, or
  user-specific resource name in the diff. Recipes should reference secrets or
  workspace configuration instead (see CONTRIBUTING.md and the repo README).

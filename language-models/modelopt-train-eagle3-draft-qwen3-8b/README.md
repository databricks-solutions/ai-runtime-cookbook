# Train an EAGLE-3 draft model for Qwen3-8B and measure it with SGLang

This recipe trains a speculative-decoding draft for `Qwen/Qwen3-8B` on UltraChat chat data, then serves the target with and without each draft. The draft proposes tokens, and the target checks them in one forward pass. The benchmark reports mean accept length (tokens produced per target forward pass), output tokens/s, and how often greedy outputs match the no-draft outputs, on 500 held-out prompts. The target runs in the training process, so no separate capture server is needed.

Training uses NVIDIA Model Optimizer's EAGLE-3 implementation (`nvidia-modelopt==0.48.0`). The draft is exported in SGLang's `LlamaForCausalLMEagle3` layout and served with SGLang 0.5.14.

The draft uses the target's RoPE settings (`rope_theta` 1,000,000, no scaling, from Qwen3-8B's config). ModelOpt's EAGLE-3 defaults train with `rope_theta` 10,000 and export a YaRN scaling (factor 32) that the draft never trained with, so `train_draft.py` overrides both.

## Before you run

The default `output_root`, `/Volumes/main/default/air_examples/modelopt-train-eagle3-draft-qwen3-8b`, requires write access to the existing Unity Catalog volume. To use another writable volume path, override it when submitting:

```bash
databricks air run -f workload.yaml --override 'parameters.output_root=/Volumes/<catalog>/<schema>/<volume>/<directory>' --watch
```

## Run

```bash
databricks air run -f workload.yaml --watch
```

The recipe uses the standard v6 environment with PyTorch 2.11.0 and transformers 5.8.1 explicitly installed. The v6 validation took about 25 minutes in the container on one H100, including 463 s in the training loop. The earlier managed AI v6 validation took about 27 minutes, including 472 s in the training loop.

## Results

`train_draft.py` logs the training loss and the held-out loss before and after training to the AIR MLflow run. `benchmark.py` logs throughput, accept length, identical-output rates and first-divergence positions. The draft is written to `<output_root>/<MLflow run ID>/draft/`, the untrained head to `draft_untrained/`, and the full benchmark to `benchmark.json` in the same directory.

The target is `Qwen/Qwen3-8B` at revision `b968826d9c46dd6066d109eabc6255188de91218`, and the reference draft is `AngelSlim/Qwen3-8B_eagle3` at revision `9629dfce7a4a10564dd48d3e5485c3976095653c`.

Measured on one H100 with 500 UltraChat test prompts, greedy decoding and up to 256 new tokens (about 117,600 generated tokens per configuration). The first run used managed AI v6; the second used standard v6. Each configuration was timed once per run, so throughput differences of a few percent are within timing noise. Standard errors are across prompts.

| Environment | Draft | Output tokens/s | Mean accept length (± SE) | Speedup vs no draft | Greedy output identical to no draft | Median first divergence (tokens) |
|---|---|---|---|---|---|---|
| Managed AI v6 | None | 1,122 | 1.00 | 1.00× | — | — |
| Managed AI v6 | None, second pass (control) | 1,125 | 1.00 | 1.00× | 100% | — |
| Managed AI v6 | Untrained draft head | 786 | 1.01 (± 0.000) | 0.70× | 17.2% | 63.5 |
| Managed AI v6 | Trained here | 1,044 | 1.39 (± 0.004) | 0.93× | 17.4% | 62 |
| Managed AI v6 | `AngelSlim/Qwen3-8B_eagle3` (reference) | 1,515 | 1.94 (± 0.011) | 1.35× | 18.4% | 62 |
| Standard v6 | None | 1,121 | 1.00 | 1.00× | — | — |
| Standard v6 | None, second pass (control) | 1,125 | 1.00 | 1.00× | 100% | — |
| Standard v6 | Untrained draft head | 788 | 1.01 (± 0.000) | 0.70× | 17.2% | 63.5 |
| Standard v6 | Trained here | 1,052 | 1.39 (± 0.004) | 0.94× | 18.2% | 62 |
| Standard v6 | `AngelSlim/Qwen3-8B_eagle3` (reference) | 1,517 | 1.94 (± 0.011) | 1.35× | 18.4% | 62 |

Both runs kept 1,960 of 2,000 training examples for 2 epochs.

The first-divergence position is the first token (after retokenizing the output text) where a draft's greedy output differs from the first no-draft output. It is computed over the prompts that diverge.

What the numbers show:

- **Trained draft.** Training raised the accept length from 1.01 (untrained head) to 1.39. In this run, the trained draft is slower than no draft. The public reference draft has a higher accept length and is faster.
- **Identical outputs.** Two no-draft passes match on every prompt, so the no-draft engine is deterministic here. Each draft's output differs from the no-draft output on about 82% of prompts. The untrained head is rejected almost every step, and its outputs diverge at the same rate and median position as the trained and reference drafts. So the divergence does not come from accepted draft tokens. It is consistent with numerical differences in the verification forward pass. We did not test that mechanism directly.
- **RoPE match.** An earlier validation run of this recipe, before the RoPE override, served the trained draft with YaRN scaling and measured accept length 1.39 (± 0.004) and 0.93×. The override makes no measurable difference here.

Held-out draft loss on 96 of the first 100 test conversations (the rest were filtered or did not fill a batch) was 13.98 (± 0.02) before training in both runs, and 3.59 (± 0.06) after training on managed AI v6 versus 3.58 (± 0.06) on standard v6. This held-out loss covers one draft step. The training loss in the MLflow run sums three train-time-test steps (ModelOpt's default), so it is larger and is not comparable with the held-out loss.

Before the validation run, a smoke run with reduced settings (16 training prompts, 1 epoch, 8 evaluation prompts, 16 new tokens) checked that the pipeline runs. Its numbers are not reported.

## Data and reference

- **Training and evaluation data:** [`HuggingFaceH4/ultrachat_200k`](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k), `train_sft` for training and `test_sft` for evaluation. The dataset card lists the licence as MIT.
- **Reference draft:** [`AngelSlim/Qwen3-8B_eagle3`](https://huggingface.co/AngelSlim/Qwen3-8B_eagle3), an EAGLE-3 draft for Qwen3-8B published by Tencent's AngelSlim team. Its model card describes the AngelSlim toolkit and reports vLLM benchmarks on a single H20 GPU at batch size 1. The card does not describe the training data for this checkpoint. This recipe loads it unchanged and does not reproduce its training.

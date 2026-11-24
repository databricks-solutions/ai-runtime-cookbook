# On-policy distillation with TRL

This recipe distills math reasoning from `Qwen/Qwen2.5-1.5B-Instruct` (the teacher) into `Qwen/Qwen2.5-0.5B-Instruct` (the student) on one H100. The student writes its own solutions, and the teacher scores every token of them. The offline arm trains on fixed teacher text instead.

Each step, the student samples one solution for each of 16 GSM8K training questions with vLLM. The teacher runs a forward pass over the same tokens. The loss is the reverse KL between the student's and the teacher's next-token distributions on the student's tokens. TRL's `DistillationTrainer` does this. The recipe also trains an offline student on the teacher's fixed solutions to the same questions for the same number of steps. It reports accuracy on all 1,319 GSM8K test questions before and after each method.

## Before you run

The recipe writes outputs to `/Volumes/main/default/air_examples/trl-on-policy-distillation-qwen2.5-gsm8k`. That volume must exist and be writable. The models and dataset are public and ungated, so no Hugging Face token is needed.

## Run

```bash
databricks air run -f workload.yaml --watch
```

## Results

The run prints each stage's results. It writes `summary.json` to `<output_root>/<MLflow run ID>/` and logs the metrics and the loss curves to the AIR MLflow run.

In the validation run, the on-policy student scored 1.7 points below the untrained student (SE 1.3). That is within noise of zero. The offline student scored 3.2 points below (SE 1.4), about 2.3 standard errors below zero. The teacher scored highest. The cause of the drop is not established. The control arms rule out the save and reload path, but not the training itself.

| Model | GSM8K test accuracy (1,319 questions) | Standard error |
|---|---|---|
| Student before training | 47.4% | 1.4 points |
| Teacher, Qwen2.5-1.5B-Instruct | 70.3% | 1.3 points |
| Offline distillation | 44.2% | 1.4 points |
| On-policy distillation | 45.7% | 1.4 points |
| Control: untrained student, saved and reloaded through the trainer's save path (on-policy path) | 47.3% | 1.4 points |
| Control: untrained student, saved and reloaded through the trainer's save path (offline path) | 47.8% | 1.4 points |

The questions are the same for every model, so the differences use paired standard errors:

| Comparison | Difference | Standard error of the difference |
|---|---|---|
| On-policy minus student before | -1.7 points | 1.3 points |
| Offline minus student before | -3.2 points | 1.4 points |
| On-policy minus offline | +1.5 points | 1.3 points |
| Control (on-policy path) minus student before | -0.1 points | 0.4 points |
| Control (offline path) minus student before | +0.4 points | 0.4 points |

Every validation run at the default settings is reported, because the results differ between runs. A first default run, before the control arms were added, gave on-policy 43.9% (minus student before: -3.5 points, SE 1.3) and offline 44.0% (minus student before: -3.4 points, SE 1.4). The student before training was 47.4% in both runs. The on-policy result changed by 1.8 points between the two runs, so single-run differences of this size are not reliable. Runs 1 to 4 of the development were smoke runs or environment failures, and they are excluded from these results.

The control arms check the save and reload path. The untrained student, saved with the same trainer call and reloaded in vLLM, scores within noise of the untrained student loaded from the Hub.

Output format in the validation run. Only 0.4% of the untrained student's answers contain the `####` marker that the prompt asks for, and no distilled student's answers contain it. The scorer therefore effectively measured last-number matches. The share of answers cut off at 512 tokens was 4.2% for the untrained student, 3.7% for on-policy, and 3.6% for offline. The mean completion length ranged from 279 tokens (teacher) to 306 tokens (untrained student).

End-of-turn tokens: the first completion of each model ends with `<|im_end|>`, Qwen's end-of-turn token. This is recorded in `last_tokens` in `summary.json`. The offline SFT training example's end is printed in the offline training log.

The validation run takes about 17 minutes in the container (17 min 07 s, including dependency setup and the control evaluations), without queueing time. The on-policy training stage takes about 5 minutes of that (306 s).

The learning rate was not tuned, and the test set was not used to choose any setting.

## How the two arms differ

Both arms start from the same student, use the same 2,048 random GSM8K training questions (seed 0), and take 128 optimizer steps at batch size 16 and learning rate 1e-5.

- **On-policy:** the student generates its own solutions at temperature 1.0 during training. The teacher scores them, and the loss is the reverse KL (`beta=1.0`).
- **Offline:** the teacher writes 4 solutions per question at temperature 0.7 before training. The recipe keeps the first correct one per question, which gave 1,897 of 2,048 questions. The student then trains on that fixed text with supervised fine-tuning.

The test set is used only for reporting. Nothing is selected or tuned on it.

## Notes

- The teacher and student must share a vocabulary. `DistillationTrainer` checks the two `vocab_size` values and raises an error on a mismatch (TRL 1.18.0, `trainer/distillation_trainer.py`). The 1.5B teacher passes this check with the 0.5B student.
- `train_opd.py` and `train_offline.py` disable PyTorch's cuDNN attention backend. Both trainers stop with an error if the gradient norm becomes non-finite.
- The offline run logs to stdout, not MLflow. Its loss keys would overwrite the on-policy curves in the same run.
- Set `parameters.use_vllm` to `0` to generate with plain Transformers instead of the colocated vLLM engine. This recipe was validated only with `use_vllm: 1`.
- For a quick check, override the sizes: `databricks air run -f workload.yaml --override parameters.num_train_prompts=32 --override parameters.num_eval_questions=64`. On an earlier version of this recipe, before the control arms were added, the smoke check took about 8 minutes in the container.

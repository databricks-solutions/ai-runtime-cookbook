#!/bin/bash
# Distillation pipeline on one GPU: each stage runs in its own process so GPU memory is freed.
set -eo pipefail
cd "$CODE_SOURCE_PATH"

echo "=== 1/4 Baseline student accuracy"
python generate.py eval student_before
echo "=== 2/4 Teacher generates solutions (batch inference with vLLM)"
python generate.py teacher
echo "=== 3/4 Student fine-tunes on the teacher's correct solutions"
python train_student.py
echo "=== 4/4 Distilled student accuracy"
python generate.py eval student_after

python summarize.py

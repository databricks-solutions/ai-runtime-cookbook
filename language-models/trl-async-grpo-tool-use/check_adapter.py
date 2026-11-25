"""Load the trained LoRA adapter onto the base model in a fresh process and answer one question.

workload.yaml runs this after train.py, so it shares no state with training. It shows the adapter loads from
disk the way a reader would load it.
"""

import os
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

with open(os.environ["HYPERPARAMETERS_PATH"]) as f:
    P = yaml.safe_load(f)

ADAPTER = Path(P["output_root"]) / os.environ["MLFLOW_RUN_ID"] / "adapter"
QUESTION = (
    "Here is a SQLite database schema:\n"
    "CREATE TABLE singers (id INTEGER, name TEXT, country TEXT);\n\n"
    "Question: How many singers are from France?\n\n"
    "Call submit_answer with one SQLite SELECT query that answers the question."
)

tokenizer = AutoTokenizer.from_pretrained(P["model_name"])
base = AutoModelForCausalLM.from_pretrained(P["model_name"], dtype=torch.bfloat16).to("cuda")
model = PeftModel.from_pretrained(base, str(ADAPTER)).eval()

text = tokenizer.apply_chat_template(
    [{"role": "user", "content": QUESTION}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to("cuda")
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
answer = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
if not answer.strip():
    raise RuntimeError(f"the adapter in {ADAPTER} produced an empty reply")
print(f"[check] loaded {ADAPTER}\n[check] reply: {answer}", flush=True)

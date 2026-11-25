"""Shared prompt format, answer parsing and parameters."""

import math
import os
import re

import yaml

with open(os.environ["HYPERPARAMETERS_PATH"]) as f:
    P = yaml.safe_load(f)

SYSTEM = "Solve the math problem step by step. End with the final answer as: #### <number>"
WORK = "/tmp/distill"


def messages(question):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]


def final_number(text):
    """Final answer after '####', else the last number in the text, as a float (None if absent)."""
    match = re.search(r"####\s*\$?(-?[\d,]*\.?\d+)", text)
    found = match.group(1) if match else (re.findall(r"-?[\d,]*\.?\d+", text) or [None])[-1]
    if found is None:
        return None
    return float(found.replace(",", "").rstrip("."))


def is_correct(pred, gold):
    """Compare numerically, so '#### 72.00' matches gold 72."""
    return pred is not None and math.isclose(pred, gold, abs_tol=1e-6)

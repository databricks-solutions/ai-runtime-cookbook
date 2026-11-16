"""Shared prompt format, answer parsing and parameters."""

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
    """Final answer after '####', else the last number in the text."""
    match = re.search(r"####\s*\$?(-?[\d,]*\.?\d+)", text)
    found = match.group(1) if match else (re.findall(r"-?[\d,]*\.?\d+", text) or [None])[-1]
    if found is None:
        return None
    found = found.replace(",", "").rstrip(".")
    return found[:-2] if found.endswith(".0") else found

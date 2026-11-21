"""Multi-turn tool-use reinforcement learning with TRL's AsyncGRPOTrainer.

The model answers questions about small SQLite databases from gretelai/synthetic_text_to_sql.
Each rollout gets a fresh in-memory database. The model explores it with a read-only
run_query tool, then calls submit_answer. The environment rewards 1.0 when the submitted
query returns the same rows as the gold query, and 0.0 otherwise.

Generation runs on a vLLM server on GPU 0, started by workload.yaml. This script trains on GPU 1.
Evaluation runs the same tool loop over the server's OpenAI-compatible API, on the same held-out
test rows before and after training. The trained LoRA adapter is loaded into the server at runtime.
"""

import json
import os
import re
import sqlite3
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlflow
import yaml
from datasets import Dataset, load_dataset
from peft import LoraConfig

from trl import AsyncGRPOConfig, AsyncGRPOTrainer

with open(os.environ["HYPERPARAMETERS_PATH"]) as f:
    P = yaml.safe_load(f)

# The AIR-provided MLflow run ID. The trainer's MLflow callback resumes the run at train begin,
# which removes MLFLOW_RUN_ID from the environment, and the spawned rollout worker then re-imports
# this module without it. The child never logs, so it can take the empty default; main() fails the
# workload when the run ID is missing, so nothing is ever logged to a placeholder.
RUN_ID = os.environ.get("MLFLOW_RUN_ID", "")
OUT = Path(P["output_root"]) / RUN_ID
WORK = Path("/tmp/trl-recipe")
VLLM_URL = "http://localhost:8000"
DATASET = "gretelai/synthetic_text_to_sql"
ADAPTER_NAME = "grpo_final"

QUERY_TIMEOUT_S = 5.0
MAX_ROWS_SHOWN = 20
MAX_TOOL_OUTPUT_CHARS = 2000
MAX_CONTEXT_CHARS = 1500  # keeps prompts short enough for the 8192-token server window
MAX_GOLD_ROWS = 200

PROMPT_TEMPLATE = (
    "Here is a SQLite database schema:\n{schema}\n\n"
    "Question: {question}\n\n"
    "Use run_query to inspect the data if you need to. When you are sure of the answer, "
    "call submit_answer with one SQLite SELECT query that answers the question."
)

# Served to the model at evaluation time. Training derives the same tools from the environment's docstrings.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "run_query",
            "description": "Run a read-only SQLite SELECT query against the database and return up to 20 rows.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "A SQLite SELECT statement."}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "Submit the final SQLite SELECT query that answers the question. This ends the task.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The final SQLite SELECT statement."}},
                "required": ["query"],
            },
        },
    },
]


def schema_only(sql_context):
    """The model sees CREATE statements. The INSERT rows stay hidden, so it has to query the data."""
    statements = re.findall(r"CREATE\s+(?:TABLE|VIEW)\b[^;]*;", sql_context, flags=re.IGNORECASE | re.DOTALL)
    return "\n".join(s.strip() for s in statements)


def run_select(conn, query):
    """Run one statement with a time limit. Raises sqlite3.Error on failure or timeout."""
    deadline = time.monotonic() + QUERY_TIMEOUT_S
    conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
    try:
        return conn.execute(query).fetchall()
    finally:
        conn.set_progress_handler(None, 0)


def normalize_rows(rows):
    """Order-insensitive, type-tolerant form of a result set, so equivalent answers compare equal."""

    def norm(value):
        if isinstance(value, float):
            value = round(value, 2)
            return int(value) if value.is_integer() else value
        if isinstance(value, str):
            return value.strip().lower()
        return value

    return sorted((tuple(norm(v) for v in row) for row in rows), key=repr)


def build_split(split, n_examples, seed):
    """Keep rows whose database builds, whose gold query runs, and whose gold result is non-empty and small."""
    rows = load_dataset(DATASET, split=split).shuffle(seed=seed)
    kept = []
    for row in rows:
        # LIMIT without a total ORDER BY has no single correct answer, so skip it.
        if len(row["sql_context"]) > MAX_CONTEXT_CHARS or " LIMIT " in row["sql"].upper():
            continue
        conn = None
        try:
            conn = sqlite3.connect(":memory:")
            conn.executescript(row["sql_context"])
            gold = run_select(conn, row["sql"])
        except sqlite3.Error:
            continue
        finally:
            if conn is not None:
                conn.close()
        if not gold or len(gold) > MAX_GOLD_ROWS:
            continue
        content = PROMPT_TEMPLATE.format(schema=schema_only(row["sql_context"]), question=row["sql_prompt"])
        kept.append(
            {
                "prompt": [{"role": "user", "content": content}],
                "sql_context": row["sql_context"],
                "sql": row["sql"],
            }
        )
        if len(kept) == n_examples:
            break
    if len(kept) < n_examples:
        raise RuntimeError(f"only {len(kept)} usable rows in {split}; wanted {n_examples}")
    print(f"[data] {split}: kept {len(kept)} rows", flush=True)
    return Dataset.from_list(kept)


class SQLEnv:
    """One environment per rollout. TRL exposes its public methods as tools.

    reset() receives the row's sql_context and sql columns as keyword arguments.
    get_reward() is the only reward source, so no reward_funcs are passed to the trainer.
    """

    def reset(self, sql_context: str, sql: str, **kwargs) -> None:
        if getattr(self, "_conn", None) is not None:
            self._conn.close()
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.executescript(sql_context)
        self._gold = normalize_rows(run_select(self._conn, sql))
        # The model cannot write to the database, and the tool results cannot change the gold answer.
        self._conn.execute("PRAGMA query_only = ON")
        self.submitted_query = None

    def run_query(self, query: str) -> str:
        """Run a read-only SQLite SELECT query against the database and return up to 20 rows.

        Args:
            query: A SQLite SELECT statement.

        Returns:
            The rows as text, or an error message when the query fails.
        """
        try:
            rows = run_select(self._conn, query)
        except sqlite3.Error as error:
            return f"Error: {error}"
        text = "\n".join(str(row) for row in rows[:MAX_ROWS_SHOWN]) or "(no rows)"
        if len(rows) > MAX_ROWS_SHOWN:
            text += f"\n... {len(rows) - MAX_ROWS_SHOWN} more rows not shown"
        return text[:MAX_TOOL_OUTPUT_CHARS]

    def submit_answer(self, query: str) -> str:
        """Submit the final SQLite SELECT query that answers the question. This ends the task.

        Args:
            query: The final SQLite SELECT statement.

        Returns:
            A confirmation message.
        """
        self.submitted_query = query
        return "Answer submitted."

    def get_reward(self) -> float:
        if self.submitted_query is None:
            return 0.0
        try:
            rows = run_select(self._conn, self.submitted_query)
        except sqlite3.Error:
            return 0.0
        return float(normalize_rows(rows) == self._gold)


def dispatch(env, name, arguments):
    tools = {"run_query": env.run_query, "submit_answer": env.submit_answer}
    if name not in tools:
        return f"Error: unknown tool {name}"
    try:
        return tools[name](**arguments)
    except TypeError:
        return "Error: invalid arguments"


def chat(model, messages, temperature):
    body = {
        "model": model,
        "messages": messages,
        "tools": TOOL_SCHEMAS,
        "temperature": temperature,
        "max_tokens": P["max_completion_length"],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{VLLM_URL}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)["choices"][0]["message"]


def run_episode(model, row, temperature):
    env = SQLEnv()
    env.reset(sql_context=row["sql_context"], sql=row["sql"])
    messages = list(row["prompt"])
    tool_calls = tool_errors = 0
    for _ in range(P["max_tool_calling_iterations"]):
        message = chat(model, messages, temperature)
        calls = message.get("tool_calls") or []
        assistant = {"role": "assistant", "content": message.get("content") or ""}
        if calls:
            assistant["tool_calls"] = calls
        messages.append(assistant)
        if not calls:
            break
        for call in calls:
            function = call["function"]
            arguments = function["arguments"]
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = None
            result = "Error: invalid arguments" if arguments is None else dispatch(env, function["name"], arguments)
            tool_calls += 1
            tool_errors += result.startswith("Error")
            messages.append({"role": "tool", "tool_call_id": call.get("id"), "name": function["name"], "content": result})
    return {
        "success": env.get_reward(),
        "submitted": env.submitted_query is not None,
        "tool_calls": tool_calls,
        "tool_errors": tool_errors,
    }


def mean_with_standard_error(values):
    """Mean of a sample and its standard error: sample standard deviation over sqrt(n)."""
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    return mean, (variance / n) ** 0.5


def evaluate(model, rows, label):
    """Greedy decoding over the held-out rows. Returns summary metrics keyed by label.

    Each metric also gets a `<name>_se` companion: the standard error over the sampled questions.
    """
    with ThreadPoolExecutor(max_workers=32) as pool:
        episodes = list(pool.map(lambda row: run_episode(model, row, 0.0), rows))
    n = len(episodes)
    summary = {}
    for name, values in {
        "success": [e["success"] for e in episodes],
        "submit_rate": [1.0 if e["submitted"] else 0.0 for e in episodes],
        "tool_calls_per_episode": [float(e["tool_calls"]) for e in episodes],
    }.items():
        mean, standard_error = mean_with_standard_error(values)
        summary[f"{label}/{name}"] = mean
        summary[f"{label}/{name}_se"] = standard_error
    total_calls = sum(e["tool_calls"] for e in episodes)
    error_rate = sum(e["tool_errors"] for e in episodes) / max(1, total_calls)
    # Treat individual tool calls as the sampled units for this pooled proportion's standard error.
    summary[f"{label}/tool_error_rate"] = error_rate
    summary[f"{label}/tool_error_rate_se"] = (error_rate * (1 - error_rate) / max(1, total_calls)) ** 0.5
    print(
        f"[eval] {label}: "
        + ", ".join(
            f"{key.split('/')[1]}={value:.3f}" + (f" +/- {summary[f'{key}_se']:.3f}" if not key.endswith("_se") else "")
            for key, value in summary.items()
            if not key.endswith("_se")
        ),
        flush=True,
    )
    return summary


def load_adapter(name, path):
    body = json.dumps({"lora_name": name, "lora_path": path}).encode()
    request = urllib.request.Request(
        f"{VLLM_URL}/v1/load_lora_adapter", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        response.read()


def log_eval_metrics(result, step):
    """Log one evaluation's metrics to the AIR run. A logging failure fails the workload."""
    client = mlflow.MlflowClient()
    for key, value in result.items():
        client.log_metric(RUN_ID, key, float(value), step=step)


def main():
    if not RUN_ID:
        raise RuntimeError("MLFLOW_RUN_ID is not set; the recipe logs to the AIR-provided MLflow run")
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    train_ds = build_split("train", int(P["train_samples"]), seed=0)
    eval_ds = build_split("test", P["eval_samples"], seed=0)

    # Baseline: the base model through the same multi-turn tool loop, on the held-out rows.
    # Logged before training: the run exists but the trainer's MLflow callback has not resumed it yet.
    base = evaluate(P["model_name"], eval_ds, "eval")
    log_eval_metrics(base, step=0)

    trainer = AsyncGRPOTrainer(
        model=P["model_name"],
        args=AsyncGRPOConfig(
            output_dir=str(WORK / "trainer"),
            max_steps=int(P["max_steps"]),
            learning_rate=float(P["learning_rate"]),
            num_generations=int(P["num_generations"]),
            max_completion_length=int(P["max_completion_length"]),
            max_tool_calling_iterations=int(P["max_tool_calling_iterations"]),
            gradient_accumulation_steps=int(P["gradient_accumulation_steps"]),
            max_concurrent_rollouts=int(P["max_concurrent_rollouts"]),
            chat_template_kwargs={"enable_thinking": False},
            temperature=1.0,
            bf16=True,
            logging_steps=1,
            save_strategy="steps",
            save_steps=int(P["save_steps"]),
            save_total_limit=1,
            # Logs the reward and rollout curves to the AIR run (MLFLOW_RUN_ID). The callback ends
            # the run when training finishes, so the final metrics below reopen it by ID.
            report_to="mlflow",
            seed=0,
        ),
        train_dataset=train_ds,
        environment_factory=SQLEnv,
        peft_config=LoraConfig(
            r=int(P["lora_r"]),
            lora_alpha=2 * int(P["lora_r"]),
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )
    trainer.train()

    adapter_dir = OUT / "adapter"
    trainer.save_model(str(adapter_dir))
    print(f"[train] adapter saved to {adapter_dir}", flush=True)

    # Load the trained adapter into the running server and repeat the evaluation on the same rows.
    load_adapter(ADAPTER_NAME, str(adapter_dir))
    after = evaluate(ADAPTER_NAME, eval_ds, "eval")
    final_step = trainer.state.global_step

    metrics = {
        "model": P["model_name"],
        "train_rows": len(train_ds),
        "eval_rows": len(eval_ds),
        "train_steps": final_step,
        **{f"before_{k}": v for k, v in base.items()},
        **{f"after_{k}": v for k, v in after.items()},
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # The MLflow callback ended the AIR run when training finished; reopen it by ID for the final metrics.
    with mlflow.start_run(run_id=RUN_ID):
        log_eval_metrics(after, step=final_step)
        mlflow.log_artifact(str(OUT / "metrics.json"))
    print(f"[done] wrote {OUT / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()

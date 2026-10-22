#!/bin/bash
set -euo pipefail

if [ -z "${RAY_ENTRYPOINT:-}" ]; then
  echo "RAY_ENTRYPOINT is not set; expected a path relative to CODE_SOURCE_PATH." >&2
  exit 1
fi

RAY_HEAD_PORT="${RAY_HEAD_PORT:-6379}"
GPUS_PER_NODE="${LOCAL_WORLD_SIZE:-1}"
EXPECTED_NODES="${NUM_NODES:-1}"
EXPECTED_GPUS=$((EXPECTED_NODES * GPUS_PER_NODE))

stop_local_ray() {
  ray stop --grace-period 5 >/dev/null 2>&1 || true
}

if [ "${NODE_RANK:-0}" = "0" ]; then
  trap stop_local_ray EXIT

  echo "Starting Ray head with ${GPUS_PER_NODE} GPU(s)..."
  ray start \
    --head \
    --port="$RAY_HEAD_PORT" \
    --num-gpus="$GPUS_PER_NODE" \
    --dashboard-host=0.0.0.0

  echo "Waiting for ${EXPECTED_NODES} node(s) and ${EXPECTED_GPUS} GPU(s)..."
  python - "$EXPECTED_NODES" "$EXPECTED_GPUS" <<'PY'
import sys
import time

import ray

expected_nodes = int(sys.argv[1])
expected_gpus = int(sys.argv[2])
ray.init(address="auto", logging_level="ERROR")

for _ in range(60):
    live_nodes = sum(node.get("Alive", False) for node in ray.nodes())
    available_gpus = int(ray.cluster_resources().get("GPU", 0))
    if live_nodes >= expected_nodes and available_gpus >= expected_gpus:
        print(f"Ray cluster ready: {live_nodes} node(s), {available_gpus} GPU(s)", flush=True)
        ray.shutdown()
        break
    time.sleep(5)
else:
    ray.shutdown()
    raise SystemExit(
        f"Ray cluster did not reach {expected_nodes} node(s) and {expected_gpus} GPU(s) within 5 minutes"
    )
PY

  echo "Running ${RAY_ENTRYPOINT} on the Ray head..."
  python "$CODE_SOURCE_PATH/$RAY_ENTRYPOINT"
else
  trap stop_local_ray EXIT

  echo "Joining Ray head at ${MASTER_ADDR}:${RAY_HEAD_PORT}..."
  joined=""
  for attempt in $(seq 1 12); do
    if ray start \
      --address="${MASTER_ADDR}:${RAY_HEAD_PORT}" \
      --num-gpus="$GPUS_PER_NODE"; then
      joined=1
      break
    fi
    echo "Join attempt ${attempt} failed; retrying in 5 seconds..."
    sleep 5
  done

  if [ -z "$joined" ]; then
    echo "Worker failed to join the Ray head after all retries." >&2
    exit 1
  fi

  echo "Worker joined; waiting for the Ray head to finish..."
  worker_deadline=$((SECONDS + ${RAY_WORKER_MAX_WAIT_SECONDS:-7200}))
  while [ "$SECONDS" -lt "$worker_deadline" ]; do
    if ! timeout 10 ray health-check --address "${MASTER_ADDR}:${RAY_HEAD_PORT}" >/dev/null 2>&1; then
      echo "Ray head stopped; worker is exiting."
      exit 0
    fi
    sleep 5
  done

  echo "Timed out waiting for the Ray head to finish." >&2
  exit 1
fi

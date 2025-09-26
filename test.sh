#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONUNBUFFERED=1

LLM_NAME=bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0
DATASET="${DATASET:-zhenghaoxu/R2E-Gym-Lite-Truncate-Heuristic}"
SPLIT="${SPLIT:-train}"
START_IDX="${START_IDX:-0}"
K="${K:-1}"
MAX_STEPS="${MAX_STEPS:-20}"
TRAJ_DIR="${TRAJ_DIR:-./traj_resume_smoke}"

echo "Using LLM: $LLM_NAME"
echo "Dataset: $DATASET ($SPLIT), start_idx=$START_IDX, k=$K"
echo "Trajectory output: $TRAJ_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is required but not found in PATH." >&2
  exit 1
fi

if [ ! -d .venv ]; then
  echo "Creating local virtualenv with uv..."
  uv venv --seed
fi

source .venv/bin/activate

echo "Synchronizing project dependencies..."
uv sync --frozen --no-dev >/dev/null

echo "Ensuring editable install for current tree..."
uv pip install -e . >/dev/null

mkdir -p "$TRAJ_DIR"

uv run --with boto3 python src/r2egym/agenthub/run/edit.py runagent_multiple \
  --dataset "$DATASET" \
  --split "$SPLIT" \
  --k "$K" \
  --start_idx "$START_IDX" \
  --traj_dir "$TRAJ_DIR" \
  --exp_name resume_smoke_test \
  --llm_name "$LLM_NAME" \
  --use_fn_calling True \
  --max_steps "$MAX_STEPS" \
  --max_workers 1 \
  --resume_from_history True

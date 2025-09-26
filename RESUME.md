# Resume From History

This note explains how to replay a task's conversation history when running R2E‑Gym agents. The feature is handy for iterating on failed trajectories captured in HuggingFace datasets such as `zhenghaoxu/R2E-Gym-Lite-Truncate-Heuristic`.

## What It Does

When the dataset entry contains a `messages` array (system → user → assistant → …), the agent can reconstruct the environment by executing the previous assistant actions before handing control back to the language model. The replay stops right before the recorded failure, giving the model another chance to continue from that point.

## How To Use It

### CLI Runner

Enable the behaviour by passing `--resume_from_history True` to the existing runner. Example:

```bash
uv run python src/r2egym/agenthub/run/edit.py runagent_multiple \
  --dataset "zhenghaoxu/R2E-Gym-Lite-Truncate-Heuristic" \
  --split train \
  --k 1 \
  --start_idx 0 \
  --llm_name bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0 \
  --use_fn_calling False \
  --resume_from_history True
```

The runner automatically extracts `messages` and `mistake_index` from each dataset entry and replays steps before querying the model.

### Quick Smoke Test

The repo ships with `test.sh` for a one-click smoke test. By default it:

1. Sets up a local `uv` virtual environment.
2. Installs project dependencies.
3. Runs `runagent_multiple` with `--resume_from_history True` on a single instance.

You can override parameters via environment variables or by supplying the LLM name as the first positional argument:

```bash
./test.sh us.anthropic.claude-3-7-sonnet-20250219-v1:0
```

Outputs land in `./traj_resume_smoke/` and include the replayed steps for inspection.

## Implementation Notes

- `ResumeConfig` (in `agent/agent.py`) carries the raw `messages` plus optional `mistake_index` from the dataset.
- `_bootstrap_from_messages` executes historical assistant actions in order, validating tool calls and recording observations.
- All parsing uses the existing `<think>…</think>` and `<function=…>…</function>` logic, so replayed steps share the same format as live interactions.
- Step-count reminders, reward computation, and trajectory serialization continue to work unchanged.

## Troubleshooting

- **Missing messages**: If a dataset entry lacks the `messages` field, the feature logs a warning and falls back to a normal fresh run.
- **Observation mismatches**: Replay compares stored user/tool outputs with actual environment responses. A warning is emitted if they differ, but execution continues with the new observation.
- **Max step limit**: If the prior conversation already exceeded `max_steps_absolute`, the replay aborts with an error to avoid re-running the entire trajectory.

Feel free to extend `ResumeConfig` with additional metadata (e.g., cached patches) as new datasets make more context available.

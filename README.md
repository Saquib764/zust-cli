# zust-cli

In-pod training runner for Zust. The [zust-ai-be](https://github.com/Saquib764/zust-ai-be) API starts this process for one run. It is not a user-facing CLI.

Python 3.10+ (Runpod training images ship 3.12; the runner itself does not require 3.12). Dependencies are managed with [uv](https://docs.astral.sh/uv/).

## What it does

1. Clones the training repo at the pinned commit into the run workspace (`repo/`).
2. Runs the setup argv, if the launch included one.
3. Writes the effective config and `.env` returned by the API.
4. Runs the training argv and ships TensorBoard scalars back as run metrics.
5. Optionally runs the inference argv.

Logs and phase updates are posted to `/api/v1/internal/runs/{id}/logs` and `/progress`. A carriage return in stdout is kept as its own record so progress bars overwrite one row in the UI.

If the API is unreachable (5xx, timeouts, connection errors), those callbacks are appended to `<workspace>/outbound/pending.jsonl` instead of failing the run. Each later successful request replays the queue in order before sending new data.

On Runpod the workspace is `/workspace/runs/<run_id>`. Local launches set `ZUST_WORKSPACE_ROOT`.

## Environment

All of these are required. The process exits 2 if any is missing.

| Variable | Purpose |
| --- | --- |
| `ZUST_API_BASE` | Backend origin, no trailing path |
| `ZUST_RUN_ID` | Run id |
| `ZUST_RUN_TOKEN` | Bearer token for internal callbacks |
| `ZUST_REPOSITORY` | `owner/name` on GitHub |
| `ZUST_COMMIT_SHA` | Commit to fetch |
| `ZUST_CONFIG_PATH` | Config path relative to the repo root |

Optional:

| Variable | Purpose |
| --- | --- |
| `ZUST_SETUP_ARGV` | JSON list. Runs before training |
| `ZUST_TRAINING_ARGV` | JSON list. The training command |
| `ZUST_INFERENCE_ARGV` | JSON list. Used when test inference is on |
| `ZUST_RUN_TEST_INFERENCE` | `1` to run inference after a successful train |
| `ZUST_WORKSPACE_ROOT` | Overrides `/workspace/runs/<run_id>` |
| `ZUST_GITHUB_TOKEN` | Optional. Authenticates `git fetch` for private training repos |
| `ZUST_RUNNER_GIT_URL` | Set on Runpod pods by the backend bootstrap (do not set manually) |
| `ZUST_RUNNER_ROOT` | Directory where the pod installs zust-cli (default `/workspace/zust-cli`) |

## Develop

```bash
uv sync
uv run python -m zust_runner
uv run pytest
```

`python -m zust_runner` only succeeds when the environment above is set, which the backend does at launch time.

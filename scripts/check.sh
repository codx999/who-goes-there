#!/usr/bin/env bash
# Session health check: run at the start and end of every session. Stops at the first failure.
set -euo pipefail
cd "$(dirname "$0")/.."

# Outside CI, the guard hooks must be enabled (relative or absolute path to .githooks). A missing
# .githooks/ fails too: an empty expected path would otherwise equal an unset hooksPath.
hooks="$(git config --get core.hooksPath || true)"
hooks_dir="$( { [ -n "$hooks" ] && cd "$hooks" 2>/dev/null && pwd -P; } || true)"
expected="$( { cd .githooks 2>/dev/null && pwd -P; } || true)"
if [ -z "${CI:-}" ] && { [ -z "$expected" ] || [ "$hooks_dir" != "$expected" ]; }; then
  echo "check.sh: git hooks are not enabled; run: git config core.hooksPath .githooks" >&2
  exit 1
fi

# The split tests on the real metadata (item 3.03) read TRAILCAM_DATA_ROOT. Outside CI it defaults
# to the main checkout's data/ when that exists, so a local check runs them instead of skipping.
if [ -z "${CI:-}" ] && [ -z "${TRAILCAM_DATA_ROOT:-}" ]; then
  root="$(bash scripts/data_root.sh)"
  if [ -n "$root" ]; then
    export TRAILCAM_DATA_ROOT="$root"
  fi
fi

uv sync --locked --quiet
uv run pytest --tb=short -q
uv run mypy trailcam/ scripts/guard_data.py scripts/kaggle_prepare_cct20.py
uv run ruff check .
uv run ruff format --check .
uv run python scripts/guard_data.py --all
uv run python scripts/guard_data.py --history
echo "check.sh: all green"

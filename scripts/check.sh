#!/usr/bin/env bash
# Session health check: run at the start and end of every session. Stops at the first failure.
set -euo pipefail
cd "$(dirname "$0")/.."

# Outside CI, the guard hooks must be enabled (relative or absolute path to .githooks).
hooks="$(git config --get core.hooksPath || true)"
hooks_dir="$( { [ -n "$hooks" ] && cd "$hooks" 2>/dev/null && pwd -P; } || true)"
if [ -z "${CI:-}" ] && [ "$hooks_dir" != "$(cd .githooks && pwd -P)" ]; then
  echo "check.sh: git hooks are not enabled; run: git config core.hooksPath .githooks" >&2
  exit 1
fi

uv sync --locked --quiet
uv run pytest --tb=short -q
uv run mypy trailcam/ scripts/guard_data.py
uv run ruff check .
uv run ruff format --check .
uv run python scripts/guard_data.py --all
uv run python scripts/guard_data.py --history
echo "check.sh: all green"

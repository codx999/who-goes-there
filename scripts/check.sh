#!/usr/bin/env bash
# Session health check: run at the start and end of every session. Stops at the first failure.
set -euo pipefail
cd "$(dirname "$0")/.."

uv sync --locked --quiet
uv run pytest --tb=short -q
uv run mypy trailcam/ scripts/guard_data.py
uv run ruff check .
uv run ruff format --check .
uv run python scripts/guard_data.py --all
uv run python scripts/guard_data.py --history
echo "check.sh: all green"

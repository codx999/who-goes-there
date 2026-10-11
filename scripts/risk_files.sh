#!/usr/bin/env bash
# Print the files changed in a git range that fall under the high-stakes paths (the trailcam
# package: splits, transforms, the box sampler, training, scoring, the demo; the Space; notebooks)
# or the control plane that protects them (tests, pytest config, check.sh, the data guard, hooks,
# CI, the Claude agent, command, rule and permission files). /review-session runs the
# risk-reviewer agent on them with the matching rows of docs/review/invariants.md.
# Usage: scripts/risk_files.sh <base>..<head>   (no output: no risky files changed; exit 2: bad range)
set -euo pipefail

usage() {
    echo "usage: $0 <base>..<head>  (both ends must name commits)" >&2
    exit 2
}

[[ $# -eq 1 && $1 == *..* ]] || usage
base=${1%%..*}
head=${1#*..}
# An empty end (e.g. a shell variable lost between calls) would diff nothing and print nothing.
[[ -n $base && -n $head ]] || usage
git rev-parse --verify --quiet "$base^{commit}" >/dev/null || usage
git rev-parse --verify --quiet "$head^{commit}" >/dev/null || usage

# Kept in step with the "where" column of docs/review/invariants.md (tests/test_risk_files.py
# checks every entry there against this regex, R44). Config files that override pyproject.toml
# (pytest.ini, tox.ini, setup.cfg, mypy.ini, ruff.toml, any conftest.py) are listed because a
# new one can silence a test or a check without touching a tracked file. Root training
# configs (config.yaml of item 5.02, a configs/ folder) set the arms, so they are listed too.
RISKY='^trailcam/|^configs?/|^config\.(ya?ml|toml|json)$|^space/|\.ipynb$|^tests/|^scripts/|^CLAUDE\.md$|^pyproject\.toml$|^uv\.lock$|^(\.?pytest\.ini|tox\.ini|setup\.cfg|\.?mypy\.ini|\.?ruff\.toml)$|(^|/)conftest\.py$|^\.githooks/|^\.github/workflows/|^\.claude/'

# --no-renames: a file moved out of a risky path shows up as its deleted old path.
# -z and quotePath=false: non-ASCII paths come out as-is, not quoted.
git -c core.quotePath=false diff --no-renames --name-only -z "$base" "$head" \
    | { grep -zE "$RISKY" || true; } \
    | tr '\0' '\n'

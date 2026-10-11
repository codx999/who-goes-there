#!/usr/bin/env bash
# Print the absolute path of the main checkout's data/, or nothing if it does not exist. A worktree
# has no data/ of its own and reaches the main checkout's by absolute path (CLAUDE.md). check.sh
# uses this to set TRAILCAM_DATA_ROOT, so the split tests on the real metadata (item 3.03) run in
# every local check instead of skipping. Prints a path only; never lists or reads data/.
set -euo pipefail
common="$(git rev-parse --path-format=absolute --git-common-dir)"
data="$(dirname "$common")/data"
if [ -d "$data" ]; then
  cd "$data" && pwd -P
fi

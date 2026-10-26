# Project Setup and Review Playbook

Exported 2026-10-10 from the live doc:
https://claude.ai/code/artifact/fdd57274-42a8-4dbd-b6d2-d5d33f6a785d

## Purpose and how to use

This playbook copies the working setup of Animal Spirits (the finance-comment stance classifier) into a new project, starting with Who Goes There, the camera-trap image classifier. It covers four things: the Python tooling and CI, the files Claude Code reads (CLAUDE.md, rules, commands, agents), the one-item-per-session protocol, and the tiered code review.

Apply it in this order, one commit per step: tooling and check script first, then the data guard and hooks, then CLAUDE.md and the progress files, and the review playbook last. The review files depend on the check script and the progress files existing. The last section is a checklist you can follow top to bottom.

The source files live in the Animal Spirits repo (a separate local project). Copy them across and edit them; do not copy anything under `docs/research/`, the package code or the data. The reviewer design is written up in `docs/review/playbook-reusable.md` (copied into this repo); Part B of that file is a prompt you can paste into Claude Code here.

## Tooling setup

Python 3.12 managed by uv, with ruff, mypy (strict) and pytest, all run by one script that CI also runs. Never use bare `python`, `pip install` or `conda` locally; Kaggle and Colab notebooks are the only exception.

| Piece | Setting | Source file to copy |
| --- | --- | --- |
| Python | 3.12, pinned | `.python-version` |
| Packages | `uv add <pkg>`, `uv add --dev <pkg>`; commit `uv.lock` | `pyproject.toml`, `uv.lock` |
| Lint and format | ruff, line length 100, rules E F I B UP | `pyproject.toml` `[tool.ruff]` |
| Types | mypy `strict = true` on the package | `pyproject.toml` `[tool.mypy]` |
| Tests | pytest, `pythonpath = ["."]`, `testpaths = ["tests"]` | `pyproject.toml` `[tool.pytest.ini_options]` |
| Check script | sync, tests, mypy, ruff check, ruff format --check, data guard; stops at first failure | `scripts/check.sh` |
| CI | GitHub Actions on push to main and manual run; runs `check.sh`; no secrets; actions pinned to commit IDs; `fetch-depth: 0` | `.github/workflows/check.yml` |
| Claude permissions | allow test, lint, type and git read commands; deny `Read` on `.env*` and `data/**` | `.claude/settings.json` |

`scripts/check.sh` in full, with the package name changed:

```bash
#!/usr/bin/env bash
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
```

Check its exit code before every commit; never pipe it into `tail` or `head` first, because the pipe hides a failure. Keep the uv version in CI equal to your local one and bump both together.

## Repo layout and CLAUDE.md template

Keep state in files, not chat: a plan, a feature list, a log, and a short CLAUDE.md that points at them.

```text
who-goes-there/
  CLAUDE.md                  # project brief, ground rules, session protocol, tooling
  pyproject.toml  uv.lock  .python-version  config.yaml  .env.example  .gitignore
  trailcam/                  # all code
  tests/
  scripts/
    check.sh  guard_data.py  risk_files.sh
  docs/
    plan.md                  # source of truth: design, schedule, decisions
    progress.json            # feature list, one item per unit of work
    progress-log.md          # dated done / next entries
    review/
      invariants.md          # rules that must always hold
      playbook-reusable.md
  .claude/
    settings.json            # permissions
    rules/data.md  rules/training.md      # path-scoped rules
    commands/review-session.md  dryrun.md  review-shapes.md
    agents/code-reviewer.md  risk-reviewer.md
  .githooks/pre-commit  pre-merge-commit  pre-push
  .github/workflows/check.yml
```

CLAUDE.md skeleton (keep it under about 80 lines; detail goes in the docs it points to):

```markdown
# <Project name>
One paragraph: what it classifies, the classes, the capstone scope and what is out of scope.

## Where the context lives
- docs/plan.md: source of truth (design, schedule, decisions).
- If documents disagree: plan.md wins, then the design doc that owns the topic.

## Ground rules
- Data and licence rules, test-set isolation, leakage alarms, budget caps, stack.

## Session protocol
The start / pick / verify / end steps, model and effort, review step (see the sections below).

## Local tooling
uv, mypy, ruff, pytest commands; check.sh; data guard and hooks; custom commands; rules files.
```

The custom commands carry over almost as they are: `/dryrun` runs one epoch on a tiny batch and stops, and `/review-shapes` prints a layer → input shape → output shape table before any training. For a CNN, `/review-shapes` matters more than it did for a stock text classifier, so keep it.

## Session protocol

Work is split into numbered sessions, one module each, and every unit of work is an item in `docs/progress.json` that flips to `passes: true` only after its steps succeed.

Item format (copy the `_rules` text too):

```json
{
  "_rules": "Flip only `passes`, and only after `steps` succeed. Never edit or delete items or steps; new items need the user's OK. Items with owner=user are never flipped by Claude and never block other work. Steps starting with 'manual:' are checked by the user.",
  "items": [
    {"id": "1.01", "session": 1, "owner": "claude", "due": "2026-10-12",
     "description": "Dataset loader with a fixed train/val/test split",
     "steps": "uv run pytest tests/test_dataset.py passes; split counts printed, no file names",
     "passes": false}
  ]
}
```

Every session follows the same five steps:

1. **Start.** `pwd`, `git log --oneline -10`, then `git merge-base --is-ancestor main HEAD`. If it fails in an app-made worktree with no commits of its own, `git merge --ff-only main` and re-read CLAUDE.md. Read the last entries of `progress-log.md`, run `bash scripts/check.sh`, and check the last CI run with `gh run list --branch main --limit 1`. Fix anything red first.
2. **Pick work.** The lowest-id item in the current session with `owner: claude` and `passes: false`; one item at a time.
3. **Verify, then flip.** Run the item's steps and set `passes: true` only if they succeed. Pipeline items need a real run on fixtures, not only unit tests.
4. **End.** `check.sh` green, a dated "done / next" entry appended to `progress-log.md`, one commit per item where practical, repo mergeable.
5. **Review.** Report the tier and the command, then wait for the user (see the review section).

Two habits that saved time: in a worktree, data paths must be absolute paths into the main checkout's `data/`; and merge and push happen only with the user's go-ahead, run by the user in the main checkout.

## Data guard and safety rules

No data, model weights or keys ever reach git, and four layers enforce it: `.gitignore`, a guard script, git hooks and CI.

- **`scripts/guard_data.py`** (standard library only) refuses blocked directories (`data/`, `models/`, `outputs/`), data and archive suffixes (`.db`, `.sqlite*`, `.csv`, `.tsv`, `.jsonl`, `.parquet`, `.pkl`, `.zip`, `.tar`, `.gz` and more) and secret patterns such as API keys. It reads blobs from git, never the working tree, and reports the path and rule only, never the matching text. Modes: `--staged` (pre-commit), `--pre-push`, `--all` and `--history` (check.sh and CI).
- **Hooks** in `.githooks/` call it with `uv run --no-project`. Enable once per clone: `git config core.hooksPath .githooks`.
- **`.gitignore`** adds `/data/`, `/models/`, `/outputs/` and the data suffixes on top of the standard Python template.
- **`.claude/settings.json`** denies `Read` on `.env`, `.env.*.local` and `data/**`, so Claude cannot open them even by accident.

Path-scoped rules in `.claude/rules/` load only when Claude touches matching files:

- `data.md` (data and dataset paths): print shapes, dtypes and counts only, never raw rows or file listings.
- `training.md` (training paths): never start long training on its own, use the dry-run config first, and stop after two failures in a row and wait for you.

For images, add `.jpg`, `.jpeg`, `.png`, `.webp`, `.pt`, `.pth`, `.ckpt`, `.safetensors` and `.h5` to the blocked suffixes, so a dataset or checkpoint never gets committed. Keep sample images for the demo outside the repo or in a tracked, allow-listed `assets/` folder.

## Model and effort policy

Implement on Opus 5.5 at `high`, raise to `xhigh` for risky items, and review on two different models so their blind spots don't overlap.

| Role | Model | Effort | When |
| --- | --- | --- | --- |
| Implementer | Opus 5.5 | high | Default for every item |
| Implementer, risky item | Opus 5.5 | xhigh | Interrupted runs, retries, resumability, splits, budget caps |
| Advisor (optional) | Opus 5.5 | high | Second opinion mid-item; re-check each warning against the finished code |
| `code-reviewer` agent | Opus 5.5 | xhigh | Tier 2: the whole range |
| `risk-reviewer` agent | Fable 5.1 | high | Tier 2: high-stakes files only; if Fable wrote the code, a second Opus run replaces it |
| Built-in `/code-review` | session model | xhigh | Tier 1 |

Pass `model` and `effort` explicitly on every agent spawn, and start every reviewer prompt with "read the agent file first". Agent files load at session start, so a worktree on an old commit runs a stale copy without any error. Because the user pays per token, prefer one item per session and no multi-agent runs they did not ask for.

## Code review playbook

A script picks the review tier from the changed paths, the user starts the review, and every finding is verified before anything is fixed. Claude never starts a review on its own.

| Tier | When | Command |
| --- | --- | --- |
| 0 | No code changed | Nothing to run; report any rows whose spec document changed |
| 1 | Code changed, `risk_files.sh` printed nothing | `/code-review xhigh --max-findings all <base>..HEAD` |
| 2 | High-stakes files changed | `/review-session`: code-reviewer (Opus 5.5 xhigh) on the whole range, risk-reviewer (Fable 5.1 high) on the printed files |
| 3 | Split code or the final test scoring | Tier 2 plus `/code-review ultra`; leave time for one fix round before the deadline |

Run items (long training runs, the final test scoring) skip this path: their ledger rows are reviewed instead (see below).

### The files

- **`scripts/risk_files.sh <base>..<head>`** prints the changed files that match one high-stakes regex (exit 2 on a bad range). It uses `--no-renames` so a file moved out of a risky path still shows, and `-z` with `core.quotePath=false` so non-ASCII paths are not quoted. The regex always covers the control plane: `tests/`, `scripts/`, `CLAUDE.md`, `pyproject.toml`, `uv.lock`, `.githooks/`, `.github/workflows/`, `.claude/`, and any `pytest.ini`, `tox.ini`, `setup.cfg`, `mypy.ini`, `ruff.toml` or `conftest.py`, because a new one can silence a check without touching a tracked file.
- **`docs/review/invariants.md`**: one row per rule that must always hold, with id, rule, source document and section, where it can break (paths and item ids), a scenario for a reviewer to trace, and the test that checks it or "none yet". 30 to 40 rows is enough. A test parses the "where" column and asserts the regex matches every entry, so the two cannot drift.
- **`.claude/agents/code-reviewer.md`** (generalist, whole range) and **`risk-reviewer.md`** (specialist, given files). Both: tools Read, Grep, Glob, Bash; read-only; never read env or data files or the progress-log narrative; severity defined (high = an invariant breaks, money is spent wrongly or data leaks; everything else low); "nothing found" is a valid result. Output: findings most severe first with file:line, failure scenario, how verified, suggested fix.
- **`.claude/commands/review-session.md`**: refuses with uncommitted changes, computes `merge-base(main, HEAD)..HEAD`, runs `risk_files.sh`, selects invariant rows by path or item id, flags "spec changed" rows, reports the tier, spawns both agents in parallel at tier 2, merges the reports with each finding tagged by reviewer, then stops.

### After a review

1. Verify each finding against the code, with a reproducing test where possible.
2. Reject a finding only with a test or a code citation; list every rejected finding above low for the user with the reason.
3. Fix the real ones, each with a regression test that fails on the old code. Never use `--fix`.
4. Run one fix round, on the fix range only, when a fix commit touches a high-stakes file (other than purely added tests). A further round needs the user's request.
5. Log fixed and skipped findings with the reviewer that raised them, and end the log entry with "Review: tier N, ran X" or "skipped by user".

### Actions with no diff

Long training runs, paid API jobs, test-set scoring and exports never show up in a diff. Make the code write a ledger row (what ran, with what parameters, cost, who approved) and have the progress-log entry quote it; the ledger is what gets reviewed.

## Who Goes There: what changes for the image classifier

Decided: **Who Goes There**, a PyTorch species classifier for Caltech Camera Traps (CCT20) photos, tested on camera locations it never saw. Public GitHub repo, built with local Claude Code sessions (not Projects). The tooling, session protocol and review machinery carry over unchanged; the rows below are what changes.

| Setting | Animal Spirits | Who Goes There |
| --- | --- | --- |
| Folder and repo | `Animal-Spirits`, private | `~/Projects/who-goes-there`, public on GitHub, MIT license for the code |
| Package | `ytstance/` | `trailcam/` with `data/` (metadata, splits), `model/`, `train/`, `evaluate/`, `demo/` |
| Data | YouTube comments via the API, 30-day rule | CCT20 from LILA: about 58k images, 15 classes plus empty, 20 camera locations, about 6 GB; CDLA-Permissive, cite Beery et al. 2018 |
| Compliance doc | `docs/compliance.md` (YouTube, Anthropic, TypeSafe terms) | `docs/data-card.md`: license, attribution, class list, counts per split |
| Labels | LLM batch labeling, $40 / $75 caps | Labels ship with the dataset; no LLM spend, so drop the budget rows |
| Split | By video, author and time | Official location split: train locations, cis test (seen cameras), trans test (unseen cameras); every burst kept in one split |
| Leakage alarm | Macro-F1 above about 0.80 | Trans accuracy close to cis accuracy, or any image or sequence id in two splits |
| Dependencies | pandas, scikit-learn; torch later | Add when training starts: `torch`, `torchvision`, `timm`, `gradio`, `grad-cam`; mypy overrides for untyped libraries |
| Compute | Kaggle or Colab | Kaggle notebook downloads CCT20 from LILA; the Mac runs code and tests only |
| Risk regex | `^ytstance/` | `^trailcam/`; keep `\.ipynb$` and the control-plane paths |
| Run ledger | LLM batches (R20-R23) | One row per training run: config, seed, val score, checkpoint hash |
| Tier 3 | Session-6 splits, evaluate, calibrate | The split code and the single final scoring on cis and trans test |
| Rules files | `data.md`, `training.md` | `data.md`: counts and shapes only, never open or show images unless asked; `training.md` unchanged |
| Demo | Small demo | Gradio app with preloaded unseen-camera images, Grad-CAM heatmaps, a live sort of about 200 photos, and a recorded backup video |

Starter rows for `docs/review/invariants.md`:

- No camera location is in both training and trans test, and no cis test image is in training.
- Every burst (sequence id) sits in one split.
- Augmentation runs on training images only; val and test get resize and normalize only.
- Normalization uses the pretrained model's published mean and standard deviation.
- The class-to-index map (15 classes plus empty) is written once and shared by training, evaluation and the demo.
- Hyperparameters and the checkpoint are chosen on val; cis and trans test are scored once, at the end, and the run is logged in the ledger.
- Demo images come from trans test locations only.
- Seeds are set and logged for every run.

## Bootstrap checklist

About one session of work: steps 1 to 6 give a green `check.sh`, and steps 7 to 10 add the review machinery.

- [ ] 1. In `~/Projects/who-goes-there` (the folder path is fixed from here on, since Claude Code keys its memory to it): `git init`, and `uv init` with Python 3.12; commit `.python-version`.
- [ ] 2. Copy `pyproject.toml` tool sections, rename the package to `trailcam`, then `uv add --dev pytest mypy ruff` and `uv sync`.
- [ ] 3. Copy `.gitignore`, `scripts/guard_data.py` (add the image and checkpoint suffixes) and `scripts/check.sh`; add a smoke test so pytest has something to run.
- [ ] 4. Copy `.githooks/` and run `git config core.hooksPath .githooks`.
- [ ] 5. Copy `.claude/settings.json`, `.claude/rules/`, and the `dryrun` and `review-shapes` commands; update the paths.
- [ ] 6. Run `bash scripts/check.sh` until green, then commit.
- [ ] 7. Write `CLAUDE.md` from the template, `docs/plan.md` (from `docs/HANDOFF.md`), `docs/progress.json` and an empty `docs/progress-log.md`.
- [ ] 8. Create the public GitHub repo `who-goes-there` with an MIT LICENSE and the CCT20 attribution in the README, copy `.github/workflows/check.yml` with the uv version matching yours, push, and confirm the first CI run is green.
- [ ] 9. Paste Part B of `docs/review/playbook-reusable.md` into Claude Code in this repo, filling in the bracketed parts: it writes `invariants.md`, `risk_files.sh` with its test, both agent files and `review-session.md`.
- [ ] 10. Run `bash scripts/risk_files.sh main..HEAD` on that branch and start its first review yourself.

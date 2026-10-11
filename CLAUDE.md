# Who Goes There

A PyTorch species classifier for Caltech Camera Traps (CCT20) photos. Input: one camera-trap JPEG.
Output: one of 12 classes (11 animals plus `car`; `empty`, badger, fox and deer are dropped).
The headline result is accuracy and macro-F1 on cameras seen in training (cis test) against
cameras never seen (trans test). Bootcamp capstone, presentation 2026-10-26; the deliverables are
the trained model, a Gradio demo on Hugging Face Spaces, slides and this public repo. Out of
scope: detection, sequence-level classification, the `empty` class, anything beyond a Kaggle GPU.

## Where the context lives

- `docs/plan.md`: source of truth (design, split, training, schedule, decisions and why).
- `docs/progress.json`: one item per unit of work; `docs/progress-log.md`: dated done / next.
- `docs/data-inspection.md`: verified CCT20 facts. `docs/course-guideline.md`: class requirements.
- `docs/setup-playbook.md`: session protocol and review tiers. `docs/references.md`: how to work.
- If documents disagree: `plan.md` wins, then the document that owns the topic. `docs/HANDOFF.md`
  and `docs/setup-review.md` are inputs to the plan, not rules.

## Ground rules

- No images, metadata dumps, weights or keys in git. `scripts/guard_data.py` runs in the
  pre-commit, pre-merge and pre-push hooks and in CI; `.gitignore` and the Read deny list in
  `.claude/settings.json` back it up. Demo samples go only in `assets/`, generated figures only in
  `docs/figures/`; both have file-count and size budgets in the guard.
- Data: counts, shapes and class distributions only; never raw rows, file listings or images from
  `data/`. In a worktree, data paths are absolute paths into the main checkout's `data/`.
- Test-set isolation: every choice is made on val. cis test and trans test are scored once, at the
  end, with the user's go-ahead, and the run is logged in `docs/ledger.md`.
- Leakage alarms: an image or sequence id in two splits, a location in both train and trans, or
  trans accuracy close to cis accuracy.
- The class-to-index map lives in `trailcam/data/classes.py` and is imported everywhere.
- Augmentation on the training transform only; val and test get resize and normalise.
- Stack: Python 3.12, uv, torch, torchvision, timm, gradio, grad-cam. Kaggle runs training; the Mac
  runs code, tests and the dry-run. Never start a long training run on your own.
- Licence: code MIT; data CDLA-Permissive 1.0, cite Beery et al. 2018 wherever the data is shown.

## Session protocol

1. Start: `pwd`, `git log --oneline -10`, `git merge-base --is-ancestor main HEAD` (in an app-made
   worktree with no own commits, `git merge --ff-only main` if it fails), read the last entries of
   `docs/progress-log.md`, `bash scripts/check.sh`, `gh run list --branch main --limit 1`. Fix red first.
2. Pick: the lowest-id item in the current session with `owner: claude` and `passes: false`.
3. Verify, then flip: run the item's steps; set `passes: true` only if they succeed. Never edit or
   delete items; new items need the user's OK. Items with `owner: user` are never flipped by Claude.
4. End: `check.sh` green, a dated entry in `docs/progress-log.md`, one commit per item, no push.
   Merge and push are done by the user in the main checkout.
5. Review: report the tier (`docs/setup-playbook.md`, "Code review playbook") and the command, then
   wait. Claude never starts a review on its own. Tier 3: the split code and the final scoring.

Model and effort: implement on Opus 5.5 at `high`; `xhigh` for the split code, the final scoring,
retries and resumability. Pass model and effort explicitly on every agent spawn. One item per
session; no multi-agent runs the user did not ask for.

## Local tooling

- `bash scripts/check.sh`: uv sync, pytest, mypy (strict), ruff check and format, data guard
  (`--all`, `--history`). Run at session start and before every commit; check the exit code, never
  pipe it through `tail`.
- Packages: `uv add <pkg>` / `uv add --dev <pkg>`, commit `uv.lock`. Never bare `python`, `pip` or
  `conda`. One exception: `space/requirements.txt` for Hugging Face Spaces, exported with
  `uv export --no-dev --no-hashes` and trimmed to what `space/app.py` imports.
- Hooks: `git config core.hooksPath .githooks` once per clone (in an app-made worktree:
  `git config --worktree core.hooksPath "$PWD/.githooks"`).
- Commands: `/dryrun` runs one epoch on a tiny fixture batch and stops; `/review-shapes` prints the
  layer, input shape, output shape and parameter table before any training.
- Rules in `.claude/rules/`: `data.md` (data, evaluate, fixtures paths) and `training.md` (train,
  model, notebooks paths; notebooks are committed with outputs cleared).
- Weights never go through GitHub: `scripts/deploy_space.py` uploads them to the Space with
  `huggingface_hub`, token from the environment.

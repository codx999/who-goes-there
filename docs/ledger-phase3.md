# Ledger, phase 3

One row per phase-3 training run (`docs/phase3-plan.md` section 8, item 13.11), appended by
`trailcam/train/train.py` with `--ledger <clone>/docs/ledger-phase3.md` (the training notebook's
`LEDGER = "docs/ledger-phase3.md"`), in the column order the script prints. The columns and the
header are those of `docs/ledger.md`, whose notes on each column hold here too. Kaggle runs are
started by the user, who pastes the printed row here; the row is the unit of review for run items.

- Only phase-3 configs, whose names start with `p3_`: `train.py` refuses a `p3_` config with any
  other ledger and any other config with this one, before training and at the append, and the
  check below refuses a row of another phase here and a `p3_` row in `docs/ledger.md` or
  `docs/ledger-phase2.md`. Phase-1 rows stay in `docs/ledger.md`, phase-2 rows in
  `docs/ledger-phase2.md`.
- `p3_convnext` and `p3_convnext_cp` are two recipes of one model, so the rows of one model may
  carry several config hashes; the rows of one config share one, and every row shares one split
  seed (phase 1's split, seed 42). `p3_baseline` is `resnet18_arm_a` exactly, so its rows carry
  arm A's config hash.
- Training still selects the checkpoint on val macro-F1 alone (R17); a candidate is kept or
  dropped on val and trans_val by the keep rule fixed before any run (`docs/phase3-plan.md`
  section 1), so trans_val here is optimistic (the plan's section 6). Test scores never go here:
  the finalist's, if any, go to `docs/test-ledger-phase3.md` (section 7).

Check it after pasting a row: `uv run python -m trailcam.train.ledger --check docs/ledger-phase3.md`
(`trailcam/train/ledger.py`) exits 1 with every problem, or 0 with the scores per config, model,
arm and seed.

Data: Caltech Camera Traps (CCT20), Beery et al. 2018, CDLA-Permissive 1.0.

| run id | model | arm | config | config hash | split seed | run seed | split hash | epochs run | best epoch | scheduler | val accuracy | val macro-F1 | trans_val accuracy | trans_val macro-F1 | checkpoint sha256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

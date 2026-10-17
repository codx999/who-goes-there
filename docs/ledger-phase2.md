# Ledger, phase 2

One row per phase-2 training run (`docs/phase2-plan.md`, item 13.02), appended by
`trailcam/train/train.py` with `--ledger <clone>/docs/ledger-phase2.md` (the training notebook's
`LEDGER = "docs/ledger-phase2.md"`), in the column order the script prints. The columns and the
header are those of `docs/ledger.md`, whose notes on each column hold here too. Kaggle runs are
started by the user, who pastes the printed row here; the row is the unit of review for run items.

- Only phase-2 configs, whose names start with `p2_`: `train.py` refuses a `p2_` config with any
  other ledger and a phase-1 config with this one, before training and at the append, and the
  check below refuses a row of either kind in the wrong ledger. Phase-1 rows stay in
  `docs/ledger.md`, which holds exactly the pre-registered runs the final scoring reads.
- Every phase-2 step is a new recipe of the same model, so the rows of one model may carry several
  config hashes; the rows of one config share one, and every row shares one split seed (phase 1's
  split, seed 42). `p2_baseline` is `resnet18_arm_a` exactly, so its rows carry arm A's config hash.
- Training still selects the checkpoint on val macro-F1 alone (R17); a step is kept or dropped on
  val and trans_val (`docs/phase2-plan.md` section 3), so trans_val here is slightly optimistic
  (the plan's honesty note). Test scores of phase 2 never go here.

Check it after pasting a row: `uv run python -m trailcam.train.ledger --check docs/ledger-phase2.md`
(`trailcam/train/ledger.py`) exits 1 with every problem, or 0 with the scores per config, model,
arm and seed.

Data: Caltech Camera Traps (CCT20), Beery et al. 2018, CDLA-Permissive 1.0.

| run id | model | arm | config | config hash | split seed | run seed | split hash | epochs run | best epoch | scheduler | val accuracy | val macro-F1 | trans_val accuracy | trans_val macro-F1 | checkpoint sha256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

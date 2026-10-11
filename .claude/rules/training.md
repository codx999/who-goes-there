---
paths:
  - "trailcam/train/**"
  - "trailcam/model/**"
  - "notebooks/**"
---
# Training Protocols

## Script constraints
- Never launch long training runs on your own. Use the dry-run config (one epoch, tiny batch,
  fixture images) first, and leave real runs to the Kaggle notebook started by the user.
- If a run or verification script fails 2 times in a row, STOP, print the short stack trace
  summary, and wait for the user. Do not loop to self-heal.
- Every run sets and logs its seeds. Every training run appends one row to `docs/ledger.md` (the
  dry run prints its row and never appends) with the fields of `docs/plan.md` section 5, "Ledger"
  row, in this order: run id, model, arm, config, split seed, run seed, split hash, epochs run,
  best epoch, scheduler, val accuracy, val macro-F1, trans_val accuracy, checkpoint hash. The
  ledger row is what gets reviewed.
- Arm A (control) and arm B (box-guided crop) configs differ only in the `arm` field
  (`docs/plan.md` section 6): same backbone, seeds, epochs, optimizer, schedule, resolution, batch
  size, loss and evaluation transform. The arm selects the training crop sampler and nothing else.
- Hyperparameters and the checkpoint are chosen on val only. cis test and trans test are scored
  once, at the end, by the user's go-ahead.
- Notebooks are committed with outputs cleared (Kaggle: Edit > Clear all outputs, or
  `jupyter nbconvert --clear-output --inplace <nb>.ipynb`). The guard refuses any embedded image,
  including plots; save figures as PNG under `docs/figures/` instead (40 files, 20 MB budget).

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
- Every run sets and logs its seed, and writes one ledger row (config, seed, epochs, best epoch,
  scheduler, val score, checkpoint hash). The ledger row is what gets reviewed.
- Hyperparameters and the checkpoint are chosen on val only. cis test and trans test are scored
  once, at the end, by the user's go-ahead.
- Notebooks are committed with outputs cleared (Kaggle: Edit > Clear all outputs, or
  `jupyter nbconvert --clear-output --inplace <nb>.ipynb`). The guard refuses any embedded image,
  including plots; save figures as PNG under `docs/figures/` instead (40 files, 20 MB budget).

---
description: Run one epoch on a tiny fixture batch to verify the training pipeline end to end
---
Run the dry-run:
`uv run python -m trailcam.train.train --max-epochs=1 --batch-size=2 --dry-run=True`

If `trailcam/train/train.py` does not exist yet, report that and STOP. Do not create it.

If the run completes:
1. Show the loss values, the output shape of the final layer and the ledger row it wrote.
2. STOP. Do NOT start a real training run; those happen in the Kaggle notebook, started by the user.

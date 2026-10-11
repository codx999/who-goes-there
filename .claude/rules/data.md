---
paths:
  - "trailcam/data/**"
  - "trailcam/evaluate/**"
  - "data/**"
---
# Data and Dataset Rules

## Verification safeguards
- Print counts, shapes, dtypes and class distributions only. Never print raw metadata rows, file
  listings of `data/`, or image file names in bulk.
- Never open, display or send an image from `data/` unless the user asks for that specific image.
  Demo samples live in the allow-listed `assets/` folder only.
- Split code is tier 3 (highest stakes). Every change to it needs the leakage tests in
  `tests/` to pass: no location in both train and trans test, no sequence id in two splits, no
  image id in two splits.
- The class-to-index map is written once in `trailcam/data/` and imported everywhere else; never
  redefine it inline.
- Val and test transforms resize so the shorter side is 256, centre crop 224 and normalise with
  the ImageNet mean and std (plan.md sections 5 and 2), never a direct resize to 224 by 224; augmentation
  belongs to the training transform alone.
- Bounding boxes are training-time augmentation (arm B) and attention scoring only; never a model
  input. The box-guided crop sampler lives in `trailcam/data/boxcrop.py`, is used by the arm B
  training transform only, and must never be reachable from the eval (val or test) transform.
- Generated figures go in `docs/figures/` and must never contain raw dataset images; demo samples
  (a few trans-test photos) go in `assets/`. Both folders have a file-count and size budget in
  `scripts/guard_data.py`.

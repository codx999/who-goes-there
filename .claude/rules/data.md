---
paths:
  - "trailcam/data/**"
  - "trailcam/evaluate/**"
  - "data/**"
  - "tests/fixtures/**"
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
- Val and test transforms are resize and normalize only; augmentation belongs to the training
  transform alone.

# Setup review against the course guideline

Written 2026-10-10 after reading `docs/HANDOFF.md`, `docs/setup-playbook.md`,
`docs/course-guideline.md` and the lecture repo, and after the CCT20 metadata inspection
(`docs/data-inspection.md`). Input for the design session that writes `docs/plan.md`.

## Already covered by the handoff and playbook

PyTorch stack, Gradio demo, a test set the model never saw (cis and trans), standard metrics with a
confusion matrix, the two-layer minimum, Kaggle for training, data never in git, tiered review.

## Gaps the guideline adds (fold into plan.md, progress.json and invariants.md)

1. **Hugging Face Spaces deployment.** Add a `space/` folder mirroring the course's chapter 09
   demo: `app.py`, `model.py` (or an import from `trailcam`), `requirements.txt`, `examples/`.
   The weights must be in the Space, but the data guard blocks `.pt`/`.pth` in this repo: upload
   weights to the Space with a deploy script using `huggingface_hub`, never through GitHub.
   Spaces free tier is CPU only, so prefer a small backbone (EfficientNet-B0, ResNet-18 or
   MobileNetV3). Hugging Face account and Space creation are owner=user items.
2. **Inference timer.** A predict function timed with `time.perf_counter()` around the forward pass
   (after one warm-up call), shown in the Gradio UI in seconds. Add an invariant row.
3. **Early stopping or LR scheduling** is required. Put it in the training rules; the ledger row
   records epochs run, best epoch and scheduler.
4. **DataLoader rules.** Shuffle on the training loader only; val and test loaders never shuffle.
   Invariant row.
5. **Justification notes.** A "decisions and why" section in plan.md for architecture, loss
   (cross-entropy vs class weights vs label smoothing under heavy imbalance), optimizer (AdamW vs
   SGD) and epoch count, so the slides can quote it.
6. **Problem statement.** An explicit paragraph: classifier, input (one camera-trap JPEG), output
   (one of the species classes), core challenge (generalising to camera locations never seen in
   training).
7. **`requirements.txt` exception.** The playbook bans pip; Spaces needs a `requirements.txt`.
   Generate it from uv (`uv export --no-dev --no-hashes`) or hand-pin a minimal list, and note the
   exception in CLAUDE.md.
8. **Risk regex.** Add `^space/` (public app that loads weights) next to `^trailcam/`.
9. **Submission format.** The Space (app.py, weights, requirements.txt) is now a known deliverable;
   whether a report, notebook or video is also required is still open.

## Data decisions from the inspection (defaults the user accepted unless changed)

- **Empty class:** train has zero `empty` images, so drop the class; filter empties out of val and
  test. Task becomes species-only.
- **Rare classes:** drop badger (9 train), fox (5 train) and deer (42 train, 0 trans test).
  Result: 12 species classes. (Alternative: merge into `other`.)
- **Multi-species images:** drop the 78 images with more than one distinct species.
- **Re-split:** the official train and cis_val files share 224 sequences, so pool them and re-split
  by `seq_id` (stratified by class if practical). cis_test, trans_val and trans_test are clean and
  stay as published. trans_val (one camera, 10 classes) is a sanity check only.
- **Headline metric:** accuracy and macro-F1 side by side, cis vs trans.
- **Class map:** category ids are non-contiguous (1, 3, 5, ... 99); write an explicit
  name-to-index map once.
- **Preprocessing on Kaggle:** resize the 1024 px images once to ~256 px and cache, before training.

## Model and session policy reminders

Implementer Opus 5.5 at high (xhigh for split code and final scoring); the bootstrap in this
session ran on Fable 5.1, so Tier 2 risk review of anything written here uses a second Opus run.
One item per session; the design document gets its own session.

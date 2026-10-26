# Who Goes There: slide outline and speaker notes

Items 12.01, 13.16 and 13.18. The outline of the capstone talk (presentation 2026-11-04) for the
AI and machine-learning bootcamp, built on the course handout ([`course-guideline.md`](course-guideline.md)):
the classifier, the dataset, the architecture with its loss, optimizer and training schedule, the
DataLoader, test metrics, the inference timer, the Gradio UI and the Hugging Face Space. The story
is how I built it, what I tried and what worked. One section per slide, each with a target time,
what goes on the slide, the figure and the speaker notes. The deck is the Slides artifact
<https://claude.ai/artifact/QGeyZXWLbB2AvL2keNczqR>, built from this file. Likely questions and
short answers are in [`talk-qa.md`](talk-qa.md).

Nothing here is claimed as original: the problem, the split and every improvement I tried come
from published research or standard practice, credited where they appear (slide 10's footnote).

**Where the numbers come from.** Phase-1 results from [`ledger.md`](ledger.md),
[`test-ledger.md`](test-ledger.md) and [`plan.md`](plan.md) section 7; phase 2 from
[`phase2-plan.md`](phase2-plan.md) section 12; phase 3 from [`phase3-plan.md`](phase3-plan.md)
section 13 and [`test-ledger-phase3.md`](test-ledger-phase3.md); dataset counts from
[`data-card.md`](data-card.md) and plan section 2; the training recipe from plan sections 4 and 5
and `config.yaml`; run times and the Space's readings from [`progress-log.md`](progress-log.md);
the papers' numbers from [`research-report.md`](research-report.md) Part 3. Headline claims use
the mean of 3 seeds.

**Data citation**, on every slide that shows a dataset photo or a chart of the data (marked
"cite"): Caltech Camera Traps (CCT20), Beery, Van Horn, Perona, "Recognition in Terra Incognita",
ECCV 2018. Data licence CDLA-Permissive 1.0. Code MIT.

**Words.** "I built", "I applied", "I tested a method proposed by", "I learned"; never a claim to
have invented a method.

Terms: **familiar cameras (cis)** = cameras seen in training; **new cameras (trans)** = cameras
never seen. **Gap** = cis minus trans.

## Timing

| slide | title | minutes | running total |
| --- | --- | --- | --- |
| 1 | Who Goes There | 1 | 1 |
| 2 | The problem | 2 | 3 |
| 3 | The dataset | 2 | 5 |
| 4 | Exploring the data | 2 | 7 |
| 5 | The data pipeline | 3 | 10 |
| 6 | Model and training choices | 4 | 14 |
| 7 | Training | 2 | 16 |
| 8 | Baseline results | 3 | 19 |
| 9 | Why it fails | 2 | 21 |
| 10 | Improving it | 4 | 25 |
| 11 | Deployment | 1 | 26 |
| 12 | Live demo | 3 | 29 |
| 13 | Challenges and lessons | 2 | 31 |
| 14 | Future work, thank you, questions | 1 | 32 |
| B1 to B4 | backup, for questions only | | |

The brief's per-slide targets (2026-10-25) summed to 36; to keep the 32-minute total, slides 2, 3
and 4 take 2 minutes instead of 3, and the demo 3 instead of 4. If running long on the day, drop
the demo's night coyote first.

---

## Slide 1. Who Goes There (1 min)

**On the slide.** Title "Who Goes There"; subtitle "A camera-trap species classifier that has to
work on cameras it has never seen"; name; the program, "AI and Machine Learning: Code, Train, &
Deploy", capstone, 2026-11-04; the Space URL
<https://huggingface.co/spaces/Codx999/who-goes-there>; the night coyote,
`assets/trans_coyote_night_05.jpg` (cite).

**Speaker notes.** Who I am in one sentence. What I built: a PyTorch image classifier that names
the animal in a camera-trap photo, trained on Kaggle and deployed as a Gradio app on Hugging Face
Spaces. Ask the room: who is in this photo? (A coyote, at night, in infrared.)

---

## Slide 2. The problem (2 min)

**On the slide.**

- Camera traps: fixed, motion-triggered cameras, day photos and infrared night shots.
- The bottleneck is labelling: 3.2 million images in Snapshot Serengeti (Norouzzadeh et al. 2018);
  4.8 million in one Idaho collection (Beery, Morris, Yang 2019).
- The task: **a classifier**. Input: one photo. Output: one of 12 classes.
- The core challenge: **work on cameras the model has never seen**. Every camera has a fixed
  background, so a model can learn "this rock means coyote". Beery et al. 2018 measured it: 19.1%
  error on familiar cameras, 41.0% on new ones.

**Speaker notes.** Ecologists move cameras all the time, so the number that matters is the one on
new cameras. That challenge, and the split that measures it, come from Beery et al. 2018; I built
my project around reproducing it and then trying to shrink it.

---

## Slide 3. The dataset (2 min)

**On the slide.**

- **Caltech Camera Traps, CCT20** (Beery et al. 2018): 57,864 images from 20 camera locations in
  the American Southwest; 10 cameras for training, 10 never seen.
- **12 classes**: 11 animals plus `car`. **4 dropped**: `empty` (zero images in the official train
  split), badger (9 training images), fox (5) and deer (42, none in the new-camera test set).
  53,535 images kept.
- Day and night: `assets/trans_coyote_day_04.jpg` beside `assets/trans_coyote_night_05.jpg` (cite).
- **Not in `torchvision.datasets`**, so it is loaded with a custom `torch.utils.data.Dataset`,
  exactly as the course's chapter 04 teaches; everything downstream is the course's pipeline.

**Speaker notes.** Why CCT20 anyway: it comes with a published split by camera (Beery et al.
2018), which is what makes "new cameras" measurable; the `torchvision` datasets I looked at have
no such split.
The custom Dataset reads the COCO-style metadata (`trailcam/data/metadata.py`) and returns an image
tensor and a class index; the class-to-index map is defined once, in `trailcam/data/classes.py`.
Images and metadata never enter git. Dropping rare classes follows Schneider et al. 2020 (about
1,000 images per class as a practical floor).

---

## Slide 4. Exploring the data (2 min)

**On the slide.** Three charts of the data, from `docs/figures/` (cite):

- `class_counts.png`: **imbalanced**. Opossum is 20% of the training images; opossum and raccoon
  are 51% of the new-camera test set.
- `day_night.png`: **mostly night**. About two in three images are infrared (0.657); the
  new-camera test set is the most night-heavy (0.716, against 0.600 to 0.626 for the familiar
  splits).
- `camera_locations.png`: **uneven cameras**. Images per camera, familiar against new.

**Speaker notes.** What the exploration changed: imbalance means accuracy alone would reward
guessing opossum, so I report macro-F1 beside it everywhere; infrared means I added random
grayscale to the training augmentation; uneven cameras mean a few cameras dominate, which is why
results are reported for familiar and new cameras separately. Numbers from `data-card.md`.

---

## Slide 5. The data pipeline (3 min)

**On the slide.**

| split | images | cameras | used for |
| --- | --- | --- | --- |
| train | 13,587 | 10 familiar | fitting |
| val | 2,394 | the same 10 | choosing epochs, models, every setting |
| cis_test | 14,478 | the same 10, other times | the final score, familiar cameras |
| trans_val | 1,603 | 1 new | a sanity check on a new camera |
| trans_test | 21,473 | 9 new | the final score, new cameras: the headline |

- **Preprocessing**: images resized once on Kaggle to 512 px (a cache); every model sees 224 x 224,
  normalised with ImageNet's mean and standard deviation.
- **Augmentation, training only**: random resized crop (scale 0.65 to 1.0), horizontal flip, mild
  colour jitter, random grayscale (p 0.1). Val and test: resize to 256, centre crop 224.
- **DataLoader**: batch 64, shuffle on the training loader only, 4 workers.

**Speaker notes.** Train and val come from the same 10 cameras and are split by burst (a camera
fires 3 frames at once; all three go to the same split), because the official train and val files
share 224 bursts, which would leak near-identical frames. Tests in `tests/test_split.py` check that
no camera, burst or image is in two splits. The test sets were scored only at the end. Augmentation
never touches val or test, so the scores measure the model, not the augmentation.

---

## Slide 6. Model and training choices (4 min)

**On the slide.** Each choice with its one-line reason (plan sections 4 and 5, `config.yaml`):

| choice | what | why |
| --- | --- | --- |
| baseline | small CNN from scratch (the course's TinyVGG pattern, 3 conv blocks) | shows what pretraining buys |
| main model | ResNet-18, ImageNet-1k weights (torchvision), new 12-way head | few training images, so borrowed features; 11.7M parameters, fast on a CPU Space |
| loss | cross-entropy | the standard loss for single-label classification |
| optimizer | AdamW; body 1e-4, head 1e-3, weight decay 1e-2 | the new head learns fast while the pretrained body moves gently |
| first epoch | head only | the random head settles before it pushes gradients into the body |
| learning rate | cosine schedule after a 1-epoch warmup | large steps early, small steps late |
| epochs | at most 10 (15 for the small CNN) | enough on a T4 within the GPU budget |
| early stopping | on val macro-F1, patience 3; best checkpoint restored | stops before overfitting, and selects on the metric that counts every class |
| metric | accuracy **and** macro-F1 | the classes are imbalanced |
| hardware | Kaggle, one T4 GPU, mixed precision, batch 64 | free, with a known weekly quota |

**Speaker notes.** This is the course's transfer-learning recipe (chapter 06) with the standard
settings; none of it is new, and every choice can be justified from the dataset: few images per
class, imbalance, a CPU deployment. The small CNN trains from scratch, so body and head share the
1e-3 rate. `/review-shapes` printed each model's layer and shape table before any run.

---

## Slide 7. Training (2 min)

**On the slide.** There is no training-curve figure in the repo; the per-epoch scores are in the
Kaggle logs and the best epochs in the ledgers. So, a short table:

| model | epochs run | best epoch | time per epoch | per run |
| --- | --- | --- | --- | --- |
| small CNN, seed 0 | 15 | 14 | about 44 s | 12.8 min |
| ResNet-18, 3 seeds | 10 each | 9, 8, 8 | about 42 s | about 8.5 min |
| ConvNeXt-Tiny + copy-paste (the final model), 3 seeds | 10 each | 9, 10, 8 | about 67 to 76 s | about 11 to 13 min |

- Every run logged as one ledger row: config, seeds, scores and the checkpoint's hash.

**Speaker notes.** Say plainly that I did not plot training curves. The small CNN was still
improving at its 15-epoch cap; two of the final model's three runs peaked at the last of 10
epochs, a note for future work. The ResNet-18 seeds peaked at epochs 8 or 9, so early stopping
with patience 3 never fired before the cap. Times from the Kaggle logs, as recorded in
`progress-log.md` and `phase3-plan.md` section 13.

---

## Slide 8. Baseline results (3 min)

**On the slide.** The test sets, scored once at the end of phase 1 (plan section 7):

| model | familiar cameras: accuracy | familiar: macro-F1 | new cameras: accuracy | new: macro-F1 |
| --- | --- | --- | --- | --- |
| small CNN, seed 0 | 0.5026 | 0.4504 | 0.3140 | 0.2041 |
| ResNet-18, mean of 3 seeds | 0.6708 | 0.6216 | 0.5137 | 0.4127 |

- Pretraining buys about 20 points on new cameras.
- **The gap reproduced**: on new cameras the ResNet-18 loses 0.157 of accuracy and 0.209 of
  macro-F1.

**Figure.** `docs/figures/test_trans_confusion_matrix_arm_a_seed0.png`, the new-camera confusion
matrix (cite).

**Speaker notes.** Read two cells off the matrix: bobcat called cat (a look-alike) and raccoon
called opossum (opossum is the largest training class). A surprise: val accuracy was 0.87 but
familiar-camera test accuracy 0.67, on the same cameras; val came from the training weeks, the test
from other times, so val was optimistic about time as well. No leakage alarm fired.

---

## Slide 9. Why it fails (2 min)

**On the slide.** Grad-CAM (Selvaraju et al. 2017, hand-written in `trailcam/evaluate/gradcam.py`)
shows which pixels drove the prediction:

- `docs/figures/gradcam_overlay_01.png`: a bobcat called cat (0.79); the heat is on the animal.
- `docs/figures/gradcam_overlay_08.png`: an opossum called coyote (0.22); the heat is on the
  background (cite).
- Blank the animal out and the ResNet-18 still names the class far above chance: accuracy falls
  from 0.5754 to 0.2470, chance is 0.083. **The background is carrying predictions.**

**Speaker notes.** Two kinds of failure: the bobcat is wrong for a good reason, the opossum for a
bad one. Geirhos et al. 2020 call this shortcut learning. The share of Grad-CAM heat inside the
animal's box was 0.2315, against 0.137 for an even map. The overlays are from the phase-1 Space
model (ResNet-18 with box-guided crops, seed 2); the blanking numbers are the baseline's, mean of 3
seeds, on images whose box lies inside the centre crop.

---

## Slide 10. Improving it (4 min)

**On the slide.** What I tried, in order, against the ResNet-18 baseline:

| step | what I tried | result on new cameras | kept? |
| --- | --- | --- | --- |
| baseline | ResNet-18 | test accuracy 51% (0.5137) | the starting point |
| 1 | box-guided training crops | test accuracy +2.5 points, on every seed | small gain; served on the first Space |
| 2 | 320 px input, head-first training, ResNet-50 | small or inconsistent: camera check +0.016 macro-F1 at 320 px (3 seeds); ResNet-50 fell on seed 1 | no |
| 3 | copy-paste animals onto empty frames | camera check +0.097 macro-F1, but familiar cameras -0.030 | no, not on ResNet-18 |
| 4 | ConvNeXt-Tiny backbone | camera check +0.046 macro-F1 | yes |
| 5 | ConvNeXt-Tiny + copy-paste | camera check +0.103 macro-F1; then **test accuracy 66% (0.6598)** | yes: the final model |

- One line: the test sets stayed untouched until the end and were used twice in all: once for
  phase 1, once for the final model only; the second use is disclosed.

**Footnote (credits).** Box crops: Beery et al. 2018. Copy-paste: Gao et al. 2023; Mesnage et al.
2025. ConvNeXt: Liu et al. 2022. Grad-CAM: Selvaraju et al. 2017. Standard practice: higher
resolution, head-first training, ResNet-50.

**Speaker notes.**

- "Camera check" is trans_val, the one new camera I could look at while choosing; changes are
  means of 3 seeds where 3 were run, against the baseline seed by seed. A change was kept only if
  it lifted the new camera by more than 0.03 on every seed without costing the familiar cameras
  more than a set amount, a rule written down before the runs (backup B4).
- Copy-paste is the most instructive: on ResNet-18 it traded familiar cameras for new ones, about
  1 point for 3. On ConvNeXt the trade remained, but ConvNeXt's own gain paid for it, so the
  familiar cameras did not drop (+0.0063).
- The final model, mean of 3 seeds: new-camera accuracy 0.5137 to 0.6598, macro-F1 0.4127 to
  0.5367; the accuracy gap shrank from 0.1571 to 0.0967. The macro-F1 gap barely moved (0.2089 to
  0.1874) because the familiar cameras improved too. Two changes at once, so the test cannot say
  which did the work. Full table: backup B1.

---

## Slide 11. Deployment (1 min)

**On the slide.**

- **Gradio app** (`space/app.py`): upload a photo; it shows the top class with its confidence, the
  two runners-up, an optional Grad-CAM heat map and the **inference time in seconds**.
- **Inference timer**: one warm-up pass, then the timed forward pass (`trailcam/demo/predict.py`).
- **Hugging Face Space**: `app.py`, `requirements.txt`, the weights uploaded by
  `scripts/deploy_space.py` (never through git). CPU only: about 0.3 to 0.6 s a click, Grad-CAM on
  included.
- Switched on 2026-10-21 from the ResNet-18 to the improved model (ConvNeXt-Tiny + copy-paste,
  seed 2, chosen on val).

**Speaker notes.** The time shown is the forward pass only; the click's wall time adds the
upload and the image preparation. The ResNet-18 is the documented rollback.

---

## Slide 12. Live demo (3 min)

**On the slide.** The Space, <https://huggingface.co/spaces/Codx999/who-goes-there>, and beside it
the after-picture of slide 9's two photos under the final model (cite):

- `docs/figures/p3_gradcam_overlay_01.png`: the bobcat, now bobcat (0.84).
- `docs/figures/p3_gradcam_overlay_08.png`: the opossum, still wrong (squirrel, 0.39), but the
  heat is now on the animal.

The script, saying before each click what I expect:

1. **Night coyote**, `trans_coyote_night_05`: coyote, 100% on the Space (2026-10-21). Optional;
   drop it first if short of time.
2. **Bobcat with Grad-CAM on**, `trans_bobcat_day_01`: bobcat 84%, cat 16% on the Space
   (2026-10-21).
3. **The opossum, an honest failure**, `trans_opossum_day_10`: squirrel (0.39) in the scoring's
   Grad-CAM run of the same weights.
4. **The live sort** (item 12.05): about 200 new-camera photos from a folder outside the repo,
   sorted into class folders with a visible timer and per-class counts:

```bash
uv run python -m trailcam.demo.sort --src <the demo photos> --dst <a new folder> --checkpoint <seed-2 checkpoint>.pt --config-name p3_convnext_cp
```

**Fallbacks, in order.** (1) The Space: it sleeps when idle, so wake it 10 minutes before the talk
and run one photo. (2) The local app's `--share` link:

```bash
uv run python -m trailcam.demo.app --checkpoint <seed-2 checkpoint>.pt --config-name p3_convnext_cp --share
```

(3) The recorded video of the Space and the sort (item 12.04, the user's).

**Speaker notes.** A wrong answer you predicted is a better demo than a right one you hoped for.
The confidence is the model's own estimate, not a calibrated probability. Rehearse on the Space
and write down each photo's answer; the opossum's Space reading is not logged yet. The final model
gets 8 of the 10 `assets/` photos right; the other wrong one is the raccoon by day, called cat
(0.92). The checkpoint's file name and sha256 are in `progress-log.md`, 2026-10-21; the local tools
default to the ResNet-18's config, so `--config-name p3_convnext_cp` is required. The live sort's
counts are per predicted class, not a score (the script reads no labels).

---

## Slide 13. Challenges and lessons (2 min)

**On the slide.**

| challenge | what I did, and the lesson |
| --- | --- |
| CCT20 is not in `torchvision.datasets` | a custom Dataset (chapter 04); the rest of the course pipeline unchanged |
| the official train and val files share 224 bursts | re-split by burst, with tests; check the data, not only the paper |
| Kaggle's torch build has no P100 kernels; a run stuck at "Starting..." for 48 minutes | moved to a T4; cancel and resubmit after 10 empty minutes |
| val said 0.87, the familiar-camera test 0.67 | val from the training weeks flatters the model; hold out a later period |
| one seed or one camera pointed the wrong way more than once | 3 seeds for every claim; one validation camera is not enough |
| most ideas did not help (5 of 8 dropped) | measure each one against a fixed baseline, one change at a time |

**Speaker notes.** The biggest lesson: start with a plain baseline and change one thing at a
time; the improvements came from published ideas, and the work was in testing them fairly.

---

## Slide 14. Future work, thank you, questions (1 min)

**On the slide.**

- **Detect, then classify**: MegaDetector (Beery, Morris, Yang 2019) finds the animal first; Beery
  et al. 2018 found box crops more than halve the error.
- **Rare species**: no rodent right on the new cameras (0 of 58).
- **More cameras** for training and validation; **burst voting** over a camera's 3 frames (Dussert
  et al. 2024); **calibrated confidence** (Guo et al. 2017).
- Thank you. Questions? Data: CCT20, Beery et al. 2018, CDLA-Permissive 1.0 · code MIT · the
  Space and repo links.

**Speaker notes.** Each next step costs something: a detector means two models at inference,
burst voting needs the bursts. Then questions; answers in `talk-qa.md`.

---

## Backup slides (for questions only)

### B1. The full experiment table

Macro-F1 change against the ResNet-18 baseline, paired by seed (`phase2-plan.md` section 12,
`phase3-plan.md` section 13):

| change | val (familiar cameras) | trans_val (one new camera) | kept? |
| --- | --- | --- | --- |
| 320 px input, 3 seeds | +0.0100 (mean) | +0.0158 (mean) | no: under the 0.03 bar |
| head-first, 3 epochs, seed 0 | -0.0078 | +0.0177 | no: val fell |
| ResNet-50, seeds 0 and 1 | +0.0128, +0.0154 | +0.0197, -0.0219 | no: not up on every seed |
| copy-paste p 0.5, seed 0 | -0.0304 | +0.0971 | no: val fell |
| copy-paste p 0.25, seed 0 | -0.0151 | +0.0493 | no: val fell |
| ConvNeXt-Tiny, 3 seeds | +0.0245 (mean) | +0.0457 (mean) | yes |
| ConvNeXt-Tiny + copy-paste, 3 seeds | +0.0063 (mean) | +0.1028 (mean) | yes: final |

Box-guided crops (phase 1) were judged on the test sets: trans_test accuracy +3.0, +1.1 and +3.3
points, macro-F1 +2.9, +1.2 and +1.3 (seeds 0, 1, 2). The two kept changes cannot be told apart on
the test sets; on trans_val, +0.0457 for ConvNeXt and +0.0571 for copy-paste on top.

### B2. Final model: confusion matrix and per class

`docs/figures/p3_test_trans_confusion_matrix_seed2.png` and
`docs/figures/p3_test_trans_per_class_seed2.png` (cite). The rare species (bird, rodent, skunk)
score 0.43 macro-F1 on the new cameras against 0.63 for the common ones (means of 3 seeds,
`p3_slices_test.md`); rodent is at zero.

### B3. Day vs night

New-camera test accuracy by light, seed 0 (plan section 7; `docs/figures/slices_test.md`):

| model | day | night |
| --- | --- | --- |
| small CNN | 0.2961 | 0.3210 |
| ResNet-18 | 0.5315 | 0.5058 |
| ResNet-18 with box-guided crops | 0.5618 | 0.5362 |

Night is a few points harder; re-weighting the new-camera test to the familiar cameras' night
share changes the gap by about 0.01 or less, so day/night does not explain it.
`docs/figures/day_night.png` (cite).

### B4. How changes were judged, and the test sets protected

- Every change was one change from the same baseline, written down with its expected result before
  the run, and kept only if trans_val macro-F1 rose by more than 0.03 on the mean of 3 seeds and on
  every seed, with val down by at most 1 point (phase 2) or 3 points (phase 3).
- A control run re-trained the baseline on new code first and matched phase 1 (bit for bit in
  phase 2, to the fourth decimal in phase 3).
- The test sets: scored once at the end of phase 1 (all 7 checkpoints), then once for the final
  model's 3 seeds, after phase 1's scores had been seen (2026-10-18); disclosed wherever the final
  numbers appear, and shown beside phase 1's, never in their place.

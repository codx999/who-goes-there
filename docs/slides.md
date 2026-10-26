# Who Goes There: slide outline and speaker notes

Items 12.01, 13.16 and 13.18. The outline of the capstone talk (presentation 2026-11-04) for the
AI and machine-learning bootcamp, built on the course handout ([`course-guideline.md`](course-guideline.md)):
the classifier, the dataset, the architecture with its loss, optimizer and training schedule, the
DataLoader, test metrics, the inference timer, the Gradio UI and the Hugging Face Space. The story
is how I built it, what I tried and what worked. One section per slide, each with a target time,
what goes on the slide, the figure and the speaker notes. The deck is the Slides artifact
<https://claude.ai/artifact/QGeyZXWLbB2AvL2keNczqR>, built from this file. Likely questions with
short answers, and the spoken script, are kept on the presenter's machine, not in the repo.

Nothing here is claimed as original: the problem, the split and every improvement I tried come
from published research or standard practice, credited where they appear (slide 10's footnote).

**Where the numbers come from.** Phase-1 results from [`ledger.md`](ledger.md),
[`test-ledger.md`](test-ledger.md) and [`plan.md`](plan.md) section 7; phase 2 from
[`phase2-plan.md`](phase2-plan.md) section 12; phase 3 from [`phase3-plan.md`](phase3-plan.md)
section 13 and [`test-ledger-phase3.md`](test-ledger-phase3.md); dataset counts from
[`data-card.md`](data-card.md) and plan section 2; the training recipe from plan sections 4 and 5
and `config.yaml`; run times and the Space's readings from the project's work log (kept locally);
the papers' numbers from [`research-report.md`](research-report.md) Part 3. Headline claims use
the mean of 3 seeds.

**Data citation**, on every slide that shows a dataset photo or a chart of the data (marked
"cite"): Caltech Camera Traps (CCT20), Beery, Van Horn, Perona, "Recognition in Terra Incognita",
ECCV 2018. Data licence CDLA-Permissive 1.0. Code MIT.

**Footers.** Each slide's footer carries only its number and citations the audience can use (the
data citation, papers, the code link on slide 11), never repository paths; the repository sources
of every number are in each slide's speaker notes and in this file.

**Colour key.** One meaning per colour, on every slide and chart:

- **Blue** (`#2F5D8A`) = new cameras, the main result; lighter blues (`#6F93B5`, `#9FB6CC`,
  `#DCE6EF`) for earlier steps or panels about new cameras.
- **Green** (`#6B8F7A`) = familiar cameras.
- **Orange** (`#B04E15`) = attention: section labels, the problem (the background shortcut, rare
  species, what falls), the training-only parts of a diagram and the keep bar. Never a camera
  group.
- **Neutral**: dark ink (`#14231C`) for text, dark boxes and night; grey (`#9AA39E`) for plain
  counts and the guessing baseline; sand (`#CDB98A`) for day.
- Figures saved as images (heat maps, B2's matrix) keep their own colours.

**Words.** "I built", "I applied", "I tested a method proposed by", "I learned"; never a claim to
have invented a method.

Terms: **familiar cameras (cis)** = cameras seen in training; **new cameras (trans)** = cameras
never seen. **Gap** = cis minus trans. The slides and the script say only "familiar" and "new";
the code names (cis_test, trans_val, trans_test) stay in this file and the speaker notes.

## Timing

| slide | title | minutes | running total |
| --- | --- | --- | --- |
| 1 | Who Goes There | 1 | 1 |
| 2 | The problem | 2 | 3 |
| 3 | The dataset | 2 | 5 |
| 4 | Exploring the data | 2 | 7 |
| 5 | The data pipeline: from photo to final score | 3 | 10 |
| 6 | Model and training choices | 4 | 14 |
| 7 | Baseline results | 3 | 17 |
| 8 | Why it fails | 2 | 19 |
| 9 | What I tried | 2 | 21 |
| 10 | What worked | 4 | 25 |
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

**On the slide.** A small line "Capstone · 2026-11-04"; title "Who Goes There"; subtitle "A
camera-trap species classifier that has to work on cameras it has never seen"; then three short
lines: "Cody" (bold), the program "AI and Machine Learning: Code, Train, & Deploy", and the Space
URL as a clickable link <https://huggingface.co/spaces/Codx999/who-goes-there>; the text column
spans the photo's height; the night coyote,
`assets/trans_coyote_night_05.jpg` (cite).

**Speaker notes.** Say hi; no formal intro. What I built: a PyTorch image classifier that names
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
  background, so a model can learn the background ("this camera's rock means coyote") instead of
  the animal.
- Beery et al. 2018 measured the gap, shown as two big numbers: **81%** right on familiar cameras
  (green), **59%** right on new cameras (blue) (their Inception-v3 classifier's 19.1% and 41.0% top-1 error, as
  accuracy: 100 minus the error); "same model, same classes; about 22 points lost on new cameras".

**Speaker notes.** Ecologists move cameras all the time, so the number that matters is the one on
new cameras. That challenge, and the split that measures it, come from Beery et al. 2018; I built
my project around reproducing it and then trying to shrink it.

---

## Slide 3. The dataset (2 min)

**On the slide.** Title "Caltech Camera Traps, CCT20"; a fact line **57,864 photos · 20 cameras**
in the American Southwest **· 16 categories**; then "Why this dataset" (cite):

- **It measures the real challenge:** a published split by camera, so "new cameras" can be tested.
  No torchvision dataset has one.
- **Real-world photos:** night infrared, blur, animals half in frame, uneven species.
- **A published reference:** the gap Beery et al. measured on this exact data (slide 2; no new
  numbers here).
- **Open:** free to use (CDLA-Permissive 1.0).
- **The trade-off** (grey box): not in `torchvision.datasets`, so I wrote a custom PyTorch
  `Dataset`. (Said aloud: the way the course's chapter 04 teaches custom datasets.)
- Day and night: `assets/trans_coyote_day_04.jpg` beside `assets/trans_coyote_night_05.jpg`,
  captioned "Coyote by day: colour" and "Same species at night: infrared, grey".

Which categories were kept is decided on slide 4, after the data has been shown.

**Speaker notes.** The custom Dataset reads the COCO-style metadata (`trailcam/data/metadata.py`)
and returns an image tensor and a class index; the class-to-index map is defined once, in
`trailcam/data/classes.py`. Everything downstream is the course's pipeline. Images and metadata
never enter git. The 16 categories: 11 animals, car, empty, badger, fox and deer.

---

## Slide 4. Exploring the data (2 min)

**On the slide.** Title "Imbalanced, and mostly at night"; chart heading "Training photos per class, the 12
kept". No image charts (the figures' small
labels and cis/trans wording do not read when projected); instead (cite):

- A bar chart drawn on the slide: training photos per class (`data-card.md`, train column, split
  seed 42): opossum 2,733, rabbit 2,494, coyote 1,437, cat 1,257, squirrel 1,250, raccoon 1,016,
  dog 817, bobcat 799, car 603, bird 509, rodent 440, skunk 232; neutral grey bars, with the three
  rare species (bird, rodent, skunk) in orange, marked "rare". Opossum is 20% of the training photos.
- A big number: **2 in 3** photos are infrared, at night (66%, from 0.657); on the new cameras
  **72%** (0.716, in blue; the familiar splits are 0.600 to 0.626).
- Under the chart, in grey: "Too few to learn (official training file): deer 42 · badger 9 · fox 5 ·
  empty 0" (`plan.md` section 2: empty has no training photos; deer has none in the new-camera
  test; Schneider et al. 2020 put the practical floor near 1,000 per class).
- "What it changed": too few photos → dropped 4 categories, 12 classes and 53,535 photos kept;
  imbalance → macro-F1 beside accuracy, everywhere; night → random grayscale in training; uneven
  cameras → familiar and new cameras reported separately.
- The original charts stay in `docs/figures/` (`class_counts.png`, `day_night.png`,
  `camera_locations.png`) for questions.

**Speaker notes.** What the exploration changed: imbalance means accuracy alone would reward
guessing opossum, so I report macro-F1 beside it everywhere; infrared means I added random
grayscale to the training augmentation; uneven cameras mean a few cameras dominate, which is why
results are reported for familiar and new cameras separately. Numbers from `data-card.md`.

---

## Slide 5. The data pipeline (3 min)

**On the slide.** Eyebrow "The data pipeline", title "From photo to final score". Three numbered
parts, in the order things happen; no table and no metaphors.

**1 · How each photo is prepared**, a one-line flow: **Photo** (JPEG) → **Resize** (224 × 224) →
**Augment** (training only; highlighted) → **Normalise** (ImageNet scale) → **DataLoader** (batches
of 64) → **Model** (12 classes).

**2 · One training run, repeated each epoch · the same for every model**, an architecture diagram
with two rows:

- Row 1, learning: **Training photos** (13,587, in batches) → **Model** (ResNet-18 (baseline) →
  ConvNeXt-Tiny (final)) → **Loss** (cross-entropy: how wrong?) → **Optimizer** (AdamW adjusts the
  weights), with an orange arrow looping back into the model: "update the weights: learning".
- Row 2, validation: **Validation photos** (2,394 · no learning) → the same **Model** → **Score**
  (macro-F1, each epoch) → **Keep the best epoch** (final model: epochs 9, 10, 8 of 10 · early stop
  never fired). No arrow loops back: validation never changes the weights.

**3 · After each training run, then once at the end** (blue = new cameras), in the order it
actually happened:

- Repeats after **every** training run: **Best model** (of this run) → **Check** (1 new camera ·
  1,603) → **Keep or drop the method** (↺ then train the next one, back to part 2). Every ledger
  row has this check score (trans_val) next to its validation score; comparing them against the
  baseline decided which methods to keep (slide 10).
- A dashed divider, then **once**, when every method is decided: **Final model** (ConvNeXt-Tiny +
  copy-paste) → **Test, once** (9 new cameras · 21,473 · main result).

- What happened, from the ledgers: early stopping (patience 3) never fired; every ResNet-18 and
  ConvNeXt run used all 10 epochs (best epochs 8 to 10) and the small CNN all 15 (best 14); the
  deployed model is seed 2's epoch 8. The same pipeline trained every model; the final model
  differs only in the backbone and one extra training-only augmentation, copy-paste (slide 10).
- In the speaker notes only: the familiar-camera test (14,478 photos, the same 10 cameras at other
  times, scored at the same moment as the new-camera test; slide 8); split by burst (a camera fires
  3 near-identical frames at once; they always stay in the same split, so validation can't cheat);
  the full prep settings: a 512 px cache built once on Kaggle; training = random resized crop
  (scale 0.65 to 1.0), horizontal flip, mild colour jitter, random grayscale (p 0.1); validation and
  test = resize to 256, centre crop 224; ImageNet mean and standard deviation; DataLoader batch 64,
  shuffle on the training loader only, 4 workers. Code names: the familiar test is cis_test, the
  check is trans_val, the new test is trans_test.

**Speaker notes.** Train and val come from the same 10 cameras and are split by burst (a camera
fires 3 frames at once; all three go to the same split), because the official train and val files
share 224 bursts, which would leak near-identical frames. Tests in `tests/test_split.py` check that
no camera, burst or image is in two splits. The test sets were scored only at the end. Augmentation
never touches val or test, so the scores measure the model, not the augmentation.

---

## Slide 6. Model and training choices (4 min)

**On the slide.** Title "A pretrained model, and why each choice". No table.

- **The model architecture**, a diagram left to right: **Photo** (224 × 224) → **Pretrained body**
  (ResNet-18 (baseline) → ConvNeXt-Tiny (final); already learned on ImageNet) → **New head** (12
  outputs, one per class; highlighted) → **Prediction** (12 class probabilities). Under the body:
  "learning rate 0.0001: moves gently, keeps what ImageNet taught it". Under the head: "learning
  rate 0.001: learns fast · epoch 1: only the head trains".
- **Four "why" cards:**
  - **Why this model:** few photos per class (skunk: 232), so start from a body that already knows
    edges, textures and shapes; both models answer in under a second on the Space's free
    CPU (the deployed ConvNeXt-Tiny: about 0.1 s a photo, slide 11). (The 20-point evidence, small CNN vs ResNet-18, is on the next slide.)
  - **Loss: cross-entropy:** measures how wrong each prediction is, and punishes confident wrong
    answers most; the standard loss for picking one of 12 classes.
  - **Optimizer: AdamW:** adjusts the weights with two speeds, slow for the body and fast for the
    head; steps grow during the first epoch, then shrink smoothly to the end.
  - **Stopping:** at most 10 epochs; keep the epoch with the best validation macro-F1; early stop
    set, never needed (best epochs 8 to 10 of 10).

**Speaker notes.** This is the course's transfer-learning recipe (chapter 06) with standard
settings; none of it is my invention. The same recipe trained the ResNet-18 baseline and the final
ConvNeXt-Tiny; only the body changes (and the final model adds copy-paste augmentation, slide 10).
Exact settings, if asked (`plan.md` sections 4 and 5, `config.yaml`): AdamW, body learning rate
1e-4, head 1e-3, weight decay 1e-2; 1 warm-up epoch, then cosine; at most 10 epochs (15 for the
small CNN), the first head-only; early stopping on validation macro-F1 with patience 3, best
checkpoint restored; batch 64, mixed precision, one Kaggle T4 GPU; the small CNN is the course's
TinyVGG pattern (3 conv blocks), trained from scratch at 1e-3. `/review-shapes` printed each model's
layer and shape table before any run. Cross-entropy is unweighted; the imbalance is handled by
reporting and selecting on macro-F1.

---

## Slide 7. Baseline results (3 min)

**On the slide.** Title "Good on familiar cameras, much worse on new ones". Percentages only; no
table and no confusion matrix (its labels do not read when projected).

- **A bar chart drawn on the slide**, test accuracy (the share of photos named correctly),
  familiar cameras (green) against new cameras (blue): small CNN, trained from scratch, **50%** vs
  **31%**; ResNet-18, pretrained, mean of 3 runs, **67%** vs **51%**. Under it: "Pretraining adds
  20 points on new cameras, but the gap stays."
- A big number: **−16 points** "on new cameras: the kind of gap Beery et al. found, reproduced."
- **Biggest mistakes on new cameras:** raccoon → opossum (similar grey animals at night); bobcat →
  cat (a look-alike).
- **A surprise:** validation said 87%, but the familiar-camera test said 67%. Same cameras,
  different weeks.

**Speaker notes.** The test sets were scored once, at the end of training the baseline. Exact
numbers (`test-ledger.md`, plan section 7): small CNN, seed 0: familiar 0.5026 accuracy, 0.4504
macro-F1; new 0.3140, 0.2041. ResNet-18, mean of 3 runs: familiar 0.6708, 0.6216; new 0.5137,
0.4127; gap 0.1571 accuracy, 0.2089 macro-F1. The mistakes, read off ResNet-18 seed 0's
new-camera confusion matrix (`docs/figures/test_trans_confusion_matrix_arm_a_seed0.png`): 884 of
5,469 raccoons called opossum; 437 of 2,329 bobcats called cat. Validation accuracy 0.8647 to
0.8676 over the 3 runs (`ledger.md`). Validation came from the training weeks, the test from other
times, so validation was optimistic about time as well as camera. No leakage alarm fired.

---

## Slide 8. Why it fails (2 min)

**On the slide.** Title "The model looks at the background".

- Two Grad-CAM heat maps (Selvaraju et al. 2017; hand-written in `trailcam/evaluate/gradcam.py`),
  from the first deployed model (ResNet-18 with box-guided crops, seed 2) (cite):
  `docs/figures/gradcam_overlay_01.png`, a bobcat called cat, 79% ("Bobcat → cat, 79%"), heat on the animal;
  `docs/figures/gradcam_overlay_08.png`, an opossum called coyote, 22% ("Opossum → coyote, 22%"), heat on the background.
- A bar chart drawn on the slide, "Blank the animal out: does the model still get it right?" (new
  cameras, photos with the animal fully in frame): normal photo **58%**, animal blanked out
  **25%**, blind guessing (1 in 12) **8%**.
- "With no animal visible, it is still right **3 times more often than guessing**: the background
  carries predictions."

**Speaker notes.** Two kinds of failure: the bobcat is wrong for a good reason, the opossum for a
bad one. Geirhos et al. 2020 call this shortcut learning. The chart is the baseline ResNet-18, mean
of 3 runs, on the 16,982 new-camera photos whose animal box lies fully inside the centre crop
(plan section 7): accuracy 0.5754 normally, 0.2470 with the box blanked to the image mean; chance
0.083. The 58% is higher than slide 8's 51% because this subset only has photos with a fully
visible animal. The share of Grad-CAM heat inside the animal's box was 0.2315, against 0.137 for an
even map.

---

## Slide 9. What I tried (2 min)

**On the slide.** Title "What I tried: six ideas, one at a time". Six cards, each with a plain
one-line description and its source (none of the ideas is mine):

- **Aims at the background** (orange): **Box-guided crops**: crop training photos around the
  animal's box, so the animal fills more of the frame (Beery et al. 2018). **Copy-paste**: paste
  the animal onto an empty photo from another camera, so animal and background stop going together
  (Gao et al. 2023; Mesnage et al. 2025).
- **A stronger model:** **ConvNeXt-Tiny**: a modern network design, with the same ImageNet
  pretraining as ResNet-18 (Liu et al. 2022). **ResNet-50**: a deeper version of ResNet-18, with
  more capacity (standard practice).
- **Training details:** **Bigger photos, 320 px**: more pixels for small or distant animals.
  **Head first**: train the new head alone for 3 epochs before the body joins in (both standard
  practice).
- **How each was tested:** run once; if promising, run twice more with different random starts
  ("seeds"), because one run can be lucky. Kept only if it clearly helped on the new camera every
  time.

**Speaker notes.** Results come on the next slide. Box-guided crops: for half of the boxed training
photos, the crop is chosen so the box fills 20% to 100% of it (plan section 6). Copy-paste: for
half of the boxed training photos, the animal's box is pasted onto an empty frame from another
training camera, day onto day and night onto night, never from a test camera (`phase2-plan.md`
section 6). ConvNeXt-Tiny: timm's `convnext_tiny.fb_in1k` (`phase3-plan.md` section 3). ResNet-50:
25.6M parameters against ResNet-18's 11.7M. Seeds fix the random choices (initial head weights,
data order, augmentation). The keep rule was written down before the runs (backup B4). This slide
replaces the earlier "Training" slide; its facts (epochs, best epochs, early stopping, time per run)
are on slides 5 and 6 and in their notes.

---

## Slide 10. What worked (4 min)

**On the slide.** Title "What worked: 51% → 66% on new cameras". Percentages, one measure for the
chart, plain verdicts for the rest; no table.

- **The journey, a bar chart drawn on the slide:** test accuracy on the 9 new cameras, means of 3
  runs: **baseline** ResNet-18 **51%** (0.5137) → **+ box-guided crops**, the first app, **54%**
  (0.5387) → **ConvNeXt-Tiny + copy-paste**, the final model, **66%** (0.6598).
- **Each idea, checked on the new camera** (verdicts only): bigger photos, 320 px: dropped, gain too
  small; head first: dropped, familiar fell; ResNet-50: dropped, not every run; copy-paste on
  ResNet-18: new up, familiar down; ConvNeXt-Tiny: kept; ConvNeXt-Tiny + copy-paste: kept, final.
- "The gap between familiar and new cameras shrank from **16** to **10 points**" (0.1571 → 0.0967;
  the final model's familiar-camera accuracy 0.7565).
- Footer: test sets used twice in all, once for the first models, once for the final model.

**Speaker notes.** The check-camera numbers behind each verdict (trans_val macro-F1 change against
the baseline, paired run by run; `phase2-plan.md` section 12, `phase3-plan.md` section 13): 320 px
+0.016 on 3 runs, under the 0.03 bar; head first −0.008 on familiar cameras; ResNet-50 +0.020, then
−0.022 on its second run; copy-paste on ResNet-18 +0.097 on the new camera but −0.030 on familiar
cameras; ConvNeXt-Tiny +0.046; ConvNeXt-Tiny + copy-paste +0.103, familiar +0.006. The story:
copy-paste helped new cameras but cost familiar ones on ResNet-18; the stronger ConvNeXt body paid
that cost. Macro-F1 on new cameras 0.4127 → 0.5367. Honest limits: two changes at once, so the test
cannot split the credit (check camera: +0.046 for ConvNeXt, +0.057 more for copy-paste); rare
species still weak (rodent 0 of 58). The second use of the test sets came after the first scores had
been seen; it is disclosed. Full numbers: backup B1 (chart). Credits on slide 9.

---

## Slide 11. Deployment (1 min)

**On the slide.** Title "From code to a live app".

- **A system diagram**, five numbered steps left to right: **1 · code: GitHub** (the code: dataset,
  training, app) → **2 · train: Kaggle** (photo cache; training on a free T4 GPU) → **3 · save:
  Weights file** (best run's model, chosen on validation) → **4 · deploy: Hugging Face Space**
  (app.py (Gradio), requirements.txt, the weights · CPU; highlighted) → **5 · use: Your browser**
  (upload a photo → class, confidence, time, heat map).
- **Three cards:** **The app (Gradio):** the top class with its confidence, two runners-up, an
  optional Grad-CAM heat map, and the time taken. **Inference timer:** one warm-up pass, then the
  timed prediction in seconds: about 0.1 s for the model, 0.3 to 0.6 s a click on a free CPU.
  **The live model:** since 2026-10-21, the final ConvNeXt-Tiny + copy-paste; the weights go up by
  script, never through git.
- Footer, both clickable: Code: github.com/codx999/who-goes-there · App:
  huggingface.co/spaces/Codx999/who-goes-there.

**Speaker notes.** These are the course's deployment requirements (`course-guideline.md`): a Gradio
UI with the inference time, deployed with app.py, the weights file and requirements.txt. The timer
measures the forward pass only (`trailcam/demo/predict.py`); measured on the Space on 2026-10-21:
forward pass about 0.08 to 0.14 s, a whole click 0.31 to 0.58 s with Grad-CAM (`progress-log.md`).
The weights are uploaded with `scripts/deploy_space.py` (huggingface_hub, token from the
environment). The ResNet-18 (box-guided crops, seed 2) is the documented rollback.

---

## Slide 12. Live demo (3 min)

**On the slide.** The audience's view only; the photo choice, expected answers and fallbacks are
the presenter's, in the speaker notes and the spoken script.

- Title: the Space's address as a clickable link,
  <https://huggingface.co/spaces/Codx999/who-goes-there>.
- Left: "Photos from cameras it has never seen": 1 · A coyote at night; 2 · A bobcat, with the
  heat map on; 3 · An opossum; 4 · A whole folder: about 200 photos, each filed under its predicted
  animal, with a timer (the live sort, in a terminal on the laptop). No expected answers on the
  face, so the clicks keep their surprise.
- Right, labelled "The app": a screenshot of the Space after classifying the bobcat example
  (`trans_bobcat_day_01`, Grad-CAM on; captured 2026-10-26 with a scripted headless Chrome, kept
  out of git as an artifact asset): bobcat 84%, cat 16%, coyote 0%, 0.151 s forward pass, the heat
  map on the bobcat (the same reading as on 2026-10-21). Caption: "The bobcat the first model
  called a cat: now bobcat, 84%, in 0.15 s". It ties back to slide 8, shows the audience the app
  and a real output before the switch, and stays up if the app is down. Step 2 then shows its
  answer in advance, so it is done live as a confirmation. The
  heat maps of slide 8's photos under the final model (`p3_gradcam_overlay_01.png`, bobcat 84%;
  `p3_gradcam_overlay_08.png`, squirrel 39%, heat on the animal) moved to the notes as the last
  resort.
- Footer: "Model: ConvNeXt-Tiny + copy-paste" and the data credit.

The Space shows the 10 `assets/` photos as clickable examples, so each step is a click on a
thumbnail. The script, saying before each click what I expect:

1. **Night coyote**, `trans_coyote_night_05`: coyote, 100% on the Space (2026-10-21). Optional;
   drop it first if short of time.
2. **Bobcat with the heat map (Grad-CAM) on**, `trans_bobcat_day_01`: bobcat 84%, cat 16% on the Space
   (2026-10-21).
3. **The opossum, an honest failure**, `trans_opossum_day_10`: on the Space (2026-10-26) squirrel 39%,
   bobcat 17%, opossum 14% (third), 0.138 s, heat on the animal; the same 39% as the scoring's
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
and check each photo's answer; all three demo photos' Space readings are now logged here. The final model
gets 8 of the 10 `assets/` photos right; the other wrong one is the raccoon by day, called cat
(0.92). The checkpoint's file name and sha256 are in `progress-log.md`, 2026-10-21; the local tools
default to the ResNet-18's config, so `--config-name p3_convnext_cp` is required. The live sort's
counts are per predicted class, not a score (the script reads no labels).

---

## Slide 13. Challenges and lessons (2 min)

**On the slide.** Title "What went wrong, and what I learned". Six cards in a 3 × 2 grid, each with
a tag, the challenge in bold and a "Lesson:" line, in plain language:

1. **The data:** the dataset is not built into PyTorch. Lesson: a small custom loader is enough;
   everything after it can stay standard.
2. **The data:** the official files leaked: 224 bursts of photos sat in both training and
   validation. Lesson: re-split by burst, with tests that fail on any overlap. Check the data, not
   only the paper.
3. **The GPU:** Kaggle's older GPU could not run its PyTorch, and one run sat 48 minutes without
   starting. Lesson: check the GPU before training (I moved to the T4), and restart a run if
   nothing has begun after 10 minutes.
4. **Testing:** validation said 87%, the familiar-camera test 67%. Lesson: photos from the
   training weeks look too easy. Test on a later period too.
5. **Testing:** a single run, or a single check camera, can point the wrong way. Lesson: three
   runs with different random starts for every claim.
6. **Ideas:** 4 of the 6 ideas did not help on their own. Lesson: same baseline, one change at a
   time, the keep rule written down before the runs.

**Speaker notes.** Tell two or three cards as stories; the GPU one and the validation-versus-test
one land best. The biggest lesson: start with a plain baseline and change one thing at a time; the
improvements came from published ideas, and the work was in testing them fairly. Details: CCT20 is
not in `torchvision.datasets` (custom Dataset, chapter 4); the released train and cis_val files
share 224 sequences, 222 after filtering (`plan.md`, `data-inspection.md`); Kaggle's torch 2.11 has
no sm_60 kernels, so no P100 (2026-10-14); the 48-minute "Starting..." has no known cause, and the
photo cache was packed into one archive (2026-10-15); validation about 0.87 against 0.67; ResNet-50
+0.020 then −0.022 on the check camera. Of slide 9's six ideas, box-guided crops and ConvNeXt-Tiny
were kept on their own; 320 px, head first, ResNet-50 and copy-paste on ResNet-18 were dropped
(copy-paste came back with ConvNeXt-Tiny in the final model). The earlier "5 of 8 dropped" counted
a different list and no longer matches the deck.

---

## Slide 14. Future work, thank you, questions (1 min)

**On the slide.**

- **Detect, then classify**: MegaDetector (Beery, Morris, Yang 2019) finds the animal first; Beery
  et al. 2018 found box crops more than halve the error.
- **Rare species**: no rodent right on the new cameras (0 of 58).
- **More cameras** for training and validation; **burst voting** over a camera's 3 frames (Dussert
  et al. 2024); **calibrated confidence** (Guo et al. 2017).
- Thank you. Questions? Two clickable links: "Try the app:"
  <https://huggingface.co/spaces/Codx999/who-goes-there> and "Read the code:"
  <https://github.com/codx999/who-goes-there>.
- Footer: Data: CCT20, Beery, Van Horn, Perona, ECCV 2018 · CDLA-Permissive 1.0 · code MIT.

**Speaker notes.** Each next step costs something: a detector means two models at inference,
burst voting needs the bursts. Then questions. Leave this slide up
during questions so both links stay on screen; backups B1 to B4 are hidden (skipped when
presenting), to jump to if a question needs one.

---

## Backup slides (for questions only)

Marked hidden in the deck: skipped when presenting, still there to jump to when a question
needs one.

### B1. Every idea, scored against the baseline

**On the slide.** A two-sided bar chart instead of the table: one row per idea, two panels sharing
the rows, "Familiar cameras (validation)" (green) and "The one new camera kept for checking"
(blue), each with a zero line, so a fall goes left and a rise goes right. Values are macro-F1
change in points (× 100); a dashed orange line marks the +3 keep bar on the new-camera panel.
ResNet-50 shows its two runs as two thin bars; the other rows say "1 run" or "mean of 3 runs". A
status column says why each was dropped or kept; the two kept rows are shaded light blue. Below:
"Box-guided crops came first and were scored on the test sets instead: new-camera accuracy up 3.0,
1.1 and 3.3 points over the 3 runs."

The numbers behind it: macro-F1 change against the ResNet-18 baseline, paired by seed (`phase2-plan.md` section 12,
`phase3-plan.md` section 13):

| change | familiar cameras (val) | one new camera (trans_val) | kept? |
| --- | --- | --- | --- |
| 320 px input, 3 seeds | +0.0100 (mean) | +0.0158 (mean) | no: under the 0.03 bar |
| head-first, 3 epochs, seed 0 | -0.0078 | +0.0177 | no: familiar fell |
| ResNet-50, seeds 0 and 1 | +0.0128, +0.0154 | +0.0197, -0.0219 | no: not up on every seed |
| copy-paste p 0.5, seed 0 | -0.0304 | +0.0971 | no: familiar fell |
| copy-paste p 0.25, seed 0 | -0.0151 | +0.0493 | no: familiar fell |
| ConvNeXt-Tiny, 3 seeds | +0.0245 (mean) | +0.0457 (mean) | yes |
| ConvNeXt-Tiny + copy-paste, 3 seeds | +0.0063 (mean) | +0.1028 (mean) | yes: final |

Box-guided crops (phase 1) were judged on the test sets: trans_test accuracy +3.0, +1.1 and +3.3
points, macro-F1 +2.9, +1.2 and +1.3 (seeds 0, 1, 2). The two kept changes cannot be told apart on
the test sets; on trans_val, +0.0457 for ConvNeXt and +0.0571 for copy-paste on top.

### B2. Final model on new cameras: which classes it mixes up

**On the slide.** The two figures, the line "Rare species (bird, rodent, skunk) score 43% macro-F1
on new cameras, against 63% for the common ones. No rodent is named right.", and the footer
"Final model, the deployed run" (seed 2).

`docs/figures/p3_test_trans_confusion_matrix_seed2.png` and
`docs/figures/p3_test_trans_per_class_seed2.png` (cite). The rare species (bird, rodent, skunk)
score 0.43 macro-F1 on the new cameras against 0.63 for the common ones (means of 3 seeds,
`p3_slices_test.md`); rodent is at zero.

### B3. Day vs night does not explain the gap

**On the slide.** Grouped bars in percentages, day (sand) and night (dark ink), for the first run
of each model: small CNN 30% / 32%, ResNet-18 53% / 51%, ResNet-18 with box-guided crops 56% / 54%.
No figure: `docs/figures/day_night.png` is left off (its colours clash, its labels use repo terms,
and its main fact, 2 in 3 photos at night, is on slide 4). Two lines to the right of the chart:
"Night photos (infrared, in grey) are about 3 points harder for the ResNet-18 models." and "Give
the new-camera test the same share of night photos as the familiar cameras, and the gap moves by
about 1 point or less."

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

**On the slide.** Four short paragraphs in plain words: "round" for phase, "run" for seed, "3
points" for 0.03 macro-F1, "trained models" for checkpoints (the mapping is in the notes).

- Every change was one change from the same baseline, written down with its expected result before
  the run, and kept only if the new-camera check (macro-F1 on one new camera, trans_val) rose by
  more than 0.03 on the mean of 3 seeds and on every seed, with the familiar cameras (val) down by
  at most 1 point (phase 2) or 3 points (phase 3).
- A control run re-trained the baseline on new code first and matched phase 1 (bit for bit in
  phase 2, to the fourth decimal in phase 3).
- The test sets: scored once at the end of phase 1 (all 7 checkpoints), then once for the final
  model's 3 seeds, after phase 1's scores had been seen (2026-10-18); disclosed wherever the final
  numbers appear, and shown beside phase 1's, never in their place.

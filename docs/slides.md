# Who Goes There: slide outline and speaker notes

Item 12.01. The outline of the capstone talk (presentation 2026-11-04), one section per slide, each
with what goes on the slide, the figure from `docs/figures/` and the speaker notes. Every result
number is from [`ledger.md`](ledger.md) or [`plan.md`](plan.md) section 7, except the phase-2
slide, whose numbers are from [`phase2-plan.md`](phase2-plan.md) section 12, and the phase-3 slide's
rule thresholds, from [`phase3-plan.md`](phase3-plan.md) section 1 (both marked there, and flagged in
`docs/progress-log.md`). Headline claims use the mean of 3 seeds. No phase-3 result exists yet;
slide 15 is a placeholder.

**Data citation**, on every slide that shows a dataset photo or a chart of the data (marked
"cite" below): Caltech Camera Traps (CCT20), Beery, Van Horn, Perona, "Recognition in Terra
Incognita", ECCV 2018. Data licence CDLA-Permissive 1.0. Code MIT.

Terms used on every slide: **cis** = cameras the model saw in training (other photos from them);
**trans** = cameras it never saw. **Gap** = cis minus trans.

---

## Slide 1. Who Goes There

**On the slide.** Title; subtitle "Telling animals apart on cameras the model has never seen";
name; the Space URL <https://huggingface.co/spaces/Codx999/who-goes-there>; the data citation
(cite).

**Speaker notes.** One sentence of what it is: a photo classifier for camera traps, and a measured
answer to whether it works on new cameras. Say up front that the talk ends with the honest
limits.

---

## Slide 2. The problem

**On the slide.**

- Camera traps: fixed, motion-triggered cameras, day photos and infrared night shots.
- Input: one photo. Output: one of 12 classes, 11 animals plus `car`.
- The catch: every camera has a fixed background. A model can learn "this rock means coyote" and
  look good on familiar cameras while failing on new ones.
- So the score that matters is on **cameras never seen in training** (trans), next to cameras it
  has seen (cis).

**Figure.** `docs/figures/camera_locations.png` (images per camera, cis vs trans) (cite).

**Speaker notes.** The dataset is Caltech Camera Traps, the CCT20 subset, from Beery et al.
2018. That paper defined cis and trans and showed that a full-image classifier loses a lot of
accuracy on unseen cameras. A field biologist moves cameras to new sites all the time, so the
trans number is the one a user lives with. `empty`, badger, fox and deer are dropped (plan
section 2): `empty` is a different problem, the other three are too rare to learn.

---

## Slide 3. The classifier

**On the slide.**

- A ResNet-18 pretrained on ImageNet, fine-tuned end to end; a small CNN trained from scratch as
  the baseline.
- Test sets, scored once at the end: cis_test 14,478 images; trans_test 21,473 images from 9
  cameras never seen.
- Accuracy **and** macro-F1: the classes are imbalanced, so a model that guesses the big classes
  can look good on accuracy alone.

**Figure.** `docs/figures/class_counts.png` (cite).

**Speaker notes.** Why ResNet-18: few training images, so pretraining supplies the low-level
features; small enough for a CPU Space; the standard Grad-CAM subject (plan section 9). Why
macro-F1: every class counts the same, so a rare class the model ignores pulls it down. Test-set
isolation: every choice was made on val; the test sets were scored once, for all 7 checkpoints
together, with my go-ahead, and logged.

---

## Slide 4. Live demo

**On the slide.** The Space, live: <https://huggingface.co/spaces/Codx999/who-goes-there>. Upload
a camera-trap photo; it returns the top class with its confidence, the two runners-up, the time of
the forward pass and an optional Grad-CAM heat map. Model: ResNet-18 arm B seed 2, picked on val
(cite on any photo shown).

**Then the live sort** (item 12.05): `trailcam/demo/sort.py` sorts a folder of about 200
trans-test photos into class folders with a visible timer and per-class counts.

**Backups, in order.** (1) The local app's `--share` link, started before the talk. (2) The
backup video of the Space and the sort (item 12.04), opened in a tab before the talk.

**Speaker notes.**

- **Pick the demo photos in advance and know which are right and which are wrong.** The Space's
  model gets 5 of the 10 `assets/` photos right (plan section 7), in line with its trans_test
  score. Rehearse on the Space and write down each photo's answer. Known from the Grad-CAM run:
  the bobcat (`trans_bobcat_day_01`) comes back as cat and the opossum (`trans_opossum_day_10`)
  as coyote; both coyotes (`trans_coyote_day_04`, `trans_coyote_night_05`) are right.
- Show one right and one wrong on purpose; say before clicking which you expect. A wrong answer
  you predicted is a better demo than a right one you hoped for.
- Say "confidence" as the model's own estimate, not a calibrated probability (plan section 8;
  calibration, item 12.03, is not done unless it lands).
- The Space runs on a CPU. If it has been idle it may need a restart: open it a few minutes
  before and run one photo.
- The live sort: say the counts out loud at the end; they are counts per predicted class, not a
  score.
- If the 12.05 rehearsal includes the 20-photo human baseline, it goes here, before the model
  answers.

---

## Slide 5. The honest test: cis vs trans

**On the slide.** The headline table, test sets scored once (plan section 7, test results):

| model | cis_test accuracy | cis_test macro-F1 | trans_test accuracy | trans_test macro-F1 | accuracy gap | macro-F1 gap |
| --- | --- | --- | --- | --- | --- | --- |
| small CNN, seed 0 | 0.5026 | 0.4504 | 0.3140 | 0.2041 | 0.1886 | 0.2463 |
| ResNet-18 arm A, mean of 3 seeds | 0.6708 | 0.6216 | 0.5137 | 0.4127 | 0.1571 | 0.2089 |
| ResNet-18 arm B, mean of 3 seeds | 0.6808 | 0.6361 | 0.5387 | 0.4309 | 0.1422 | 0.2051 |

One line under it: "On cameras it has never seen, the model loses 0.1571 of accuracy and 0.2089
of macro-F1" (arm A, mean of 3 seeds).

**Figure (optional, beside the table).** `docs/figures/test_trans_per_camera_arm_b_seed0.png`,
accuracy per unseen camera.

**Speaker notes.** Lead with the gap, not the top-left number. Pretraining buys about 20 points of
trans_test accuracy over the small CNN, but the gap stays: the ResNet-18 is better everywhere and
still much worse on new cameras. No leakage alarm fired: no image or sequence in two splits, no
trans camera in training. The per-camera bars show the spread across the 9 unseen cameras; the
worst camera has no photo in `assets/`, so it is shown as a bar only.

---

## Slide 6. Two numbers that should be the same, and are not

**On the slide.**

- Val accuracy 0.87; cis_test accuracy 0.67. **Same cameras.**
- Val was drawn from the same weeks as training; cis_test from other times.
- So val was optimistic about **time** as well as location. Seasons, light, vegetation, and which
  animals pass by all change.
- A second warning: trans_val (one unseen camera) said arm A was ahead; the 9 cameras of
  trans_test said the opposite (slide 8).

**Speaker notes.** This is the finding I did not plan for. The leakage alarm list includes
"trans close to cis"; nothing like that fired. The drop is the time shift inside the cis cameras.
The lesson for anyone building one of these: a random val split from the training weeks flatters
the model; a val set from a later period would have predicted the test score better. And one
camera is not a test of "new cameras" (trans_val's spread is mostly that one camera's bobcats).

---

## Slide 7. The experiment: boxes at training time only

**On the slide.**

- The question: if the training crops are steered onto the animal, does the model look at the
  animal and lose less on new cameras, with nothing extra at inference?
- **Arm A** (control): standard random resized crops.
- **Arm B** (treatment): for images with a box, half the time the crop is sampled so it contains
  the box and the box fills 20% to 100% of it; otherwise as arm A.
- Everything else identical: backbone, seeds, epochs, optimiser, schedule, evaluation. 3 seeds
  per arm. At test both see the whole photo; the box is never an input.
- Measured: scores and the gap; **attention-in-box** (share of the Grad-CAM heat inside the
  animal's box); the **animal-removed test** (blank the box, see if the prediction survives).

**Figure.** `docs/figures/box_fill.png`, how much of the frame the boxes cover, with arm B's 20%
to 100% range shaded (cite).

**Speaker notes.** Prior work compares whole-image against crop-at-test, which needs a detector
when deployed. This uses the free boxes only while training, so the deployed model stays one
classifier. One change, one variable, controls on everything else. The attention score and the
animal-removed test are there so a score change can be read: did the model move its eyes, or
just get lucky?

---

## Slide 8. Result: arm B is ahead on new cameras, by a little

**On the slide.**

- Arm B beats arm A on trans_test **on every seed**: accuracy +3.0, +1.1 and +3.3 points,
  macro-F1 +2.9, +1.2 and +1.3 (seeds 0, 1, 2).
- Mean gain: +2.5 points of accuracy, +1.8 of macro-F1.
- The accuracy gap shrinks from 0.157 to 0.142. The macro-F1 gap barely moves (0.2089 to 0.2051):
  arm B rose on cis too.
- **Small but consistent.** Not a fix for new cameras.

**Speaker notes.** Say it in that order: consistent, small, the gap mostly still there. The
val-side camera (trans_val, one camera) had pointed the other way; trans_test's nine cameras
overturned it, which is why a single camera never decided anything. The Space serves arm B
because it beat arm A on all three seeds; the seed was picked on val (open question 7). One
caution from the slices (seed 0): on the rare species (bird, rodent, skunk) arm B was lower,
0.3066 against 0.3754 accuracy, so the gain is in the common classes.

---

## Slide 9. Did the model look at the animal?

**On the slide.** Mean of 3 seeds, images whose box lies inside the centre crop (plan section 7):

| arm | attention-in-box, trans_test | trans_test accuracy | trans_test accuracy, box blanked |
| --- | --- | --- | --- |
| arm A | 0.2315 | 0.5754 | 0.2470 |
| arm B | 0.2331 | 0.6039 | 0.2613 |

- Attention-in-box: an even heat map scores 0.137. Both arms look at the box more than chance;
  arm B is higher on every seed, but by about 0.002.
- Animal removed: accuracy falls from about 0.57 to 0.61 to 0.22 to 0.29, chance is 0.083. With
  the animal blanked out, a large share of predictions survive: **the background still carries
  them**, for both arms.

**Figures.** `docs/figures/attention_test_arm_a_seed0.png` and
`docs/figures/attention_test_arm_b_seed0.png`, side by side.

**Speaker notes.** This is the mechanism check, and it says the honest thing: the scores moved a
little, the attention barely moved, and the shortcut is still there. Reading it with the plan's
rule (section 6): a positive result needed the attention to rise; it rose by a hair. So this is a
mixed result, reported as mixed. Arm B's crops did not remove the background shortcut, which is
what phase 2's copy-paste arm went after (slide 14). If asked: blanking uses the image mean, so a
model that learned animals should fall towards chance; this one stays well above it.

---

## Slide 10. Grad-CAM: where the heat goes

**On the slide.** Four overlays of the Space's model (arm B seed 2) on `assets/` photos, all from
unseen cameras (cite: Beery et al. 2018, CDLA-Permissive 1.0):

| figure | photo | prediction | where the heat is |
| --- | --- | --- | --- |
| `docs/figures/gradcam_overlay_01.png` | bobcat, day | cat (0.79) | on the animal |
| `docs/figures/gradcam_overlay_08.png` | opossum, day | coyote (0.22) | on the background |
| `docs/figures/gradcam_overlay_06.png` | coyote, night (infrared) | coyote (1.00) | on the animal |
| `docs/figures/gradcam_overlay_05.png` | coyote, day | coyote (1.00) | partly on the sand below it |

**Speaker notes.** Show 01 and 08 large, 06 and 05 small. Two different failures: the bobcat is
wrong *for a good reason*, the model looked at the animal and saw a cat; the opossum is wrong *for
a bad reason*, it looked at the scenery, the shortcut the animal-removed test measures. The day
coyote is right, but part of the heat is on the sand: a right answer is not always for the right
reason. The numbers in parentheses are the model's confidence. The overlay numbers are the
photo's place in `assets/` in name order, not the index in the file name. Grad-CAM is
hand-written (`trailcam/evaluate/gradcam.py`), the same function that scores attention-in-box.

---

## Slide 11. Two errors from the confusion matrix

**On the slide.** `docs/figures/test_trans_confusion_matrix_arm_b_seed0.png`, with two cells
circled:

1. **Bobcat predicted as cat.** A look-alike: similar shape, both often at night. It matches
   overlay 01 (heat on the animal) and the trans_val camera, whose bobcats were mostly called
   cat.
2. **Raccoon predicted as opossum.** Both are grey, similar in size and mostly nocturnal, and
   opossum is the largest training class, so in an infrared frame the model leans to it.

**Speaker notes.** Point at the circled cells and read their counts off the figure; they are not
on the slide as text. The first error is about the animal's appearance, the second about the
class prior in hard (infrared) frames. Neither is fixed by looking at the box more, which is
consistent with slide 9: the gap is not only background, the animals themselves look different
across cameras, as Koh et al. (WILDS) found. Rare classes (rodent, bird) are also visible as
nearly empty diagonals; mention them in one sentence if time allows. (Data: Beery et al. 2018.)

---

## Slide 12. Day vs night

**On the slide.** trans_test accuracy by light, seed 0 (plan section 7; every seed in
`docs/figures/slices_test.md`):

| model | accuracy, day | accuracy, night |
| --- | --- | --- |
| small CNN | 0.2961 | 0.3210 |
| ResNet-18 arm A | 0.5315 | 0.5058 |
| ResNet-18 arm B | 0.5618 | 0.5362 |

- Night (infrared, grey) is a few points harder for the ResNet-18s.
- The trans cameras shoot more at night than the cis ones, so could that be the gap? No:
  re-weighting trans_test to the cis night share changes the gap by about 0.01 or less.

**Figure.** `docs/figures/day_night.png`, the night share per split and the photos per hour (cite).

**Speaker notes.** Infrared frames lose colour, which an ImageNet backbone leans on, so night
being harder is expected. The useful result is the second bullet: a tempting explanation for
the gap, checked and ruled out with the predictions already in hand, no extra training. The
small CNN is better at night, but it is near the floor everywhere.

---

## Slide 13. What the features say

**On the slide.** `docs/figures/features_by_camera_test.png` beside
`docs/figures/features_by_species_test.png`: the backbone's features of the test photos in two
dimensions, coloured by camera and by species.

**Speaker notes.** Optional slide; cut it first if short of time. On val, the features cluster by
camera (plan section 7): the model's own representation knows which camera a photo came from.
That is the background shortcut, seen from inside the network. Point at the clusters; no numbers
on this slide.

---

## Slide 14. Phase 2: five changes tried, none kept

**On the slide.** One baseline (phase 1's arm A recipe), one change per variant, judged on val
and on trans_val (one unseen camera) by a screen, confirm and keep rule **written before any
run**. Numbers from `docs/phase2-plan.md` section 12 (macro-F1 changes):

| change | val | trans_val | verdict |
| --- | --- | --- | --- |
| copy-paste animals onto other cameras' empty frames, p 0.5 | -0.0304 | +0.0971 | screened out (val fell) |
| copy-paste, p 0.25 | -0.0151 | +0.0493 | screened out (val fell) |
| head-first training | -0.0078 | +0.0177 | screened out (undertrained) |
| 320 px input, 3 seeds | +0.0100 mean | +0.0158 mean | not kept (under the 0.03 bar) |
| ResNet-50 | +0.0128 | +0.0197, then -0.0219 on seed 1 | not kept (not up on every seed) |

**Frozen at the baseline.** The test sets were not touched again; the Space keeps arm B seed 2.

**Speaker notes.** The finding worth a slide is copy-paste's trade-off: about 3 points of
familiar-camera macro-F1 for about 10 on the new camera, near-linear in the paste rate (half the
rate gives half of each). The rule would not trade val for one camera, so it was screened out; a
longer training run is the follow-up. Higher resolution and ResNet-50 gave small val gains that
the one-camera noise could not confirm. The point of the slide: a pre-declared rule that rejected
everything is still a result, and believable because the numbers could not bend it. Caveat to
say out loud: trans_val is one camera, so these are one-camera results.

---

## Slide 15. Phase 3: a stronger backbone (results pending)

> **PLACEHOLDER: results pending.** From [`phase3-plan.md`](phase3-plan.md) (item 13.10). The
> freeze is about 2026-10-28 (its section 11). No phase-3 number goes on this slide until it is in
> `docs/ledger-phase3.md` and section 13 of that plan.

**On the slide.**

- **What it tests**: does a stronger ImageNet-1k backbone help on the unseen camera? Two
  candidates against the same baseline as phase 2 (phase 1's arm A, 3 seeds):
  (a) **ConvNeXt-Tiny**, everything else the phase-1 recipe; (b) **ConvNeXt-Tiny plus phase 2's
  copy-paste** (paste probability 0.5), betting the stronger backbone absorbs part of
  copy-paste's val cost.
- **The keep rule, fixed before any run**: kept only if trans_val macro-F1 rises by more than
  0.03 on the mean of 3 seeds **and** on every seed, and val macro-F1 falls by no more than 0.03.
- **The screen** (seed 0 first): trans_val macro-F1 at least 0.03 up and val at most 0.03 down;
  only then seeds 1 and 2.
- **One finalist at most**, and only it is ever scored on test, once, with the user's go-ahead.
- Results: pending.

**If a phase-3 recipe is kept and scored on test:** slide 5 shows phase 1's and phase 3's test
numbers side by side, never one in place of the other, and says the test sets were used a
**second time**, for one recipe chosen on val and trans_val only, after phase 1's test scores had
been seen (`phase3-plan.md` section 7). If nothing is kept, there is no test scoring: this slide
says so in one line, like slide 14, and the Space keeps arm B seed 2.

**Speaker notes.** Why the rule is written first: once a result is seen, a bar can always be
drawn just under it. Why the val tolerance is 3 points here (phase 2 allowed 1): phase 3 aims at
new cameras, and the user accepts up to 3 points on familiar cameras for a clear gain on the new
one, no more. The honest limit to say out loud: trans_val is still one camera, now judged
against seven recipes, so a kept gain is optimistic about the next new camera. Fill in the
verdicts from `phase3-plan.md` section 13 when they exist.

---

## Slide 16. Limitations and next steps

**On the slide.**

- **Limitations.** One dataset, one region, one species set. 3 seeds per arm. trans_val is one
  camera. Confidences are not calibrated. Rare species are weak.
- **The biggest lever: detect, then classify.** A detector (MegaDetector) finds the animal, the
  classifier sees only the crop. The dataset paper found crops more than halve the error. It
  changes the product: two models, a detector at inference.
- **BioCLIP**, a backbone pretrained on the tree of life, once its pretraining data is checked
  for CCT photos (a backbone that saw the trans cameras would inflate every trans score).
- **SWA** (averaging weights over the late epochs), and copy-paste with a longer training budget.
- **Sequence-level voting**: a camera fires a burst of frames; voting over the burst instead of
  one frame.

**Speaker notes.** Detection is out of scope by design (plan section 1): the question was what a
single classifier can do. Each next step costs something: the detector changes deployment,
BioCLIP needs a data audit, sequence voting needs the bursts at inference.

---

## Slide 17. The idea is not new. The measurement is.

**On the slide.**

- Using boxes to steer a classifier is a known idea; detect-then-classify is the deployed answer.
- What this project adds: a controlled measurement on CCT20, two arms, 3 seeds, unseen cameras,
  with an attention score and an animal-removed test, the test sets scored once.
- The answer: a small, consistent gain on new cameras; the attention barely moved; the
  background still carries the predictions.
- Data: Caltech Camera Traps (CCT20), Beery et al. 2018, CDLA-Permissive 1.0. Code MIT. Repo
  and Space links.

**Speaker notes.** End on the sentence in the title. Then questions. Likely questions and where
the answer is: "why not use the boxes at test?" (slide 16, detection); "is the gain real?"
(slide 8, every seed, but small); "why is val so much higher than test?" (slide 6, time shift);
"what did phase 2 find?" (slide 14).

---

## Backup slides

- **Val results** (plan section 7, val results): the 9-row val table and the val figures
  `docs/figures/val_confusion_matrix_arm_b_seed0.png`, `docs/figures/attention_val_arm_b_seed0.png`.
- **Per class on trans_test**: `docs/figures/test_trans_per_class_arm_a_seed0.png` and
  `docs/figures/test_trans_per_class_arm_b_seed0.png`.
- **Small CNN confusion matrix**: `docs/figures/test_trans_confusion_matrix_small_cnn.png`, to
  show what no pretraining looks like.
- **How the test sets were protected** (plan section 7, protocol): one scoring, a script that
  refuses a second run, the user's go-ahead, logged in `docs/test-ledger.md`.
- **The data**: `docs/figures/burst_length.png` (why the split is by sequence) (cite).

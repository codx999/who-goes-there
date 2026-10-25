# Who Goes There: slide outline and speaker notes

Items 12.01 and 13.16. The outline of the capstone talk (presentation 2026-11-04), told as a
learning journey: a known problem reproduced, then published methods tested one at a time under
rules written before the runs. One section per slide, each with a target time, what goes on the
slide, the figure from `docs/figures/` and the speaker notes. Nothing here is claimed as
original: the problem, the split and every method come from published research or standard
practice, and slide 13 credits each one.

**Where the numbers come from.** Every result number is from [`ledger.md`](ledger.md) or
[`plan.md`](plan.md) section 7, except: phase 2's, from [`phase2-plan.md`](phase2-plan.md)
section 12 (rows in [`ledger-phase2.md`](ledger-phase2.md)); phase 3's val and trans_val numbers,
from [`phase3-plan.md`](phase3-plan.md) section 13 (rows in [`ledger-phase3.md`](ledger-phase3.md));
phase 3's test numbers, from [`test-ledger-phase3.md`](test-ledger-phase3.md) and plan section 7,
"Phase 3's test results", always shown beside phase 1's ([`test-ledger.md`](test-ledger.md)) with
the disclosure (slide 9); the published papers' numbers, from
[`research-report.md`](research-report.md) Part 3 and [`phase2-plan.md`](phase2-plan.md) section
6, "The evidence"; the Space's readings, from [`progress-log.md`](progress-log.md), 2026-10-21.
Headline claims use the mean of 3 seeds.

**Data citation**, on every slide that shows a dataset photo or a chart of the data (marked
"cite" below): Caltech Camera Traps (CCT20), Beery, Van Horn, Perona, "Recognition in Terra
Incognita", ECCV 2018. Data licence CDLA-Permissive 1.0. Code MIT.

**Words.** "I reproduced", "I tested a method proposed by", "in my setup", "I learned". The
methods are other people's; what is mine is the work of testing them carefully.

Terms used on every slide: **cis** = cameras the model saw in training (other photos from them);
**trans** = cameras it never saw. **Gap** = cis minus trans. **val** = held-out photos from the
training cameras; **trans_val** = one unseen camera kept for choosing; **cis_test** and
**trans_test** = the test sets, locked until the end.

## Timing

| slide | title | minutes | running total |
| --- | --- | --- | --- |
| 1 | Who goes there? The labelling problem | 2 | 2 |
| 2 | What this project is, and what it is not | 2 | 4 |
| 3 | The known problem | 3 | 7 |
| 4 | The process | 3 | 10 |
| 5 | Step 0: the plain baseline | 3 | 13 |
| 6 | Understanding the failure | 3 | 16 |
| 7 | The method ladder | 5 | 21 |
| 8 | What the failures taught | 2 | 23 |
| 9 | The result, with the disclosure | 3 | 26 |
| 10 | Before and after, then the live demo | 6 | 32 |
| 11 | Limits and next steps | 2 | 34 |
| 12 | What I learned | 2 | 36 |
| 13 | Credits | 1 | 37 |
| | Q&A, from the backup slides | after | |

About 37 minutes. If the slot is nearer 32: cut slide 6's second figure, give slide 7 four
minutes (talk through the kept and the most instructive dropped rungs only), and give slide 10's
demo four minutes (one photo fewer). That brings it to about 33.

---

## Slide 1. Who goes there? The labelling problem (2 min)

**On the slide.**

- Title "Who Goes There"; subtitle "Reproducing a known problem in camera-trap classification,
  and learning to test published fixes"; name; the Space URL
  <https://huggingface.co/spaces/Codx999/who-goes-there>.
- One night photo, `assets/trans_coyote_night_05.jpg`, full bleed (cite).
- Camera traps take photos faster than people can label them: 3.2 million images in Snapshot
  Serengeti (Norouzzadeh et al. 2018); 4.8 million in one Idaho collection, where a detector took
  about 80% of the images out of manual review (Beery, Morris, Yang 2019).

**Speaker notes.** Ask the room: who is in this photo? (A coyote, at night, in infrared.) Now
imagine a million of these. That is why people train classifiers for camera traps, and why
ecologists care whether the classifier still works when the camera moves. Say up front: this is
a learning project; the talk ends with honest limits and a list of whose ideas I used.

---

## Slide 2. What this project is, and what it is not (2 min)

**On the slide.**

| this project is | this project is not |
| --- | --- |
| a learning project for a bootcamp capstone | a new method or a research contribution |
| a reproduction of a published problem (Beery et al. 2018) | a new dataset or benchmark |
| published methods tested one at a time, under rules written before the runs | tuning until the numbers look good |
| a classifier deployed as a live demo | a detector, a sequence model or a production system |

- The problem, the split, the dataset and every method I tested come from published research or
  standard practice. Slide 13 lists each one with its source.

**Speaker notes.** Set expectations in one breath: if you came for a new algorithm, there isn't
one. What I can show is the process: start from a plain trained model, pick methods others
proposed, test each one fairly, keep the few that work, and report the ones that didn't. Most of
the learning was in the second part.

---

## Slide 3. The known problem (3 min)

**On the slide.**

- **The source:** Beery, Van Horn, Perona, "Recognition in Terra Incognita", ECCV 2018. They
  built the CCT20 benchmark from Caltech Camera Traps and split it **by camera**: cis cameras
  (seen in training) and trans cameras (never seen).
- **Their finding:** a full-image classifier (Inception-v3) had 19.1% top-1 error on seen cameras
  and 41.0% on unseen ones. Classifying the animal's box crop more than halved both (8.1%, 19.6%).
- **Why:** every camera has a fixed background. A model can learn "this rock means coyote" and
  look good on familiar cameras while failing on new ones.
- **My setup:** CCT20, 12 classes (11 animals plus `car`), one photo in, one class out. Test sets:
  cis_test 14,478 images; trans_test 21,473 images from 9 cameras never seen.
- **"I set out to reproduce it":** does the gap show up with a modern small backbone, a clean
  split and today's tools?

**Figure.** `docs/figures/camera_locations.png` (images per camera, cis vs trans) (cite).

**Speaker notes.** The cis/trans measurement is Beery et al.'s, not mine; I adopted their
benchmark. Koh et al. 2021 (WILDS) later made unseen-camera shift a standard benchmark, and
Schneider et al. 2020 saw the same effect on a Canadian dataset (95.6% at trained locations,
68.7% at new ones), so this is a well-established problem. A field biologist moves cameras all the
time, so the trans number is the one a user lives with. `empty`, badger, fox and deer are dropped
(plan section 2): `empty` is a different problem, the other three are too rare to learn (Schneider
et al. put the practical floor at about 1,000 images per class). Two changes from the paper,
for hygiene: I re-split train and val by burst (`seq_id`), because the published files share
some bursts between them, and I report macro-F1 beside accuracy, as WILDS does.

---

## Slide 4. The process (3 min)

**On the slide.** A loop, drawn left to right:

1. **Baseline:** a plain trained model, scored on val and trans_val.
2. **Pick a published method**, one change only.
3. **Write it down first:** a hypothesis line (what I expect and why), committed before the run.
4. **Test on val and trans_val:** a screen on seed 0, then a confirm on seeds 1 and 2.
5. **Keep or drop**, by a rule written before any run: trans_val macro-F1 up by more than 0.03 on
   the mean of 3 seeds **and** up on every seed, with val macro-F1 down by at most 1 point (phase
   2) or 3 points (phase 3).
6. **Repeat.**

Around the loop:

- **The test sets stay locked** until the end; scored once in phase 1 and once more, disclosed,
  for phase 3's one recipe (slide 9).
- **3 seeds** for every claim; one seed is not enough (slide 8).
- **A control run** each phase: the baseline re-trained on the new code must match phase 1 before
  any result is read. Phase 2's matched bit for bit; phase 3's to the fourth decimal.

**Speaker notes.** None of this is my invention: one change at a time, a held-out test set,
several seeds and rules written in advance are standard experimental practice. What I learned is
how much discipline it takes to follow them when a result is tempting. Why trans_val and not the
test sets: the test sets are for the end only; trans_val is the one unseen camera I am allowed to
look at. Its weakness: it is one camera (46% bobcats), so its noise is about 0.03, which is where
the 0.03 bar comes from. Phase 3 loosened the val condition from 1 to 3 points; that was decided
after phase 2's results and before any phase-3 run, and it is written in `phase3-plan.md` section
1 (backup Q&A).

---

## Slide 5. Step 0: the plain baseline (3 min)

**On the slide.** The phase-1 baseline, scored once on the test sets at the end of phase 1 (plan
section 7):

| model | cis_test accuracy | cis_test macro-F1 | trans_test accuracy | trans_test macro-F1 | accuracy gap | macro-F1 gap |
| --- | --- | --- | --- | --- | --- | --- |
| small CNN from scratch, seed 0 | 0.5026 | 0.4504 | 0.3140 | 0.2041 | 0.1886 | 0.2463 |
| ResNet-18 (ImageNet-pretrained), arm A, mean of 3 seeds | 0.6708 | 0.6216 | 0.5137 | 0.4127 | 0.1571 | 0.2089 |

- **The gap, reproduced:** on cameras it has never seen, the model loses 0.157 of accuracy and
  0.209 of macro-F1. The same shape as Beery et al.'s result, not the same number: other
  classes, another backbone, another split.
- **A second gap I had not planned for:** val accuracy 0.87, cis_test accuracy 0.67, on the **same
  cameras**. Val came from the same weeks as training; cis_test from other times.

**Figure.** `docs/figures/test_trans_per_camera_arm_a_seed0.png`, accuracy per unseen camera.

**Speaker notes.** The baseline is the standard transfer-learning recipe from the course
(chapter 06): an ImageNet-pretrained ResNet-18 from torchvision, fine-tuned end to end, AdamW,
a cosine schedule, random resized crops. The small CNN from scratch shows what pretraining buys:
about 20 points of trans_test accuracy. Accuracy **and** macro-F1, because the classes are
imbalanced and macro-F1 counts every class the same. On the time shift: I learned this the hard
way, but it is well known; WILDS builds its in-distribution test from the training cameras on
different days (Koh et al. 2021). A random val split from the training weeks flatters the model.
No leakage alarm fired: no image or burst in two splits, no trans camera in training. The
per-camera bars show the spread across the 9 unseen cameras.

---

## Slide 6. Understanding the failure (3 min)

**On the slide.** Three tools to ask "where does the model look?", ResNet-18 arm A, trans_test,
mean of 3 seeds, images whose box lies inside the centre crop (plan section 7):

- **Grad-CAM** (Selvaraju et al. 2017): a heat map of the image regions that drove the
  prediction. Hand-written in `trailcam/evaluate/gradcam.py`.
- **Attention-in-box:** the share of the Grad-CAM heat inside the animal's box. Arm A: 0.2315; an
  even heat map would score 0.137. Above chance, but most of the heat is outside the animal.
- **Animal removed:** blank the box and predict again. Accuracy falls from 0.5754 to 0.2470;
  chance is 0.083. With the animal gone, the scenery alone still names the class far above
  chance: **the background is carrying predictions**.

**Figures.** `docs/figures/attention_test_arm_a_seed0.png`; beside it, small,
`docs/figures/gradcam_overlay_05.png` (a day coyote, right at 1.00, part of the heat on the sand
below it) (cite).

**Speaker notes.** Geirhos et al. 2020 call this shortcut learning: a rule that works on the
benchmark and fails under shift. The two scores are simple checks that the project's literature
review suggested (`research-report.md` Part 4), so a score change can be read: did the model move
its eyes, or just get lucky? Grad-CAM is a heuristic, not ground truth, which is why it is paired
with the animal-removed test. The overlay is from arm B seed 2, the model the Space served in
phase 1; arm A's attention is within 0.002 of arm B's. A right answer is not always for the right
reason: the coyote is right, and part of the heat is on the sand.

---

## Slide 7. The method ladder (5 min)

**On the slide.** One rung per method: the source, what it suggested, what happened in my
setup, and the verdict by the rule of slide 4. Changes are in macro-F1 against the baseline (arm
A), paired by seed. Rung 1 was phase 1's pre-planned experiment, judged on the test sets; rungs 2
to 8 were judged on val and trans_val only.

| rung | method | source | what the source suggested | what happened here | verdict |
| --- | --- | --- | --- | --- | --- |
| 1 | box-guided training crops (arm B): half the time, the crop contains the animal's box | Beery et al. 2018 (crops at test); the training-only variant from my literature review | classifying the box crop more than halves the error, but needs a detector at test | trans_test accuracy +2.5 and macro-F1 +1.8 points, up on every seed; attention-in-box +0.002 | mixed: small, consistent gain; the attention barely moved |
| 2 | input at 320 px instead of 224 (`p2_res320`) | Koh et al. 2021 (WILDS baselines at 448 px) | higher resolution may help; part of WILDS' numbers may be resolution | val +0.0100, trans_val +0.0158 (means of 3 seeds, up on every seed) | dropped: under the 0.03 bar |
| 3 | train the new head for 3 epochs before the body (`p2_head3`) | standard fine-tuning practice | a settled head first can protect the pretrained features | val -0.0078, trans_val +0.0177 (seed 0) | dropped at the screen: undertrained in 10 epochs |
| 4 | ResNet-50 instead of ResNet-18 (`p2_r50`) | Koh et al. 2021 (WILDS uses ResNet-50); my literature review | a bigger backbone as a check on whether the gap depends on it | val +0.0128, +0.0154; trans_val +0.0197, then -0.0219 on seed 1 | dropped: not up on every seed |
| 5 | copy-paste, p 0.5 (`p2_arm_c`): paste the boxed animal onto an empty frame from another training camera | Gao et al. 2023; WILDS leaderboard; Mesnage et al. 2025 | randomise the background, keep the animal; the leaderboard's copy-paste entry beat plain training (36.5 against 30.8 OOD macro-F1) | val -0.0304, trans_val +0.0971 (seed 0) | dropped at the screen: val fell |
| 6 | copy-paste, p 0.25 (`p2_arm_c_p25`) | the same | the same, gentler | val -0.0151, trans_val +0.0493 (seed 0) | dropped at the screen: val fell |
| 7 | ConvNeXt-Tiny instead of ResNet-18 (`p3_convnext`) | Liu et al. 2022 | a modernised ConvNet; same ImageNet-1k pretraining as the baseline | val +0.0245, trans_val +0.0457 (means of 3 seeds, up on every seed) | **kept** |
| 8 | ConvNeXt-Tiny plus copy-paste, p 0.5 (`p3_convnext_cp`) | Liu et al. 2022 with Gao et al. 2023, the WILDS leaderboard, Mesnage et al. 2025 | the stronger backbone might pay copy-paste's val cost | val +0.0063, trans_val +0.1028 (means of 3 seeds, +0.1007, +0.1021, +0.1057) | **kept, the finalist** |

**Speaker notes.**

- Walk the ladder top to bottom; spend the time on rungs 1, 5, 7 and 8.
- **Rung 1.** Beery et al. classified the box crop at test time, which needs a detector when
  deployed. My literature review suggested spending the free boxes during training only, so the
  deployed model stays one classifier. I do not know whether someone has tested exactly this, and
  I make no claim either way; using boxes to steer a classifier is a well-known idea. Everything
  else identical: backbone, seeds, epochs, optimiser, schedule. The box is never an input at
  test. Per seed: accuracy +3.0, +1.1 and +3.3 points, macro-F1 +2.9, +1.2 and +1.3. The
  background shortcut stayed (both arms' blanked accuracy 0.25 to 0.26), which pointed to
  copy-paste.
- **Rungs 5 and 6, my adaptation of copy-paste.** The box rectangle, not a segmented animal;
  backgrounds only from the empty frames of training cameras (995 frames), never from a test
  camera, unlike Mesnage et al.; night pasted onto night and day onto day, because Mesnage et al.
  saw nocturnal species get worse when night animals landed on day backgrounds; the paste's
  bottom edge in the lower half of the frame, as in their section 2.4.
- **Rung 3.** The phase-2 plan pointed to a published version of this idea (LP-FT), but that
  reference was never verified, so I credit it as standard practice.
- **Rung 7.** timm's `convnext_tiny.fb_in1k` weights, chosen so the pretraining data stays
  ImageNet-1k like the baseline's: the one change is the architecture.
- Five of the eight rungs were dropped. That is the normal outcome, and it is the next slide.

---

## Slide 8. What the failures taught (2 min)

**On the slide.**

- **One seed can lie.** ResNet-50 passed the screen on seed 0 (trans_val +0.0197) and fell on
  seed 1 (-0.0219).
- **One seed would have picked the wrong finalist.** On seed 0, ConvNeXt with copy-paste led
  ConvNeXt alone by 0.0240 on trans_val, inside the 0.03 band, so the tie-break on val would have
  chosen ConvNeXt alone. On 3 seeds the lead was 0.0571.
- **One camera can lie.** In phase 1, trans_val (one camera) put arm A ahead; trans_test's nine
  cameras put arm B ahead on every seed.
- **A trade-off can move.** On ResNet-18, copy-paste cost about 1 point of val for 3 on the
  unseen camera (-0.030 for +0.097), and the rule stopped it. On ConvNeXt the same trade appeared
  (-0.018 for +0.057 against ConvNeXt alone), but ConvNeXt's own val gain paid for it: against
  the baseline, the val cost was gone (+0.0063).

**Speaker notes.** The lesson is not "these methods are bad"; it is that small effects need
several seeds, and that a dropped result is still information. Copy-paste was dropped in phase 2
and became half of the finalist in phase 3, without changing the rule's trans_val bar. The
phase-2 rule would not trade val for one camera; phase 3 allowed up to 3 points of val, written
before its runs, and the finalist did not need it (its val rose).

---

## Slide 9. The result, with the disclosure (3 min)

**On the slide.** Phase 1's one test scoring, then phase 3's below it (plan section 7):

| model | cis_test accuracy | cis_test macro-F1 | trans_test accuracy | trans_test macro-F1 | accuracy gap | macro-F1 gap |
| --- | --- | --- | --- | --- | --- | --- |
| phase 1: small CNN, seed 0 | 0.5026 | 0.4504 | 0.3140 | 0.2041 | 0.1886 | 0.2463 |
| phase 1: ResNet-18 arm A, mean of 3 seeds | 0.6708 | 0.6216 | 0.5137 | 0.4127 | 0.1571 | 0.2089 |
| phase 1: ResNet-18 arm B, mean of 3 seeds | 0.6808 | 0.6361 | 0.5387 | 0.4309 | 0.1422 | 0.2051 |
| phase 3: ConvNeXt-Tiny + copy-paste, mean of 3 seeds (test sets' 2nd use) | 0.7565 | 0.7241 | 0.6598 | 0.5367 | 0.0967 | 0.1874 |

- **Disclosure, on the slide:** "Phase 3: the test sets' second use, for one recipe chosen on val
  and trans_val only, after phase 1's test scores had been seen (2026-10-18)."
- trans_test macro-F1 +0.12 over arm A, up on every seed; accuracy +0.15.
- The accuracy gap shrank by more than a third (0.1571 to 0.0967). The macro-F1 gap barely moved
  (0.2089 to 0.1874): the familiar cameras improved too.

**Figure (optional).** `docs/figures/p3_test_trans_per_class_seed2.png`: rodent at zero, bird the
next weakest (cite).

**Speaker notes.**

1. **The disclosure first.** Looking at a test set twice costs some of its honesty. The recipe
   was fixed on val before the scoring, only it was scored, once, and it sits beside phase 1's
   rows, never in their place.
2. **Then the gain.** Every phase-3 seed beats every phase-1 checkpoint on trans_test. On
   trans_val the gain had been +0.10; this time one camera predicted the nine well, which is not
   the same as being reliable.
3. **Then the limits.** Two things changed at once, the backbone and copy-paste, so the test
   cannot split the credit; only trans_val, one camera, can (+0.0457 and +0.0571). Rare species on
   new cameras: macro-F1 0.43 against 0.63 for the common ones, and no rodent is right on any
   seed. The background shortcut is still there: with the animal blanked, 47% of the trans_test
   accuracy survives, against 43% for both phase-1 arms.

---

## Slide 10. Before and after, then the live demo (6 min)

**On the slide, part 1 (about 2 min): the same two photos, before and after.** Both from cameras
never seen in training (cite: Beery et al. 2018, CDLA-Permissive 1.0):

| photo | before: phase 1, ResNet-18 arm B seed 2 | after: phase 3, ConvNeXt-Tiny + copy-paste seed 2 |
| --- | --- | --- |
| bobcat, day | `docs/figures/gradcam_overlay_01.png`: cat (0.79), heat on the animal | `docs/figures/p3_gradcam_overlay_01.png`: bobcat (0.84) |
| opossum, day | `docs/figures/gradcam_overlay_08.png`: coyote (0.22), heat on the background | `docs/figures/p3_gradcam_overlay_08.png`: squirrel (0.39), still wrong, heat now on the animal |

**Speaker notes, part 1.** The bobcat was wrong for a good reason (it looked at the animal and
saw a cat) and is now right. The opossum is still wrong, but for a better reason: the heat moved
off the scenery and onto the animal. Two photos are anecdotes, not evidence; the evidence is
slide 9's numbers. The numbers in parentheses are the model's confidence, its own estimate, not
a calibrated probability (calibration, item 12.03, is not done). The overlay number is the
photo's place in `assets/` in name order, not the index in the file name.

**Part 2 (about 4 min): the live demo.** The Space,
<https://huggingface.co/spaces/Codx999/who-goes-there>, serving phase 3's ConvNeXt-Tiny +
copy-paste seed 2 (switched on 2026-10-21, item 13.15; about 0.3 to 0.6 s a click on the Space's
CPU, Grad-CAM on included).

The script, saying before each click what I expect:

1. **Night coyote**, `trans_coyote_night_05`: coyote, 100% on the Space (2026-10-21). "Infrared,
   no colour, a camera it never saw."
2. **The bobcat with Grad-CAM on**, `trans_bobcat_day_01`: bobcat 84%, cat 16% on the Space
   (2026-10-21). "The photo from the slide before."
3. **The opossum, an honest failure**, `trans_opossum_day_10`: squirrel (0.39) in the scoring's
   Grad-CAM run of the same weights. "It will get this wrong, and here is where it looks."
4. **The live sort** (item 12.05): about 200 trans-test photos from a folder outside the repo,
   sorted into class folders with a visible timer and per-class counts:

```bash
uv run python -m trailcam.demo.sort --src <the demo photos> --dst <a new folder> --checkpoint <seed-2 checkpoint>.pt --config-name p3_convnext_cp
```

**Fallbacks, in order.**

1. **The Space.** It sleeps when idle: **wake it 10 minutes before the talk** and run one photo.
2. **The local app's `--share` link**, started before the talk if the Space is slow to wake:

```bash
uv run python -m trailcam.demo.app --checkpoint <seed-2 checkpoint>.pt --config-name p3_convnext_cp --share
```

3. **The recorded video** of the Space and the sort (item 12.04, the user's), open in a tab.

**Speaker notes, part 2.**

- The checkpoint's file name and sha256 are in `progress-log.md`, 2026-10-21. The local tools
  default to arm B's config, so `--config-name p3_convnext_cp` is required with this checkpoint.
- Rehearse on the Space and write down each photo's answer; the opossum's Space reading is not
  logged yet. On the scoring's run, the model gets 8 of the 10 `assets/` photos right; the other
  wrong one is the raccoon by day (`trans_raccoon_day_07`), called cat (0.92).
- A wrong answer you predicted is a better demo than a right one you hoped for.
- The live sort: read the counts out loud at the end; they are counts per predicted class, not a
  score (the script reads no labels).
- If the 12.05 rehearsal includes the 20-photo human baseline, it goes before the model answers.
- If the Space is ever rolled back to arm B seed 2 (the documented rollback): 5 of 10 right; the
  bobcat comes back as cat and the opossum as coyote; both coyotes are right.

---

## Slide 11. Limits and next steps (2 min)

**On the slide.**

- **Limits.** One dataset, one region, one species set. 3 seeds per recipe. trans_val is one
  camera. Confidences are not calibrated. The test sets have been used twice. Phase 3 changed two
  things at once.
- **Rare species:** no rodent right on trans_test, 0 of 58, on any seed.
- **Next steps, each from published work:**
  - **Detect, then classify:** MegaDetector (Beery, Morris, Yang 2019). Beery et al. 2018 found
    box crops more than halve the error. It changes the product: two models at inference.
  - **More cameras**, for training and for validation: one validation camera is too few. WILDS'
    iWildCam has 243 training cameras and still shows a gap, so more cameras help but do not cure
    it.
  - **Wildlife pretraining** (backbones trained on camera-trap or tree-of-life data), once their
    pretraining data is checked for CCT photos: a backbone that saw the trans cameras would inflate
    every trans score.
  - **Burst voting:** a camera fires a burst of frames; Dussert et al. 2024 found that averaging
    the logits over a camera-trap sequence before the softmax worked best in their calibration
    study.
  - **Rare species:** Beery et al. 2020 added synthetic animals for a rare class; on this dataset
    that route needs data from outside it.

**Speaker notes.** Detection was out of scope by design (plan section 1): the question was what
a single classifier can do. Each next step costs something: the detector changes deployment,
wildlife pretraining needs a data audit, burst voting needs the bursts at inference. Calibration
(Guo et al. 2017) would make the demo's confidence honest; it is item 12.03, not done.

---

## Slide 12. What I learned (2 min)

**On the slide.**

- **Reproduce before improving.** The baseline and its gap came first; every later number is a
  change from it.
- **One change at a time.** Phase 3's finalist broke this on purpose, and its result cannot say
  which change did the work.
- **Rules before results.** Written before the runs, the rule decided; I could not bend it after
  seeing a number I liked.
- **Control runs.** Re-train the baseline on new code before trusting any comparison.
- **Several seeds, several cameras.** One seed and one camera each pointed the wrong way at least
  once.
- **Most ideas don't work, so measure.** Of eight rungs, two were kept.

**Speaker notes.** These are process lessons, not results, and none of them is mine: they are
standard experimental practice. Learning them by doing was the point of the project. If I did it
again: a val set from a later time period, more than one validation camera, and the per-class
numbers read before quoting any gain.

---

## Slide 13. Credits (1 min)

**On the slide.** Every borrowed idea and its source:

| what I used | source |
| --- | --- |
| the problem, the CCT20 benchmark, the cis/trans split by camera, the box-crop result | Beery, Van Horn, Perona. "Recognition in Terra Incognita." ECCV 2018 |
| unseen-camera shift as a benchmark, macro-F1, different-day test from training cameras, ResNet-50 and 448 px baselines | Koh et al. "WILDS: A Benchmark of in-the-Wild Distribution Shifts." ICML 2021 |
| the copy-paste entry and the leaderboard numbers | WILDS leaderboard, iWildCam (wilds.stanford.edu/leaderboard) |
| the gap outside CCT; the floor of about 1,000 images per class | Schneider, Greenberg, Taylor, Kremer. "Three critical factors affecting automated image species recognition performance for camera traps." Ecology and Evolution, 2020 |
| the scale of the labelling problem | Norouzzadeh et al. "Automatically identifying, counting, and describing wild animals in camera-trap images with deep learning." PNAS, 2018 |
| detect-then-classify (MegaDetector); the review-time figures | Beery, Morris, Yang. "Efficient Pipeline for Camera Trap Image Review." 2019 |
| Grad-CAM | Selvaraju et al. "Grad-CAM." ICCV 2017 |
| the word "shortcut", and testing out of distribution | Geirhos et al. "Shortcut Learning in Deep Neural Networks." Nature Machine Intelligence, 2020 |
| targeted augmentation (copy-paste) | Gao, Sagawa, Koh, Hashimoto, Liang. "Out-of-Domain Robustness via Targeted Augmentations." ICML 2023 |
| copy-paste on camera traps; paste placement; day/night matching | Mesnage et al. "Copy-Paste Augmentation Improves Automatic Species Identification in Camera Trap Images." Ecology and Evolution, 2025 |
| ConvNeXt-Tiny | Liu, Mao, Wu, Feichtenhofer, Darrell, Xie. "A ConvNet for the 2020s." CVPR 2022 |
| burst voting (next steps) | Dussert, Chamaillé-Jammes, Dray, Miele. "Being confident in confidence scores: calibration in deep learning models for camera trap image sequences." Remote Sensing in Ecology and Conservation, 2024 |
| synthetic examples for rare classes (next steps) | Beery et al. "Synthetic Examples Improve Generalization for Rare Classes." WACV 2020 |
| calibration (next steps) | Guo et al. "On Calibration of Modern Neural Networks." ICML 2017 |
| ImageNet-pretrained ResNet-18 and ResNet-50, fine-tuning, AdamW, cosine schedule, random resized crops, head-first training, several seeds, rules written in advance | standard practice; the course (skillspire pytorch-deep-learning, chapters 04, 06 and 09) |
| the box-guided training crop and the attention-in-box score | suggested by the project's literature review, `docs/research-report.md` (written with AI research tools, checked against the papers above) |
| tools | PyTorch, torchvision, timm, Gradio, Hugging Face Spaces, Kaggle; built with Claude Code as a coding assistant |

- **Data:** Caltech Camera Traps (CCT20), Beery, Van Horn, Perona, ECCV 2018, via LILA BC.
  Licence CDLA-Permissive 1.0. **Code:** MIT. Repo and Space links.

**Speaker notes.** Read the first line and the last two aloud; leave the table up for questions.
Then: "Questions?"

---

## Backup slides

### B1. Phase 2 in detail

One baseline (phase 1's arm A recipe), one change per variant, judged on val and on trans_val by a
screen, confirm and keep rule written before any run. Numbers from `docs/phase2-plan.md` section 12
(macro-F1 changes; per seed where run):

| change | val | trans_val | verdict |
| --- | --- | --- | --- |
| copy-paste, p 0.5 | -0.0304 | +0.0971 | screened out (val fell) |
| copy-paste, p 0.25 | -0.0151 | +0.0493 | screened out (val fell) |
| head-first training | -0.0078 | +0.0177 | screened out (undertrained) |
| 320 px input, 3 seeds | +0.0163, +0.0119, +0.0017 (mean +0.0100) | +0.0271, +0.0088, +0.0116 (mean +0.0158) | not kept (under the 0.03 bar) |
| ResNet-50 | +0.0128, +0.0154 | +0.0197, -0.0219 | not kept (not up on every seed) |

**Notes.** The control re-run reproduced phase 1's arm A seed 0 bit for bit. Copy-paste's
trade-off on ResNet-18 was about 1 to 3, near-linear in the paste rate (half the rate gave half of
each). No step was kept, so phase 2 froze at the baseline and the test sets were not touched.
Caveat: trans_val is one camera, so these are one-camera results.

### B2. Arm B in detail

- Arm A (control): standard random resized crops. Arm B: for images with a box, half the time the
  crop contains the box and the box fills 20% to 100% of it; otherwise as arm A.
- `docs/figures/box_fill.png`, how much of the frame the boxes cover, arm B's range shaded (cite).
- Attention and animal-removed test, mean of 3 seeds (plan section 7):

| arm | attention-in-box, trans_test | trans_test accuracy | trans_test accuracy, box blanked |
| --- | --- | --- | --- |
| arm A | 0.2315 | 0.5754 | 0.2470 |
| arm B | 0.2331 | 0.6039 | 0.2613 |

- `docs/figures/attention_test_arm_a_seed0.png` and `docs/figures/attention_test_arm_b_seed0.png`.
- From the slices (seed 0): on the rare species (bird, rodent, skunk) arm B was lower, 0.3066
  against 0.3754 accuracy, so its gain is in the common classes.

### B3. Day vs night

trans_test accuracy by light, seed 0 (plan section 7; every seed in `docs/figures/slices_test.md`):

| model | accuracy, day | accuracy, night |
| --- | --- | --- |
| small CNN | 0.2961 | 0.3210 |
| ResNet-18 arm A | 0.5315 | 0.5058 |
| ResNet-18 arm B | 0.5618 | 0.5362 |

- Night (infrared, grey) is a few points harder for the ResNet-18s.
- The trans cameras shoot more at night, but re-weighting trans_test to the cis night share
  changes the gap by about 0.01 or less: not the explanation.
- `docs/figures/day_night.png` (cite).

### B4. Per class

`docs/figures/test_trans_per_class_arm_a_seed0.png`, `docs/figures/test_trans_per_class_arm_b_seed0.png`
and `docs/figures/p3_test_trans_per_class_seed2.png` (cite). Rare species (bird, rodent, skunk) are
under 4% of the training images; phase 3's rare-species macro-F1 on trans_test is 0.43 against
0.63 for the common ones (`docs/figures/p3_slices_test.md`).

### B5. Two errors from the confusion matrix

`docs/figures/test_trans_confusion_matrix_arm_b_seed0.png` (phase 1) and
`docs/figures/p3_test_trans_confusion_matrix_seed2.png` (phase 3), with two cells circled (cite):

1. **Bobcat predicted as cat**: a look-alike, both often at night; trans_val's camera called most
   of its bobcats cat.
2. **Raccoon predicted as opossum**: both grey, similar in size, mostly nocturnal; opossum is the
   largest training class, so in an infrared frame the model leans to it.

Read the counts off the figure; they are not on the slide as text. Neither error is fixed by
looking at the box more: the animals themselves look different across cameras, as Koh et al. 2021
found. `docs/figures/test_trans_confusion_matrix_small_cnn.png` shows what no pretraining looks
like.

### B6. What the features say

`docs/figures/features_by_camera_test.png` beside `docs/figures/features_by_species_test.png`, and
phase 3's `docs/figures/p3_features_by_camera_test.png`: the backbone's features of the test
photos in two dimensions. They still group by camera: the model's own representation knows which
camera a photo came from, the background shortcut seen from inside the network.

### B7. How the test sets were protected

- Phase 1: one scoring of all 7 checkpoints, a script that refuses a second run, the user's
  go-ahead, logged in `docs/test-ledger.md`.
- Phase 3 (`phase3-plan.md` section 7): one recipe, all three seeds, scored once; the finalist
  recomputed by the code from the ledgers, never typed in; its own test ledger,
  `docs/test-ledger-phase3.md`, so phase 1's run-once guard stayed as it was. Every seed in
  `docs/figures/p3_headline_test.md`.
- The data: `docs/figures/burst_length.png`, why the split is by burst (cite).

### B8. Q&A prep

Short, honest answers:

- **"What's new here?"** Nothing in the methods. The problem is Beery et al.'s and every method is
  published; the learning is in testing them properly, one at a time, under rules written before
  the runs.
- **"Isn't scoring the test set twice cheating?"** It costs some honesty, so it is disclosed: one
  recipe, chosen on val and trans_val only, scored once, shown beside phase 1, never instead
  (slide 9).
- **"Did you move the goalposts in phase 3?"** The val condition was loosened from 1 to 3 points
  after phase 2's results and before any phase-3 run, in writing. The finalist did not need it: its
  val rose (+0.0063).
- **"Was it the backbone or copy-paste?"** The test cannot say. On trans_val, one camera: +0.0457
  for the backbone, +0.0571 for copy-paste on top.
- **"Why not use the boxes at test, or a detector?"** Out of scope by design; detect-then-classify
  (MegaDetector, Beery, Morris, Yang 2019) is the deployed answer, and the first next step.
- **"Is arm B's gain real?"** Up on every seed, but small (+2.5 accuracy points), and the
  attention barely moved.
- **"Why is val so much higher than cis_test?"** Time shift: val came from the training weeks
  (slide 5).
- **"Why start with ResNet-18?"** Few training images, so pretraining matters; small enough for a
  CPU Space; the course's transfer-learning recipe; a standard Grad-CAM subject.
- **"Are the percentages probabilities?"** No; the model's own confidence, not calibrated.
- **"Why not BioCLIP or another wildlife backbone?"** Their pretraining may include CCT photos,
  which would inflate every trans score; that needs a data audit first.
- **"How did you use AI tools?"** The literature review and much of the code were written with AI
  assistance; the facts were checked against the papers, every decision went through rules
  written before the runs, and every number is traceable to a ledger row.

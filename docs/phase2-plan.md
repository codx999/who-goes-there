# Phase 2: improving generalisation to new cameras

Written 2026-10-17 (item 13.01), from the points agreed with the user through the coordinator
that day. It is a plan: nothing in it runs before the presentation on 2026-10-26, except the
optional exploratory run in section 8. `docs/plan.md` stays the source of truth for phase 1 (the
split, the reported runs, the single test scoring); this file owns phase 2. Phase 2 lifts two of
plan.md section 1's scope limits, for phase 2 only: backbones larger than ResNet-18, and more than
one training-time change.

Data: Caltech Camera Traps (CCT20), Beery et al. 2018, CDLA-Permissive 1.0. Phase-1 numbers below
are counts and scores from `docs/ledger.md`, `docs/plan.md` and `docs/data-card.md`, or derived
from them; model sizes are the published ones; hypotheses and run-time multipliers are estimates
and say so.

## 1. Goal

Raise the scores on cameras the model never saw (trans macro-F1 and accuracy) while losing no
more than about 1 point of val macro-F1 against the baseline, and keep the model deployable on the
Space's CPU: a prediction in under about one second per photo (plan.md sections 4 and 8, item
11.03). The bar stays the CPU's after the Space moves to ZeroGPU, so the model also runs on the
free fallback and on a laptop.

Baseline: phase 1's `resnet18_arm_a`, run seeds 0, 1 and 2 (section 2). Because phase 1 spends
the one test scoring, phase 2 measures "trans" on trans_val (section 4, the honesty note).

## 2. Phase-1 results so far, and the noise they show

The means of `docs/ledger.md` (`uv run python -m trailcam.train.ledger --check docs/ledger.md`
prints them). trans_val is one unseen camera (1,725 images, 9 of the 12 classes, macro-F1 over
the classes present); in phase 1 it is reported and never selected on (invariant R17). The test
results arrive with item 9.02 and go into plan.md section 7.

| model | runs (run seeds) | val accuracy | val macro-F1 | trans_val accuracy | trans_val macro-F1 |
| --- | --- | --- | --- | --- | --- |
| small CNN | 1 (0) | 0.6625 | 0.6216 | 0.1965 | 0.0775 |
| ResNet-18 arm A | 3 (0, 1, 2) | 0.8661 | 0.8557 | 0.3783 | 0.3101 |
| ResNet-18 arm B | 3 (0, 1, 2) | 0.8537 | 0.8428 | 0.3337 | 0.3026 |

On trans_val, arm B's mean macro-F1 is within the seed noise of arm A's, and its val macro-F1 is
lower on every seed (by 0.0097 to 0.0175). The experiment's verdict comes from the test scoring,
not from these rows. On both arms, trans_val sits far below val (arm A: 0.38 against 0.87
accuracy): that gap is what phase 2 works on.

**The noise.** Arm A's three seeds differ only in the run seed:

| run seed | val macro-F1 | trans_val accuracy | trans_val macro-F1 | best epoch (of 10) |
| --- | --- | --- | --- | --- |
| 0 | 0.8542 | 0.3724 | 0.2915 | 9 |
| 1 | 0.8559 | 0.3525 | 0.3187 | 8 |
| 2 | 0.8569 | 0.4099 | 0.3200 | 8 |
| range | 0.0027 | 0.0574 | 0.0285 | |

The bands used below, rounded up: **val macro-F1 ±0.003** and **trans_val ±0.03** (accuracy 0.35
to 0.41; the macro-F1 range is also about 0.03). Val is about ten times steadier because it holds
2,394 images from the ten training cameras; trans_val is one camera, where about 50 photos
changing class move accuracy by 0.03. The bands are arm A's, the baseline's: arm B's seeds spread
about three times wider on trans_val macro-F1 (0.2588 to 0.3472), so a recipe's own spread is
checked when it becomes the baseline (section 3).

## 3. How a change is judged: screen, confirm, keep

Every comparison is paired by run seed: the change's seed k against the current baseline's seed
k, on phase 1's split (split seed 42, split hash `c78a9c10…`).

- **Screen** on run seed 0. Continue to the confirm only if val macro-F1 does not drop (by more
  than its noise, 0.003) and trans_val macro-F1 is not clearly worse (not more than 0.03 below the
  baseline's seed 0). Otherwise the step stops: its row stays in the phase-2 ledger, marked
  screened out in the progress log, and the baseline is unchanged. (Open questions 1 and 2 ask
  whether these two conditions should change.)
- **Confirm** on run seeds 1 and 2 (seed 0 is the screen's run), so three seeds in all.
- **Keep** a change only if all three hold:
  1. the three-seed mean trans_val macro-F1 rises by more than the noise, 0.03;
  2. it rises on every seed (0, 1 and 2), paired with the baseline's same seed;
  3. the three-seed mean val macro-F1 drops by less than 1 point (0.01).

  Steps that change the cost of a prediction (1, 6 and 7) also pass the CPU latency check.
  Accuracy is reported beside macro-F1 everywhere, but the rule is on macro-F1, as in phase 1.

**Kept changes stack.** A kept change's three seeds become the baseline for the next step. If
their spread is clearly wider than the bands above, the bands are re-measured from them and the
progress log says so. The goal's val limit is cumulative: the stacked recipe stays within about 1
point of phase 1's 0.8557 val macro-F1, so two kept changes that each cost 0.7 points fail it.

Why 0.03 is strict but not arbitrary: from arm A's seeds, one run's trans_val macro-F1 varies by
about 0.016 (standard deviation), so a three-seed mean moves by about 0.009 by chance and the
difference of two such means by about 0.013. A rise of 0.03 is a little over two of those, and
"up on every seed" stops one lucky seed from carrying the mean. With three seeds the spread itself
is only a rough estimate.

## 4. Rules carried over from phase 1

- **One change per step**, so that when a score moves, one thing moved it (an ablation). Steps
  that name two settings are split: 2a and 2b, 3a and 3b.
- **A pinned commit per step.** The step's config and its hypothesis line are committed first;
  that commit is the notebook's `COMMIT` for every seed of the step, and the step's progress-log
  entry quotes it.
- **Each run is a row in `docs/ledger-phase2.md`**, with the columns of `docs/ledger.md`, and
  never in `docs/ledger.md`. The final scoring (`trailcam/evaluate/score_test.py`) relies on
  `docs/ledger.md` holding exactly the pre-registered runs: it refuses a ledger without a small
  CNN row and an arm A and an arm B row for each reported seed, or with a row of another seed, and
  scores every row's checkpoint in one run. The phase-1 checker also refuses one model under two
  config hashes, which is exactly what phase 2's steps are.
- **Config names start with `p2_`** (for example `p2_r320_arm_a`). The phase-1 configs and their
  config hashes stay as they are: `score_test.py` refuses a config whose hash no longer matches its
  ledger row, and `config_hash` hashes every field of `Config` but `arm` and `run_seed`, so adding
  a field changes every phase-1 hash. Each code step that adds a setting gives it a form that
  leaves the phase-1 hashes unchanged, with a test that pins them.
- **Choices on val and trans_val only.** Training is unchanged: early stopping and the restored
  checkpoint still follow val macro-F1 (R17). Only the keep-or-drop decision of a step looks at
  trans_val. cis test and trans test are not touched.
- **Same split and seeds**: split seed 42 and run seeds 0, 1 and 2, as in phase 1.

**Honesty note.** Phase 1 spends the one test scoring (9.02). Phase 2's numbers are therefore
trans_val, and since phase 2 selects on trans_val, they are slightly optimistic: the best of
several tries looks better than it will on the next new camera. WILDS flags the same bias when
hyperparameters are tuned on the out-of-distribution val split (Koh et al. 2021,
`docs/research-report.md`). trans_val is also one camera, so a phase-2 gain is a gain on one
unseen camera, and every slide or README line that quotes one says so. This changes trans_val's
role from phase 1's R17 on purpose, for phase-2 rows only. Any second test scoring would need the
user's explicit decision, disclosed wherever its numbers appear (the recipe was chosen after the
first scoring was seen); `score_test.py` refuses a test ledger that already holds a row, so it
cannot happen by accident.

## 5. A hypothesis line before every run

Before a step's first run, one line goes into the step's progress-log entry, in the pinned
commit, so it is dated before any result:

```text
Step 1a, resolution 320. Expect: trans_val macro-F1 up about 0.03 (+0.02 to +0.05), val
macro-F1 up 0.005 to 0.01, run time about 2x. Because: at 224 px many animals are a few pixels wide.
```

After the screen (and the confirm, if it runs), the result goes next to it with one word
(matched, smaller, none, opposite) and three to five lines in the learning journal's topic "What
we'd improve next, and how": what was expected, what came out, and what that teaches. A wrong
guess is the useful case: it shows where the mental model of the problem is off.

## 6. The ladder

**Provisional, pending the literature review** of improvement methods for new-location shift (the
user's decision of 2026-10-17: an external research pass, to become Part 7 of
`docs/research-report.md`). The steps and their order will be re-ranked from its evidence.

In this order for now. Each step is judged against the baseline of the moment (section 3).
`config.yaml` (and a `configs/` folder) is in `scripts/risk_files.sh`'s regex, so even a
config-only step is tier 2 by the script; its review is one config block and its test.

| step | change | needs | review tier | runs (min to max) |
| --- | --- | --- | --- | --- |
| 1a | input 320 px | code | 2 | 1 to 3 |
| 1b | input 384 px, only if 1a is kept | config, on 1a's code | 2 | 0 to 3 |
| 2a | label smoothing 0.1 | code (`train.py`) | 2 | 1 to 3 |
| 2b | weight decay 5e-2 (from 1e-2) | config only | 2 | 1 to 3 |
| 3a | stronger colour jitter | code (transforms) | 2 | 1 to 3 |
| 3b | random erasing | code (transforms) | 2 | 1 to 3 |
| 4 | class-balanced loss (low expected value) | code (`train.py`), high-stakes | 2 | 1 to 3 |
| 5 | longer, gentler training | config only | 2 | 1 to 3 |
| 6 | ConvNeXt-Tiny or ResNet-50 | code (model, Grad-CAM, Space) | 2 | 1 to 3 |
| 7 | seed ensemble, test-time flip | code (evaluation) | 2 | 0 (no training) |
| 8 | arm B on the final recipe | config only | 2 | 3 |

Our own report already cites WILDS (`docs/research-report.md`, Part 3): on unseen cameras, plain
training (ERM) was not meaningfully beaten by the shift-robustness methods tested (CORAL, IRM,
Group DRO), and reweighting did not help (Part 4). So step 4 is marked low expected value until
the review says otherwise.

**1a. Resolution 320.** Train crop and eval centre crop at 320 px, the eval resize at 366
(320 x 256/224, the same margin as now). *Needs code*: the 224 px input is a constant shared by
the train transform of both arms, the eval transform, the attention frame, Grad-CAM, the demo and
the Space, so the input size becomes a setting that the checkpoint carries. *Hypothesis*: at 224 px
many animals are a few pixels wide (`docs/research-report.md`, part 2, which names resolution the
first thing to try); more pixels on the animal should help unseen cameras most: trans_val macro-F1
+0.02 to +0.05, val +0.5 to 1 point, about 2x the run time and the CPU latency (still far under a
second for ResNet-18).

**1b. Resolution 384**, only if 1a is kept. *Needs*: a config on 1a's code. *Hypothesis*: a
smaller further gain, +0.01 to +0.03 over 320, at about 3x the 224 px run time; every later step
would then cost about 3x too, which the budget (section 7) has to absorb. The 512 px cache (longer
side 512, so the shorter side is 373 or 384 for the two capture sizes) covers 320 fully; at 384
the train crops are fine, but the eval resize to 439 upsamples by about 15%. 1b's item decides
between that and resizing straight to 384.

**2a. Label smoothing 0.1.** *Needs code*: a loss setting in `trailcam/train/train.py`. Plan.md
section 9 kept it out of phase 1 as a second variable and a calibration tool. *Hypothesis*: softer
targets make the model less sure of camera-specific cues: trans_val macro-F1 flat to +0.02, val
within ±0.5 point; the demo's confidences drop, which item 12.03's calibration would see.

**2b. Weight decay 5e-2.** *Config only* (`weight_decay`). *Hypothesis*: no change beyond the
noise. AdamW shrinks each weight by the learning rate times the decay at each step; at a body rate
of at most 1e-4 over about 1,900 body steps (nine epochs of about 212 batches) that is about 1% in
total at 5e-2, against 0.2% now. A
cheap screen of a small predicted effect; a large move would mean something else is going on.

**3a. Stronger colour jitter.** *Needs code*: the jitter strength is a constant in
`trailcam/data/dataset.py`. For example brightness and contrast from 0.2 to 0.4; saturation does
nothing on the infrared frames, about two thirds of the images. *Hypothesis*: cameras differ in
exposure and infrared lighting, so trans_val macro-F1 flat to +0.02, val 0 to -0.5 point.

**3b. Random erasing.** *Needs code*: a random patch of the normalised training tensor blanked
(for example with probability 0.25), never in the eval transform (R20). *Hypothesis*: the
model can rely less on any one region, background included: trans_val flat to +0.02, val about
-0.5 point.

**4. Class-balanced loss.** *Needs code*: cross-entropy weighted per class in `train.py` (inverse
frequency, or the effective-number weights of Cui et al., CVPR 2019; the item chooses before the
run). High-stakes: the unweighted loss is a fixed control of phase 1 (plan.md sections 5 and 9).
*Hypothesis*: the rare classes (skunk 229, rodent 375, bird 484 training images) gain recall, so
val macro-F1 0 to +1 point and val accuracy 0 to -1 point; trans_val macro-F1 flat, since WILDS
found reweighting does not beat plain training on unseen cameras and the research report predicts
"no change" for this direction. Low expected value until the literature review says otherwise.

**5. Longer, gentler training.** *Config only*: `max_epochs` 20 (from 10), `patience` 5 (from 3),
`lr_body` 5e-5 (from 1e-4). Counted as one change, "more, smaller steps" (open question 3).
*Hypothesis*: phase 1's best epochs were 8 to 10 of 10 on every run, so training stopped near
the cap (partly because cosine annealing makes its smallest steps last); a lower body rate keeps
more of the ImageNet features. Val +0.5 point, trans_val flat to +0.02, run time up to 2x (early
stopping may end it sooner).

**6. Backbone.** ConvNeXt-Tiny (timm, already a dependency; 28.6M parameters, about 4.5 GFLOPs
at 224 px) or ResNet-50 (torchvision; 25.6M, about 4.1 GFLOPs), against ResNet-18's 11.7M and 1.8
GFLOPs. One of them, picked by the CPU latency check with random weights before any GPU time.
*Needs code*: the model factory, the Grad-CAM target layer, the parameter groups and the Space's
checks. ImageNet-1k weights only, so plan.md section 4's argument (ImageNet holds no CCT images)
still holds; ImageNet-22k weights would need the same check. *Hypothesis*: a stronger ImageNet
backbone transfers better: trans_val +0.02 to +0.05, val +0.5 to 1 point, about 2.5x the run time
and the CPU latency. At 384 px that is about 7x ResNet-18 at 224 px, and Grad-CAM doubles a
request, so the latency check can decide this step on its own.

**7. Seed ensemble and test-time flip**, at evaluation only. *Needs code* in
`trailcam/evaluate/`: average the softmax of the kept recipe's three seed checkpoints, and of each
photo and its mirror image. No training run. *Hypothesis*: ensemble +0.01 to +0.03 on trans_val
macro-F1, flip 0 to +0.01, val +0.5 point. The flip is judged per seed with the usual rule. The
ensemble is one model with no seed spread, so it is reported beside its three members and kept
only if it beats each of them on trans_val macro-F1 and val. A request costs six forward passes (3
models x 2 views), so the latency check decides whether the Space serves it, and which member its
Grad-CAM shows.

**8. Arm B on the final recipe.** *Config only*: a `p2_` arm B config identical to the final arm
A recipe apart from `arm` (R25), seeds 0, 1 and 2, compared with arm A's seeds by the same rule.
*Hypothesis*: no difference beyond the noise, as on phase 1's trans_val; at a higher resolution
the animal already fills more pixels, which may leave box-guided crops less to add. This asks
phase 1's question again on a stronger recipe; it is trans_val evidence, not the phase-1 result.

## 7. Budget

About 12 to 18 runs at 10 to 25 minutes each, so 3 to 5 GPU hours of the Kaggle quota's 30 a week.
Each code step is one item with its own review; config-only steps (2b, 5, 8, and 1b) can share one
item and one review.

Run time, measured in phase 1 and estimated after: a ResNet-18 run at 224 px took about 12
minutes, start-up included (the gaps between consecutive start times in item 7.03's run ids). The
multipliers below assume the GPU is the bottleneck; the first run of each step measures them.

| change | run time against now |
| --- | --- |
| 320 px | about 2x |
| 384 px | about 3x |
| 20 epochs, patience 5 | up to 2x |
| ConvNeXt-Tiny or ResNet-50 | about 2.5x |

They multiply when changes stack: 384 px with ConvNeXt-Tiny is roughly 7x, over an hour a run.

The run count depends on the screen. The ladder costs 11 runs if every screen stops a step and 30
if every step reaches its confirm. With the screen of section 3, a change that does nothing passes
most of the time (one seed seldom falls 0.03 below the baseline by chance), so the budget, not the
screen, is likely to end the ladder. The rule: before each run, write its projected minutes (the
last run's time times the step's multiplier) and the running total in the progress log; the ladder
stops when the next step's confirm would take the total past 5 GPU hours, and the steps not reached
go to future work (section 10). The user can raise the cap; the quota allows it.

## 8. Timing

Phase 1 finishes first: the evaluation on val and trans_val (8.06), the single test scoring
(9.02), the Space (11.02, 11.03), the slides and README (12.01, 12.02), the presentation on
2026-10-26. Phase 2 starts after that: the infrastructure (13.02, due 2026-10-28), then one item
per code step.

**Option: step 1a alone before the presentation**, as one labelled exploratory run (seed 0),
reported on trans_val only and labelled "exploratory, one seed, one unseen camera, not test" on
the slide. It is not config-only: it needs step 1a's code (tier 2) and 13.02's separate ledger,
both in the week of 9.02, the Space and the slides; and its new input-size setting must leave the
phase-1 config hashes unchanged, or 9.02's scoring refuses to run. Recommended only once 9.02 is
scored and 12.01 and 12.02 are done; otherwise the slides name resolution as the first next step,
with the research report's reasoning.

## 9. Infrastructure first (item 13.02, not built here)

Before the first phase-2 run, item 13.02 (tier 2: `trailcam/train/`, the notebook and tests)
adds:

- `docs/ledger-phase2.md`, with the header and columns of `docs/ledger.md` (`append_row` refuses
  a ledger whose header differs);
- the ledger checker (`trailcam/train/ledger.py --check`) accepting it: the same row checks, but
  several config hashes per model, since every step is a new recipe; R25's arm pairing still
  holds for step 8;
- `notebooks/kaggle_train.ipynb` taking an optional `LEDGER` parameter (today the notebook sets
  `LEDGER` to `docs/ledger.md` in the clone; phase 1's default stays);
- the `p2_` prefix for phase-2 config names;
- tests that a phase-2 row can never land in `docs/ledger.md`: neither appended by `train.py` nor
  pasted by hand (rows are pasted by the user, so the check of `docs/ledger.md` refuses a `p2_`
  row and fails `check.sh`).

Not in 13.02 as agreed, and needed by the first code step that adds a setting (section 4): the
form that keeps the phase-1 config hashes unchanged, and the test that pins them. CLAUDE.md's
"every choice is made on val" describes phase 1; whether it gets a phase-2 line is open question 5.

## 10. Beyond the ladder (future work, for the slides' closing)

- **A detector crop at test time.** The literature's fix: Beery et al. 2018 found that classifying
  box crops more than halves both errors, and the deployed pipeline is detect, crop, then classify
  (Beery, Morris, Yang 2019). It changes the product: two models, a detector on the CPU, boxes at
  inference, all out of phase 1's scope.
- **Camera-trap-pretrained backbones.** Promising, but they need an audit of their pretraining
  data for CCT images first: several are trained on LILA collections, where CCT itself is hosted,
  and a backbone that saw the trans cameras in pretraining would inflate every trans score. That
  is the leakage trap.
- **Calibration** (item 12.03): temperature scaling fit on val, with expected calibration error on
  cis and trans. It would tell whether the demo's confidences can be trusted on new cameras.

## 11. Open questions for phase 2

1. **The screen's val condition.** As agreed, the screen stops a change whose val macro-F1 drops
   by more than the noise (0.003), but the keep rule accepts a drop of up to 1 point. Regularisation
   and augmentation (steps 2 and 3) are exactly the changes that may trade a little val for trans,
   so the screen could stop a change the keep rule would accept. Recommended: the screen's val
   condition uses the keep rule's 1 point. The user's decision.
2. **The screen's trans_val condition.** "Not more than 0.03 worse" passes most changes that do
   nothing (section 7). "Not below the baseline's seed 0" would stop about half of them and keep
   the ladder near the 12 to 18 runs budgeted, at the cost of sometimes stopping a real but small
   gain. The user's decision.
3. **Step 5 as one change.** Twenty epochs, patience 5 and a lower body rate are counted as one
   idea; split into three steps if a kept step 5 needs explaining and the budget allows.
4. **The Space after phase 2.** Whether a phase-2 model replaces the phase-1 model on the Space;
   if it does, the Space's card quotes its trans_val numbers with the honesty note, not phase 1's
   test numbers.
5. **CLAUDE.md and R17.** Both say every choice is made on val; phase 2 decides steps on val and
   trans_val (section 4). Whether CLAUDE.md and `docs/review/invariants.md` get a phase-2 line
   (in 13.02 or before the first step) is the user's decision.

## Sources

- Beery, Van Horn, Perona. "Recognition in Terra Incognita." ECCV 2018.
  https://arxiv.org/abs/1807.04975
- Koh et al. "WILDS: A Benchmark of in-the-Wild Distribution Shifts." ICML 2021.
  https://arxiv.org/abs/2012.07421
- Beery, Morris, Yang. "Efficient Pipeline for Camera Trap Image Review." 2019.
  https://arxiv.org/abs/1907.06772
- Cui, Jia, Lin, Song, Belongie. "Class-Balanced Loss Based on Effective Number of Samples." CVPR
  2019.
- The literature review behind phase 1, with the numbers: `docs/research-report.md`.

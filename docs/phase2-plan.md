# Phase 2: improving generalisation to new cameras

Written 2026-10-17 (item 13.01), from the points agreed with the user through the coordinator
that day; the ladder (section 6) was revised the same day from an external literature review, at
the user's request. It is a plan: nothing in it runs before the presentation on 2026-10-26, except
the optional exploratory result in section 8. `docs/plan.md` stays the source of truth for phase 1
(the split, the reported runs, the single test scoring); this file owns phase 2. Phase 2 lifts two
of plan.md section 1's scope limits, for phase 2 only: backbones larger than ResNet-18, and more
than one training-time change.

Data: Caltech Camera Traps (CCT20), Beery et al. 2018, CDLA-Permissive 1.0. Phase-1 numbers below
are counts and scores from `docs/ledger.md`, `docs/plan.md` and `docs/data-card.md`, or derived
from them; model sizes are the published ones; hypotheses and run-time multipliers are estimates
and say so. Literature claims are marked **verified** (checked against the original page by the
coordinator on 2026-10-17) or **to verify in 13.03**; nothing else is cited as verified.

## 1. Goal

Raise the scores on cameras the model never saw (trans macro-F1 and accuracy) while losing no
more than about 1 point of val macro-F1 against the baseline, and keep the model deployable on the
Space's CPU. The budget is a prediction in under about one second per photo (plan.md sections 4
and 8, item 11.03); it is a budget, not today's cost: ResNet-18 at 224 px is expected to take tens
of milliseconds on the CPU (to be measured at 11.03), so the CPU is not the blocker. The bar stays
the CPU's after the Space moves to ZeroGPU, so the model also runs on the free fallback and on a
laptop.

Baseline: phase 1's `resnet18_arm_a`, run seeds 0, 1 and 2 (section 2). Because phase 1 spends
the one test scoring, phase 2 measures "trans" on trans_val (section 4, the honesty note).

## 2. Phase-1 results so far, and the noise they show

The means of `docs/ledger.md` (`uv run python -m trailcam.train.ledger --check docs/ledger.md`
prints them). trans_val is one unseen camera: 1,603 kept images of the 1,725 released (the data
card's filter table), 9 of the 12 classes, macro-F1 over the classes present. In phase 1 it is
reported and never selected on (invariant R17). The test results arrive with item 9.02 and go
into plan.md section 7.

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
- **Confirm** on run seeds 1 and 2 (seed 0 is the screen's run), so three seeds in all; on five
  seeds (run seeds 0 to 4) for steps 2 to 6 (see "Resolvability" below).
- **Keep** a change only if all three hold:
  1. the mean trans_val macro-F1 over the confirm's seeds rises by more than the noise, 0.03;
  2. it rises on every seed, paired with the baseline's same seed;
  3. the mean val macro-F1 drops by less than 1 point (0.01).

  Steps that change the cost of a prediction (0, 4 and 5) also pass the CPU latency check.
  Accuracy is reported beside macro-F1 everywhere, but the rule is on macro-F1, as in phase 1.

**Kept changes stack.** A kept change's seeds become the baseline for the next step. If their
spread is clearly wider than the bands above, the bands are re-measured from them and the progress
log says so. The goal's val limit is cumulative: the stacked recipe stays within about 1 point of
phase 1's 0.8557 val macro-F1, so two kept changes that each cost 0.7 points fail it.

**Resolvability.** From arm A's seeds, one run's trans_val macro-F1 varies by about 0.016
(standard deviation), so the difference of two three-seed means varies by about 0.013 by chance;
with arm B's spread (about 0.044) it would be about 0.036. So with three seeds, **a gain under
about 4 points cannot be confirmed** on trans_val, and the 0.03 threshold plus "up on every seed"
is the least that separates a gain from a lucky seed. Steps 2 to 6 expect smaller gains (section
6), so they confirm on five seeds before any gain is claimed; the baseline then needs run seeds 3
and 4 too, two extra runs each time the baseline changes. A per-camera breakdown cannot replace
the extra seeds: trans_val is one camera (plan.md section 3). With three seeds the spread itself is
only a rough estimate.

## 4. Rules carried over from phase 1

- **One change per step**, so that when a score moves, one thing moved it (an ablation).
- **A pinned commit per step.** The step's config and its hypothesis line are committed first;
  that commit is the notebook's `COMMIT` for every seed of the step, and the step's progress-log
  entry quotes it.
- **Each run is a row in `docs/ledger-phase2.md`**, with the columns of `docs/ledger.md`, and
  never in `docs/ledger.md`. The final scoring (`trailcam/evaluate/score_test.py`) relies on
  `docs/ledger.md` holding exactly the pre-registered runs: it refuses a ledger without a small
  CNN row and an arm A and an arm B row for each reported seed, or with a row of another seed, and
  scores every row's checkpoint in one run. The phase-1 checker also refuses one model under two
  config hashes, which is exactly what phase 2's steps are.
- **Config names start with `p2_`** (for example `p2_copypaste_arm_c`). The phase-1 configs and
  their config hashes stay as they are: `score_test.py` refuses a config whose hash no longer
  matches its ledger row, and `config_hash` hashes every field of `Config` but `arm` and
  `run_seed`, so adding a field changes every phase-1 hash. Each code step that adds a setting
  gives it a form that leaves the phase-1 hashes unchanged, with a test that pins them.
- **Choices on val and trans_val only.** Training is unchanged: early stopping and the restored
  checkpoint still follow val macro-F1 (R17). Only the keep-or-drop decision of a step looks at
  trans_val. cis test and trans test are not touched.
- **Same split and seeds**: split seed 42 and run seeds 0 to 2 (to 4 where section 3 says), as in
  phase 1.

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
Step 1, copy-paste arm C v1. Expect: trans_val macro-F1 up about 0.03 (0 to +0.06), val flat
to -0.005. Because: animals on new backgrounds weaken the background shortcut; our backgrounds
are training cameras only, so less than the published gains.
```

After the screen (and the confirm, if it runs), the result goes next to it with one word
(matched, smaller, none, opposite) and three to five lines in the learning journal's topic "What
we'd improve next, and how": what was expected, what came out, and what that teaches. A wrong
guess is the useful case: it shows where the mental model of the problem is off.

## 6. The ladder

Revised 2026-10-17 from an external literature review (the user ran three AI research tools; the
coordinator checked the facts marked verified). Item 13.03 writes that review up as Part 7 of
`docs/research-report.md`, verifies the rest, and re-checks this ladder against it.

### The evidence

| source | what it shows | status |
| --- | --- | --- |
| WILDS leaderboard, iWildCam, OOD test macro-F1, ResNet-50 at 448 px | ERM (grid search) 30.8 (1.3); ERM + RandAugment 32.2 (1.2); ERM + targeted augmentation (copy-paste using MegaDetector; the entry is marked as deviating from the guidelines) 36.5 (0.9); ABSGD 33.0 (0.6); CORAL 32.7 (0.2); Group DRO 23.8; IRM 15.1; MixUp 13.8 (0.8); JTT 11.0. With unlabeled data: Connect Later 36.8, ICON 34.5 | verified |
| Gao, Sagawa, Koh, Hashimoto, Liang, ICML 2023 | targeted augmentations randomise the spurious, domain-dependent features and keep the robust ones; a new OOD state of the art by 3.2 to 15.2% on three real-world datasets | verified |
| Mesnage et al. 2025, Ecology and Evolution | copy-paste on Snapshot Serengeti, location-split cross-validation, a YOLOv5s **detector** (mAP, not classification); animals segmented (U²-Net) from training locations and pasted onto empty backgrounds **from the test locations**; mean mAP +0.0156 ± 0.005 (an 8% ± 2% gain); 8 of 47 species worse, mostly nocturnal, put down to night animals pasted onto day backgrounds | verified |
| Qu and Xie, Connect Later, ICML 2024 | fine-tuning with targeted augmentations, using unlabeled data; +0.9% on iWildCam with ResNet-50 | verified |
| SWAD (Cha et al., NeurIPS 2021) | TerraIncognita (built from Caltech Camera Traps locations): ERM 46.1, SWAD 50.0 | to verify in 13.03 |
| LP-FT (Kumar et al., ICLR 2022) | mixed on camera traps: +0.9 in Connect Later's ablation, -0.3 in a 2025 CLIP study | to verify in 13.03 |
| Gadot et al. 2024, IET Computer Vision | about +25% relative macro-F1 from detector crops | to verify in 13.03 |
| Model soups (Wortsman et al., ICML 2022) | averaging the weights of fine-tuned models | to verify in 13.03 |

Every WILDS baseline ran at 448 px, so part of those numbers may be resolution. Our report quotes
the WILDS paper's figures (ERM 31.0, CORAL 32.8, Group DRO 23.9, IRM 15.1; Part 3); the
leaderboard's differ by 0.1 to 0.2. Two cautions for this project: the leaderboard's copy-paste
entry used a detector, and Mesnage's backgrounds came from the test locations, which our rules
forbid; and Mesnage measured a detector, not a classifier.

### The steps

In this order. Each step is one change, judged against the baseline of the moment (section 3).
`config.yaml` (and a `configs/` folder) is in `scripts/risk_files.sh`'s regex, so even a
config-only step is tier 2 by the script; its review is one config block and its test. Two of
the review's labels are corrected here from the code: resolution (step 4) and the infrared-aware
augmentation (step 6) need code, since the input size and the augmentation strengths are
constants in `trailcam/data/dataset.py`, not config fields.

| step | change | evidence | needs | review tier | runs |
| --- | --- | --- | --- | --- | --- |
| 0 | weight soup and softmax ensemble of arm A's 3 seeds | soups to verify | code (evaluation) | 2 | 0 (evaluation only) |
| 1 | copy-paste, arm C (v1 rectangles; v2 masks if v1 is promising) | strongest | code (new arm, background pool, tests) | 2 at xhigh; 3 recommended | v1 1 to 3, v2 0 to 3 |
| 2 | SWA over the late epochs | to verify | code (`train.py`) | 2 | 1 to 5 |
| 3 | longer head-only phase (2 to 3 epochs) | mixed | config only | 2 | 1 to 5 |
| 4 | input 320 px | weak | code (input size) | 2 | 1 to 5 |
| 5 | ResNet-50 at 224 px | weak | code (model entry) and config | 2 | 1 to 5 |
| 6 | infrared-aware greyscale and photometric augmentation | weak | code (transforms) | 2 | 1 to 5 |

**0. Soup and ensemble of arm A's three seeds**, evaluation only, about 0 GPU. *Needs code* in
`trailcam/evaluate/`: average the three checkpoints' weights into one model (a soup), and average
their softmax outputs (an ensemble), each scored on val and trans_val. *Caveat*: the seeds started
from different random heads, so a plain soup may fail; test it, don't assume it. *Hypothesis*:
ensemble +0.01 to +0.03 trans_val macro-F1 over its members' mean; the soup either near the
ensemble or far below its members. A soup or ensemble is one model with no seed spread, so it is
reported beside its three members, and the soup is kept only if it beats each member on trans_val
macro-F1 and val. The ensemble is report-only (three forward passes per photo; see "Dropped").
Step 0 changes no training, so it does not stack: a soup that works is applied to the final
recipe's seeds at the end. Its scores go in the progress log with the checkpoints' sha256.

**1. Copy-paste, arm C**, the strongest evidence (the leaderboard's targeted augmentation, Gao
2023, Mesnage 2025). For about half of the boxed training images (probability 0.5, as in arm B),
the box region of the training animal is pasted onto an empty frame from a training camera, night
onto night and day onto day by the sidecar's night flag (Mesnage's nocturnal failures). v1 pastes
the box rectangle (cheap); v2, a separate step only if v1's screen is promising, pastes a
segmentation mask (its segmentation model needs a check of its own pretraining data). *Needs code*:
a new arm and transform with tests, and the background pool. **The background pool is a leakage
boundary**, so it gets its own rules and tests:

- only the 995 empty frames of the official cis_val file: official train has none, cis_test's
  1,180 are test images (on the training cameras, but test), and trans_val's 61 and trans_test's
  1,778 are on unseen cameras (the data card's filter table); unlike Mesnage, never a test-location
  or test-split frame;
- minus any frame whose sequence (`seq_id`) has a frame in the val split, since a burst's empty
  frame shows the val photo's scene seconds apart;
- tests: every background's location is a training location; no background's image id or
  `seq_id` is in val, cis_test, trans_val or trans_test; the pool is a function of the metadata and
  the split seed, its size and hash printed (counts only).

The cache and sidecar already hold every annotated image, empties included
(`scripts/kaggle_prepare_cct20.py`, `scripts/kaggle_pack_cache.py`), so no cache rebuild; the
metadata loader drops empties today, so the pool needs a reviewed path to them. R23 and R24 name arm B as
the only box user at training time; arm C's item extends them. Arm C is compared with arm A on
identical configs apart from `arm` (R25's rule). Tier 2 at xhigh by the playbook; tier 3
(`/code-review ultra` as well) is recommended because the pool is a new data boundary like the
split, and is automatic if the code touches `trailcam/data/split.py` or its tests. *Hypothesis*:
trans_val macro-F1 0 to +0.06 (about +0.03), val flat to -0.005; smaller than the leaderboard's
+5.7 because our backgrounds are training cameras only, no detector, 224 px. Three seeds; if the
result is positive but under about 4 points, seeds 3 and 4 are added before it is claimed.

**2. SWA over the late epochs** (`torch.optim.swa_utils`: a running average of the weights over
the last epochs, batch-norm statistics recomputed on the training data). *Needs code* in
`train.py`; the item decides how the average meets early stopping (for example, scored on val as
one more candidate at the end). Evidence: SWAD on TerraIncognita (to verify). *Hypothesis*:
trans_val 0 to +0.03, val flat. Five seeds.

**3. A longer head-only phase**, 2 or 3 epochs instead of 1 (`head_only_epochs`), *config only*.
Head-only epochs count inside `max_epochs` (plan.md section 5), so `max_epochs` rises with them and
the body still trains 9 epochs: the one change is the head-only phase. Evidence mixed (LP-FT, to
verify). *Hypothesis*: trans_val flat to +0.01, val flat. Five seeds.

**4. Resolution 320.** Train crop and eval centre crop at 320 px, the eval resize at 366 (320 x
256/224, the same margin as now; the 512 px cache's shorter side is 373 or 384, so no upsampling).
*Needs code*: the 224 px input is a constant shared by the train transform of both arms, the eval
transform, the attention frame, Grad-CAM, the demo and the Space, so the input size becomes a
setting that the checkpoint carries. Evidence weak (the WILDS baselines' 448 px may be part of
their numbers); at 224 px many animals are a few pixels wide (`docs/research-report.md`, Part 2).
*Hypothesis*: trans_val 0 to +0.03, val +0.5 point, about 2x the run time and the CPU latency.
Five seeds.

**5. ResNet-50 at 224 px** (torchvision, ImageNet-1k weights; 25.6M parameters and about 4.1
GFLOPs, against ResNet-18's 11.7M and 1.8). *Needs code*: a model entry in
`trailcam/model/factory.py`, its parameter groups and Grad-CAM layer, the Space's checks; and a
config. ImageNet-1k weights keep plan.md section 4's argument (ImageNet holds no CCT images).
*Hypothesis*: trans_val 0 to +0.03, val +0.5 point, about 2.3x the run time and the CPU latency,
measured before the GPU run. Five seeds.

**6. Infrared-aware augmentation**: greyscale and photometric changes aimed at infrared frames
(about two thirds of the images), for example a higher greyscale probability and contrast or gamma
jitter. *Needs code*: the strengths are constants in `trailcam/data/dataset.py`; never in the
eval transform (R20). Evidence weak. It is never combined with arm C's day/night matching in one
run: if arm C is kept, step 6 is screened on the recipe without arm C and the two are not stacked
(the user picks if both pass). *Hypothesis*: trans_val flat to +0.01, val flat. Five seeds.

### Dropped, with reasons

- **Class-balanced loss, focal loss, logit adjustment, LDAM**: no gain expected under location
  shift. Our report already cites WILDS (Part 3: plain training not meaningfully beaten by CORAL,
  IRM or Group DRO) and that reweighting did not help (Part 4); the review's claims about the other
  three are to verify in 13.03, as is whether ABSGD (33.0 on the leaderboard, 2.2 above ERM) is a
  reweighting method.
- **MixUp**: 13.8 on the leaderboard against ERM's 30.8 (verified).
- **IRM, Group DRO and other domain-generalisation objectives**: 15.1 and 23.8 against 30.8
  (verified); CORAL's 32.7 is within two points of ERM.
- **Wildlife- or biology-pretrained backbones** (SpeciesNet, BioCLIP, DINOv2): their pretraining
  overlap with CCT is unknown, so a leakage risk (CCT is published on LILA, beside the other
  camera-trap collections such models may train on); the review says SpeciesNet is a crop
  classifier (to verify in 13.03).
- **Test-time augmentation and seed ensembles for the demo**: three to five forward passes per
  photo on the CPU; step 0's ensemble is reported, not deployed.
- **Not carried over from the first draft**: label smoothing, weight decay 5e-2, stronger colour
  jitter, random erasing, longer and gentler training, ConvNeXt-Tiny; and arm B on the final recipe,
  which the revised ladder leaves out (open question 4).

## 7. Budget

About 12 to 18 runs at 10 to 25 minutes each, so 3 to 5 GPU hours of the Kaggle quota's 30 a week.
Each code step is one item with its own review; config-only steps can share one item and one
review.

Run time, measured in phase 1 and estimated after: a ResNet-18 run at 224 px took about 12
minutes, start-up included (the gaps between consecutive start times in item 7.03's run ids). The
multipliers below assume the GPU is the bottleneck; the first run of each step measures them.

| change | run time against now |
| --- | --- |
| arm C (copy-paste) | about 1x (the paste is data loading on the CPU) |
| SWA | about 1.1x (one extra pass for the batch-norm statistics) |
| head-only 2 to 3 epochs | about 1.1x to 1.2x (one or two more epochs, cheaper ones) |
| 320 px | about 2x |
| ResNet-50 | about 2.3x |

They multiply when changes stack: 320 px with ResNet-50 is roughly 4.6x, about 55 minutes a run.

The run count depends on the screen and on the five-seed confirms. The ladder costs 6 runs if
every screen stops a step, and about 40 if every step reached its full confirm (with the
baseline's extra seeds). With the screen of section 3, a change that does nothing passes most of
the time (one seed seldom falls 0.03 below the baseline by chance), so the budget, not the screen,
is likely to end the ladder. The rule: before each run, write its projected minutes (the last
run's time times the step's multiplier) and the running total in the progress log; the ladder stops
when the next step's confirm would take the total past 5 GPU hours, and the steps not reached go to
future work (section 10). The user can raise the cap; the quota allows it.

## 8. Timing

Phase 1 finishes first: the evaluation on val and trans_val (8.06), the single test scoring
(9.02), the Space (11.02, 11.03), the slides and README (12.01, 12.02), the presentation on
2026-10-26. The literature write-up (13.03, due 2026-10-24) is docs only. Phase 2's runs start
after the presentation: the infrastructure (13.02, due 2026-10-28), then one item per code step.

**Option: one labelled exploratory result before the presentation**, reported on trans_val only
and labelled "exploratory, one unseen camera, not test" on the slide. With the revised ladder it
falls on step 0, which needs no training run but does need a tier-2 evaluation item in the week of
9.02, the Space and the slides. Arm C (step 1) is leakage-critical code and is not recommended
before 10-26. Recommended only once 9.02 is scored and 12.01 and 12.02 are done; otherwise the
slides name copy-paste as the first next step, with the evidence above.

## 9. Infrastructure first (item 13.02, not built here)

Before the first phase-2 run, item 13.02 (tier 2: `trailcam/train/`, the notebook and tests)
adds:

- `docs/ledger-phase2.md`, with the header and columns of `docs/ledger.md` (`append_row` refuses
  a ledger whose header differs);
- the ledger checker (`trailcam/train/ledger.py --check`) accepting it: the same row checks, but
  several config hashes per model, since every step is a new recipe; R25's arm pairing still
  holds for arm C against arm A;
- `notebooks/kaggle_train.ipynb` taking an optional `LEDGER` parameter (today the notebook sets
  `LEDGER` to `docs/ledger.md` in the clone; phase 1's default stays);
- the `p2_` prefix for phase-2 config names;
- tests that a phase-2 row can never land in `docs/ledger.md`: neither appended by `train.py` nor
  pasted by hand (rows are pasted by the user, so the check of `docs/ledger.md` refuses a `p2_`
  row and fails `check.sh`).

Not in 13.02 as agreed, and needed by the first code step that adds a setting (section 4): the
form that keeps the phase-1 config hashes unchanged, and the test that pins them. CLAUDE.md's
"every choice is made on val" describes phase 1; whether it gets a phase-2 line is open question 7.

## 10. Beyond the ladder (future work, for the slides' closing)

- **A detector crop at test time.** The literature's fix: Beery et al. 2018 found that classifying
  box crops more than halves both errors, and the deployed pipeline is detect, crop, then classify
  (Beery, Morris, Yang 2019); the review cites about +25% relative macro-F1 from detector crops
  (Gadot et al. 2024, to verify in 13.03). It changes the product: two models, a detector on the
  CPU, boxes at inference, all out of phase 1's scope.
- **Camera-trap-pretrained backbones**, once their pretraining data is audited for CCT images:
  a backbone that saw the trans cameras in pretraining would inflate every trans score (section 6,
  "Dropped").
- **Calibration** (item 12.03): temperature scaling fit on val, with expected calibration error on
  cis and trans. It would tell whether the demo's confidences can be trusted on new cameras.
- **Unlabeled trans-camera data** (Connect Later, ICON on the leaderboard) would use the new
  cameras' photos without labels; it changes what the trans scores mean, so it needs its own
  decision.

## 11. Open questions for phase 2

1. **The screen's val condition.** As agreed, the screen stops a change whose val macro-F1 drops
   by more than the noise (0.003), but the keep rule accepts a drop of up to 1 point, so the screen
   could stop a change the keep rule would accept. Recommended: the screen's val condition uses the
   keep rule's 1 point. The user's decision.
2. **The screen's trans_val condition.** "Not more than 0.03 worse" passes most changes that do
   nothing (section 7). "Not below the baseline's seed 0" would stop about half of them and keep
   the ladder nearer the runs budgeted, at the cost of sometimes stopping a real but small gain.
   The user's decision.
3. **The keep threshold with five seeds.** Five seeds shrink the chance difference of two means to
   about 0.010 (arm A's spread), so a threshold near 0.02 would be two standard errors; as agreed,
   it stays 0.03 until the user decides.
4. **Arm B on the final recipe**, step 8 of the first draft, is not in the revised ladder. Rerun
   phase 1's question on the improved recipe (3 runs), or leave it, now that arm C uses the boxes?
5. **Arm C's review tier**: tier 2 at xhigh by the playbook, tier 3 recommended (section 6).
6. **The Space after phase 2.** Whether a phase-2 model (or a soup) replaces the phase-1 model on
   the Space; if it does, the Space's card quotes its trans_val numbers with the honesty note, not
   phase 1's test numbers.
7. **CLAUDE.md and R17.** Both say every choice is made on val; phase 2 decides steps on val and
   trans_val (section 4). Whether CLAUDE.md and `docs/review/invariants.md` get a phase-2 line
   (in 13.02 or before the first step) is the user's decision.

## Sources

Verified against the original pages by the coordinator, 2026-10-17:

- WILDS leaderboard, iWildCam: https://wilds.stanford.edu/leaderboard/
- Gao, Sagawa, Koh, Hashimoto, Liang. "Out-of-Domain Robustness via Targeted Augmentations." ICML
  2023, PMLR 202:10800-10834.
- Mesnage et al. "Copy-Paste Augmentation Improves Automatic Species Identification in Camera Trap
  Images." Ecology and Evolution 15(11):e72357, 2025 (PMC12588685).
- Qu and Xie. "Connect Later." ICML 2024. https://arxiv.org/abs/2402.03325

To verify in 13.03: SWAD (Cha et al., NeurIPS 2021); LP-FT (Kumar et al., ICLR 2022); Gadot et
al. 2024 (IET Computer Vision); model soups (Wortsman et al., ICML 2022); the review's claims on
focal loss, logit adjustment, LDAM, ABSGD and SpeciesNet.

Already in the project's literature review (`docs/research-report.md`):

- Beery, Van Horn, Perona. "Recognition in Terra Incognita." ECCV 2018.
  https://arxiv.org/abs/1807.04975
- Koh et al. "WILDS: A Benchmark of in-the-Wild Distribution Shifts." ICML 2021.
  https://arxiv.org/abs/2012.07421
- Beery, Morris, Yang. "Efficient Pipeline for Camera Trap Image Review." 2019.
  https://arxiv.org/abs/1907.06772

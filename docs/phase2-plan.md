# Phase 2: improving generalisation to new cameras

Written 2026-10-17 (item 13.01), from the points agreed with the user through the coordinator
that day; the ladder (section 6) was revised the same day from an external literature review, at
the user's request. Updated the same day by item 13.02 with the user's decisions of 2026-10-17
(through the coordinator): phase 2 runs this week, before the presentation on 2026-10-26, from
2026-10-18 to a freeze on 2026-10-22 (section 8), and its one test scoring is declared in section
4 before any phase-2 result exists. `docs/plan.md` stays the source of truth for phase 1 (the
split, the reported runs, phase 1's single test scoring); this file owns phase 2. Phase 2 lifts two
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

Baseline: phase 1's `resnet18_arm_a`, run seeds 0, 1 and 2 (section 2), re-run unchanged at the
phase-2 commit as `p2_baseline` (section 3). Phase 2 measures "trans" on trans_val (section 4,
the honesty note); its one test scoring, of the frozen final recipe only, is declared in section 4.

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

**trans_val is one camera and a bobcat-heavy class mix.** Its 1,603 images come from one
location, 125: bobcat 734 (46%), opossum 444, raccoon 129, dog 96, cat 72, skunk 63, coyote 49,
rabbit 9 and bird 7, with no squirrel, car or rodent (the data card's per-class table). 8.06's
evaluation (plan.md section 7, "Val results") puts most of phase 1's trans_val spread on bobcat
recall at that camera, 0.19 to 0.40 across the runs, the misses mostly predicted as cat; and
macro-F1 there
averages in two classes of under 10 images, where one or two photos swing a class's F1. So
choosing a recipe on trans_val is choosing for one camera and this class mix, and a trans_val
gain is weak evidence of a gain on new cameras in general: it can be a better bobcat-versus-cat
boundary at camera 125 and nothing more.

## 3. How a change is judged: screen, confirm, keep

Every comparison is paired by run seed: the change's seed k against the baseline's seed k at
the same commit, on phase 1's split (split seed 42, split hash `c78a9c10…`). The baseline is
`p2_baseline`, which is `resnet18_arm_a` exactly (same fields, same config hash).

**The control re-run.** The code has moved since phase 1's training commit `61f1ecc` (item
13.02's progress-log entry lists the training-path files changed). So `p2_baseline` seed 0 runs
in the screening batch, at the phase-2 commit, and is checked against phase 1's arm A seed 0
before any screen is read: val macro-F1 within the val band of section 2 (0.8542 ± 0.003) and
trans_val macro-F1 within the trans_val band (0.2915 ± 0.03). Inside both, the code changes did
not move the recipe, and a variant's difference from `p2_baseline` is the variant's. Outside
either, the screen stops until the difference is explained. A band and not equality, because a
GPU run is not guaranteed to repeat bit for bit even at one commit and seed.

**This week's order: screen in parallel, then confirm, then stack** (the user's decision of
2026-10-17; the dates are section 8's). Every variant is one change from the same baseline, so
the screens are independent: they can run side by side or in any order, each compared with
`p2_baseline` alone and never with another variant, and each comparison still changes one thing.
The variants ready on 10-19 are `p2_head3`, `p2_res320`, `p2_r50` and, once item 13.04 merges,
`p2_arm_c`; SWA (step 2) is screened when its code item merges. Only the confirmed winners are
then combined (the stack), so the stack is the one comparison that changes several things, and
it is judged as one recipe: it shows what the winners give together, not what each gives (each
winner's own confirm shows that).

- **Screen** on run seed 0. Continue to the confirm only if val macro-F1 does not drop (by more
  than its noise, 0.003) and trans_val macro-F1 is not clearly worse (not more than 0.03 below the
  baseline's seed 0). Otherwise the step stops: its row stays in the phase-2 ledger, marked
  screened out in the progress log, and the baseline is unchanged. (Open questions 1 and 2 ask
  whether these two conditions should change.)
- **Confirm** on run seeds 1 and 2 (seed 0 is the screen's run), so three seeds in all, with
  `p2_baseline`'s seeds 1 and 2 run at the same commit (open question 8). This week every step
  confirms on three seeds (section 8); the five seeds below are not in the timeline, so a gain
  under about 4 points is reported as not confirmed rather than claimed (see "Resolvability").
- **Keep** a change only if all three hold:
  1. the mean trans_val macro-F1 over the confirm's seeds rises by more than the noise, 0.03;
  2. it rises on every seed, paired with the baseline's same seed;
  3. the mean val macro-F1 drops by less than 1 point (0.01).

  Steps that change the cost of a prediction (0, 4 and 5) also pass the CPU latency check.
  Accuracy is reported beside macro-F1 everywhere, but the rule is on macro-F1, as in phase 1.

  Val macro-F1 must hold: condition 3 here and the screen's val condition are requirements, never
  traded against a trans_val gain, since val's ten cameras are the steadier evidence (section 2).
  Every screen, confirm and stack reports trans_val accuracy and macro-F1 together, with bobcat
  recall and trans_val's per-class table beside them as context, so a gain that is all bobcat at
  camera 125 shows as one. That context selects nothing: the rule stays on val macro-F1 and
  trans_val macro-F1, and no selection metric is added. The per-class table and bobcat recall
  come from `trailcam/evaluate/evaluate.py` on val and trans_val; its notebook (8.08) reads
  `docs/ledger.md` only, and the evaluation tools run at 224 px only, so for phase-2 checkpoints
  (and `p2_res320` in particular) that context needs a small follow-up item; until it exists it is
  reported where it can be.

**Kept changes stack.** The kept changes are combined into one recipe (a config with several
changes, so `tests/test_config.py`'s one-change rule needs a named exception for it) and run on
seeds 0, 1 and 2; the stack is kept by the same rule against `p2_baseline`. If it fails, the final
recipe is chosen on 10-22 among the single kept changes and `p2_baseline`, by the same rule. If a
kept recipe's spread is clearly wider than the bands above, the bands are re-measured from it and
the progress log says so. The goal's val limit is cumulative: the stacked recipe stays within
about 1 point of phase 1's 0.8557 val macro-F1, so two kept changes that each cost 0.7 points
fail it.

**Resolvability.** From arm A's seeds, one run's trans_val macro-F1 varies by about 0.016
(standard deviation), so the difference of two three-seed means varies by about 0.013 by chance;
with arm B's spread (about 0.044) it would be about 0.036. So with three seeds, **a gain under
about 4 points cannot be confirmed** on trans_val, and the 0.03 threshold plus "up on every seed"
is the least that separates a gain from a lucky seed. Steps 2 to 6 expect smaller gains (section
6), which five seeds (run seeds 0 to 4, and the baseline's seeds 3 and 4) could resolve; this
week's timeline has three, so such a gain, if seen, is reported as not confirmed, and seeds 3 and
4 are left for after the presentation. A per-camera breakdown cannot replace the extra seeds:
trans_val is one camera (plan.md section 3). With three seeds the spread itself is
only a rough estimate.

## 4. Rules carried over from phase 1

- **One change per step**, so that when a score moves, one thing moved it (an ablation). In the
  parallel screen, one change per variant against the same baseline (section 3).
- **A pinned commit per batch of runs.** The configs and the hypothesis lines of every variant in
  a batch (the 10-19 screen, the 10-20 confirm, the 10-21 stack) are committed first; that commit
  is the notebook's `COMMIT` for every run of the batch, `p2_baseline`'s included, and the batch's
  progress-log entry quotes it. A variant run at a later commit than its `p2_baseline` seed is
  paired with it only if no training-path file changed between the two commits (the diff 13.02's
  log takes from `61f1ecc`); otherwise that `p2_baseline` seed re-runs at the later commit.
- **Each run is a row in `docs/ledger-phase2.md`** (built in 13.02), with the header and columns
  of `docs/ledger.md`, and never in `docs/ledger.md`. The final scoring
  (`trailcam/evaluate/score_test.py`) relies on `docs/ledger.md` holding exactly the
  pre-registered runs: it refuses a ledger without a small CNN row and an arm A and an arm B row
  for each reported seed, or with a row of another seed, and scores every row's checkpoint in one
  run. The phase-1 checker also refuses one model under two config hashes, which is exactly what
  phase 2's steps are. `train.py` refuses a `p2_` config with any ledger but
  `docs/ledger-phase2.md`, and a phase-1 config with that one, before training and at the append;
  the ledger checker refuses a `p2_` row in `docs/ledger.md` (so a pasted one fails `check.sh`) and
  a phase-1 row in `docs/ledger-phase2.md`. `.claude/rules/training.md`'s "every training run
  appends one row to `docs/ledger.md`" describes phase 1.
- **Config names start with `p2_`** (`p2_baseline`, `p2_head3`, `p2_res320`, `p2_r50`; item 13.04
  adds `p2_arm_c`). The phase-1 configs and their config hashes stay as they are: `score_test.py`
  refuses a config whose hash no longer matches its ledger row, and `config_hash` hashes every
  field of `Config` but `arm` and `run_seed`, so adding a field would change every phase-1 hash.
  So a field added for phase 2 is listed in `HASH_NEUTRAL` (`trailcam/train/config.py`) with the
  value phase 1 used, and is left out of the hash at that value: 13.02's `input_size` at 224.
  `tests/test_config.py` pins every phase-1 hash to the rows of `docs/ledger.md`.
- **Choices on val and trans_val only.** Training is unchanged: early stopping and the restored
  checkpoint still follow val macro-F1 (R17). Only the keep-or-drop decision of a step looks at
  trans_val. cis test and trans test are touched once, by the scoring declared below.
- **Same split and seeds**: split seed 42 and run seeds 0 to 2, as in phase 1.
- **The phase-2 transform exception.** For phase-2 configs only, `input_size` sets the side of the
  training crop and the eval transform's resize (`round(size x 256 / 224)`) and centre crop:
  `p2_res320` trains on 320 px crops and is evaluated with the shorter side resized to 366 and a
  320 px centre crop, everything else identical. The 256 and 224 of CLAUDE.md, `.claude/rules/`
  and R20 stay true for every phase-1 config, the demo and the Space (the user approved this
  plan, 2026-10-17). The evaluation tools, the demo and the Space run at 224 only, and
  `load_checkpoint` (`trailcam/evaluate/attention.py`), which they all use, refuses a config of
  another input size, so a 320 px model's val and trans_val scores are `train.py`'s, and its test
  scoring needs 13.06's mode. The checkpoint carries its input size too (review 13.02, the
  user's decision): `train.py` saves `input_size` beside the model, the class list and the
  weights, and `load_checkpoint` refuses a checkpoint whose size is not the config's, so a
  `p2_res320` checkpoint loaded under a 224 config (`resnet18_arm_a` on a command line, or the
  Space's default) is refused instead of scored or served at 224; ResNet-18's weights alone could
  not tell, being the same shape at 320 px. A checkpoint without a size, the format of phase 1's
  seven, counts as 224 and loads as before.

**The pre-declared test scoring** (the user's decision of 2026-10-17, written here before any
phase-2 result exists):

- Phase 1 is scored once: item 9.02, on 2026-10-19, as `docs/plan.md` section 7 says.
- Phase 2 gets **exactly one** further test scoring, on **2026-10-22**, of **exactly one** recipe:
  the frozen final recipe, all its seeds, each checkpoint scored once on cis test and trans test.
  No other phase-2 recipe, variant, screen or seed is ever scored on a test split.
- Every phase-2 choice (the screens, the confirms, what stacks, the final recipe) is made on val
  and trans_val. The test scores choose nothing.
- That scoring needs a phase-2 mode in `score_test.py` with its own test ledger, a later tier-3
  item, **13.06** (not built in 13.02, and not yet in `docs/progress.json`). Until it exists,
  `score_test.py` scores only a ledger holding phase 1's pre-registered runs (it requires the
  small CNN row and both arms per seed, which no phase-2 ledger has) and refuses any test ledger
  that already holds a row; 13.06 should also make its phase-1 mode refuse the phase-2 ledger by
  name.
- If the freeze slips past 2026-10-22, phase 2 has no test scoring: it is reported on trans_val
  only, with the honesty note.
- That one scoring is the real check of a phase-2 gain. trans_test holds 21,473 images from 9
  unseen cameras (the data card), against trans_val's 1,603 from one (section 2), and nothing in
  phase 2 is chosen on it; a gain that holds there is evidence about new cameras, and one that
  does not was a camera-125 effect.

**Honesty note.** Phase 2 selects on trans_val, so its trans_val numbers are slightly
optimistic: the best of several tries looks better than it will on the next new camera. WILDS
flags the same bias when hyperparameters are tuned on the out-of-distribution val split (Koh et
al. 2021, `docs/research-report.md`). trans_val is also one camera, so a phase-2 gain there is a
gain on one unseen camera, and every slide or README line that quotes one says so. This changes
trans_val's role from phase 1's R17 on purpose, for phase-2 rows only. The pre-declared scoring
above is the user's explicit decision on a second test scoring, taken before any phase-2 result
exists: one recipe, chosen without its test scores, so its test numbers are not selected on.
Wherever they appear, they are disclosed as phase 2's, of a recipe chosen after phase 1's test
scores were seen (9.02 on 10-19), though on val and trans_val only. `score_test.py` refuses a
test ledger that already holds a row, so no further scoring can happen by accident.

## 5. A hypothesis line before every run

Before a step's first run, one line goes into the step's progress-log entry, in the pinned
commit, so it is dated before any result. For the parallel screen, every variant's line is in the
one commit pinned for the batch, before its first run:

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

Each step is one change from `resnet18_arm_a`, screened against `p2_baseline` (section 3); this
week steps 1 to 5 are screened in parallel, and steps 0 and 6 are not in the timeline (section
8). `config.yaml` (and a `configs/` folder) is in `scripts/risk_files.sh`'s regex, so even a
config-only step is tier 2 by the script; its review is one config block and its test. Item 13.02
built the code of steps 3 to 5 and their configs (the input size as a config field, the
ResNet-50 entry), so they are config-only now; the infrared-aware augmentation (step 6) still
needs code, since the augmentation strengths are constants in `trailcam/data/dataset.py`.

| step | change | evidence | needs | review tier | runs |
| --- | --- | --- | --- | --- | --- |
| 0 | weight soup and softmax ensemble of arm A's 3 seeds | soups to verify | code (evaluation) | 2 | 0 (evaluation only) |
| 1 | copy-paste, arm C (v1 rectangles; v2 masks if v1 is promising), `p2_arm_c` | strongest | code (new arm, background pool, tests; item 13.04) | 2 at xhigh; 3 recommended | v1 1 to 3, v2 0 to 3 |
| 2 | SWA over the late epochs | to verify | code (`train.py`) | 2 | 1 to 3 |
| 3 | longer head-only phase (3 epochs), `p2_head3` | mixed | config only | 2 (13.02's review) | 1 to 3 |
| 4 | input 320 px, `p2_res320` | weak | config (its code in 13.02) | 2 (13.02's review) | 1 to 3 |
| 5 | ResNet-50 at 224 px, `p2_r50` | weak | config (its model entry in 13.02) | 2 (13.02's review) | 1 to 3 |
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
2023, Mesnage 2025). Our own evidence points the same way: in 8.06's animal-removed test (plan.md
section 7, "Val results"), blanking every box drops ResNet-18's val accuracy from about 0.90 only
to 0.50 to 0.61, against 0.083 by chance (1 in 12): with the animal gone, the model still names
the class about half the time from the scene, which is background reliance, the shortcut arm C
targets. For about half of the boxed training images (probability 0.5, as in arm B), the box
region of the training animal is pasted onto an empty frame from a training camera, night
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
the only box user at training time; arm C's item extends them. Arm C (`p2_arm_c`) is compared
with `p2_baseline` on identical configs apart from `arm` (R25's rule and its phase-2 line). Tier 2 at xhigh by the playbook; tier 3
(`/code-review ultra` as well) is recommended because the pool is a new data boundary like the
split, and is automatic if the code touches `trailcam/data/split.py` or its tests. *Hypothesis*:
trans_val macro-F1 0 to +0.06 (about +0.03), val flat to -0.005; smaller than the leaderboard's
+5.7 because our backgrounds are training cameras only, no detector, 224 px. Three seeds; if the
result is positive but under about 4 points, seeds 3 and 4 are added (after the presentation)
before it is claimed.

**2. SWA over the late epochs** (`torch.optim.swa_utils`: a running average of the weights over
the last epochs, batch-norm statistics recomputed on the training data). *Needs code* in
`train.py`; the item decides how the average meets early stopping (for example, scored on val as
one more candidate at the end). Evidence: SWAD on TerraIncognita (to verify). *Hypothesis*:
trans_val 0 to +0.03, val flat. Three seeds this week.

**3. A longer head-only phase**, `p2_head3`: 3 epochs instead of 1 (`head_only_epochs`), *config
only*. Head-only epochs count inside `max_epochs` (plan.md section 5), and `max_epochs` stays 10
(the user's decision of 2026-10-17), so the body trains 7 of the 10 epochs instead of 9: one field
changes. The warmup (1 epoch) then no longer coincides with the head-only phase: the head warms
up in epoch 1 and trains alone near the top of the schedule in epochs 2 and 3, and the body joins
in epoch 4 with the cosine already at about 0.88 of its peak. Evidence mixed (LP-FT, to verify).
*Hypothesis*: trans_val flat to +0.01, val flat. Three seeds this week.

**4. Resolution 320**, `p2_res320`. Train crop and eval centre crop at 320 px, the eval resize
at 366 (round(320 x 256/224), the same margin as now; the 512 px cache's shorter side is 373 or
384, so no upsampling), everything else identical. *Built in 13.02*: `input_size` in the config
(hash-neutral at 224, section 4), taken by the training transform of both arms and the eval
transform through `train.py` and `make_loaders`. The checkpoint records its input size, and
`load_checkpoint` refuses a checkpoint of another size than its config's (section 4). The
evaluation tools, the demo and the Space stay at 224 and refuse a config of another size, so
deploying a 320 px model (open question 6) would need the attention frame, Grad-CAM, the demo and
the Space to take the size (`load_checkpoint`'s `eval_size`). Evidence weak (the
WILDS baselines' 448 px may be part of their numbers); at 224 px many animals are a few pixels
wide (`docs/research-report.md`, Part 2). *Hypothesis*: trans_val 0 to +0.03, val +0.5 point,
about 2x the run time and the CPU latency. Three seeds this week.

**5. ResNet-50 at 224 px**, `p2_r50` (torchvision, ImageNet-1k weights; 25.6M parameters and
about 4.1 GFLOPs, against ResNet-18's 11.7M and 1.8). *Built in 13.02*: `resnet50` in
`trailcam/model/factory.py` with torchvision's `IMAGENET1K_V1` weights, the original recipe as
ResNet-18's only weights are (V2's newer, heavier-augmented recipe would be a second change); its
parameter groups (head `fc`, 2048 features) and Grad-CAM layer (`layer4`); its shape table
tested (`tests/test_models.py`, 23.5M parameters with our 12-way head). The Space is not changed
(open question 6); the weights download on Kaggle (Internet on). ImageNet-1k weights keep plan.md
section 4's argument (ImageNet holds no CCT images). *Hypothesis*: trans_val 0 to +0.03, val +0.5
point, about 2.3x the run time and the CPU latency, measured before it can be kept. Three seeds
this week.

**6. Infrared-aware augmentation**: greyscale and photometric changes aimed at infrared frames
(about two thirds of the images), for example a higher greyscale probability and contrast or gamma
jitter. *Needs code*: the strengths are constants in `trailcam/data/dataset.py`; never in the
eval transform (R20). Evidence weak. It is never combined with arm C's day/night matching in one
run: if arm C is kept, step 6 is screened on the recipe without arm C and the two are not stacked
(the user picks if both pass). *Hypothesis*: trans_val flat to +0.01, val flat. Not in this week's timeline.

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
review (13.02's review covers steps 3 to 5).

This week, from the multipliers below and about 12 minutes a ResNet-18 run: the 10-19 screen is
five runs (`p2_baseline`, `p2_head3`, `p2_res320`, `p2_r50`, `p2_arm_c`), about 90 minutes, plus
SWA's screen, about 13; the 10-20 confirm is `p2_baseline`'s two seeds and two per winner, from
about 25 minutes (no winner) to about 3.5 hours (every variant wins); the 10-21 stack is three runs,
about 40 minutes for a ResNet-18 stack and nearly 3 hours if it holds 320 px and ResNet-50 (about
55 minutes a run). The worst case, about 8 GPU hours, is above the 5-hour cap of the rule below:
open question 9.

Run time, measured in phase 1 and estimated after: a ResNet-18 run at 224 px took about 12
minutes, start-up included (the gaps between consecutive start times in item 7.03's run ids). The
multipliers below assume the GPU is the bottleneck; the first run of each step measures them.

| change | run time against now |
| --- | --- |
| arm C (copy-paste) | about 1x (the paste is data loading on the CPU) |
| SWA | about 1.1x (one extra pass for the batch-norm statistics) |
| head-only 3 of the 10 epochs (`p2_head3`) | about 0.9x to 1x (two body epochs become cheaper head-only ones) |
| 320 px | about 2x |
| ResNet-50 | about 2.3x |

They multiply when changes stack: 320 px with ResNet-50 is roughly 4.6x, about 55 minutes a run.

Over the whole ladder, after the presentation too, the run count depends on the screen and on
the five-seed confirms. The ladder costs 6 runs if
every screen stops a step, and about 40 if every step reached its full confirm (with the
baseline's extra seeds). With the screen of section 3, a change that does nothing passes most of
the time (one seed seldom falls 0.03 below the baseline by chance), so the budget, not the screen,
is likely to end the ladder. The rule: before each run, write its projected minutes (the last
run's time times the step's multiplier) and the running total in the progress log; the ladder stops
when the next step's confirm would take the total past 5 GPU hours, and the steps not reached go to
future work (section 10). The user can raise the cap; the quota allows it.

## 8. Timing

Phase 2 runs this week, before the presentation on 2026-10-26, not after (the user's decision of
2026-10-17, through the coordinator):

| date | phase 2 |
| --- | --- |
| 2026-10-18 | this infrastructure (13.02), and arm C's code (13.04, a parallel session) |
| 2026-10-19 | screening runs, seed 0 of each variant against `p2_baseline` seed 0 (and the control check, section 3); then the SWA code |
| 2026-10-20 | confirm the winners on seeds 1 and 2 |
| 2026-10-21 | stack the winners, 3 seeds |
| 2026-10-22 | freeze: the final recipe chosen on val and trans_val only; then its one test scoring (section 4), given item 13.06 |

Screening independent one-change variants in parallel against the same baseline, then stacking
the winners, keeps one change per comparison: every screen and every confirm compares a variant
with `p2_baseline` and nothing else, and only the stack combines changes, once, after each has
been confirmed alone (section 3).

Phase 1's own items are unchanged and run beside it: the evaluation on val and trans_val (8.06),
the single test scoring (9.02, 2026-10-19), the Space (11.02, 11.03), the slides and README
(12.01, 12.02), the literature write-up (13.03, due 2026-10-24, docs only). Phase 2 touches no
phase-1 config, ledger row or checkpoint, so it changes nothing 9.02 scores. Steps 0 (soup and
ensemble) and 6 (infrared-aware augmentation) are not in this week's timeline and stay on the
ladder for after the presentation. The slides can show phase 2 on trans_val with the honesty
note, and its test scores only if the 10-22 scoring ran.

## 9. Infrastructure first (item 13.02, built)

Item 13.02 (tier 2: `trailcam/`, `config.yaml`, the notebook and tests) built, before the first
phase-2 run:

- `docs/ledger-phase2.md`, with the header and columns of `docs/ledger.md` (`append_row` refuses
  a ledger whose header differs);
- the ledger checker (`trailcam/train/ledger.py --check`) accepting it, by its file name: every
  cell check, plus the `p2_` prefix on every row; several config hashes per model (every step is
  a new recipe), but one per config and one split seed for the whole ledger; the phase-2 summary
  by config first. `docs/ledger.md`'s check refuses a `p2_` row;
- `train.py` refusing a `p2_` config with any ledger but `docs/ledger-phase2.md`, and a phase-1
  config with that one, before training and again at the append;
- `notebooks/kaggle_train.ipynb` taking an optional `LEDGER` parameter, `docs/ledger.md` by
  default, passed to `--ledger`; its first check cell refuses a config of the other phase in
  seconds, before the clone;
- the `p2_` prefix (`PHASE2_PREFIX`, `is_phase2` in `trailcam/train/config.py`);
- the form that keeps the phase-1 config hashes unchanged (`HASH_NEUTRAL`, section 4) and the test
  that pins them to `docs/ledger.md`'s rows;
- `input_size` (step 4's code) and the `resnet50` model entry (step 5's), and the configs
  `p2_baseline`, `p2_head3`, `p2_res320` and `p2_r50`, each checked by a test to be
  `resnet18_arm_a` with its one change;
- the evaluation tools, the demo and the Space refusing a config of another input size than 224,
  and, after review 13.02, every new checkpoint recording its input size, which `load_checkpoint`
  checks against the config (a checkpoint without one, phase 1's, counts as 224);
- the phase-2 lines of R17, R20, R25, R30, R31 and R40 in `docs/review/invariants.md`.

Not in 13.02: `score_test.py`'s phase-2 mode and its test ledger (13.06, section 4).

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
   phase 1's test numbers. Since review 13.02 the checkpoint records its input size and the
   deploy's load refuses a `p2_res320` checkpoint under the Space's 224 px config, so a 320 px
   model cannot reach the Space by mistake; serving one on purpose needs the demo and the Space
   to take the size.
7. **CLAUDE.md and R17** (answered 2026-10-17). CLAUDE.md and `.claude/rules/` keep their
   phase-1 rules; the phase-2 exceptions (choices on val and trans_val, the transform exception,
   the second ledger, the one pre-declared test scoring) are stated in section 4 of this file, and
   `docs/review/invariants.md` has phase-2 lines in R17, R20, R25, R30, R31 and R40 (item 13.02).
8. **`p2_baseline`'s seeds 1 and 2.** As written (section 3), they run on 10-20 at the confirm's
   commit, so every pair is within one commit (two runs, about 25 minutes). The cheaper option:
   pair the confirms with phase 1's arm A seeds 1 and 2 if `p2_baseline` seed 0 passes the control
   check. Recommended: run them. The user's decision.
9. **This week's GPU cap.** The worst case of section 7 is about 8 GPU hours against the 5-hour cap
   of the stopping rule; the quota (30 a week) allows it. Raise the cap for this week, or confirm
   the cheapest winners first and drop what does not fit? The user's decision.

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

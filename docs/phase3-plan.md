# Phase 3: a stronger backbone for new cameras

Written 2026-10-19 (item 13.10), from the user's decisions of 2026-10-19, relayed by the
coordinator: phase 3 is approved as a bounded improvement round aimed at new cameras, the
presentation moves from 2026-10-26 to **2026-11-04**, and the keep rule below is fixed now,
before any phase-3 run. Phase 1 is done and was scored on test once (`docs/test-ledger.md`,
scored at 20261018T193117Z). Phase 2 is frozen at the baseline (`docs/phase2-plan.md` section 12).
The Space serves ResNet-18 arm B seed 2 and stays the safety net whatever phase 3 gives.

This file owns phase 3. `docs/plan.md` stays the source of truth for phase 1, and
`docs/phase2-plan.md` for phase 2; phase 3 reuses phase 2's machinery and rules except where a
section below says otherwise. Like phase 2, phase 3 lifts two of plan.md section 1's scope
limits, for phase 3 only: a backbone larger than ResNet-18, and (candidate b) more than one
training-time change.

Data: Caltech Camera Traps (CCT20), Beery et al. 2018, CDLA-Permissive 1.0. Every score below
comes from `docs/ledger.md`, `docs/ledger-phase2.md`, `docs/test-ledger.md`, `docs/plan.md` or
`docs/phase2-plan.md`, or is derived from them (a difference or a mean of their rows).
Hypotheses and run times are estimates and say so. Model sizes are not quoted here: item 13.11's
shape table (`/review-shapes`) measures them.

## 1. Goal and the keep rule

Raise **trans_val macro-F1**, the score on the one validation camera never seen in training,
with a stronger ImageNet-pretrained backbone, at a bounded cost on the familiar cameras, and keep
the model deployable on the Space's CPU (a prediction in well under a second, plan.md sections 4
and 8).

**The keep rule** (the user's choice of 2026-10-19, fixed before any phase-3 run). A candidate is
**kept** only if all three hold, each delta paired by run seed against the baseline (section 2):

1. the mean trans_val macro-F1 over run seeds 0, 1 and 2 rises by **more than 0.03**;
2. trans_val macro-F1 **rises on every one** of the 3 seeds;
3. the mean val macro-F1 drops by **no more than 0.03** (3 points).

Against the baseline's rows that reads: a mean trans_val macro-F1 above 0.3401 (the baseline's
0.3101 plus 0.03), seed by seed above 0.2915, 0.3187 and 0.3200, and a mean val macro-F1 of at
least 0.8257 (0.8557 minus 0.03). Accuracy is reported beside macro-F1 everywhere; the rule is on
macro-F1, as in phases 1 and 2.

Conditions 1 and 2 are phase 2's keep conditions unchanged (`docs/phase2-plan.md` section 3), so
phase 2's verdicts stand under them: `p2_res320`'s mean gain of +0.0158 is still under 0.03.
Condition 3 is looser than phase 2's 1 point, on purpose: phase 3 aims at new cameras, and phase
2 measured that copy-paste, the one change that moved trans_val clearly, costs val macro-F1
(-0.0304 at paste probability 0.5, `docs/phase2-plan.md` section 12). The user accepts up to 3
points on the familiar cameras for a clear gain on the new one; more than 3 is not traded.

**The screen** (run seed 0, against the baseline's seed 0). A candidate goes on to the confirm
only if both hold:

- trans_val macro-F1 is **at least 0.03 above** the baseline's seed 0 (at least 0.3215);
- val macro-F1 is **at most 0.03 below** the baseline's seed 0 (at least 0.8242).

Why this screen:

- **It asks of seed 0 what the keep rule asks of the mean.** Phase 2's screen and keep rule
  disagreed: the screen stopped a val drop over 0.003 while the keep rule allowed 0.01, so
  `p2_head3` (-0.0078) stopped at a screen the keep rule would have let through (phase 2's open
  question 1). Here the two bars are the same.
- **The val half is reliable on one seed.** Val holds 2,394 images from the ten training cameras
  and its seed noise is about 0.003 (`docs/phase2-plan.md` section 2), so a seed-0 drop beyond
  0.03 means the mean almost surely fails condition 3 as well.
- **The trans_val half saves runs that are unlikely to pay.** Phase 2's trans_val screen ("not
  more than 0.03 worse") passed every variant and decided nothing (its open question 2). If seed 0
  gains under 0.03, seeds 1 and 2 must together make up the shortfall while each also rises; one
  run's trans_val macro-F1 varies by about 0.016 (`docs/phase2-plan.md` section 3), so that is
  possible but not likely, and a ConvNeXt confirm costs two of the longest runs of this phase
  (section 10).
- **The cost, said plainly.** A real gain close to the bar, about 0.03 to 0.04, fails this screen
  fairly often by chance. That is the case three seeds cannot settle anyway: phase 2 found that a
  gain under about 4 points cannot be confirmed on trans_val with three seeds (its section 3,
  "Resolvability").
- **It was written after phase 2's results were seen**, though not fitted to them, so here is its
  hindsight: of phase 2's five seed-0 runs only `p2_arm_c_p25` (val -0.0151, trans_val +0.0493)
  would pass it. `p2_arm_c` would miss on val by 0.0004 (-0.0304), and `p2_res320` (+0.0271),
  `p2_r50` (+0.0197) and `p2_head3` (+0.0177) on trans_val.

**The confirm**: run seeds 1 and 2 of each candidate that passes the screen, so three seeds in
all, paired with the baseline's seeds 1 and 2. The keep rule above is applied to the three
seeds; nothing else is kept.

**One finalist.** Only one recipe can be phase 3's result (and the only one ever scored on test,
section 7). If both candidates are kept, the finalist is the one with the higher mean trans_val
macro-F1; if the two means are within 0.03 of each other (trans_val's noise band), the one with
the higher mean val macro-F1. Declared now, before any run, and confirmed by the user on
2026-10-19 (open question 1).

## 2. The baseline

The same baseline as phase 2, so that phase 2's and phase 3's deltas are on one footing: phase
1's `resnet18_arm_a`, run seeds 0, 1 and 2 (`docs/ledger.md`). Its seed 0 is also `p2_baseline`
seed 0, which reproduced it bit for bit at the phase-2 commit (`docs/phase2-plan.md` section 3).

| run seed | val accuracy | val macro-F1 | trans_val accuracy | trans_val macro-F1 |
| --- | --- | --- | --- | --- |
| 0 | 0.8647 | 0.8542 | 0.3724 | 0.2915 |
| 1 | 0.8659 | 0.8559 | 0.3525 | 0.3187 |
| 2 | 0.8676 | 0.8569 | 0.4099 | 0.3200 |
| mean | 0.8661 | 0.8557 | 0.3783 | 0.3101 |

**Also reported, deciding nothing: arm B**, the deployed model's recipe (`resnet18_arm_b`, the
Space serves its seed 2). Every candidate's seed k is also shown against arm B's seed k, and its
best-val seed against arm B seed 2, so the slides can say what replacing the Space's model would
change. Arm B never enters the keep rule: phase 1 kept arm A as the control, and phase 2 judged
everything against it.

| arm B run seed | val macro-F1 | trans_val macro-F1 |
| --- | --- | --- |
| 0 | 0.8427 | 0.2588 |
| 1 | 0.8384 | 0.3019 |
| 2 (the Space) | 0.8472 | 0.3472 |
| mean | 0.8428 | 0.3026 |

**One control re-run, `p3_baseline` seed 0**, and no other baseline run. `p3_baseline` is
`resnet18_arm_a` exactly (same fields, so the same config hash `483aa055…`), run once in the
screening batch at the phase-3 commit. Why it is worth its one ResNet-18 run (about 12 minutes,
`docs/phase2-plan.md` section 7):

- item 13.11 changes training-path files (`trailcam/model/`, `config.yaml`, the ledger routing in
  `trailcam/train/train.py`), and phase 2's own rule (its section 4) re-runs the baseline seed
  when a training-path file changed since the baseline's commit;
- it is the cheapest end-to-end test of the new phase-3 ledger on Kaggle (section 8) before the
  first long ConvNeXt run lands in it.

It is a check, not a new baseline. It must land within phase 2's bands of arm A seed 0: val
macro-F1 0.8542 ± 0.003 and trans_val macro-F1 0.2915 ± 0.03 (`docs/phase2-plan.md` sections 2
and 3); `p2_baseline` reproduced it bit for bit, so bit for bit is expected. Outside either band,
no screen is read until the difference is explained. Inside both, the keep rule still uses arm
A's rows, so phase 2 and phase 3 share one baseline; if the control is not bit for bit, the
progress log shows each screen's deltas against both, and a verdict that would differ between
the two goes to the user. Seeds 1 and 2 are phase 1's, not re-run, as phase 2 decided (its open
question 8).

## 3. The candidates

Two candidates, screened in this order, each judged against the baseline by section 1's rule.

| | config | change from `resnet18_arm_a` | why |
| --- | --- | --- | --- |
| (a) | `p3_convnext` | the backbone: ConvNeXt-Tiny, ImageNet-1k weights; everything else the phase-1 recipe | a stronger backbone is the change phase 2 left untested at the right size: ResNet-50 lifted val on both seeds (+0.0128, +0.0154) but its trans_val did not hold (+0.0197, -0.0219) |
| (b) | `p3_convnext_cp` | (a) plus arm C's copy-paste at paste probability 0.5 | phase 2's clearest new-camera signal: with ResNet-18 at seed 0, val macro-F1 -0.0304 and trans_val macro-F1 +0.0971 |

**(a) `p3_convnext`.** ConvNeXt-Tiny from timm, already a dependency (`pyproject.toml`), so no
new package. The weights are pinned by tag to **`convnext_tiny.fb_in1k`**: in the locked timm
(1.0.30), the bare name `convnext_tiny` resolves to `convnext_tiny.in12k_ft_in1k`, pretrained on
ImageNet-12k first. ImageNet-1k only keeps plan.md section 4's argument (ImageNet holds no CCT
images), and it is the same pretraining data as ResNet-18's, so the one change is the
architecture, not the pretraining data. The rest is the phase-1 recipe as `config.yaml`'s
`common` block and `resnet18_arm_a` give it: AdamW, `lr_body` 1e-4 and `lr_head` 1e-3, weight
decay 1e-2, batch 64, mixed precision, 10 epochs with 1 head-only, cosine after 1 warmup epoch,
patience 3, 224 px, arm A. No retuning: tuning on trans_val with three seeds is what the budget
cannot afford and what would make trans_val even more optimistic (section 6). timm's defaults
stay: no stochastic depth (`drop_path_rate` 0). The head's layer norm (`head.norm`) keeps its
pretrained weights; it is outside `head_module`, so it trains with the body.

*Hypothesis line* (the step's pinned commit carries it, dated before the run):

```text
Phase 3 (a), ConvNeXt-Tiny ImageNet-1k (p3_convnext). Expect: val macro-F1 up about 0.01 to
0.02, trans_val macro-F1 up about 0.02 (-0.02 to +0.06). Because: a modern backbone of the same
pretraining data separates bobcat and cat better on any camera, but one unseen camera and three
seeds leave the trans_val gain within reach of the noise, as ResNet-50's was.
```

**(b) `p3_convnext_cp`.** `p3_convnext` with arm C (`arm: C`, `paste_p: 0.5`): phase 2's
copy-paste sampler and background pool unchanged (cis_val's 995 empty frames at the 10 training
cameras, day 531 and night 464, hash `87c5d379…`, `docs/phase2-plan.md` section 6). It is two
changes from the baseline and one from (a): against the baseline it is judged as one recipe by
section 1's rule; paired with (a) by run seed, it shows what copy-paste does on ConvNeXt, a
report that decides nothing. The bet: ConvNeXt's val gain absorbs part of copy-paste's val cost,
so the trade that stopped `p2_arm_c` at phase 2's screen fits inside condition 3.

*Hypothesis line*:

```text
Phase 3 (b), ConvNeXt-Tiny with copy-paste at p 0.5 (p3_convnext_cp). Expect: trans_val
macro-F1 up about 0.08 (+0.03 to +0.12) over the baseline, val about -0.015 (-0.035 to 0). Because:
copy-paste's trade on ResNet-18 was about 1 val point for 3 trans_val points, and a stronger
backbone should give back some of the val cost without losing the new-camera gain.
```

(b) is screened whatever (a)'s verdict: its question does not need (a) to pass. Two exceptions,
declared now: if the control fails (section 2), nothing is screened; if (a)'s training is broken
(a run that fails, or val macro-F1 more than 0.03 below the baseline, which would point at the
recipe rather than the backbone), (b) waits until that is explained, since it carries the same
setup plus copy-paste's own val cost.

**Out of phase 3** (the user's decision of 2026-10-19), listed as future work:

- **BioCLIP** and other biology- or camera-trap-pretrained backbones: their pretraining overlap
  with CCT is unknown, so a leakage risk (`docs/phase2-plan.md` section 6, "Dropped").
- **MegaDetector detect-then-classify**: two models and boxes at inference, out of scope since
  plan.md section 1 (`docs/phase2-plan.md` section 10).
- Phase 2's unrun steps: SWA (never built), the soup or ensemble of arm A's seeds,
  infrared-aware augmentation, copy-paste with a longer training budget, and seeds 3 and 4 of
  `p2_res320` (`docs/phase2-plan.md` sections 10 and 12).

## 4. Rules carried over from phase 2

Phase 2's rules hold for phase 3 with `p3_` in place of `p2_` (`docs/phase2-plan.md` sections 3
to 5), except where this file says otherwise:

- **Same split and seeds**: split seed 42 (split hash `c78a9c10…`), run seeds 0 to 2.
- **Choices on val and trans_val only.** Training still selects the checkpoint on val macro-F1
  (R17); only the keep-or-drop decision and the finalist look at trans_val.
- **A pinned commit per batch.** The screening batch (`p3_baseline`, `p3_convnext`,
  `p3_convnext_cp`, seed 0) runs at one commit, the notebook's `COMMIT`, after 13.11 merges; its
  progress-log entry quotes it and carries both hypothesis lines, dated before the first run. A
  confirm run at a later commit is paired with the screen's control only if no training-path file
  changed in between (phase 2's rule); otherwise `p3_baseline` seed 0 re-runs there.
- **Config names start with `p3_`**, and the phase-1 and phase-2 configs and their config hashes
  stay as they are (`tests/test_config.py` pins them). A field phase 3 adds goes into
  `HASH_NEUTRAL` at its earlier value, as `input_size` and `paste_p` did; none is expected.
- **A hypothesis line before every run**, and after the screen and the confirm, the result beside
  it in one word (matched, smaller, none, opposite) and 3 to 5 journal lines under topic 15
  (`docs/phase2-plan.md` section 5).
- **Context, never selection**: every screen and confirm reports trans_val accuracy and macro-F1
  together, with bobcat recall and trans_val's per-class rows beside them, so a gain that is all
  bobcat at camera 125 shows as one. The rule stays on the two macro-F1s.

What changes: the val tolerance of the keep rule (section 1), the screen (section 1), the ledger
(section 8), and one test scoring at most, of one finalist (section 7).

## 5. The configs: left to item 13.11

The `p3_` configs are **not** added in this item. They live in `config.yaml` (the configs
`trailcam/train/config.py` loads), and `load_configs` refuses a model name that is not in
`MODEL_NAMES` of `trailcam/model/factory.py`; `convnext_tiny` is not there yet, so a `p3_convnext`
block would break every config load until the model code exists. And a `p3_` name today is a
phase-1 name to `train.py` (only `p2_` routes to a phase-2 ledger), so a run would be accepted
with `docs/ledger.md`, the ledger the final scoring relies on. Both need code, so the configs go
with it into 13.11: `p3_baseline`, `p3_convnext` and `p3_convnext_cp`, each tested for its one
change (`p3_baseline` none from `resnet18_arm_a`, `p3_convnext` the model, `p3_convnext_cp` the
arm and `paste_p` from `p3_convnext`).

## 6. The honest limits

- **trans_val is one camera.** Its 1,603 kept images all come from one location, 125; 46% are
  bobcats (734), it holds 9 of the 12 classes, and two of them have under 10 images, where one or
  two photos swing a class's F1 (`docs/phase2-plan.md` section 2). A trans_val gain is a gain at
  camera 125 with this class mix, and can be a better bobcat-versus-cat boundary and nothing more.
- **It has misled once already.** On trans_val, arm B looked no better than arm A; on trans_test's
  nine cameras arm B won on every seed (plan.md section 7, test results).
- **It is noisy.** One run's trans_val macro-F1 varies by about 0.016 across seeds (arm A), about
  0.044 with arm B's spread (`docs/phase2-plan.md` section 3). The keep rule's two trans_val
  conditions are the guard: a mean gain over 0.03 that also holds on every seed. With three seeds
  a gain under about 4 points still cannot be confirmed; phase 3 reports such a gain as not
  confirmed.
- **It is optimistic.** Phase 3 selects on trans_val, after phase 2 selected on it too: seven
  recipes have now been judged on camera 125, and the best of several tries looks better than it
  will on the next new camera (phase 2's honesty note, WILDS, Koh et al. 2021).
- **It cannot grow.** Every other unseen camera belongs to trans_test (21,473 images from 9
  cameras, plan.md section 7). Moving any of them into validation would change the test set that
  phase 1 already scored, so trans_val stays as it is.

## 7. The test-set rule

The test sets were scored once, by phase 1 (item 9.02, `docs/test-ledger.md`). Phase 2 declared a
second scoring and did not run it, since nothing was kept (`docs/phase2-plan.md` section 12).
Phase 3's rule, written before any phase-3 run:

- **At most one recipe, once.** Only the finalist (section 1), all three of its seeds, each
  checkpoint scored once on cis test and trans test. No other candidate, screen or seed is ever
  scored on a test split. If nothing is kept, there is **no test scoring**, and phase 3 is
  reported on val and trans_val only.
- **With the user's go-ahead**: `APPROVED_BY` in `notebooks/kaggle_score_test.ipynb`, and the run
  logged as phase 1's was.
- **In its own file**, `docs/test-ledger-phase3.md`, not a section of `docs/test-ledger.md`:
  `score_test.py` refuses a test ledger that already holds a row, the guard that keeps phase 1 at
  one scoring, and a separate file leaves that guard as it is.
- **It needs code that does not exist.** `score_test.py` scores only `docs/ledger.md`'s
  pre-registered phase-1 set (a small CNN row and both arms per seed), and the notebook's `LEDGER`
  is fixed to that file. A phase-3 mode (the finalist's rows of `docs/ledger-phase3.md`, the new
  test ledger, the phase-1 mode refusing both by name, and a refusal unless the recipe was kept)
  is a tier-3 item of its own, **item 13.12** (approved by the user on 2026-10-19, due
  2026-10-27). It is built only after a screen passes; if none passes, it is recorded as
  skipped. Phase 2 named the same idea 13.06 and never added it; 13.12 takes a new id so that
  phase 2's references to 13.06 keep their meaning.
- **The scores choose nothing.** Not the finalist (chosen before), not the Space's seed (chosen on
  val, as in plan.md's open question 7).
- **Disclosure.** The README and the slides report phase 1's test results (`docs/test-ledger.md`)
  and phase 3's (`docs/test-ledger-phase3.md`) side by side, never one for the other, and say that
  the test sets were used a second time, for one recipe chosen on val and trans_val only, after
  phase 1's test scores had been seen (2026-10-18).

## 8. The ledger: a new `docs/ledger-phase3.md`

Phase-3 runs go into a new file, `docs/ledger-phase3.md`, with the header and columns of
`docs/ledger.md`, built by item 13.11. Not rows in `docs/ledger-phase2.md`, for four reasons:

- **Phase 2 is frozen.** `docs/ledger-phase2.md`'s header, its checker and `docs/phase2-plan.md`
  section 4 all say it holds `p2_` configs only; its rows are the record of a closed phase.
- **Neither option is free.** Today the ledger checker refuses any non-`p2_` row in
  `ledger-phase2.md`, and `train.py` treats a `p3_` config as a phase-1 config, accepting it with
  `docs/ledger.md` and refusing it with `ledger-phase2.md`. So the routing code changes in either
  case; the new file adds one entry to the same mechanism, a prefix-to-ledger table where today
  there is the single `p2_` / `ledger-phase2.md` pair.
- **The checks fit.** The phase-1 checks (any ledger but `ledger-phase2.md`) refuse two config
  hashes of one model, which `p3_convnext` and `p3_convnext_cp` are; 13.11 gives
  `ledger-phase3.md` phase 2's checks (several hashes per model, one per config, one split seed,
  the `p3_` prefix on every row) and refuses a `p3_` row in either other ledger. `check.sh` runs
  these checks through `tests/test_ledger.py`, which runs the checker on the repository's ledgers;
  13.11 adds the new file there.
- **The tools follow.** The training notebook's `LEDGER` parameter is passed to `--ledger`, and
  its first cell refuses a config of the other phase; the evaluation notebook accepts exactly two
  ledger values and pairs each phase-2 config with `p2_baseline`. With its own file and its own
  control, `p3_baseline`, phase 3 avoids phase 2's known limitation of two rows of one config and
  seed in one ledger (`docs/phase2-plan.md` section 9). 13.11 adds `docs/ledger-phase3.md` to
  both notebooks' allowed values, with `p3_baseline` as the default pair.

The cost: the baseline's seeds 1 and 2 are in `docs/ledger.md`, so a confirm's deltas are
computed across two ledgers, as phase 2's were.

## 9. Item 13.11: the code

Item 13.11 (tier 2: `trailcam/`, `config.yaml`, the notebooks, `space/`), due 2026-10-22, before
any phase-3 run:

- `convnext_tiny` in `trailcam/model` (`build_model`), timm, pretrained with the pinned tag
  `convnext_tiny.fb_in1k`, its head sized from the class map; `head_module` (timm's
  `get_classifier`, the head's final linear layer) and `parameter_groups` working; its shape table
  tested on the 12-way head; ImageNet's mean and standard deviation checked (R21).
- Grad-CAM's target layer for ConvNeXt (the last stage, before the head's pooling), and every
  evaluation tool working with it: `evaluate.py`, `attention.py`, `slices.py` (the feature maps
  read the head's input, which `head_module` gives) and `score_test.py`'s checkpoint loading.
- The `p3_` configs (section 5), checkpoints carrying `input_size` as 13.02's already do, and the
  phase-3 ledger and its routing (section 8).
- The Space able to load a ConvNeXt checkpoint. That needs more than the config:
  `space/requirements.txt` has no timm today, and `tests/test_space.py` asserts it, so timm and
  its own dependencies (`huggingface_hub`, `safetensors`) join it, pinned to the lock; and the
  Space's model is `DEFAULT_CONFIG` in `trailcam/demo/predict.py`, which `deploy_space.py` checks
  the weights against, so it must be able to name a `p3_` config. No deploy in 13.11.
- Tests on the synthetic fixture: a ConvNeXt dry run, its checkpoint evaluated end to end, the
  `p3_` routing refused against the other ledgers, the phase-1 and phase-2 hashes unchanged.
- The phase-3 lines of the invariants that phase 2 extended for the same reasons
  (`docs/review/invariants.md`: R17, R25, R30 and R40).

## 10. Budget

About 2 to 3 Kaggle GPU hours of the weekly 30. A ResNet-18 run takes about 12 minutes
(`docs/phase2-plan.md` section 7); a ConvNeXt-Tiny run is estimated at about 2 to 3 times that,
25 to 35 minutes, to be measured by the first one.

| batch | runs | estimate |
| --- | --- | --- |
| screen | `p3_baseline`, `p3_convnext`, `p3_convnext_cp`, seed 0 | about 1 to 1.4 hours |
| confirm | seeds 1 and 2 of each candidate that passes the screen: 0, 2 or 4 runs | 0 to about 2.3 hours |

Every outcome but both candidates confirming fits about 2.5 hours (one confirm: about 1.9 to 2.5);
both confirming is about 2.7 to 3.7 hours. Phase 2's rule holds: before each run, its projected
minutes (the last ConvNeXt run's time) and the running total go into the progress log. **The cap is 3 GPU hours**: if the next
confirm would pass it, that confirm waits for the user, who can raise the cap (the quota allows
it), and (a)'s confirm runs before (b)'s.

## 11. Timeline

The presentation is on **2026-11-04** (Wednesday), moved from 2026-10-26 by the user on
2026-10-19; 10-26 is no longer a deadline.

| date | phase 3 |
| --- | --- |
| 2026-10-19 to 10-21 | this plan (item 13.10, due 10-21) |
| 2026-10-22 | item 13.11, the code (section 9); its review |
| 2026-10-22 to 10-25 | the screening batch on Kaggle, run by the user (section 4) |
| 2026-10-25 to 10-27 | the confirms; item 13.12, the test-scoring code (section 7), once a screen passes |
| about 2026-10-28 | the freeze: the verdicts and the finalist; then, only if a finalist exists and the user approves, its one test scoring and the Space redeploy |
| 2026-10-29 to 11-03 | slides, backup video, rehearsal |
| 2026-11-04 | presentation |

**The Space redeploy** is optional and needs a kept finalist. Its seed is chosen on val (the
finalist's best val macro-F1), never on a test score. Before the switch, the Space's CPU must
still answer in well under a second: a ConvNeXt-Tiny forward pass costs more than ResNet-18's,
so it is measured on the Space (the app reports the time), with and without Grad-CAM. If it
fails, or no finalist is kept, the Space keeps arm B seed 2. The Space's card then quotes the
deployed model's numbers with the trans_val honesty note, and its test numbers only if section
7's scoring ran.

## 12. Open questions

All four were answered before any phase-3 run (through the coordinator).

1. **The finalist rule** (section 1): higher mean trans_val macro-F1, unless within 0.03, then
   higher mean val macro-F1. **Decided 2026-10-19, the user: as written.**
2. **The screen** (section 1): +0.03 trans_val and at most -0.03 val on seed 0. **Decided
   2026-10-19, the user: as written.** The keep rule was already fixed.
3. **(b) after a broken (a)** (section 3): both screens run regardless of (a)'s verdict, and (b) is
   held back only if (a)'s training is broken. **Decided 2026-10-19, the user: as the plan has
   it**; (b) does not wait on (a)'s screen.
4. **The test-scoring item** (section 7). **Decided 2026-10-19, the user: approved**, added as
   item 13.12 (tier 3, due 2026-10-27). It is built only after a screen passes; if none passes,
   it is recorded as skipped.

## 13. Results

None yet. The screens, the confirms, the verdicts and the freeze go here, every delta recomputed
from `docs/ledger-phase3.md` and `docs/ledger.md`.

## Sources

The scores: `docs/ledger.md`, `docs/ledger-phase2.md`, `docs/test-ledger.md`, `docs/plan.md`
section 7 and `docs/phase2-plan.md` sections 2, 3, 6, 7 and 12. The literature behind the
candidates (copy-paste, WILDS) is listed in `docs/phase2-plan.md`'s sources and
`docs/research-report.md`.

- Liu, Mao, Wu, Feichtenhofer, Darrell, Xie. "A ConvNet for the 2020s." CVPR 2022.
  https://arxiv.org/abs/2201.03545 (ConvNeXt; the weights are timm's `convnext_tiny.fb_in1k`).

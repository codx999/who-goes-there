# Test ledger, phase 3

One row per checkpoint of phase 3's finalist scored on cis_test and trans_test by phase 3's one
test scoring (`trailcam/evaluate/score_test.py --phase 3`, item 13.12; `docs/phase3-plan.md`
section 7), in the column order the script prints. The columns are those of `docs/test-ledger.md`,
whose notes on each column hold here too. This file is separate so that phase 1's run-once guard
on `docs/test-ledger.md` stays as it is, and it is committed with its header and no row so that
the scoring can refuse a missing file instead of creating one (a mistyped path never passes for
an empty test ledger).

The scoring runs once, for the finalist's three checkpoints (run seeds 0, 1 and 2) together, on
Kaggle, by the user, with the go-ahead (`APPROVED_BY` in `notebooks/kaggle_score_test.ipynb`,
with `PHASE = 3`). The finalist is not typed in anywhere: the script recomputes the plan's
screen, keep rule and finalist rule from `docs/ledger-phase3.md` and `resnet18_arm_a`'s rows of
`docs/ledger.md`, and refuses if nothing was kept. It prints each checkpoint's row as soon as it is
scored, and at the end appends all the rows to the copy in the Kaggle clone (`--test-ledger
<clone>/docs/test-ledger-phase3.md`). The user then pastes the printed rows once into this file in
the repo. A run that stops part way appends nothing and is run again from the start.

Disclosure (section 7): these scores are the test sets' second use, for one recipe chosen on val
and trans_val only, after phase 1's test scores (`docs/test-ledger.md`, 2026-10-18) had been seen.
They are reported beside phase 1's, never in their place, and they choose nothing: not the
finalist, chosen before, and not the Space's seed, chosen on val.

Data: Caltech Camera Traps (CCT20), Beery et al. 2018, CDLA-Permissive 1.0.

| run id | model | arm | config | split hash | checkpoint sha256 | cis_test accuracy | cis_test macro-F1 | trans_test accuracy | trans_test macro-F1 | accuracy gap | macro-F1 gap | scored at | approved by |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

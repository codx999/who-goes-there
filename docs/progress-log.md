# Progress log

Dated "done / next" entries, newest last. One entry per session.

## 2026-10-10: session 0, bootstrap (Fable 5.1)

**Done.** Playbook steps 1 to 6 and 8: git and uv (Python 3.12), pyproject with ruff, strict
mypy and pytest, data guard extended with image and checkpoint suffixes and an `assets/`
allow-list (25 tests), git hooks installed and verified against a fake JPEG, Claude settings,
rules and the `/dryrun` and `/review-shapes` commands, `check.sh` green. CCT20 metadata
downloaded (3 MB) and inspected: `docs/data-inspection.md`. Class guideline and setup review saved
to `docs/course-guideline.md` and `docs/setup-review.md`. Public repo `codx999/who-goes-there`
created by the user (MIT, CCT20 citation in README); first CI run green (run 38096639477).

**Next.** Step 7 in a dedicated design session: `docs/plan.md` from HANDOFF.md, data-inspection.md,
course-guideline.md and setup-review.md; then CLAUDE.md, `docs/progress.json`. Then step 9
(Part B of `docs/review/playbook-reusable.md`) in its own session, and step 10.

Review: tier 0 for the docs; the guard script and tests are control-plane files and get their
first review with the step 9 machinery.

**Review of the bootstrap (same day).** `/code-review xhigh --max-findings all 9788ae4..HEAD`,
11 findings, none high. Fixed (8): notebook image outputs now refused; `assets/` images capped at
24 files and 12 MB over the index; JSON with more than 100 image or annotation records refused as
dataset metadata; video suffixes and the MP4 marker blocked; `.gitignore` regenerated from the
guard's lists with a drift test; `assets/` allow-list narrowed to jpg, jpeg, png, webp; `.env.local`
added to the Read deny list; `check.sh` fails outside CI when hooks are not enabled; unit tests for
`_parse_raw`, `pre_push_revs`, `read_blobs`. Each fix has a test that fails on the old guard (9 of
39 failed against the pre-fix copy). Skipped (3, low): mypy targets listed twice, version string in
three places, importlib loading in the test. Review: tier 1, ran /code-review xhigh. The fix commit
touches control-plane files, so it gets one fix-round review of its own range.

**Fix-round review (same day).** `/code-review xhigh --max-findings all 257bbfa..HEAD`, 9
findings, none high. Fixed (9): the notebook and metadata checks were rewritten as byte-level
scans on every text blob regardless of suffix (image MIME keys and data URIs followed by at least
1000 base64 characters; more than 100 dataset record keys), which also closes markdown
attachments, HTML outputs, top-level lists, nested dicts and renamed dumps; a second allow-listed
image folder `docs/figures/` (40 files, 20 MB) for generated plots, mirrored in `.gitignore` and
the drift test; image budgets enforced per commit in history mode; HEIC/AVIF recognised and
labelled as images; `check.sh` accepts an absolute hooks path; the set difference hoisted; the
clear-outputs workflow documented in the training rule. 49 tests; 12 of the new ones fail against
the previous guard. One judgment call: the payload threshold was added after the stricter scan
retroactively refused the previous commit's test file (a 400-character fake payload); the
threshold is a real improvement (docs and icons do not count) and avoided a force-push.
Review: tier 1, ran /code-review xhigh, two rounds; no further round unless requested.

## 2026-10-10: session 1, design document (Fable 5.1)

**Done.** Playbook step 7 in a dedicated session, explore then plan then implement: all seven
inputs read, the outline of `docs/plan.md` with its conflicts presented and approved in chat, then
four commits, `check.sh` green before each. `docs/plan.md` is now the source of truth: problem
statement (12 classes, 11 animals plus `car`), the explicit class-to-index map, the split design
(official location split kept; train and cis_val pooled and re-split by sequence because they share
224 sequences; cis and trans test scored once), the four leakage tests, small CNN plus fine-tuned
EfficientNet-B0 from timm, AdamW with cosine warmup and early stopping on val macro-F1, accuracy
and macro-F1 side by side, Gradio demo and Hugging Face Space with weights uploaded via
`huggingface_hub`, the "decisions and why" section, a dated schedule for sessions 2 to 12, open
questions, sources. `docs/HANDOFF.md` retired with a note pointing to the plan. `CLAUDE.md` (69
lines) from the playbook template. `docs/progress.json` with 36 items (after the review fixes) for sessions 2 to 12, 12 of
them owner=user (Kaggle runs, Hugging Face account and Space, the final scoring, example photos,
video, rehearsal); sessions 0 and 1 are logged here only. Worktree note: the app sets an absolute
`core.hooksPath` to the main checkout, so `check.sh` reported hooks disabled until
`git config --worktree core.hooksPath "$PWD/.githooks"`; recorded in CLAUDE.md.

Judgment calls, approved: `car` kept as the twelfth class; the playbook's "15 classes plus empty"
invariant rows are superseded by the plan and get corrected in session 2; fixture images are
generated at test time because the guard blocks them in git; the split is reproduced from seed
plus hash rather than committed.

**Next.** Session 2 (2026-10-11): Part B of `docs/review/playbook-reusable.md` writes
`invariants.md`, `risk_files.sh` with its test, both reviewer agents and `/review-session`
(items 2.01 to 2.03); then the user runs the first review (2.04) over the bootstrap and design
range. Open question 1 (the TA's reading of the torchvision dataset rule) is due 2026-10-13.

**Review (same day).** CLAUDE.md is a control-plane file written by Fable 5.1 and the review
machinery does not exist yet, so the user ran `/code-review xhigh --max-findings all main..HEAD`:
21 findings, none high, all fixed in one commit. Steps that could not pass after the merge (2.02
and 2.04 now name the explicit range 9788ae4..HEAD; 2.01 counts rows without assuming an id
prefix; 11.02 relies on the guard's Hugging Face token pattern instead of `git grep`); the dry run
now accepts `--dry-run=True` as `/dryrun` passes it and writes its own synthetic fixture
(`trailcam/data/synthetic.py`, shared with the tests, replacing a `tests/fixtures/` generator and
its needless 100-record cap); `docs/ledger.md` is created with its header in 5.03 before the first
Kaggle row is pasted; a packaging item 5.04 (build-system table, `uv build`, wheel install in a
fresh venv) precedes the Kaggle pip install; 5.01 adds the torch, torchvision and timm
dependencies with the mypy overrides; 6.01 uses `uvx --from nbconvert`; CLAUDE.md names the
notebook pip exception; Grad-CAM overlays are made only from the `assets/` photos so no dataset
image lands in `docs/figures/` (the asset selection moved from 10.02 to 8.03, before the val
evaluation 8.04; 8.02 writes fixture figures to a temporary `--out`); 11.01 pins grad-cam only if
open question 5 says so and adds `deploy_space.py` to the mypy targets; the TA question is due
2026-10-11 with Oxford-IIIT Pet named as the fallback; the sequence label for stratification is
the first kept frame; four numbers corrected (57,864 images, about 21 MB of weights, 9 of 12
classes in trans_val, 407 same-species multi-annotation images). Rejected: none.
Review: tier 0 for the docs plus CLAUDE.md, ran /code-review xhigh; one round.

## 2026-10-11: session 1 continued, design revision after the literature review (Fable 5.1)

**Done.** A parallel session wrote `docs/research-prompt.md` and `docs/research-report.md` (CCT20
literature, task options, a ranked list of research questions, an experimental design) and a draft
`scripts/kaggle_prepare_cct20.py`. Both documents are now committed under `docs/` (the report with
a note that plan.md wins where they differ). Decisions taken with the user and written into
`docs/plan.md` (now with a section 6, "Research question and experiment", and renumbered sections
after it): keep 12 classes, no `empty` (the report's 13-class option becomes an add-if-early run);
adopt the research question, box-guided crop augmentation at training time only, arm A vs arm B,
seed 0 first and seeds 1 and 2 only if arm B seed 0 is done by 10-16; measure accuracy, macro-F1,
attention-in-box (Grad-CAM mass inside the ground-truth box) and an animal-removed test (box
blanked, does the prediction survive); ResNet-18 from torchvision as the backbone with
EfficientNet-B0 as the Space fallback; 512 px longer-side cache as the prepare script already does;
stratify the re-split by sequence majority class and location with at least 30 val images per
class; Food-101 as the fallback dataset (the prompt's choice, replacing Oxford-IIIT Pet); keep
boxes, capture hour and a pixel-based day/night flag in the metadata (verified today: every image
has a full timestamp and about half are night shots; hour 12 is doubled and looks like a clock
default, so the pixel test is the label); descriptive slices (day/night, common/rare, per unseen
camera) and a two-colour feature map of backbone features; temperature scaling as add-if-early.
Rejected from the report: scoring the test set mid-way (plan keeps one scoring at the end) and a
committed split CSV (the guard refuses it; seed plus hash stays). CLAUDE.md gained two lines (the
report in the context list; boxes never a model input), 72 lines. `docs/progress.json` rewritten:
42 items, 13 owner=user; new items for the box-guided sampler (4.04), the prepare script with the
day/night test (6.01), arm A and arm B runs (6.05, 7.02, 7.03), attention and animal-removed
(8.03), slices and feature map (8.04), calibration if early (12.03). The draft Kaggle script
from the parallel session was deleted at the user's request; item 6.01 writes it fresh, type-checked and
with a test. The untracked copy of the research prompt in the main checkout was removed too (identical
to the committed one) so the merge is not blocked.

**Next.** Session 2 (2026-10-11, today): the review machinery (2.01 to 2.03), then the user's first
review (2.04) over 9788ae4..HEAD. Ask the TA about the torchvision dataset rule today (open
question 1).

Review: tier 0 for the docs plus CLAUDE.md (two added lines). The user decides whether the
CLAUDE.md change waits for the session 2 review or gets `/code-review xhigh --max-findings all
9c0c2e4..HEAD` now.

## 2026-10-11: session 2, the review machinery (Fable 5.1)

The user asked for Opus 5.5 at high; the session ran on Fable 5.1 and said so before writing
anything. Item 2.04 therefore replaces the Fable risk-reviewer with a second Opus run.

**Done.** Step 0, approved by the user: the two rules files brought in line with the final design.
`.claude/rules/training.md` lists the ledger-row fields of plan.md section 5 in order (run id,
model, arm, config, split seed, run seed, split hash, epochs run, best epoch, scheduler, val
accuracy, val macro-F1, trans_val accuracy, checkpoint hash) and states that arm A and arm B
configs differ only in the `arm` field; `.claude/rules/data.md` gains the rule that boxes are
training-time augmentation and attention scoring only, never a model input, with the sampler in
`trailcam/data/boxcrop.py` unreachable from the eval transform, and drops the `tests/fixtures/**`
path (the generator is `trailcam/data/synthetic.py`, covered by `trailcam/data/**`). Then items
2.01 to 2.03, one commit each, Part B of `docs/review/playbook-reusable.md` filled for this repo:
`docs/review/invariants.md` with 38 rows in one table (R01 to R07 what never enters git, R10 to R17
splits and test-set isolation, R20 to R27 transforms, boxes and the experiment, R30 to R38 model,
training, evaluation, demo and Space, R40 to R45 the ledger row and the control plane; the
playbook's "15 classes plus empty" starter row superseded by the 12-class map); `scripts/risk_files.sh`
(regex: `^trailcam/`, `^space/`, `\.ipynb$` plus the control plane; exit 2 on a bad range) with
`tests/test_risk_files.py` (12 tests: risky and safe samples, bad ranges, a move out of a risky
path, a non-ASCII path, the invariants table shape, and every "where" path matched by the regex);
`.claude/agents/code-reviewer.md` (Opus 5.5, xhigh, whole range), `.claude/agents/risk-reviewer.md`
(Fable 5.1, high, given files) and `.claude/commands/review-session.md` (tier 3 for 3.03 and 9.01,
run items through the ledger row R40). Acceptance steps run and passed: the grep count is 40, the
test file passes, `risk_files.sh 9788ae4..HEAD` prints CLAUDE.md, `scripts/guard_data.py` and
`scripts/risk_files.sh` and no `docs/` path, a bad range exits 2, `check.sh` green (61 tests).
Not changed: `.claude/settings.json` and CLAUDE.md (Part B steps 5 and 6 were done at bootstrap
and in session 1).

**Next.** Item 2.04 (user): the first review over the explicit range `9788ae4..HEAD`
(`/review-session` alone computes merge-base(main, HEAD)..HEAD and would cover only this session's
commits after the merge). Then session 3 (2026-10-12): the class map, the metadata loader, the
re-split by sequence with the leakage tests (tier 3, xhigh) and the data card.

Review: tier 2 (every changed file is control plane: `.claude/`, `scripts/`, `tests/`). Command
for the user: `/review-session 9788ae4..HEAD 2.01 2.02 2.03`, with a second Opus `code-reviewer`
run in place of the Fable `risk-reviewer` because Fable wrote the files; or
`/code-review xhigh --max-findings all 9788ae4..HEAD`. Not started by this session.

## 2026-10-11: item 2.04, the first review and its fixes (Opus 5.5)

**Done.** The user started `/review-session 9788ae4..HEAD 2.01 2.02 2.03` in this session after
fast-forwarding to the session 2 tip (e777e14). Tier 2: `risk_files.sh` printed 22 control-plane
files, no split or scoring code; 14 invariant rows selected (R01 to R04, R07, R14, R30, R37, R38,
R41 to R45), every row "spec changed" because the range creates the cited documents. The agent
types were not registered, so both reviewers ran as general-purpose agents on Opus at xhigh
following `.claude/agents/code-reviewer.md`: one on the whole range, one on the risk files in
place of the Fable risk-reviewer. 23 findings, 6 made by both; every one checked against the code,
reproduced with a scratchpad test or a temporary repository where possible, and every fix has a
regression test that fails on the old code. Eight fix commits: (1) the guard counts JSON-escaped
and single-quoted record keys and refuses any notebook output or attachment (high, both
reviewers); (2) more than 100 UUIDs in a blob is refused (every CCT image and sequence id is a
UUID, 57,864 of 57,864, counted) and trailing `.`, `~` or space no longer hides a suffix; (3) an
allowed image name must hold a JPEG, PNG or WebP, and WebP, TIFF and BMP signatures are refused
elsewhere; (4) `--pre-push` checks the image budgets over the whole pushed tree, with
`tests/test_guard_git.py` running the guard modes in a temporary repository; (5) `.gitignore`
ignores `.env.*` and `secrets.toml` anywhere, and the suffix set is pinned in both directions;
(6) `check.sh` fails when `.githooks/` is missing, and `tests/test_control_plane.py` pins hook
modes, CI, check.sh, pytest config, deny lines and reviewer tools; (7) `risk_files.sh` lists root
`config.yaml` and `configs/`, R25 names it; (8) `/review-session` makes the tier-3 tests tier 3
and falls back to the read-only `Plan` type, `/review-shapes` covers both models and the eval
transform, `/dryrun` says printed, rules `data.md` quotes the plan's eval transform. `check.sh`
green before each commit (97 tests).

Not fixed, the user's call: the Read deny rules are relative to the session's directory, so in a
worktree session the main checkout's `data/`, `models/`, `outputs/` and `.env*` are not denied
(high, both reviewers, reasoning only; needs absolute rules in the user's settings, which Claude
does not edit); the CLAUDE.md line "val and test get resize and normalise"; item 3.03's real-data
check may be skipped when `data/` is absent (items are not Claude's to edit); plan section 7 puts
a camera's background photo next to the per-camera chart, which R04 forbids unless it comes from
`assets/`; `.pdf` is not restricted by the guard.

Then, at the user's go-ahead: absolute Read deny rules for this project's `data/`, `models/`,
`outputs/`, `.env*` and any `kaggle.json` added to the user's `~/.claude/settings.json` (outside
the repo; the probe file could not be written from this session, so the user checks it); the
CLAUDE.md eval transform line (shorter side 256, centre crop 224, normalise; 73 lines); plan
section 7's per-camera chart made bars only; `.pdf` blocked by suffix and signature. Item 3.03's
steps edited by Claude at the user's explicit request in chat (an exception to "never edit
items"): the real-metadata coverage checks read `TRAILCAM_DATA_ROOT` and the item flips only when
`pytest -rs tests/test_split.py` reports 0 skipped; the skip is allowed only in CI.

Fix round, started by the user: `/review-session e777e14..HEAD`, tier 2, 12 risk files. Both
agent types still unregistered, so both ran as the built-in `Plan` type: code-reviewer on Opus at
xhigh over the range, risk-reviewer on Fable at high over the risk files (Opus wrote the fixes,
so Fable is the different model). Nothing above low; every first-round fix confirmed working
with a test that fails without it. Seven findings, all verified and fixed, plus two residual gaps
closed: pytest.toml and dot-prefixed override configs in the regex and the control-plane test,
no conftest.py (pytest 9.1.1's lookup order read from its source); the workflow folder, every
secrets form and the image budgets pinned; the Plan fallback's limits stated in its prompt; a
new `scripts/data_root.sh` so check.sh points `TRAILCAM_DATA_ROOT` at the main checkout's
`data/` outside CI; the dry run's never-append rule in plan section 5 and rules training.md;
citations in rules data.md and the UUID count in data-inspection.md; the whole pytest config and
every agent file pinned. `check.sh` green before each commit. No third round unless the user asks.

**Next.** Merged and pushed by Claude at the user's go-ahead (d80be8c). Item 2.04 (owner: user)
flipped to `passes: true` by Claude at the user's explicit request in chat. Open: the user checks
the Read deny rules with a probe file in `data/`. Then session 3 (2026-10-12). Then session 3 (2026-10-12): class map,
metadata loader, re-split with the leakage tests (tier 3).

Review: tier 2, ran /review-session 9788ae4..HEAD (code-reviewer Opus xhigh on the range, second Opus run on the risk files)

## 2026-10-11: session 3, item 3.01, the class map (Opus 5.5)

**Done.** `trailcam/data/classes.py`: the 12 classes of plan section 2 in index order 0 to 11
with their CCT ids, read-only name-to-index and CCT-id-to-index maps, `INDEX_TO_CCT_ID`,
`NUM_CLASSES`, the four dropped ids (30 empty, 21 badger, 51 fox, 34 deer) and lookups that
raise `DroppedClassError` for a dropped id or name and `KeyError` for an unknown one.
`tests/test_classes.py` reads the expected order and ids from the plan's class table and the
category table in `docs/data-inspection.md`, checks kept plus dropped equals the 16 categories,
and that the maps are read-only. R05's test cell names the file; the two-species filter stays
"none yet" until 3.02. `uv run pytest tests/test_classes.py` passes; item 3.01 flipped.

Unblocker, edited by the user: the project Read deny rules `./data/**`, `./models/**` and
`./outputs/**` were applied to folders of that name at any depth and refused `trailcam/data/`
(probe: `tests/data/` refused, `trailcam/model/` allowed). The user changed them to `/data/**`,
`/models/**`, `/outputs/**` (anchored at the project root) in this worktree's and in the main
checkout's `.claude/settings.json`; an app-made worktree also loads the main checkout's project
settings, so the worktree edit alone did not help. Probes after both edits: nested folders
readable; top-level `data/`, `models/`, `outputs/` in the worktree and the main checkout still
refused. `tests/test_control_plane.py` pins the new lines. The main checkout's copy is an
uncommitted edit: `git restore .claude/settings.json` there before merging this branch.

**Next.** Item 3.02, the metadata loader and the synthetic metadata writer.

## 2026-10-11: session 3, item 3.02, the metadata loader and the synthetic metadata (Opus 5.5)

**Done.** `trailcam/data/metadata.py`: finds the one `<split>_annotations.json` per official file
under `--root` (recursive), checks every category against the class map by id and name, and
builds one frozen `Record` per kept image (label index from `trailcam/data/classes.py`, location,
seq_id, frame_num, seq_num_frames, width, height, capture hour, every box as COCO x, y, w, h in
the original pixel size, and `night` from a `day_night.json` sidecar when present, else None).
Drop rules in order: two_species, empty, rare, unannotated, counted per file; repeated image ids,
orphan annotations, malformed boxes and unknown or renamed categories raise. The sidecar format
(image id to `{"night": bool}`) is defined here for item 6.01. `trailcam/data/synthetic.py`
(metadata part): 12 classes, cis locations 101 and 102, trans location 201, 3-frame sequences,
7 pool sequences per class and cis location, boxes on about 90% of animal annotations, planted
empty, rare, two-species and two-opossum frames and one burst split across train and cis_val; it
writes only the five files and the sidecar into the given directory.

`tests/test_metadata.py`: 23 tests, expectations read back from the written JSON; one runs on the
real metadata when `TRAILCAM_DATA_ROOT` is set (file totals, 78 two-species images, 4,014
empties, all as in `docs/data-inspection.md`). The CLI on the real `data/` printed counts only:
kept 53,535 (train 13,496, cis_val 2,485, trans_val 1,603, cis_test 14,478, trans_test 21,473);
dropped two_species 78, empty 4,014, rare 237, unannotated 0; boxed fraction 0.67 (cis_val) to
0.92 (trans_val); hour 12 holds 4,525 images against about 1,100 to 1,800 for its neighbours, as
plan section 2 says. Invariant test cells filled: R03, R05, R10, R15, R37.

Follow-up, at the user's go-ahead after reviewing the two open decisions: boxes are now stored as
fractions of the image's `width` and `height`, clipped to the frame in the loader
(`normalise_box`) and counted per file; on the real metadata 100 of 47,729 kept boxes are clipped
(train 33, cis_val 3, trans_val 2, cis_test 11, trans_test 51), and the CLI prints the count.
The sidecar format, `day_night.json` mapping image id to `{"night": bool, "hour": int}` with
extra keys allowed and its path passed with `--sidecar`, is written into plan section 2, along
with the box format; R24's scenario names the double-scaling failure. The synthetic sidecar
carries the hour too. `tests/test_metadata.py`: 26 tests (fractions against the written pixel
boxes, `normalise_box` cases, two planted overrunning boxes clipped and counted).

Review, started by the user: `/review-session 3.02`, tier 2, range 72613c9..bcc04ec, risk files
`tests/test_metadata.py`, `trailcam/data/metadata.py`, `trailcam/data/synthetic.py`;
code-reviewer on Opus at xhigh, risk-reviewer on Fable at high. Nothing above low; every
scenario held (R01, R03, R05, R10, R15, R23, R37, R42 to R45; the split side of R10 and R15 and
the dry-run side of R37 skipped as known gaps). Spec changed: plan section 2 (rows R01, R05, R06,
R15, R21, R23, R32 re-read; still accurate). Three low findings, all verified and fixed in
fe6c301:
- both reviewers: `load_metadata` read a `day_night.json` next to the annotation files on its
  own, against plan section 2; the flag now comes only from `--sidecar`;
- code-reviewer: a sidecar keyed wrongly (by file name) left every flag unset and the summary
  said "no sidecar"; a sidecar missing any kept image id now raises;
- code-reviewer: the real-metadata test did not pin the box facts plan section 2 states; it now
  checks rare drops per file (50, 4, 0, 168, 15), 47,729 kept boxes, 100 clipped, and that boxes
  are in the original-size pixel space (some end past 0.9 of the frame, impossible if they were
  1024 px boxes over 2048 to 2560 px originals). That test passed on the old code, as a test-gap
  fix should; the four sidecar tests failed on it.
`check.sh` green (152 passed). The fix commit touches `trailcam/data/metadata.py` and modifies
one existing test, so one fix round on fe6c301 is due.

Fix round, started by the user: `/review-session bcc04ec..HEAD 3.02`, tier 2, risk files
`tests/test_metadata.py` and `trailcam/data/metadata.py`; code-reviewer on Opus at xhigh,
risk-reviewer on Fable at high. Nothing found in the range by either; every scenario held (R01,
R03, R05, R10, R15, R42 to R45). Spec changed: plan section 2's two sidecar rules, which the code
matches. One low note outside the range from code-reviewer (R03, reasoning only), fixed: two
real-metadata asserts called `len()` inside the assert, so a failure would have printed a real
record (with its image id and file name) or real box values; the counts are now taken first.
Checked by breaking both expectations in a scratch copy: the failures print counts only. No
third round.

**Next.** Item 3.03, the re-split and the leakage tests (tier 3, xhigh).

Review: tier 2, ran /review-session 3.02 (72613c9..bcc04ec) and the fix round /review-session
bcc04ec..HEAD 3.02 (code-reviewer Opus xhigh, risk-reviewer Fable high)

## 2026-10-11: session 3, item 3.03, the re-split and the leakage tests (Opus 5.5, xhigh)

**Done.** `trailcam/data/split.py`: `make_split(meta, seed)` pools the official train and cis_val
records, groups them by `seq_id` and re-splits whole sequences into train and val; trans_val,
cis_test and trans_test are the tuples `load_metadata` returned, passed through untouched. A
sequence's class is the majority label over its kept frames (ties to the lower class index; every
image keeps its own label); a pool sequence spanning two locations is refused. Strata are (class,
location): each class gets 15% of its sequences rounded half up (at least 1, at most n - 1 when
n >= 2), spread over its locations by largest remainder, so every stratum gets the floor or the
ceiling of 15%; within a stratum the val sequences are those with the smallest SHA-256 of (seed,
seq_id). No random generator is used, so the split does not depend on record order, dict or set
order, the global `random` state or the Python version: the hash computed on the Mac is the one a
Kaggle run will record. The split hash is a SHA-256 over (split, image id, label) for all five
splits, sorted by image id. Before returning, `check_split` refuses (with counts only) an image id
twice, a `seq_id` in two splits, a location in both a cis and a trans split, or a cis location
missing from train or val. Output splits are `train`, `val`, `trans_val`, `cis_test`,
`trans_test`; each record's `split` field names the split it is in (one docstring line in
`trailcam/data/metadata.py` says so). `Split.records` is read-only and `Split`'s repr shows the
seed, the hash and the sizes only. Nothing is written to a file.

`uv run python -m trailcam.data.split --root <main checkout data/> --seed 42` printed counts only.
Pool: 15,981 images in 6,274 sequences (222 of them with kept frames in both official files),
3 mixed-label sequences, 0 majority ties. Sizes: train 13,587, val 2,394 (0.150 of the pool's
images, 940 of 6,274 sequences, 0.147 to 0.154 per cis location), trans_val 1,603, cis_test
14,478, trans_test 21,473. Smallest val class: skunk with 40 images (rodent 78, bird 89); over 200
split seeds tried, the smallest val class never fell below 40. Locations: 10 in train and val,
1 in trans_val, 10 in cis_test, 9 in trans_test. Split hash for seed 42:
c78a9c1048f63aa52f34578ca17a6391315afdc2e478b4dbfd8a3aa954d1b009.

`tests/test_split.py`: 58 tests. Every leakage and coverage rule (plan section 3 tests 1 to 4)
runs on the synthetic metadata and on the real metadata; sizes, per-class counts, sequences and
the hash are pinned for seed 42 on both (test 5), so CI catches a change to the algorithm without
data. Also: same seed same split, different seeds differ, record order, the global `random` state
and `PYTHONHASHSEED` (in a subprocess) leave the hash unchanged, the hash covers assignment and
labels, the stratum sizes follow the majority class (a spy on `val_quotas`), `val_quotas` on 300
random size tables, six refusals of planted leaks, and a CLI run that prints no id and writes no
file. `TRAILCAM_DATA_ROOT=<abs path> uv run pytest -rs tests/test_split.py`: 58 passed, 0 skipped;
without it, 37 passed and 21 skipped (the CI case). Mutation check in a scratch copy: 15 planted
bugs (per-image draw, pooling every file, first-frame strata, hash over a set or over ids only,
subsampled trans_test, no runtime check, ties to the higher index, banker's rounding, per-stratum
rounding, the global `random`, set-order draws, unseeded tie-breaks, no location check, val taken
from the cis_val file) each fail at least one test both with and without data, and the leakage
tests catch the three leaking bugs on their own with the runtime check removed. R03 check of the
failure output with bugs planted, under the default, `--tb=long` and `-vv` styles: the first
version printed a real image id, because pytest prints a failing test's arguments (`Metadata` and
`Split` reprs) and the arguments of the frame that raised; fixed by a counts-only `Split` repr,
tests that take a `Case` whose repr names its source only, and `quietly`, which re-raises a failure
in `make_split` or the CLI as its type and message. After the fix: no id, `Record(` or
`Metadata(` in any failure output. Invariant test cells filled: R10, R11, R12, R13, R14, R15, R27.

Open: `Record` and `Metadata` keep the dataclass repr, which prints image ids. A later real-data
test that takes them as fixture arguments, or a notebook cell that echoes one, would show ids; an
id-free repr on both closes that for every caller. Added as item 3.05 with the user's OK (it
changes 3.02's module); not built in this session, so it stays out of the 3.03 review range.

Review, started by the user: `/code-review ultra` (cloud) and `/review-session 3.03`, tier 3,
range d889a44..e3a2cb5, risk files `tests/test_split.py`, `trailcam/data/metadata.py`,
`trailcam/data/split.py`; code-reviewer on Opus at xhigh, risk-reviewer on Fable at high. Nothing
above low; every scenario held (R01, R03, R05, R10 to R15, R27, R42 to R45; skipped as known gaps:
the loader side of R12, the ledger and data card of R14, the run seed of R27, the `Record` and
`Metadata` reprs of R03 (item 3.05), the notebook, Kaggle and assets parts of R01 and R43). Spec
changed: none. Two findings, both verified and fixed in 9d3221d:
- both local reviewers, low: the real-metadata tests skipped whenever `TRAILCAM_DATA_ROOT` was
  unset, though item 3.03 allows the skip only in CI. At the user's decision they now skip only
  when `CI` is set and fail otherwise, in `tests/test_split.py` and `tests/test_metadata.py`. A
  subprocess test runs them without the root as a local run (all 23 fail) and as CI (all 23
  skip); on the old code it failed (0 failed locally). R42's test cell names it.
- ultrareview, nit: `make_split` built each sequence's label `Counter` twice (inline and in
  `majority_label`); it now builds it once. Behaviour unchanged: the pinned hashes hold, and the
  first-frame, tie-rule and mixed-count mutations are still caught.
Notes, not fixed: the 30-val-image floor is tested for seed 42 only and not enforced at run time,
to be decided in 5.03 if the split seed can differ from 42 (code-reviewer); the pinned real values
have no independent source, so the data card (3.04) cross-checks them; the code-reviewer
reconciled the pool and trans_val totals with `docs/data-inspection.md` (risk-reviewer).
`check.sh` green (211 passed). The fix commit modifies existing test lines and touches
`trailcam/data/split.py`, so one fix round on it is due.

Fix round, started by the user: `/review-session e3a2cb5..HEAD 3.03`, tier 3, risk files
`tests/test_metadata.py`, `tests/test_split.py`, `trailcam/data/split.py`; code-reviewer on Opus
at xhigh, risk-reviewer on Fable at high. Nothing found by either; every scenario held (R01, R03,
R10 to R15, R27, R42 to R45; the same known gaps skipped). Spec changed: none. Both reproduced the
new subprocess test by hand (locally 3 failed and 20 errors, as CI 23 skipped). Notes, not fixed:
any non-empty `CI` counts as CI, so `CI=false` would skip too, the same rule `check.sh` uses (both
reviewers); item 3.05's step `uv run pytest tests/test_metadata.py` now needs
`TRAILCAM_DATA_ROOT` exported, or a run through `check.sh` (code-reviewer). No third round.

**Next.** Item 3.04, the data card (counts from the seed 42 run above), then item 3.05, the
id-free reprs (tier 2).

Review: tier 3, ran `/code-review ultra` and `/review-session 3.03` (d889a44..e3a2cb5) and the fix
round `/review-session e3a2cb5..HEAD 3.03` (code-reviewer Opus xhigh, risk-reviewer Fable high)

## 2026-10-11: session 3, item 3.04, the data card (Opus 5.5)

**Done.** `docs/data-card.md`: licence (CDLA-Permissive 1.0) and the Beery et al. 2018 citation,
the 12 classes with CCT ids, the four drop rules with images dropped per official file, the
per-split per-class counts after the re-split (seed 42) with sequences and locations, the boxed
fraction per split, the capture-hour histogram, the seed and split hash, and the paper's two
contradictions with the released files (57,868 vs 57,864 images; 224 sequences shared by train
and cis_val). Counts come from `python -m trailcam.data.metadata` and `python -m
trailcam.data.split --seed 42` on the main checkout's `data/`; the boxed fraction per re-split
split was counted with the same modules (the metadata CLI prints it per official file only). A
scratch script parsed the card and compared it with both CLI outputs: the 15 rows of the split
table (12 classes, images, sequences, locations), the hash, the drop table and the hour table
match exactly. Hash `c78a9c10...b009`, as logged for 3.03 and pinned in `tests/test_split.py`.

Cross-check of the 3.03 pins against plan sections 2 and 3 and `docs/data-inspection.md`: the
pool is 13,553 + 3,484 = 17,037 released, 13,496 + 2,485 = 15,981 kept = train 13,587 + val
2,394; per file, released minus two_species, empty and rare gives the kept count in all five
files, and 57,864 - 78 - 4,014 - 237 = 53,535; empties 995 + 61 + 1,180 + 1,778 = 4,014 and
two-species 78 match the inspection; rare in train is 50, not 56, because the six badger + coyote
images are dropped as two_species first (coyote 1,409 to 1,403); trans_val's rabbit falls from 70
to 9 (the 59 bobcat + rabbit and 2 coyote + rabbit images); clipped boxes total 100 of 47,729, as
the plan says. No disagreement found.

Pending in the card, by design: the day/night share (the sidecar comes from 6.01, the user
records the share in 6.03) and the trans-test locations used for `assets/` (chosen by the user in
8.05; `assets/` does not exist yet). The item's steps (file exists, counts equal 3.03's for seed
42, `check.sh` green) all hold, so `passes` is flipped. `check.sh` green (211 passed).

**Next.** Item 3.05, the id-free reprs for `Record` and `Metadata` (tier 2).

## 2026-10-11: session 3, item 3.05, id-free reprs for Record and Metadata (Opus 5.5)

**Done.** `trailcam/data/metadata.py`: `Record` and `Metadata` are `repr=False` dataclasses with
their own `__repr__`, after the counts-only `Split` repr of 3.03. A record prints as
`Record(class=..., split=..., location=..., frame=n/m, boxes=k, night=...)` (a label outside the
class map prints as `label n` rather than raising); the metadata prints the records per split,
the drops per rule and the clipped boxes, each summed over the splits. No image id, file name or
seq id in either. Equality and hashing are unchanged (still every field, `eq` was not touched).
New test `test_reprs_show_no_image_id_seq_id_or_file_name` in `tests/test_metadata.py`: no
synthetic record's repr holds its image id, seq id or file name (nor `syn-` or `.jpg`), the exact
format of one record and of the metadata, two records differing only in ids print the same but
compare unequal, and a copy compares and hashes equal. R03's test cell in
`docs/review/invariants.md` names the test. `uv run pytest tests/test_metadata.py` 32 passed
(with `TRAILCAM_DATA_ROOT` exported); `check.sh` green (212 passed), so `passes` is flipped.

Not covered at first, noted: pytest's own explanation of a failing `assert record == other`
drills into the differing attributes and prints their values (checked on a scratch dataclass at
default verbosity), so an id field that differs is shown; this goes through pytest's dataclass
comparison, not the repr. Fixed after the review, below.

Review, started by the user: `/review-session 3.05`, tier 2, range 56bf5b0..6860117, risk files
`tests/test_metadata.py`, `trailcam/data/metadata.py`; code-reviewer on Opus at xhigh,
risk-reviewer on Fable at high. Nothing above low; every scenario held (R01, R03, R05, R10, R15,
R42 to R45). Spec changed: none. Known gaps: none. One finding, verified and fixed in a99065c:
- both reviewers, low: a failed `==` between two `Record`s still printed the differing image id,
  file name and seq id, because pytest explains a dataclass with a generated `__eq__` field by
  field without calling the repr. `Record` now writes its `__eq__` out (every field in order, as
  before; the dataclass still adds the frozen `__hash__`), so pytest shows the id-free repr.
  Took the code-reviewer's fix over the risk-reviewer's docstring note, which would have left
  the gap open; a `conftest.py` hook was ruled out by R42 (both). New subprocess test
  `test_failed_record_equality_shows_no_id` runs a failing comparison at default and `-vv`
  verbosity; on the old code the ids appeared 2 and 24 times. R03's test cell names it.
Notes, not fixed: displays that go around the repr (`vars(r)`, `dataclasses.asdict(r)`,
`pd.DataFrame(records)`) still show ids, outside a repr change; how IPython and Rich show a
`Record` on Kaggle was reasoned, not run (code-reviewer). `check.sh` green (213 passed). The fix
commit touches `trailcam/data/metadata.py`, so one fix round on it is due.

Fix round, started by the user: `/review-session 6860117..HEAD 3.05`, tier 2, risk files
`tests/test_metadata.py`, `trailcam/data/metadata.py`; code-reviewer on Opus at xhigh,
risk-reviewer on Fable at high. Nothing found by either; every scenario held (R01, R03, R05, R10,
R15, R42 to R45). Spec changed: none. Both traced pytest 9.1.1 and confirmed that the written-out
`__eq__` turns off its field-by-field listing; neither ran the new test against the old code
(read-only), which this session did before the fix (ids shown 2 and 24 times). No third round.

**Next.** Session 4, item 4.01, the Dataset and transforms.

Review: tier 2, ran `/review-session 3.05` (56bf5b0..6860117) and the fix round
`/review-session 6860117..HEAD 3.05` (code-reviewer Opus xhigh, risk-reviewer Fable high)

## 2026-10-12: session 4, item 4.01, the Dataset and transforms (Opus 5.5)

**Done.** Dependencies first, brought forward from 5.01 by the user's decision at session start
(the Dataset needs them; asked rather than widened): `uv add torch torchvision pillow` (torch
2.14.1, torchvision 0.29.1, Pillow 12.3.0), with a CPU-only PyTorch index for every platform but
macOS in `[tool.uv.sources]`, so the lock gives Linux `torch 2.14.1+cpu` and holds no nvidia, cuda
or triton package; a mypy override (`ignore_missing_imports`) for torchvision, which ships no
`py.typed` (torch and Pillow ship types). timm and its override stay in 5.01. Plan section 10,
"Setup maintenance", records the move (spec changed).

`trailcam/data/dataset.py`: `CCT20Dataset`, the chapter 04 pattern (`classes`, `class_to_idx`,
`load_image`, `__len__`, `__getitem__`) over one split's records and an image folder, returning
`(float32 tensor (3, 224, 224), class index)`; a grayscale JPEG loads as three equal channels.
The training transform is called with the image and the record's boxes and hands the boxes to
its crop sampler; the eval transform is called with the image alone, and its `__call__` takes
nothing else. `TrainTransform` (arm A): the sampler's crop resized to 224, horizontal flip
p=0.5, colour jitter, random grayscale p=0.1, ImageNet normalisation; `ControlCropSampler`
draws torchvision's `RandomResizedCrop.get_params` at scale 0.65 to 1.0, aspect 3/4 to 4/3, and
ignores the boxes. `EvalTransform`: shorter side to 256, centre crop 224, normalise. The dataset
refuses the training transform for records outside train and any transform of another type
(R20). An unreadable image raises `ImageReadError` with the item's position, class and split,
never the file name (the image id), and the original exception suppressed (R03). All randomness
comes from torch's generator.

Chosen here, since the plan does not pin it: "mild" colour jitter is brightness, contrast and
saturation up to 20%, no hue. Left to their items: the box sampler and the `arm` switch (4.04,
which plugs in through `TrainTransform(sampler=...)`) and the fixture JPEG writer with drawn
boxes (4.03); the tests write their own noise JPEGs, grayscale for night records, for the
synthetic metadata's seed 42 split.

`tests/test_dataset.py`, 17 tests: shape, dtype and label (0 to 11, all 12 present, in record
order) for every train and val item; the eval transform equal over two calls under different
torch and Python seeds; six train outputs pairwise different; the train transform reproducible
under one torch seed whatever Python's `random` state; the eval transform's signature and a spy
in the dataset show it never receives a box; a spy sampler receives every record's boxes in
order; over 2,000 draws the control sampler's area fraction stays within 0.65 to 1.0 and spans
it; a black-third image shows the shorter-side resize and centre crop (a direct 224 resize
fails it); a grey image normalises to the ImageNet constants; the transform refusals; a missing
and a corrupt file raise without the file name; the dataset pickles. The pickling test found a
bug, fixed before the commit: `class_to_idx` held the class map's `mappingproxy`, which cannot
be pickled, so DataLoader workers started with spawn (macOS, the dry run) would have failed; it
is now a dict copy. Eight mutations of `dataset.py` (a direct 224 resize, a flip in eval,
torchvision's default scale from 0.08, boxes passed to eval, boxes dropped before the sampler,
no normalisation, Python's `random` in the train path, the path in the error) each fail their
test. The R20, R21 and R23 test cells name the file.

`uv run pytest tests/test_dataset.py` 17 passed; `check.sh` green (230 passed), so `passes` is
flipped.

Notes for later: the first CI run with torch is the user's push of this commit (the CPU wheels
download once, then `setup-uv` caches them). For 5.04, `uv add` wrote the lower bounds
`torch>=2.14.1` and `torchvision>=0.29.1`; if Kaggle's preinstalled torch is older, `pip install`
of the package would replace it with a PyPI CUDA build, so check the bounds against Kaggle's
versions there.

Review, started by the user: `/review-session 4.01`, tier 2, range c5b2a74..8ef7d6f, risk files
`pyproject.toml`, `tests/test_dataset.py`, `trailcam/data/dataset.py`, `uv.lock`; code-reviewer
on Opus at xhigh, risk-reviewer on Fable at high. Nothing above low; every scenario held (R01,
R03, R20, R21, R23, R24 hook only, R38, R42 to R45). Known gaps passed: the R24 sampler (4.04),
the R38 build-system table (5.04). Spec changed: plan section 10 only, cited by no row. One
finding, verified and fixed in 4bea66b:
- code-reviewer, low: the runtime minimums equalled the versions locked that day
  (`torch>=2.14.1`, `torchvision>=0.29.1`, `pillow>=12.3.0`), so pip on Kaggle, which ignores
  `[tool.uv.sources]`, would replace an older preinstalled CUDA torch with PyPI's build. Before
  lowering them, `tests/test_dataset.py` was run in a scratch venv outside the repo on torch
  2.3.0, torchvision 0.18.0 and Pillow 10.1.0 (the first torch built for NumPy 2, the first
  Pillow with Python 3.12 wheels; NumPy 2.5.4, deprecation warnings as errors): 17 passed. The
  minimums are now `torch>=2.3`, `torchvision>=0.18`, `pillow>=10.1`; the lock resolves the same
  versions. New `tests/test_dependencies.py` pins them and checks they sit below the locked
  versions; both tests fail on the old `pyproject.toml`. R38's test cell names it.
Noted, not a finding (code-reviewer): the dataset's refusal of the training transform checks
`split != "train"`, which cannot tell the re-split train from the official train file; 4.02 must
test that the train loader is built from `Split.records["train"]` (R12). `check.sh` green (232
passed). The fix commit touches `pyproject.toml`, so one fix round on it is due.

Fix round, started by the user: `/review-session 8ef7d6f..HEAD 4.01`, tier 2, risk files
`pyproject.toml`, `tests/test_dependencies.py`, `uv.lock`; code-reviewer on Opus at xhigh,
risk-reviewer on Fable at high. Nothing found by either; every scenario held (R01, R03, R20,
R21, R23, R38, R42 to R45). Spec changed: none. The code-reviewer re-ran `tests/test_dataset.py`
and `tests/test_dependencies.py` on torch 2.3.0, torchvision 0.18.0 and Pillow 10.1.0 from the
uv cache (19 passed, no warnings), confirming the scratch-venv run above; the minimums were
checked on macOS arm64 only. Noted for 5.04 and 6.02 (code-reviewer): `requires-python =
">=3.12"` would block the pip install outright if Kaggle's image still runs Python 3.11, so check
Kaggle's Python as well as its torch when the wheel install is tested. No third round.

**Next.** Item 4.02, the DataLoaders (tier 2); its tests show the train loader is built from
`Split.records["train"]` (R12).

Review: tier 2, ran `/review-session 4.01` (c5b2a74..8ef7d6f) and the fix round
`/review-session 8ef7d6f..HEAD 4.01` (code-reviewer Opus xhigh, risk-reviewer Fable high)

## 2026-10-13: session 4, item 4.02, the DataLoaders (Opus 5.5)

**Done.** `trailcam/data/loaders.py`: `make_loaders(split, image_root, batch_size, *,
num_workers, pin_memory, train_transform=None, generator=None)` returns a read-only mapping with
one loader per split in `SPLIT_NAMES` order. It takes the `Split` from `make_split` (a mapping of
records is refused with `TypeError`) and builds the train loader from `split.records["train"]`
itself, with the training transform (control arm by default; 4.04 passes the arm B transform), a
`RandomSampler` and `drop_last=True`, so batch norm never gets a batch of one; the reshuffle each
epoch still draws every image. val, trans_val, cis_test and trans_test get the eval transform, a
`SequentialSampler` and every image, the last short batch included. Before building anything it
refuses a train image id that also sits in another split, with split names and counts only (R12,
R16). `num_workers` and `pin_memory` are arguments, as the config file arrives in 5.02, which
wires them; `persistent_workers` is on when there are workers. `generator` seeds the shuffle and
the workers' base seed.

`tests/test_loaders.py`, 28 tests on the synthetic metadata's split for seed 42 with noise JPEGs
in a temporary directory: a batch is `(16, 3, 224, 224)` float32 with int64 labels for every
loader; the samplers (`RandomSampler` for train, `SequentialSampler` for the rest); R12, the train
loader holds `Split.records["train"]` record for record, equals the pool minus val, differs from
the official train file and shares no id with another split (a mutation that builds it from the
pool with the overlap check off fails the test); the eval loaders give labels in record order and
the same tensors on two passes; a train epoch is a permutation; a seeded generator reproduces the
first train batch; num_workers and pin_memory pass through; with two spawned workers (forced, so
Linux CI runs spawn too) the val batches equal the single-process ones, and the train dataset
survives a pickle round trip with the same augmentation for a seed; bad arguments and a tampered
split are refused. The spawn test takes about 15 s (torch imported in each worker), so the train
loader is not iterated with workers. `docs/review/invariants.md`: test cells filled for R12, R22
and, for the loader check, R16. `check.sh` green (260 passed).

Decision for the user to confirm: `drop_last=True` on the train loader is not in plan section 5.
With batch 64 it drops at most 63 images per epoch, a different set each epoch.

Review, started by the user: `/review-session 4.02`, tier 2, range c6c65f5..b11eff2, risk files
`tests/test_loaders.py`, `trailcam/data/loaders.py`; code-reviewer on Opus at xhigh,
risk-reviewer on Fable at high. Nothing above low; every scenario held (R01, R03, R12, R16 loader
part, R22, R42 to R45). Known gap passed: the R16 run alarms (5.03). Spec changed: none.
- code-reviewer, low, fixed in a76cb74: the overlap check caught a train id still present in
  another split, but not an image moved out of a test split into train and relabelled
  `split="train"`; the dataset check passes such a record and `split.hash` went stale unchecked.
  `make_loaders` now refuses a split whose recomputed `split_hash(split.records)` differs from
  `split.hash`, with the split sizes only. New
  `test_a_split_that_no_longer_matches_its_hash_is_refused` moves a cis_test image into train; it
  failed on the old code (DID NOT RAISE). The overlap test now recomputes the hash so it still
  reaches the overlap check. R12 and R16 test cells name the hash check.
- risk-reviewer, note folded into the same fix: the docstring said `generator` makes the
  augmentation reproducible; with `num_workers=0` the transform draws from torch's global
  generator, so `torch.manual_seed` is needed too. Docstring corrected.
- code-reviewer, low, resolved by the user's decision below: "num_workers and pin_memory from
  config" was handed to 5.02, but no item description named them.

Decisions, relayed from the coordinator session and confirmed by the user in this session
(c545b61):
- Item 5.02's scope now includes `num_workers` and `pin_memory` in every config (4 and true on
  Kaggle, 0 and false for the dry-run), with a step that checks them.
- `drop_last=True` on the train loader is the user's decision: a batch of one would fail batch
  norm in training mode, at most 63 images per epoch are skipped and a different set each epoch,
  and every step has the same batch size. Val and the test loaders keep every image.
`check.sh` green (261 passed). The fix touches `trailcam/data/loaders.py` and modifies a test
line, so one fix round on it was due.

Fix round, started by the user: `/review-session b11eff2..HEAD 4.02`, tier 2, risk files
`tests/test_loaders.py`, `trailcam/data/loaders.py`; code-reviewer on Opus at xhigh,
risk-reviewer on Fable at high. Nothing found by either; every scenario held (R01, R03, R12, R16
loader part, R22, R42 to R45). Spec changed: none. Noted, not findings (both reviewers): a
`Split` rebuilt with a recomputed hash that moves a test image into train passes both checks;
only `make_split` builds a `Split` today, and the pinned hash per seed (`tests/test_split.py`)
and the ledger's split hash (5.03) cover it. If a later item adds a helper that subsets a split
and recomputes the hash, `make_loaders` should also call `check_split` (code-reviewer). The
code-reviewer asked whether the 5.02 edit was approved; the user approved it in this session
before c545b61. No third round.

**Next.** Item 4.03, the synthetic fixture images (tier 2).

Review: tier 2, ran `/review-session 4.02` (c6c65f5..b11eff2) and the fix round
`/review-session b11eff2..HEAD 4.02` (code-reviewer Opus xhigh, risk-reviewer Fable high)

## 2026-10-13: session 4, item 4.03, the synthetic fixture images (Opus 5.5)

**Done.** `trailcam/data/synthetic.py`, the images part: `write_synthetic_fixture(out_dir, seed)`
writes the 3.02 metadata (five annotation files and the sidecar) into `out_dir` and one JPEG per
image of those five files into `out_dir/images/`, under the image's `file_name` at its `width` x
`height` (96 x 72), read back from the written files and the sidecar, not from the writer's state.
A JPEG is a dark per-location background (each camera sees one scene) blended with 25% noise,
and a bright rectangle in the annotation's class colour, outlined in white, at each box; a night
record (the sidecar's flag) is saved as a grayscale mode L JPEG, as 4.01's own writer did. Every
image of the files gets one, filtered ones included (684 JPEGs, 312 grayscale, 606 with a box,
610 rectangles, about 2.4 MB, under half a second). An image depends only on the seed and its
id; the same seed gives the same bytes. `out_dir` must be new or empty, so a mistyped path cannot
mix the fixture into real data. It returns a `Fixture` (the paths, the image root, the counts) for
5.03's dry run. `python -m trailcam.data.synthetic <dir>` or `--root <dir>` (both, since the
item names `--root` and its step a bare directory; giving neither or both is a usage error, as is
a non-empty directory) prints counts only: no ids, file names or paths.

New `tests/test_synthetic.py`, 13 tests: the fixture writes only into its directory (five files,
the sidecar, `images/` holding exactly the files' `file_name`s), refuses a non-empty directory and
accepts an empty one; every JPEG is a JPEG at its image's size, mode L exactly for night images,
and the counts match; every box's one-pixel edge averages at least 220 in luminance (measured
239 or more) and the ring just outside it at most 140 (measured 114 or less), an unboxed image
has no pixel at 160 or above; a day box's centre is its class colour within 12; same seed same
bytes, a different seed different images; the fixture loads through 3.02 (each record's night
flag matches its JPEG's mode), splits through 3.03 to the hash pinned in `tests/test_split.py`
(images leave the metadata unchanged) and every record of the five splits reads through 4.01's
`CCT20Dataset` as a (3, 224, 224) float32 tensor with its label; the CLI in both forms, its two
usage errors and `python -m` print counts only. Eleven mutations of the writer each fail a test:
exclusive corners (one pixel off), width and height taken as corners, x and y swapped, no
grayscale, grayscale saved as RGB, width and height swapped, the JPEGs flat in `out_dir`, no
non-empty refusal, colour by location, the path printed, only the first box of an image drawn.

`tests/test_dataset.py` now writes its images with `write_synthetic_fixture` instead of its own
noise writer; no assertion changed, its 17 tests pass, and the 4.01 mutations whose detection
depends on the images (a flip in eval, Python's `random` in the train path, a deterministic
training transform) still fail it. R03's and R37's test cells name `tests/test_synthetic.py`.

Steps: `uv run python -m trailcam.data.synthetic <scratch dir>` wrote the fixture and printed
counts only; `uv run python scripts/guard_data.py --all` passed; `python -m trailcam.data.metadata`
with `--root` and `--sidecar` on it loaded 661 kept images (302 night) and `python -m
trailcam.data.split --root <dir> --seed 42` split it (hash `7a13478e…`, the pinned synthetic
hash), both exit 0. `check.sh` green (245 passed), so `passes` is flipped.

Review, started by the user: `/review-session 4.03`, tier 2, range c6c65f5..0a46547, risk files
`tests/test_dataset.py`, `tests/test_synthetic.py`, `trailcam/data/synthetic.py`; code-reviewer
on Opus at xhigh, risk-reviewer on Fable at high. Nothing found by either; every scenario held
(R01, R03, R37 for the fixture, R42 to R45). Known gap passed: R37's dry run (5.03). Spec changed:
none. The code-reviewer's `check.sh` run was green (245 passed, none skipped).

Could not verify (code-reviewer): the new image code on the declared minimum Pillow 10.1. Checked
afterwards at the user's request, as in the 4.01 fix round: a scratch venv outside the repo,
offline from the uv cache, torch 2.3.0, torchvision 0.18.0, Pillow 10.1.0, NumPy 2.5.4,
deprecation warnings as errors. `tests/test_dataset.py` passes (17) and so do 12 of the 13 tests
in `tests/test_synthetic.py`, but one fails in every run (ten runs):
`test_a_rectangle_is_drawn_exactly_where_each_box_is` (an unboxed night image with a pixel at
166, above the 160 the test allows) or `test_same_seed_same_bytes` (two writes in one process
differ). The cause, isolated in scratch scripts, is Pillow 10.1's JPEG encoder on this Mac (macOS
arm64): over 300 drawn images the pixels before saving hash the same in every process, but the
saved mode L JPEGs differ between processes (2 of 6 runs), always in the last 8 x 8 block, by up
to 95 grey levels; the same grey images saved as three-channel JPEGs differ too (2 of 6 runs),
and 8 of 300 decode with unequal channels. On Pillow 12.3, the locked version, every hash is the
same in every run. So the writer is correct and Pillow 10.1 cannot write it reproducibly; writing
night frames as three-channel JPEGs would not avoid it. Only 10.1.0 and 12.3.0 are in the uv
cache, so the first fixed version is not known offline. `check.sh` and CI use the lock and are
unaffected; Kaggle never writes the fixture (the dry run is Mac only), and the Dataset there only
decodes, which passes on 10.1.

Kept by the user's decision (2026-10-13): the minimum `pillow>=10.1`, with no change to
`pyproject.toml`, `uv.lock` or `tests/test_dependencies.py`. The floor covers what trailcam does
at runtime (decode, which passes on 10.1); the fixture writes only on the Mac, with the locked
12.3. The Pillow or libjpeg-turbo version that fixed the encoder is not known (not found in the
release notes), and the nondeterminism was seen on macOS arm64 only, not shown on Kaggle's
x86_64.

Handoff to 6.01 (the prepare script is the only code that writes JPEGs on Kaggle): print Kaggle's
Pillow version; before writing the cache, encode the same image in two separate processes and
require identical bytes, refusing to continue otherwise; if it fails, upgrade Pillow in the
notebook, and raise the `pyproject.toml` minimum only to a version that passes the same check.

Noted, not fixed (risk-reviewer), by the user's decision: `mypy tests/test_synthetic.py` reports 7
errors (Pillow's union return types compared as ints at lines 135, 138 and 146; the loop variable
`split` reused for a `Split` at lines 196 to 207). Tests are outside `check.sh`'s mypy scope and
the code is correct at runtime; fixing them would touch a reviewed high-stakes file and need
another round. No fix commit, so no fix round.

**Next.** Item 4.04, the arm B box-crop sampler (tier 2); its tests can draw on the fixture's
rectangles to check a crop contains its box. Item 4.02 runs in a parallel session.

Review: tier 2, ran `/review-session 4.03` (c6c65f5..0a46547; code-reviewer Opus xhigh,
risk-reviewer Fable high)

## 2026-10-13: session 4, item 4.04, the arm B crop sampler (Opus 5.5)

**Done.** `trailcam/data/boxcrop.py`: `BoxCropSampler(p=0.5)`, the arm B crop sampler, a
`CropSampler` like 4.01's control sampler. For an image with at least one box, a coin with
probability `p` sends it to the box path (`box_crop`), else to arm A's `ControlCropSampler`
(`self.control`); an image without boxes never reaches the box path. The box path converts each
box from fractions of the image to pixels once (`to_pixels`: x and width times the width, y and
height times the height; a box outside 0 to 1 is refused, so a box in pixels cannot pass), draws
one usable box uniformly, keeps the crop in arm A's aspect-ratio range (3/4 to 4/3) whenever a
crop in it can hold the box, draws the fill (the box's area over the crop's) uniformly over what
that allows within 0.2 to 1.0, the aspect ratio log-uniform among the shapes that hold the box and
fit the image, and the position uniformly among the placements that keep the whole box in the crop
and the crop in the image. In whole pixels the crop runs from the box's first pixel or
before to its last or after; the drawn shape is rounded down, never the box. Every draw comes
from torch's generator; at `p=0` no coin is drawn.

`trailcam/data/dataset.py`: `TrainTransform(arm=None, *, sampler=None)`. `arm`, "A" by default
or "B", picks the sampler through `make_sampler` and nothing else; `sampler` replaces it for the
tests' spies. Both at once, or an unknown arm, are refused. `make_sampler` imports `boxcrop` on
use, so loading `dataset.py`, and with it the eval transform the demo and the Space use, never
loads the box sampler, and the two modules do not import each other at load time. The docstrings
of `dataset.py` and `loaders.py` say so (the training script passes `TrainTransform(arm)`).

Decisions, written in the module docstring. I recommended keeping the first, second, third and
fifth and changing the fourth; the user agreed in this session, and the fourth was changed before
this commit:
- Several boxes: one is drawn uniformly per crop, not their union. They are animals of one
  species (two-species images are dropped), so the fill stays about one animal; the union of two
  animals at opposite edges would be most of the frame. 425 of the 11,633 boxed train images
  (seed 42) have more than one box.
- The fill is uniform in the fill itself, the plan's "20% to 100%", not in the crop area, which
  would put three quarters of the box-guided crops below a fill of 50%.
- A box covering most of the image (810 train boxes over 20% of it, 615 over half) cannot fill
  less than its share of the largest crop allowed, so its range starts there; its loosest crop is
  the largest crop in arm A's range (497 x 373 of a 512 x 373 frame, which arm A never exceeds
  either). A box covering the whole image gets the whole image.
- Aspect ratio, changed from the first draft: crops stay in arm A's 3/4 to 4/3, so the resize to
  224 x 224 never stretches the animal more than arm A does, and the fill's upper end drops for an
  elongated box (a 2:1 box tops out at 67%). 6,676 of the 12,177 train boxes lie outside that
  range, most wider than 4:3. The first draft let the crop follow the box's shape at high fills:
  on the real train boxes 6,895 of 46,532 sampled box-path crops (15%) were stretched beyond arm
  A's range and 1,118 beyond 2:1, a second difference between the arms. Now 25 are outside it
  and 8 beyond 2:1: the range is left only when no crop in it holds the box at a fill of 0.2
  inside the image (a box spanning nearly the whole width, or more elongated than about 6.7:1),
  for the closest shape.
- A box too small for any whole-pixel crop to reach a fill of 0.2 (under about two pixels a side)
  is skipped; an image whose boxes are all skipped gets the control sampler. No train box is
  under 8 px a side at 512 px, so this is a guard only.
The real-data counts come from a scratch script outside the repo that printed counts only.

`tests/test_boxcrop.py`, 31 tests. Over 1,000 samples for each of 13 cases (centre; top-left and
bottom-right corners; a 2:1 box touching the right edge; the bottom edge; the whole image; 81% of
it; a wide and a tall strip beyond the aspect range; a small box off the pixel grid; 2048 x 1494,
512 x 373, 96 x 72 and a portrait size) every box-guided crop lies in the image, holds the whole
box, and the box's area is 20% to 100% of the crop's (to 1e-12 relative, float rounding); where a
crop in arm A's aspect-ratio range can hold the box, found by trying every whole-pixel crop size,
every crop is within a pixel of that range. Over 300 random boxes of 1% to 100% of each side, 20
crops each, the same holds, and a 2:1 box tops out at a fill of 2/3. The fill reaches 0.2 on the
area, tops out above 0.95 and averages about 0.6 for a near-square box; the position is jittered;
the aspect ratio spans 3/4 to 4/3 for a near-square box; the 81% box starts at a fill of 0.83 and
its loosest crop is about 497 x 373. `to_pixels` is checked on a non-square image and at
four times the size, and a box in pixels is refused. Two boxes at opposite corners: each crop
holds exactly one, each about half the time. A sub-pixel box is never drawn next to a usable one
and, alone, gives the control crop seed for seed. Half of 1,000 boxed calls take the box path; an
image without boxes never calls it, at `p=1` too, and gets the control crop; through the dataset
only the boxed train records reach it, with their boxes. With the box path off (`p=0`) arm B
gives arm A's tensor for every train record and seed; arm B's own transform changes some boxed
tensors and no unboxed one. `TrainTransform("A")` and `("B")` differ only in the sampler. The eval
transform takes the image alone, names no box or sampler in its code, and its pipeline is exactly
resize, centre crop, tensor and normalise; in a fresh interpreter, loading the dataset and loader
modules and an eval pass leave `boxcrop` unloaded, and `TrainTransform("B")` loads it. Arm B draws
from torch only and pickles; bad `p`, arms and boxes, and arm B outside train, are refused.
Seventeen mutations each fail a test: in `boxcrop.py` the fill on the side, no pixel conversion,
boxes scaled twice, width and height swapped, jitter ignoring the box, jitter not clamped to the
image, the crop rounded up, the box path for unboxed images, a coin drawn at `p=0` and `p=1`,
always the first box, the fill uniform in area, the aspect ratio ignored, arm A's shapes never
kept, the largest crop taken as the whole image, the smallest as the box itself, the box rounded
instead of floored; in `dataset.py` a top-level `boxcrop` import. The R24 test cell and the eval part of
R23's name the file.

`uv run pytest tests/test_boxcrop.py` 31 passed; `check.sh` green (292 passed), so `passes` is
flipped. After merging main with 4.03 into this branch, `check.sh` is green with 305 passed, and
`mypy --strict tests/test_boxcrop.py` is clean (tests are outside `check.sh`'s mypy scope; one
line was tightened for it: the eval pipeline is asserted to be a `Compose` before its steps are
read).

Note for 5.02 and 5.03: `TrainTransform(config.arm)` is the whole switch. The transform half of
R25 is tested here (`test_the_arms_differ_only_in_the_crop_sampler`); the config half is 5.02's.
Note for 8.03: arm B crops around one animal at a time, so attention-in-box should count the
Grad-CAM mass inside any of the image's boxes. 4.03's handoff suggested checking containment on the
fixture's drawn rectangles; the tests check it geometrically on every crop instead, which is
exact, where the drawn pixels after JPEG and the resize would need a tolerance.

Review, started by the user: `/review-session 4.04`, tier 2, range 3feec64..d257ca2 (the item
commit 1722369 and the merge of main with 4.03; the diff against main holds only the 4.04 files),
risk files `tests/test_boxcrop.py`, `trailcam/data/boxcrop.py`, `trailcam/data/dataset.py`,
`trailcam/data/loaders.py`; code-reviewer on Opus at xhigh, risk-reviewer on Fable at high.
Nothing found by either; every scenario held (R01, R03, R12, R16 loader part, R20, R21 and R23
data side, R22, R24, R42 to R45). Known gaps passed: R16's run alarms (5.03); the model, demo,
Space and attention parts of R20, R21 and R23 (5.01, 8.03, 10.01, 11.01). Spec changed: none.
The code-reviewer's `check.sh` run was green (305 passed). The risk-reviewer probed 400 random
image sizes from 8 to 2048 px with one to three boxes in a scratch script: 3,990 of 4,000
box-path crops held a box at a fill of 0.2003 to 0.9925, and the other 10 were sub-pixel boxes
handed to the control sampler. Noted, not findings: `make_loaders` falls back to arm A when no
training transform is passed, so the 5.03 review should check that the training script passes
`TrainTransform(arm)` (risk-reviewer); arm B's sampler is pickled in-process only, not through
spawned workers, where 4.02's spawn test covers arm A's dataset (code-reviewer). No fix commit,
so no fix round.

**Next.** Session 5, item 5.01.

Review: tier 2, ran `/review-session 4.04` (3feec64..d257ca2; code-reviewer Opus xhigh,
risk-reviewer Fable high)

## 2026-10-13: session 6, item 6.01, the Kaggle prepare script (Opus 5.5)

Taken ahead of session 5 by the user's decision in the coordinator session (2026-10-13), so the
cache (6.03) can be built while session 5 is written.

**Done.** `trailcam/data/daynight.py`: `channel_spread(image)`, the mean over pixels of the
largest channel minus the smallest (0 to 255; 0 for a grey-mode image such as L), and
`is_night(image)`, a spread of at most `NIGHT_MAX_SPREAD = 2.0`. Measured in a scratch script: a
grey image saved as an RGB JPEG (quality 75, 90 or 95, before and after the 512 px resize)
decodes with a spread of exactly 0, since its chroma planes are flat; a washed-out colour scene
(HSV saturation 0.03, blended half with grey) scores 3.0. The threshold sits between the two and
is not yet checked on real images: see the note for 6.03.

`scripts/kaggle_prepare_cct20.py`, written fresh. It downloads the two archives from LILA's GCP
mirror (`.../public-datasets-lila/caltechcameratraps/eccv_18_annotations.tar.gz` and
`eccv_18_all_images_sm.tar.gz`) into `/tmp/cct20`. A complete file is reused and a partial one is
resumed with a range request (restarted if the server ignores the range); free space is checked
first. It prints the Python, platform, Pillow and libjpeg-turbo versions, then runs the encoder
check from the 4.03 handoff: 15 synthetic JPEGs (five camera-trap sizes, two not multiples of 8,
in RGB, L and grey-as-RGB) plus the first 16 annotated JPEGs of the tarball. These are resized and
encoded twice in each of two fresh spawned processes. All four SHA-256 hashes per image must
match and the two processes must be distinct, or it stops before writing anything and says to
upgrade Pillow. It then streams the tarball once (`r|gz`, never extracted to disk). Each JPEG an
annotation names (matched on the base name, which must equal the annotation's `file_name`, flat
and `.jpg`) is decoded and resized so the longer side is 512 px (Lanczos; never enlarged; L stays
L, other modes become RGB). It is saved as JPEG quality 90 under `--out/images/<file_name>` and
flagged from its resized pixels. The five annotation files are copied to `--out/annotations/`.
Only when every count checks is `--out/day_night.json` written, last, via a temporary name, then
read back: all 57,864 image ids (dropped classes and empties too) to `{"night", "hour",
"spread"}`, the hour from `date_captured` through `metadata.capture_hour`. It exits 1 when the
annotations or the tarball do not hold exactly 57,864 images, or an annotated image is missing,
repeated or unreadable, or a JPEG in the tarball is not annotated. It refuses to start without
`/kaggle` unless `--off-kaggle` is passed. Default output is `/kaggle/working/cct20_512`, which
must be new or empty. Printed, counts only (R03): the tarball layout (member types, file
extensions, depth histogram, the number of top-level entries and of folders holding JPEGs; no
names), images annotated and written per split, the night share per split and per capture hour,
the spread histogram (bins at 0.5, 1, 2, 4, 8, 16, 32) with the count within a factor of two of
the threshold, source modes, longer-side ranges and cache bytes. trailcam imports:
`daynight` and `metadata` (which imports `classes`).

Tests: `tests/test_daynight.py` (15): grayscale JPEGs, mode L and grey saved as RGB at three
qualities, are night; colour ones, including the washed-out scene, are day; the spread formula;
grey modes score 0; RGBA and P are measured in colour; every JPEG of the 4.03 fixture is flagged
exactly as its sidecar says. `tests/test_kaggle_prepare.py` (15): the fixture is packed as two
LILA-shaped tar.gz files and run through `prepare`. The sidecar's night flag and hour equal the
fixture's for every image, the copied annotations load with the sidecar through `load_metadata`,
images keep 96 x 72, and the output shows depth 1, one JPEG folder, a 0-differ encoder check
and no id, file name or folder name. A missing, repeated, extra or corrupt JPEG fails with its
count and leaves no sidecar; the 57,864 default is enforced before any write. Also covered: the
512 px resize, the mismatch counter, the encoder check on the locked Pillow and its refusal, the
off-Kaggle guard, the URLs, and `download` against a local HTTP server (fetch, reuse, a
restarted partial file, a short file refused). Seven mutations each fail a test: the night flag
inverted, the repeat check removed, no resize, the tarball count check removed, the encoder
refusal removed, the image write removed, the hour replaced by 0. A run on the fixture (scratch script) printed 684
images, 312 night, 0 encoder mismatches.

Setup maintenance: the script is in `[tool.mypy] files` and in `check.sh`'s mypy call; the
`check.sh` line in `tests/test_control_plane.py` is updated, and a new test there requires
`check.sh`'s mypy paths to equal the pyproject list. `scripts/risk_files.sh` already covers
`^scripts/` and `^trailcam/`; `trailcam/data/daynight.py` is added to the risky list in
`tests/test_risk_files.py`, which already listed the script. R03's test cell names
`tests/test_kaggle_prepare.py`.

Steps: `uv run pytest tests/test_daynight.py` 15 passed; `check.sh` green (336 passed, after
fast-forwarding to main with 4.04); `curl -sI` on both URLs answers `HTTP/2 200` (annotations
2,997,071 bytes, images 6,492,615,601). `passes` is flipped.

Kaggle's Python: the Kaggle/docker-python `Dockerfile.tmpl` on main (last changed 2026-09-28,
Colab base image `20260917`) installs into `/usr/local/lib/python3.13`, so `requires-python
>=3.12` does not block the install. This is read from the repository, not from a running
notebook. The script prints the version anyway.

Notes for 6.03 (user):
- Needs 5.04 merged first: the script imports `trailcam`, installed by pip from the public repo.
  Run it as its own process, `!python kaggle_prepare_cct20.py` (from a clone, or the file fetched
  from the repo), not `%run`: the encoder check's spawned processes re-import the file. Internet
  on; about 6.5 GB in `/tmp` and 2 to 3 GB in `/kaggle/working`.
- If the encoder check reports a difference, `pip install -U pillow` in the notebook and rerun.
  Raise the pyproject minimum only to a version that passes the same check (4.03 decision).
- Check the spread histogram: night frames should sit in the lowest bins, day frames from about
  8 up, with few within [1, 4]. If many are near the threshold, or the night share per hour is
  not near 1 at night and near 0 at midday, the threshold needs revisiting before the flag is
  used. Each image's spread is in the sidecar, so a new threshold needs no re-run.
- Record the layout lines in plan.md open question 3 and the day/night share in the data card.
  The cache layout for 6.02: images under `images/`, annotations under `annotations/`, the
  sidecar beside them (`--sidecar <dataset>/day_night.json`).

Review, started by the user: `/review-session 6.01`, tier 2, range 4f90760..4055abe (the item
commit only), risk files `pyproject.toml`, `scripts/check.sh`, `scripts/kaggle_prepare_cct20.py`,
`tests/test_control_plane.py`, `tests/test_daynight.py`, `tests/test_kaggle_prepare.py`,
`tests/test_risk_files.py`, `trailcam/data/daynight.py`; code-reviewer on Opus at xhigh,
risk-reviewer on Fable at high. Nothing found by either; every scenario held (R01, R03, R38,
R42 to R45), and so did the requirements: the sidecar read by `load_night_flags` and
`load_metadata`, the Pillow version printed and the two-process encoder check before the first
write, the script's trailcam imports free of torch, the mypy lists equal. Known gaps passed: the
real tarball and the `NIGHT_MAX_SPREAD` calibration (6.03), the pip install on Kaggle (5.04).
Spec changed: none. The code-reviewer's `check.sh` run was green (336 passed).

Noted, not findings: the parent process that writes the cache is not itself compared with the two
check processes; the 4.03 decision asks for two processes, which is what runs (risk-reviewer). No
test pins that the encoder check comes before the first write in `prepare`; it holds by reading
the code (code-reviewer). A full disk during the cache write would put one image's cache path,
which holds its id, in the traceback: one id, on a crash, in Kaggle output that is never committed
(code-reviewer). An unreadable first copy of an image followed by a readable repeat gives a
confusing message, but the run still fails (code-reviewer).

Could not verify (code-reviewer): the encoder check started through the real
`python scripts/kaggle_prepare_cct20.py` entry point (spawned processes re-import the script as
`__mp_main__`); the tests import the script by module name. By the user's decision, no
entry-point test: the user's 6.03 run on Kaggle covers it, and a failure there stops before any
write, so it is loud and harmless. The LILA URLs were checked with `curl -sI` before the item
commit (both 200); the reviewers make no network calls. No fix commit, so no fix round.

**Next.** Session 5 items in order; 6.03 after 5.04 is merged.

Review: tier 2, ran `/review-session 6.01` (4f90760..4055abe; code-reviewer Opus xhigh,
risk-reviewer Fable high)

## 2026-10-13: session 5, item 5.04, packaging for the Kaggle pip install (Opus 5.5)

Taken ahead of 5.01 to 5.03 by the user's decision in the coordinator session: the Kaggle image
cache (6.03) needs 6.01 and this item merged.

**Done.** `pyproject.toml` has a `[build-system]` table (`hatchling>=1.27`, the first with SPDX
licence expressions) and `license = "MIT"`. The wheel target is `packages = ["trailcam"]`; the
sdist target is `only-include = ["trailcam", "README.md", "LICENSE"]`, so tests, docs and anything
untracked in a checkout (a data symlink, a scratch table) never enter either archive; hatchling
adds the `.gitignore` and `pyproject.toml` to the sdist itself. The runtime dependencies are the
4.01 floors, unchanged. `uv.lock` changes in one line: the project source is now `editable`
instead of `virtual`, so `uv sync` installs `trailcam` into `.venv`.

Kaggle's Python, checked as the 4.01 fix round asked: Kaggle's GPU image (Kaggle/docker-python,
last base bump 2026-09-28) is built on Colab's runtime of 2026-09-17 and installs into
`python3.13`; Colab's published package list from that date (googlecolab/backend-info) has
Python 3.13, torch 2.11.0+cu128, torchvision 0.26.0 and Pillow 11.3.0. So `requires-python =
">=3.12"` does not block the install and every runtime floor is met; recorded in the comment
above the build table.

Steps: `uv build` wrote `dist/trailcam-0.1.0.tar.gz` and `dist/trailcam-0.1.0-py3-none-any.whl`
(ignored; `git status` shows neither). The wheel holds the nine `trailcam/` modules and its
dist-info (METADATA, WHEEL, RECORD, `licenses/LICENSE`) only; METADATA lists `Requires-Python:
>=3.12` and the three floors, with no index URL. In fresh `uv venv`s in the scratchpad (outside
the repo), `uv pip install dist/*.whl` then `python -I -c 'import trailcam.data.classes'` succeeded
on Python 3.12.9 and 3.13.9, importing from site-packages. Beyond the steps, a Kaggle-like check:
a seeded 3.13 venv with torch 2.11.0, torchvision 0.26.0 and Pillow 11.3.0 preinstalled, then
real `pip install <worktree>` (the source build `git+` takes): the import succeeded and pip kept
torch 2.11.0, torchvision 0.26.0 and Pillow 11.3.0.

New tests in `tests/test_dependencies.py` (R38's build half, which had none) run `uv build` into
a temp directory once per module: the wheel holds exactly the `trailcam/**/*.py` files plus its
dist-info; its `Requires-Dist` lines equal `[project].dependencies` and `Requires-Python` equals
`requires-python`, with no `download.pytorch.org` and no direct `@` reference; the sdist holds the
package, README, licence, `pyproject.toml` and `PKG-INFO` only. Mutation: `packages = ["trailcam",
"tests"]` with the sdist `only-include` removed fails the wheel and sdist tests. `mypy --strict
tests/test_dependencies.py` is clean. R38's test cell in `docs/review/invariants.md` names the
build tests. `check.sh` green, so `passes` is flipped.

Notes: CI runs Python 3.12 only; the 3.13 evidence is the import above, not the test suite. The
`git+https` install itself is first possible after the user pushes this commit (6.02's notebook).

Review, started by the user (go relayed by the coordinator session): `/review-session 5.04`,
tier 2, range 4f90760..38742e2 (the item commit), risk files `pyproject.toml`,
`tests/test_dependencies.py`, `uv.lock`; code-reviewer on Opus at xhigh, risk-reviewer on Fable
at high. Nothing found by either; every scenario held (R01, R03, R38, R42 to R45). Known gaps
passed: R38's notebook line (6.02), `space/requirements.txt` (11.01), timm (5.01), the live
`git+https` install (after the push). Spec changed: none. The code-reviewer's `check.sh` run was
green (308 passed); the risk-reviewer ran the dependency, control-plane, risk-files and guard
tests (102 passed). Noted, not a finding (code-reviewer): R38's test cell no longer says the
fresh-venv install has no automated test; the build tests check the wheel's files and
`Requires-Dist`, not an install and import, which this session checked by hand (Python 3.12,
3.13 and the Kaggle-like venv above). The reviewers could not check the Kaggle image versions
(no network) or run the fresh-venv install (their rules). No fix commit, so no fix round.
After the review, main with 6.01 (5bdd41e) was merged into this branch; the log conflict was
resolved with 6.01's entry first.

**Next.** Session 5, item 5.01 (or whichever item the coordinator assigns).

Review: tier 2, ran `/review-session 5.04` (4f90760..38742e2; code-reviewer Opus xhigh,
risk-reviewer Fable high)

## 2026-10-13: session 5, item 5.01, the small CNN, ResNet-18 and the factory (Opus 5.5)

**Done.** `uv add timm`: timm 1.0.30 locked (with huggingface-hub, safetensors, PyYAML), the
same version Kaggle's image has (Colab's GPU pip freeze of 2026-10-02, googlecolab/backend-info,
the base Kaggle builds on; read from the repository, not a running notebook). The lock still holds
no nvidia, cuda or triton package and every non-macOS torch and torchvision entry is `+cpu`; a
new `tests/test_dependencies.py` test pins that, since timm depends on torch too. timm ships
`py.typed`, so it needs no mypy override (only torchvision has one, from 4.01); the pyproject
comment says so. Runtime floor `timm>=1.0.3`, not the locked version (R38): in a scratch venv
outside the repo on Python 3.12 with torch 2.3.0, torchvision 0.18.0, Pillow 10.1.0, timm 1.0.3
(the first 1.0 release), NumPy 2.5.4 and huggingface-hub 2.2.0, `tests/test_models.py` and
`tests/test_dataset.py` passed (38, deprecation warnings as errors). `VERIFIED_MINIMUMS` and the
module docstring record it.

`trailcam/model/small_cnn.py`: `SmallCNN`, the chapter 04 TinyVGG pattern, three blocks of (conv
3x3, ReLU, conv 3x3, ReLU, max-pool 2) at a constant `hidden_units` (32 by default, chosen here:
the plan does not pin it) and a flatten-linear head (`classifier`) sized from the image size;
348,204 parameters. `trailcam/model/backbone.py`: `build_resnet18` (torchvision,
`ResNet18_Weights.IMAGENET1K_V1` unless `pretrained=False`, `fc` replaced by a fresh 512-to-n
linear layer) and the fallback `build_efficientnet_b0` (timm `efficientnet_b0.ra_in1k`,
ImageNet-1k, fresh 1280-to-n classifier; timm is imported inside it only, so the ResNet-18 path
never loads it). `trailcam/model/factory.py`: `build_model(name, *, pretrained=True)` for
`small_cnn`, `resnet18`, `efficientnet_b0`; it reads `classes.NUM_CLASSES` at call time and takes
no class count. `parameter_groups(model)` returns `(body, head)`, the head being the final linear
layer (`classifier`, `fc` or timm's `get_classifier()`), every parameter once in
`model.parameters()` order, for the two AdamW learning rates and the head-only first epoch (5.02,
5.03).

`tests/test_models.py`, 21 tests, no download (all built with `pretrained=False`; an autouse
fixture refuses socket connections): all three names map (2, 3, 224, 224) to (2, 12); each head is
one linear layer with `len(CLASSES)` outputs on the pooled features (25,088, 512, 1,280); with
`NUM_CLASSES` patched to 5 every model gives 5 logits; the groups partition every parameter once
and the ResNet-18 head group is `fc.weight` and `fc.bias`; spies show the default requests the
ImageNet-1k weights of both backbones and `pretrained=False` none; both weight sets expect the
dataset's ImageNet mean and std (R21); the small CNN's exact layer types, no batch norm or
dropout, and its 112/56/28 sides. Six mutations (a literal 12, either backbone without weights,
batch norm in a block, the head also in the body group, the wrong head module) each fail a test.
The R05, R21, R30 and R38 test cells name the files.

`/review-shapes` (shown in the session): the hand-traced tables for all three models, then
checked with forward hooks: shapes and parameter counts match (ResNet-18 11,182,668, body
11,176,512, head 6,156; EfficientNet-B0 4,022,920; small CNN 348,204). No mismatch flagged. The
command now names the factory and its three names. `trailcam/` is already in
`scripts/risk_files.sh` and `trailcam/model/backbone.py` in its test, so the regex is unchanged.

Steps: `uv run pytest tests/test_models.py` 21 passed; `check.sh` green (358 passed); the
`/review-shapes` table flags no mismatch. `passes` is flipped.

Notes for later: 5.03 should call `parameter_groups` for the optimizer and freeze `body` for the
head-only epoch; the Grad-CAM target is `model.layer4` on the unwrapped torchvision model.

Main merge (91e4667): 5.04 reached main after the item commit, so main (4319bd3) was merged in
with `sync_with_base_branch` before the review fix. `pyproject.toml`, `tests/test_dependencies.py`,
`docs/review/invariants.md` and this log conflicted; each keeps both sides (5.04's licence line,
build tests and R38 build cell beside 5.01's timm floor, lock test and R38 timm clauses; the 5.04
entry above this one). `uv.lock` merged cleanly and `uv lock` rewrote it byte for byte. 5.04's
`Requires-Dist` test sees `timm>=1.0.3`; `check.sh` green (361 passed).

Review, started by the user: `/review-session 5.01`, tier 2, range 5bdd41e..e63fe7d (the item
commit only), risk files `.claude/commands/review-shapes.md`, `pyproject.toml`,
`tests/test_dependencies.py`, `tests/test_models.py`, `trailcam/model/__init__.py`,
`trailcam/model/backbone.py`, `trailcam/model/factory.py`, `trailcam/model/small_cnn.py`,
`uv.lock`; code-reviewer on Opus at xhigh, risk-reviewer on Fable at high. Every scenario held
(R01, R03, R05, R21, R23, R30, R38, R42 to R45). Known gaps passed: the Space label list (R05,
11.01), the demo and the Space (R21, 10.01, 11.01), the demo and the animal-removed test (R23),
the build table and the notebook (R38, 5.04, 6.02). Spec changed: none. The risk-reviewer found
nothing. One finding, verified and fixed in e70e057:
- code-reviewer, low: the no-network fixture in `tests/test_models.py` was per test, so pytest
  set it up after the module-scoped `models` fixture, and the docstring's promise that a
  forgotten `pretrained=False` fails did not hold for it. Reproduced before the fix: with the flag
  dropped there, the ResNet-18 weights (45 MB) downloaded into an empty scratch `TORCH_HOME`
  (deleted afterwards). The guard is now module-scoped and autouse, `models` depends on it and
  asserts it is up, and a new test shows it refuses `socket.create_connection`. With the old
  guard the fixture assertion errors the ten tests on `models`; with the flag dropped those ten
  fail on the guard and nothing downloads. `check.sh` green (362 passed).
Could not verify (both): the timm 1.0.3 floor and the `/review-shapes` table, both done in this
session (scratch venv, 38 passed; hand trace checked by forward hooks); a real weight download.

Fix round, started by the user: `/review-session 91e4667..HEAD 5.01`, tier 2, the fix commit
only (the merge before it brought 5.04, already reviewed), risk file `tests/test_models.py`;
code-reviewer on Opus at xhigh, risk-reviewer on Fable at high. Nothing found by either; every
scenario held (R01, R03, R05, R21, R30, R38, R42 to R45). Both checked that the patch catches
torchvision's `urlopen` and timm's `httpx2` paths and is undone at module teardown; the
code-reviewer's `check.sh` run was green (362 passed). Noted, not findings: with the ResNet-18
weights already in the local torch cache, a forgotten flag loads them without a connection, so
the guard does not fire but nothing downloads and the spy tests still pin the default (both); a
DNS lookup still happens before the refused connect (code-reviewer). No third round.

**Next.** Item 5.02, the configs; it syncs with main after this branch is merged.

Review: tier 2, ran `/review-session 5.01` (5bdd41e..e63fe7d) and the fix round
`/review-session 91e4667..HEAD 5.01` (code-reviewer Opus xhigh, risk-reviewer Fable high)

## 2026-10-13: session 5, item 5.02, the training configs (Opus 5.5)

**Done.** The root `config.yaml` holds a `common` block (the recipe every config shares) and four
configs under `configs`, each `common` updated with its own block: `dryrun`, `small_cnn`,
`resnet18_arm_a`, `resnet18_arm_b`. `trailcam/train/config.py` loads them into a frozen `Config`
dataclass of 18 fields (model, arm, pretrained, split_seed, run_seed, max_epochs,
head_only_epochs, batch_size, lr_body, lr_head, weight_decay, scheduler, warmup_epochs, patience,
mixed_precision, num_workers, pin_memory, fixture). A config's name is its key, not a field, so
arm A and arm B differ in `arm` alone (R25); the ledger's config column takes the name. Nothing
is defaulted in code: a missing, unknown or repeated key (plain YAML keeps the last of two, so a
second `arm:` would silently switch arms), a value of the wrong type (bool for int, a string such
as `1e-4`, which YAML 1.1 reads as text), an unknown model, arm or scheduler, or an out-of-range
value raises `ConfigError` naming the config and the field. The arm values are `ARMS` of
`trailcam/data/dataset.py`; the model names are a copy of 5.01's `MODEL_NAMES` (not merged yet).

Values: split seed 42 (the pinned split tests and the data card), run seed 0, AdamW weight decay
1e-2, cosine after a one-epoch warmup, patience 3, mixed precision on, batch 64, 4 workers with
pinned memory. ResNet-18 arms: ImageNet weights, body 1e-4, head 1e-3, up to 10 epochs with one
head-only epoch. Dry run: ResNet-18 arm B with random weights (offline), one epoch, batch 2,
`fixture: true`, 0 workers, no pinned memory, no head-only epoch (so its one epoch trains the body
too). PyYAML is a runtime dependency (`>=6.0.1`, the first release with Python 3.12 wheels;
`tests/test_config.py` passed on 6.0.1 with Python 3.12.9 in a scratch venv; timm, item 5.01,
depends on PyYAML anyway) and `types-pyyaml` a dev one for mypy.

**Small CNN rate.** The small CNN trains body and head at 1e-3, not 1e-4 / 1e-3: plan.md
section 9 gives the lower body rate to protect pretrained weights, and the small CNN has none.
Kept by the user's decision (via the coordinator, after the review): no pretrained weights to
protect. Plan section 5 now says so (see the review below).

**Verify.** `uv run pytest tests/test_config.py`: 57 passed. All configs load; the dry run has
max_epochs 1, batch 2, 0 workers, pin_memory false and fixture true; every config has both seeds;
the full configs have 4 workers and pinned memory; arm A and arm B differ in exactly `["arm"]`;
the full configs carry the recipe values of R31; every arm builds a `TrainTransform`; each
refusal above on an otherwise valid copy of the file. Mutations: arm B's head rate at 2e-3 fails
the arm test and the recipe test; the dry run with 4 workers fails the dry-run test; the plain
`SafeLoader` fails the repeated-key test. `scripts/risk_files.sh` already matched `config.yaml`
(4.02 fix round) and `tests/test_risk_files.py` passes. R25's and R31's test cells name the new
tests; R31's "where" column adds `config.yaml`; R38's names the config tests for the PyYAML floor.
`check.sh` green (396 passed), so `passes` is flipped.

Notes: `config.yaml` sits at the repository root, outside the wheel; `CONFIG_PATH` finds it from a
checkout or an editable install, so the Kaggle notebook (6.02) must pass the file's path (for
example from a clone at the same commit). `MODEL_NAMES` in `trailcam/train/config.py` stays
a copy of `trailcam/model/factory.py`'s for now (user's decision): item 5.03 switches it to an
import and adds a test that the lists match. How the head-only epoch and the warmup epoch
interact is 5.03's.

Review, started by the user: `/review-session 5.02`, tier 2, range 4319bd3..6dc606f (the item
commit), risk files `config.yaml`, `pyproject.toml`, `tests/test_config.py`,
`tests/test_dependencies.py`, `trailcam/train/__init__.py`, `trailcam/train/config.py`,
`uv.lock`; code-reviewer on Opus at xhigh, risk-reviewer on Fable at high. Known gaps passed:
R14, R16, R17 (train.py, 5.03), R25's ledger half (7.01), R31's loop half (5.03). Every other
scenario held (R01, R03, R05, R25 and R31 config halves, R38, R42 to R45). The code-reviewer's
`check.sh` run was green (396 passed); the risk-reviewer ran the config, dependency, risk-files
and control-plane tests (90 passed). Two low findings, no code change, by the user's decisions:

- Both reviewers, low: the small CNN's body rate 1e-3 was not in plan.md, which gives 1e-4 / 1e-3
  for every model. The user kept 1e-3; plan.md section 5 (Optimizer row) now says the small CNN
  trains from scratch, so its body and head both use 1e-3, and the 1e-4 body rate applies to the
  pretrained backbones. R31's rule text says the same. Spec changed: plan §5.
- Risk-reviewer, low: `config.yaml` is outside the wheel, so Kaggle loads it by a second channel
  and the ledger's config name alone cannot show that both arms ran the committed values. Handoff,
  approved by the user as a scope addition: 5.03's `train.py` records in the ledger row a sha256
  of the resolved config with `arm` and `run_seed` removed; 7.01's ledger checker requires arm A
  and arm B rows with the same run seed to share it; 6.02's notebook loads `config.yaml` from a
  clone at the same commit as the installed package.

Notes for 5.03 from the code-reviewer, not findings: `train.py` must pass `config.arm` to
`TrainTransform` (`make_loaders` defaults to arm A); the head-only epoch counts inside
`max_epochs` (10 in all), though the docstring of `test_the_full_configs_carry_the_recipe` reads
as 11; command-line overrides other than `run_seed` would bypass R25. Risk-reviewer: a YAML merge
key (`<<`) would bypass the repeated-key check; the file uses none. No fix commit touches a
high-stakes file, so no fix round.

After the review, main with 5.01 (018a6bf) was merged into this branch (a9557d4): the
`pyproject.toml`, `tests/test_dependencies.py` and `docs/review/invariants.md` conflicts keep
both sides (PyYAML and timm floors; R30 and R38 from 5.01 with R31 and the PyYAML note from
5.02), `uv.lock` was regenerated with `uv lock` from main's copy, and the log keeps 5.01's entry
first. 5.01's `MODEL_NAMES` equals the copy in `config.py`.

**Next.** Session 5, item 5.03.

Review: tier 2, ran `/review-session 5.02` (4319bd3..6dc606f; code-reviewer Opus xhigh,
risk-reviewer Fable high)

## 2026-10-13: session 5, item 5.03, the training loop, its command line and the ledger (Opus 5.5)

**Done.** `trailcam/train/train.py`: `python -m trailcam.train.train --config-name <name>
--data-root <cache> --out-dir <dir>` trains one config on the 512 px cache of 6.01
(`annotations/`, `images/`, `day_night.json`), prints its ledger row and appends it to
`docs/ledger.md`; `--dry-run=True` takes the `dryrun` config, writes the synthetic fixture into a
temporary directory itself, prints its row and never appends. `docs/ledger.md` holds the header
and a short column guide, no rows. `trailcam/evaluate/metrics.py` (new) holds `accuracy` and
`macro_f1` (over the classes present in the labels) with hand-computed tests in
`tests/test_metrics.py`, so 8.01 extends the module instead of duplicating it (8.01 had not
landed on main).

Decisions:

- **Epochs.** The head-only epoch counts inside `max_epochs` (ten in all for ResNet-18), and the
  one-epoch warmup coincides with it: the head warms up while the body is frozen, and the body
  joins in epoch 2 at the top of the cosine. Stated in the module docstring and logged at the
  start of every run. The test docstring at `tests/test_config.py` no longer reads as 11 epochs.
- **Head-only.** The body's parameters are frozen and its batch-norm layers kept in eval mode, so
  the body leaves the head-only epoch exactly as it entered it.
- **Schedule.** Per optimizer step (`LambdaLR`): linear warmup to 1 over the warmup epochs, then
  cosine towards 0 at `max_epochs`. Early stopping truncates the cosine.
- **Selection.** Best val macro-F1, a tie is no improvement; patience epochs in a row without a
  higher score stop the run; the best epoch's weights are restored, saved and hashed. trans_val
  is scored once, after the restore (accuracy and macro-F1 printed, accuracy in the row), so it
  cannot steer anything. The cis_test and trans_test loaders are dropped as soon as
  `make_loaders` returns.
- **Arm.** `train_run` passes `TrainTransform(config.arm)` to `make_loaders`; a test spies on
  `make_loaders` and checks arm A gets the control sampler and arm B the box sampler.
- **Config hash** (user's decision on review 5.02). `config_hash` in `config.py`: SHA-256 of the
  resolved config's sorted-key JSON without `arm` and `run_seed` (`HASH_EXCLUDED`). Column 5 of
  the ledger, right after the config name; plan.md section 5, R40's rule text and
  `.claude/rules/training.md` list it there.
- **Model names.** `config.py` imports `MODEL_NAMES` from `trailcam/model/factory.py`; a test
  checks it is the same object.
- **Overrides (R25).** `--run-seed` for any config; `--max-epochs` and `--batch-size` with
  `--dry-run=True` only; no other option exists (a test pins the option set). `with_overrides`
  in `config.py` checks the result as a loaded config is. A full config with `--dry-run=True`,
  the dry-run config without it, a dry run with `--data-root` or `--out-dir`, and a real run
  without them are refused. Booleans: `True` / `False` (any case), nothing else; no option
  abbreviations. `--config` takes the `config.yaml` path for Kaggle (5.02 handoff).
- **Leakage alarms (R16).** Before training, `check_loaders` stops a run whose train image ids
  reappear in val or trans_val or that is handed a test loader (it re-checks the loaders the loop
  reads; `make_loaders` checks too). After training, trans_val accuracy within 0.02 of val
  accuracy (or above it) and val accuracy above 0.95 (beyond Beery et al.'s 91.9% cis on
  ground-truth crops) are alarms: a real run prints and appends its row first, so nothing is
  lost, then exits 3; a dry run prints them and exits 0 (its random weights on synthetic images
  fire the trans_val alarm).
- **Seeds (R27).** The run seed seeds `random`, NumPy, torch (every device) and the train
  shuffle generator before the model is built; both seeds are logged. NumPy is imported directly
  but not declared: torchvision requires it.
- **Device.** CUDA when present, otherwise CPU (the Mac's dry run); mixed precision (float16
  autocast and a gradient scaler) on CUDA only.
- **Checkpoint.** `<out-dir>/<run id>.pt` holds the model name, the class list and the best
  state dict; the row carries its file SHA-256. The run id is
  `<config>-seed<run seed>-<UTC start>`.

`/dryrun` (`.claude/commands/dryrun.md`) and `/review-shapes` match the module path and flags as
written; neither changed. `.claude/rules/training.md` changed by one line (the config hash in the
column list): a control-plane edit.

**Verify.** With no fixture anywhere in the checkout, `uv run python -m trailcam.train.train
--max-epochs=1 --batch-size=2 --dry-run=True` exits 0 in about 25 s on the Mac, prints the split
(seed 42, hash 7a13478e...), `logits shape (2, 12)`, train loss 2.6539, val loss 2.4667, val
accuracy 0.1667, val macro-F1 0.1037, trans_val accuracy 0.1944, the row and the expected dry-run
alarm; two runs give the same losses and scores. `uv run pytest tests/test_train.py`: 38 passed
(early stopping stops after patience epochs without improvement and restores the best epoch's
weights; the end-to-end dry run with the small CNN reads train and val until training ends,
trans_val after, never a test split, leaves no temporary file and does not touch the ledger).
`tests/test_metrics.py` 9 passed, `tests/test_config.py` 76 passed. Mutations, each caught: a
tie counted as an improvement, the last epoch restored, `TrainTransform()` without the arm, batch
norm left training in the head-only epoch, trans_val scored before training, a loose boolean
parser, `--max-epochs` on a full run, a dry run that appends. `docs/ledger.md` holds the header
and no rows. Test cells filled: R14, R16, R17, R25, R27, R30, R31, R32, R37, R40. `check.sh` green
(485 passed), so `passes` is flipped.

Review, started by the user: `/review-session 5.03`, tier 2, range bda1990..0eb3368 (the item
commit), risk files `.claude/rules/training.md`, `tests/test_config.py`, `tests/test_metrics.py`,
`tests/test_train.py`, `trailcam/evaluate/__init__.py`, `trailcam/evaluate/metrics.py`,
`trailcam/train/config.py`, `trailcam/train/train.py`; code-reviewer on Opus at xhigh,
risk-reviewer on Fable at high. Known gaps passed: R14's data card (3.04); the notebook halves of
R16, R17, R27 and R37 (6.02 on); R17's final scoring (9.01); R25's and R40's ledger checker
(7.01); R32's per-class report and R33's timer (8.01). Every other scenario held (R01, R03, R05,
R14, R16, R17, R25, R27, R30, R31, R32, R33, R37, R40, R42 to R45). Two low findings, fixed in
e6caf63 by the user's decisions (via the coordinator):

- Code-reviewer, low: a real run checked `--out-dir` and `--ledger` only after training. On a
  wheel install the default ledger path lies under site-packages, so a run without `--ledger`
  would train, print its row and die in the append before printing its leakage alarms; an
  unwritable out dir would fail after `fit` with no row at all. Fix: `preflight` (out dir created
  and written to, ledger present with the header of `LEDGER_COLUMNS` and ending with its table)
  runs before the metadata is loaded, and a failure is a `parser.error` (exit 2, "before
  training: ..."); the row and the alarms print before the append; the docstring says to pass
  `--config` and `--ledger` from a clone on Kaggle, with `--out-dir` under `/kaggle/working`.
  Tests for each refusal, for a real run stopping before training, for the alarms showing when
  the append fails, and for an alarmed run appending then exiting 3.
- Risk-reviewer, low: plan.md section 5 read "up to 10 for the backbone (one head-only epoch
  first)", which reads as 1 + 10. No config change: the Epochs row now says the head-only epoch
  counts inside the 10, so the body trains at most 9 epochs, the same for both arms.
- Scope addition approved by the user: a trans_val macro-F1 column (over the classes present,
  R17, R32) after trans_val accuracy, since the ledger had no rows yet. `LEDGER_COLUMNS` (16
  columns), the `docs/ledger.md` header and guide, plan section 5, R40's rule text and
  `.claude/rules/training.md` list it; a test checks the row carries both trans_val scores.

Spec changed: plan §5 (the Epochs and Ledger rows) and `.claude/rules/training.md` (the column
list; a control-plane edit, one line each round). `check.sh` green (493 passed) and `/dryrun`
passed after e6caf63.

Fix round, started by the user: `/review-session 0eb3368..e6caf63 5.03`, tier 2, risk files
`.claude/rules/training.md`, `tests/test_train.py`, `trailcam/train/train.py`; same reviewers,
models and gaps. Every scenario held; the five column lists agree. Risk-reviewer: nothing.
Code-reviewer, one low, fixed in 8cb7db0 by the user's decision, no further round: `preflight`
read the ledger but did not check it could be written, so a read-only ledger (a copy under
`/kaggle/input`) still failed only at the append after training. `preflight` now opens it for
append and closes it, writing nothing; `test_preflight_refuses_a_bad_ledger` gains a read-only
case, which fails with the check removed. Notes, not findings: the 5.03 and 6.04 descriptions in
`docs/progress.json` list the older columns (items are never edited); a config name holding `|`
would only be refused after training (names are the fixed keys of `config.yaml`); mypy on
`tests/` reports one pre-existing export error (tests are outside mypy's scope).

R45 note, to revisit after the deadline: in the fix round the code-reviewer ran one read-only
`uv run python -c` on `docs/progress.json`, outside its Bash allow-list (nothing was written).
Its agent file's limits rest on the prompt.

After 8cb7db0: `check.sh` green (493 passed); `/dryrun` exits 0 and prints `logits shape
(2, 12)`, train loss 2.6539, val loss 2.4667 and the row (the expected dry-run alarm follows it):

```
| dryrun-seed0-20261014T025936Z | resnet18 | B | dryrun | bb400372536aa3088a5b3d855249f4db64dbde96d7d15ac97702b47a6c75a2f4 | 42 | 0 | 7a13478e3a541b4ddbc4493d4b802450a9422aa24f0e454ff7b676ee9f99df0c | 1 | 1 | cosine warmup 1 | 0.1667 | 0.1037 | 0.1944 | 0.1220 | 579650cc47d1edf5124c4bff94e4acd80d8c3e9356de898ae045dc2561c4dc11 |
```

**Next.** Session 6, item 6.02 (the Kaggle notebook: pass `--config` and `--ledger` from a clone
at the installed commit, `--data-root` and `--out-dir` under `/kaggle/working`, and print the
row), or item 7.01 (its checker reads `LEDGER_COLUMNS`, 16 columns).

Review: tier 2, ran `/review-session 5.03` (bda1990..0eb3368; code-reviewer Opus xhigh,
risk-reviewer Fable high), fix round `/review-session 0eb3368..e6caf63 5.03` (same reviewers)

## 2026-10-13: session 6, item 6.02, the Kaggle training notebook (Opus 5.5)

**Done.** `notebooks/kaggle_train.ipynb`, outputs cleared: a first markdown cell with the item,
a "How to run on Kaggle" list (import, Add Input the 6.03 cache, set the parameters, GPU on,
internet on, Save & Run All, paste the row; stop on a LEAKAGE ALARM) and the CDLA-Permissive 1.0
licence with the Beery et al. 2018 citation; then six code cells, each under a short heading:

1. Parameters: `CONFIG_NAME` (default `small_cnn`, the 6.04 run), `RUN_SEED` (`None` keeps the
   config's), `COMMIT` (default `dc05dad68ff8a5c99980e6dd6778375bb1f5557b`, main today, checked
   against `git ls-remote` of the public repo), `DATA_ROOT` (a placeholder,
   `/kaggle/input/cct20-512`), `REPO_URL` (the public repo), the clone and `OUT_DIR`
   (`/kaggle/working/models`) under `/kaggle/working`.
2. Session check, in seconds: `COMMIT` is 40 hex, `CONFIG_NAME` is a training config, a GPU is
   there (otherwise `train_run` would silently fall back to the CPU); prints Python, torch, GPU.
3. Clone the public repo into `/kaggle/working/who-goes-there`, `git checkout --detach COMMIT`,
   check HEAD is `COMMIT` and no tracked file but `docs/ledger.md` changed (a second run in one
   interactive session reuses the clone), print the commit.
4. `python -m pip install --quiet <clone>` (the one pip install, R38); the torch, torchvision,
   Pillow, timm and PyYAML versions (`importlib.metadata`) must be the same before and after.
5. Import `trailcam`; every module of the installed package must hash (`file_sha256`) the same as
   the clone's; `DATA_ROOT` must hold `CACHE_ANNOTATIONS`, `CACHE_IMAGES` and
   `DAY_NIGHT_SIDECAR` (otherwise it fails naming what is missing and the folders under
   `/kaggle/input`, three levels deep, that hold a `day_night.json`); `load_metadata` prints its
   counts repr and the cell prints the number of files under `images/` (R03, counts only).
6. `train.main([...])` in the kernel (live output, exceptions propagate): `--config` and
   `--ledger` from the clone, `--data-root DATA_ROOT`, `--out-dir OUT_DIR`, `--run-seed` only if
   set. A `SystemExit` (argparse or `preflight` refusing, before training), `ALARM_EXIT` (the row
   and alarms are printed above) or any other non-zero exit raises, so Save & Run All stops.
7. The ledger row: exactly one new line at the end of the clone's ledger, read back into
   `LEDGER_COLUMNS` and checked with `ledger_row`; the checkpoint under `OUT_DIR` is hashed again
   and must match the row; prints the commit, config, run seed, checkpoint path and sha256, then
   `ledger_header()` and the row to paste. No cell scores cis_test or trans_test.

No code cell defines a function, class or lambda; the four cells before the install cannot call
`trailcam` yet and use the standard library and torch only. The notebook is generated once with
`nbformat` (stable cell ids `cell-00` to `cell-12`) and is a fixpoint of both `ruff format` (ruff
lints notebooks, so `check.sh` covers its code) and `nbconvert --clear-output`.

`tests/test_notebook.py` (11 tests) reads the notebook without running it: guard-clean, no
outputs, execution counts or attachments (R41); every code cell parses as plain Python and
defines nothing; every name taken from `trailcam` exists; the parameters (40-hex `COMMIT`, the
public `REPO_URL`, `RUN_SEED` `None`, the config list equals the training configs of
`config.yaml`); the one pip install is of the clone with no upgrade, editable or no-deps flag,
plus the before/after version check (R38); one `train.main` call with the clone's config and
ledger, the checkpoint under `/kaggle/working`, its flags exactly the six real-run options, each
an option of `train.py`'s parser, no `split_seed` (R37, R25, R27); no `cis_test`, `trans_test`
or `evaluate(` in code (R17); the alarm and non-zero exits raise (R16); licence, citation and
the How-to steps in the markdown (R06); no token or key file named (R07). Mutations checked: an
added `--upgrade`, a fork URL, `config.yaml` not from the clone, the alarm compare removed, and
`--dry-run True` in the argv each fail a test. Test cells filled in `docs/review/invariants.md`
for R06, R07, R16, R17, R27, R37, R38 and R41.

Exercised outside Kaggle, in the scratchpad (no training cell run, no data read): the clone cell
against a local clone source (clean clone passes, a used ledger is allowed on reuse, an edited
`config.yaml` is refused); the check cell with a missing `DATA_ROOT` (fails naming the three
parts, "none" under `/kaggle/input`); the row cell on a fabricated ledger row and checkpoint
(prints the header and row; a changed checkpoint is refused). The training cell itself first
runs in 6.04, on Kaggle, by the user.

Steps: `uv run python scripts/guard_data.py --all` exits 0 with the notebook staged; `uvx --from
nbconvert jupyter nbconvert --clear-output --inplace notebooks/kaggle_train.ipynb` leaves it
unchanged (`git diff --exit-code`); no code cell defines anything and each `trailcam` name it uses
exists (`tests/test_notebook.py`). `scripts/risk_files.sh` matches `\.ipynb$`. Spec changed:
none (invariant test cells only). `check.sh` green (504 passed).

For the user (6.03 to 6.05): set `DATA_ROOT` to the attached dataset's folder (the check cell
lists candidates) and `COMMIT` to the main commit to run, at or after this item's merge if the
notebook itself should come from that commit (the clone only needs dc05dad or later). The run's
ledger row lacks the commit; note it with the row in the progress log.

**Next.** Session 6 is the user's (6.03 the cache dataset, 6.04 the small CNN, 6.05 arm A seed
0), or item 7.01 (the ledger checker).

Review, started by the user: `/review-session 6.02` (dc05dad..f30e0c1), tier 2, risk files
`notebooks/kaggle_train.ipynb`, `tests/test_notebook.py`; code-reviewer on Opus at xhigh,
risk-reviewer on Fable at high. Known gaps passed: R06 Space and slides (11.01, 12.02), R07 deploy
script (11.01), R17 final scoring (9.01), R38 `space/requirements.txt` (11.01), R41 figures
(8.06), the training cell not run off Kaggle (6.04). Every scenario held (R01, R03, R06, R07, R16,
R17, R27, R37, R38, R41 to R45). Risk-reviewer: nothing. Code-reviewer, two lows, both fixed in
9c2260d by the user's decisions (via the coordinator):

- The run steps did not carry plan section 5's accelerator, the single P100 (GPU T4 x2 uses quota
  twice as fast and training uses one GPU). Fix: How-to-run step 4 and the no-GPU error name GPU
  P100; more than one GPU prints a warning to stderr and does not refuse.
- In an interactive session, a run at a new `COMMIT` after deleting the clone (as the clone cell
  then suggested) would pass the byte-for-byte check on disk yet train with the modules Python had
  already imported from the earlier commit. Fix: the clone cell offers only a fresh session (Run >
  Factory reset); the first cell that imports `trailcam` refuses a kernel where it is already in
  `sys.modules`. Side effect: a cell-8 failure (a wrong `DATA_ROOT`) also needs a Factory reset.

Two new tests in `tests/test_notebook.py` (no existing test line changed); both fail on the
f30e0c1 notebook. `check.sh` green (506 passed); nbconvert leaves the notebook unchanged; the
guard passes.

Fix round, started by the user: `/review-session f30e0c1..9c2260d 6.02`, same reviewers, models,
scenarios and gaps. Nothing found by either; every scenario held. Notes, not findings: the four
`# noqa: E402` comments are unused (ruff passes with `--ignore-noqa`); one assert in
`test_trailcam_is_imported_once_per_kernel` cannot fail by construction; the clone cell's
"reuses the clone" comment now applies only after a kernel restart that keeps `/kaggle/working`.

Open for the user, unverified (no network in the review): whether Kaggle's preinstalled torch
(2.11) still has kernels for the P100 (sm_60); recent PyTorch CUDA 12.8+ builds dropped Pascal,
in which case the training cell would fail at its first CUDA call, after the setup cells. Also
whether Run > Factory reset empties `/kaggle/working`. Both are checked in the first Kaggle
session (6.03 or 6.04).

R45 note, to revisit after the deadline: in the first round the code-reviewer again ran a
read-only `uv run python -c` (printing the notebook's cells), outside its Bash allow-list; nothing
was written. Same pattern as review 5.03; its agent file's limits rest on the prompt.

Review: tier 2, ran `/review-session 6.02` (dc05dad..f30e0c1; code-reviewer Opus xhigh,
risk-reviewer Fable high), fix round `/review-session f30e0c1..9c2260d 6.02` (same reviewers)

## 2026-10-14: session 7, item 7.01, the ledger checker (Opus 5.5)

Taken ahead of its session by the user's decision in the coordinator session, alongside 6.02.

**Done.** `trailcam/train/ledger.py`: `python -m trailcam.train.ledger --check <ledger>` prints
every problem to stderr and exits 1, or prints the val summary and exits 0. It imports
`LEDGER_COLUMNS`, `LEDGER_PATH` and `ledger_header` from `train.py` and keeps no copy (review
5.03). Checks, after both review rounds:

- Table: the header once, followed by its separator; every later line a row of 16 cells (a short
  row is "a column is missing"), since `append_row` writes at the end.
- Cells: none empty; config hash, split hash and checkpoint sha256 64 lower-case hex digits (what
  `hexdigest` writes); seeds and epoch counts canonical whole numbers, 1 <= best epoch <= epochs
  run; model in `MODEL_NAMES`, arm in `ARMS`; scheduler `<SCHEDULERS name> warmup <n>`; the four
  scores `d.dddd` in [0, 1]; the run id `<config>-seed<run seed>-<valid UTC stamp>` of its own
  row; config `dryrun` refused (R37).
- Across rows: one row per run id (R40); the rows of one config share model and arm; one split
  seed has one split hash and back (R14); the rows of one model and run seed, arm A and B alike,
  share split hash and config hash (R25; the config hash leaves out `arm` and `run_seed`, review
  5.02); every row of one model shares the config hash, whatever its split seed, run seed and arm
  (review 7.01); the rows of one config hash agree on model, split seed and scheduler (review
  7.01). By the user's decision, any logged run of a model with a changed recipe or split seed
  fails the check; the docstring and `docs/ledger.md` say so.
- Summary (R17, R03): one line per model, arm, split seed and run seed, sorted by those keys only,
  with val accuracy and macro-F1 and trans_val beside them as reported; a model and arm with more
  than one row gets an "all" line of means. Nothing is ranked and no "best" is shown.

Not checked, for the user to decide if wanted: the R16 score alarms (the run already exits 3 on
them; failing an alarmed row needs a rule for acknowledging it). Left to 9.01 by the user: two
runs of one config and seed both pass, and the summary averages them (its rows column shows it);
which checkpoint stands for that arm and seed is the scoring script's question.

Steps: `uv run pytest tests/test_ledger.py` 53 passed. `--check docs/ledger.md` exits 0
(`ledger: 0 rows, every check passed`); a copy with one 15-cell row exits 1 (`line 23: 15 cells,
expected 16`); a copy with an arm A and arm B pair of run seed 0 whose config hashes differ exits
1; the same pair with one config hash exits 0 with the summary. The real dry-run row of the 5.03
entry, appended to a copy, fails on `config 'dryrun'` alone, so every other cell matches what
`train.py` writes. Each cross-row check fails its own test when removed. `docs/review/invariants.md`:
test cells of R25 and R40 filled, `tests/test_ledger.py` added to R14 and R17. `docs/ledger.md`
gains the checker paragraph. `check.sh` green (546 passed before the merge of main, 559 after).

Review, started by the user: `/review-session 7.01` (dc05dad..247bad8), tier 2, risk files
`tests/test_ledger.py`, `trailcam/train/ledger.py`; code-reviewer on Opus at xhigh, risk-reviewer
on Fable at high. Scenarios R01, R03, R05, R14, R16, R17, R25, R31, R40, R42 to R45 held; known
gaps passed: R14's data card (3.04), the notebook halves of R16 and R17 (6.02), R17's and R40's
final scoring (9.01), the R16 score alarms in the checker. Three low findings, all fixed in
f203f8b by the user's decisions (via the coordinator):

- Code-reviewer, low: the config hash was compared only within one model and run seed, so a seed
  pair trained to another recipe passed and the "all" line averaged two recipes. Fix: the rows of
  one model share the config hash across run seeds and arms; test with both seed-1 rows on
  another hash, and `test_ab_pairs_are_matched_by_run_seed` now asserts what its docstring says.
- Risk-reviewer, low: rows sharing a config hash could disagree on the cells it stands for (an
  arm B row reading `cosine warmup 2`). Fix: one config hash, one model, split seed and scheduler;
  test with that arm B row.
- Both reviewers (risk as a finding, code as a note), low: no test for two models under one
  config, though R40's cell claimed it. Fix: the test is parametrised over arm and model.

Fix round, started by the user: `/review-session 247bad8..f203f8b 7.01`, same reviewers, models,
scenarios and gaps (plus the duplicate-runs note). Risk-reviewer: nothing. Code-reviewer, one
low, fixed in 9bd3045 by the user's decision (option a), no further round: the cross-seed check
grouped by model and split seed, so a seed pair on another split seed formed its own group and
passed. Plan section 5 fixes one split seed, so the check now groups by model alone; a test puts
seed 1's pair on split seed 7 and fails on the old grouping. Notes, not findings: mypy on
`tests/test_ledger.py` reports three errors (tests are outside mypy's scope, unchanged since
247bad8).

R45 notes, to revisit after the deadline: in the fix round both reviewers wrote into the session
scratchpad, outside the repository (the code-reviewer a probe test it ran to confirm its finding,
the risk-reviewer a `git show` of the parent commit for a mypy comparison); nothing in the
repository was written. Earlier (reviews 5.03 and 6.02) the code-reviewer ran a read-only
`uv run python -c` outside its Bash allow-list. Its agent file's limits rest on the prompt.

Merged main at 15ce874 (6.02) into this branch: conflicts in this log (both entries kept, in item
order) and in `docs/review/invariants.md` (R17's test cell keeps both `tests/test_notebook.py`
and `tests/test_ledger.py`; every other row as its side left it).

**Next.** Item 7.02 (user): run arm B seed 0, paste the row into `docs/ledger.md` and run
`uv run python -m trailcam.train.ledger --check docs/ledger.md`.

Review: tier 2, ran `/review-session 7.01` (dc05dad..247bad8; code-reviewer Opus xhigh,
risk-reviewer Fable high), fix round `/review-session 247bad8..f203f8b 7.01` (same reviewers).

## 2026-10-14: session 6, item 6.06, GPU T4 x2 and the architecture check (Opus 5.5)

New item, approved by the user in the coordinator session on 2026-10-14 and added to
`docs/progress.json` as 6.06 (session 6, owner claude, due 2026-10-15). Evidence, run by the user
on Kaggle in a GPU T4 x2 session on 2026-10-14: torch 2.11.0+cu128 (CUDA build 12.8), two Tesla
T4 at capability (7, 5), driver 580.178.04, `torch.cuda.get_arch_list()` = sm_75, sm_80, sm_86,
sm_90, sm_100, sm_120. No sm_60, so a P100 session cannot run Kaggle's torch: this closes the
open question of review 6.02 (the P100 would have failed at the first CUDA call of training).

**Done.** Plan section 5 (spec changed: plan §5): the accelerator is GPU T4 x2, training on one
T4, the same device for every run and arm; why (no sm_60 in Kaggle's torch 2.11, checked
2026-10-14); quota 30 GPU hours a week; the GPU budget line says one T4; `num_workers` stays 4.
Added at the coordinator's request, within this item's scope: after the user's short check
session the quota page showed 29h 49m of 30h available; the rate at which GPU T4 x2 draws on the
quota is unverified and is measured on the first real run (6.04, the quota page before and after
and the version's run time); the quota rules (GPU sessions only for training and scoring runs,
the cache build and everything else on CPU, Save & Run All so no GPU time is idle, interactive
GPU sessions stopped right after use). The GPU budget paragraph (under 10 h) is unchanged.

`trailcam/train/device.py` (new): `sm_arch`, `arch_supported` (exact match of the device's
`sm_XY` in the arch list), `check_gpu` (refuses no GPU, other than one visible device, or an
architecture not in torch's build, each naming GPU T4 x2 or the Factory reset; returns a line
naming the device, `CUDA_VISIBLE_DEVICES`, the torch and CUDA build and its arch list) and
`workers_warning` (a warning naming `num_workers` and the CPU count when the first exceeds the
second; `None` when the count is unknown). The exact match refuses devices CUDA could still serve
from a same-major cubin (an sm_89 on sm_86 code) or PTX; noted in the module docstring, not
relevant to the T4.

`notebooks/kaggle_train.ipynb` (outputs cleared): How-to-run step 4 says Accelerator GPU T4 x2,
why not the P100, that training uses the first T4 and the 30-hour quota; the 6.02 multi-GPU
warning and the P100 no-GPU message are gone. Cell order: the parameter cell (section 1) sets
`CUDA_VISIBLE_DEVICES=0` and no longer imports torch, so torch is first imported by the
`trailcam` import after the stale-import guard of review 6.02, which stays the first statement of
that cell. The GPU check moved after the install (it needs `trailcam`): section 3 is now three
cells, the guard and module match, then the GPU and CPU cell (`print(check_gpu())`, the chosen
config's `num_workers` from the clone's `config.yaml`, `os.cpu_count()`, `workers_warning` to
stderr), then the cache check, so a P100 or a missing GPU fails after the clone and pip install
(tens of seconds) but before the cache is read or training starts. Cell ids renumbered
`cell-00` to `cell-14`.

Tests: `tests/test_device.py` (new, 18 tests): `sm_arch`; `arch_supported` against Kaggle's list
(sm_75, sm_80, sm_120 in; sm_60 and sm_89 out) and against a given list (sm_60 in an older list,
sm_75 out of it, an empty list, a `compute_75` entry only); `check_gpu` on a patched `torch.cuda`
accepts one T4, refuses a P100 naming sm_60 and the arch list, no GPU, and 0 or 2 visible
devices; `workers_warning` at, below and above the CPU count and with an unknown count.
`tests/test_notebook.py`: the 6.02 P100 test is replaced by two, the T4 x2 markdown and the one
`CUDA_VISIBLE_DEVICES` assignment (equal to `VISIBLE_DEVICES`) in a cell before any cell that
imports torch or `trailcam`; and `check_gpu` and `workers_warning` called once, in one cell, not
before the first `trailcam` import and before `load_metadata` and `train.main`. Both fail on the
6.02 notebook. Mutations checked: `arch_supported` always true fails four device tests; the pin
moved after the `trailcam` imports fails the pin test.

Steps: `check.sh` green (525 passed); the tests above cover the architecture check, sm_60
refused against Kaggle's list; `uvx --from nbconvert jupyter nbconvert --clear-output --inplace
notebooks/kaggle_train.ipynb` leaves the notebook unchanged (`cmp`); `uv run python
scripts/guard_data.py --all` exits 0 with the changes staged. No invariants row touched (none
cites the accelerator); `config.yaml` unchanged.

For the user (6.03 to 6.05): 6.03 on a CPU session; for 6.04, note the quota page before and
after the run and the version's run time; choose GPU T4 x2; the GPU and CPU cell prints the
device and the CPU count, and warns if 4 workers outnumber the CPUs (a change to `num_workers`
changes the config hash, your decision after the first run). Still open from review 6.02:
whether Run > Factory reset empties `/kaggle/working`.

Review, started by the user: `/review-session 6.06` (15ce874..604aa5f), tier 2, risk files
`notebooks/kaggle_train.ipynb`, `tests/test_device.py`, `tests/test_notebook.py`,
`trailcam/train/device.py`; code-reviewer on Opus at xhigh, risk-reviewer on Fable at high.
Scenarios R03, R05, R06, R07, R14, R16, R17, R27, R31, R37, R38, R41, R42, every one held. Spec
changed: plan section 5 only (rows citing it: R14, R17, R20, R22, R27, R31, R37, R38, R40; none
contradicted). Known gaps passed: R06 Space and slides (11.01, 12.02), R07 deploy script
(11.01), R17 final scoring (9.01), R38 `space/requirements.txt` (11.01), R41 figures (8.06), the
training cell and `check_gpu` not run on a real GPU (6.04). Nothing found by either reviewer.
Neither could run nbconvert (outside their commands); it was run in this session before the
commit and left the notebook unchanged, and the amend touched `docs/` only.

Notes, recorded and not fixed (the user's decision, via the coordinator, no code change):

- CPU count: `workers_warning` reads `os.cpu_count()`, which may report the host's CPUs in a
  quota-limited container, while the DataLoader uses `len(os.sched_getaffinity(0))`. Compare
  both on the first real run (6.04).
- Default `COMMIT`: the notebook's default (dc05dad) predates `trailcam/train/device.py`, so with
  the default left in place cell 8 fails with `ModuleNotFoundError` after the clone and install.
  For 6.04 the user sets `COMMIT` to a main commit at or after the 6.06 merge (the coordinator
  gives the exact hash).
- Same device: `check_gpu` accepts any GPU whose architecture is in the build, not only a T4;
  Kaggle offers T4 x2 and the P100 and the P100 is refused, so every run is on a T4 in practice.
  The device name is printed, not in the ledger row (plan section 5 does not ask for it).
- Pin value: `check_gpu` checks one visible device, not that `CUDA_VISIBLE_DEVICES` is `0`; a
  hand edit to `1` would train on the second, identical T4. The committed value is tied to
  `VISIBLE_DEVICES` by a test.

R45 note, to revisit after the deadline (third occurrence): the code-reviewer again ran a
read-only `uv run python -c` printing the notebook's cells, outside its Bash allow-list; nothing
was written.

Evidence from 6.03 that bears on this item (pasted by the user): the cache build ran on Kaggle
at 15ce874 in a CPU session with Python 3.13.15, Pillow 12.3.0, libjpeg-turbo 3.1.4.1, timm
1.0.29 (not 1.0.30 as 5.01 read from Colab's list) and torch 2.11.0+cpu.

Merged main at 46e4330 (7.01) into this branch: one conflict, in this log (both entries kept,
7.01's first since it merged first, so this entry's review line ends the log);
`docs/progress.json`, `docs/review/invariants.md` and `docs/ledger.md` merged cleanly.

**Next.** Session 6 is the user's (6.03, 6.04, 6.05); then 7.02 (user).

Review: tier 2, ran `/review-session 6.06` (15ce874..604aa5f; code-reviewer Opus xhigh,
risk-reviewer Fable high), nothing found, no fix round

## 2026-10-14: session 8, item 8.01, the per-class report, the confusion matrix and the timer (Opus 5.5)

Taken ahead of its 10-18 date by the user's decision in the coordinator session, alongside 8.03.

**Done.** `trailcam/evaluate/metrics.py` extended; `accuracy` and `macro_f1` keep their signatures
and behaviour (5.03's tests unchanged and passing):

- `confusion_matrix(predictions, labels)`: a 12 x 12 list of counts in the class order of
  `trailcam/data/classes.py`, rows true, columns predicted; an absent class has a zero row, and
  predictions of an absent class show in its column.
- `per_class_scores(predictions, labels)`: `{class index: ClassScores(precision, recall, f1,
  support, predicted)}` for the classes present in the labels only, ascending, with the module's
  conventions (a present class never predicted has precision 0; F1 = 2TP / (2TP + FP + FN)).
  `macro_f1` is now the mean of these F1 values, so the identity holds by construction and is
  tested on 200 random cases.
- `time_one_prediction(model, image) -> float` (R33): takes the eval transform's output, (3, H, W)
  or (1, 3, H, W), moves it to the model's device, makes one untimed warm-up call, then reads
  `time.perf_counter()` around a second forward pass in eval mode under `torch.no_grad()`, with a
  device synchronise before each reading on CUDA or MPS; the model's train mode is restored. The
  choice, stated in the module docstring: decoding, the eval transform and the move to the device
  are outside the timed region, so the seconds the demo shows are the model's forward pass only.

Note for 10.01: the timer returns seconds only, as the item asks, so a demo that also needs the
logits runs the model a third time (warm-up, timed, its own predict). If that matters on the
Space's CPU, 10.01 can propose returning the logits beside the seconds.

Steps: `uv run pytest tests/test_metrics.py` 20 passed: hand-computed per-class scores and
confusion matrix on the 12-class toy case and on a split with four present classes and
predictions of absent ones; rows true and columns predicted; the bad-input cases refused by the
two new functions too; the timer returns a positive float for one synthetic-fixture JPEG through
the small CNN (eval transform, `build_model("small_cnn")`); with a fake clock, the order is
forward, clock, forward, clock and the difference is returned; no grad and eval mode in both
calls; a CPU input reaches a `meta`-device model on `meta`; a batch of two is refused.
`docs/review/invariants.md`: test cells of R32 and R33 filled (reporting on trans test stays
9.01's, the demo and Space calls 10.01's and 11.01's). `check.sh` green (589 passed).

**Next.** 8.02 (figures, Grad-CAM, the evaluate entry point), with 8.03 in parallel.

Notes, recorded and not fixed (the user's decision, via the coordinator, no code change):

- Mode restore: `time_one_prediction` restores with `model.train(was_training)`, one mode for
  every submodule, so a model passed in with mixed modes (the head-only epoch's frozen batch norm)
  would come back with that batch norm in train mode. No current or planned caller times a model
  mid-training; the demo and the Space pass eval-mode models.
- Demo cost, handed to 10.01: every call runs two forward passes and returns the seconds only, so
  a demo request costs its own prediction plus those two.

Review: tier 2, ran `/review-session 8.01` (17ca508..0270f80; code-reviewer Opus xhigh,
risk-reviewer Fable high), nothing found, no fix round. Scenarios R01, R03, R05, R32, R33, R42 to
R45 held; known gaps passed: R32 reporting on trans test (9.01), R33 the demo and Space calling
the timer (10.01, 11.01). Neither reviewer could run the CUDA and MPS synchronise branches.

## 2026-10-15: session 6, item 6.07, the notebook as imported and the cache completeness check (Opus 5.5)

New item, requested by the user in the coordinator session on 2026-10-15 and added to
`docs/progress.json` as 6.07 (session 6, owner claude, due 2026-10-15). Evidence, from the user's
Kaggle rehearsal on 2026-10-14/15 (GPU T4 x2, the private `cct20-512` dataset made from the 6.03
cache): Kaggle mounts the dataset at `/kaggle/input/datasets/codx888/cct20-512/cct20_512`, one
level deeper than the notebook's `/kaggle/input/cct20-512` and below the old three-level candidate
search; one session found 57,864 JPEGs, but a later session start loaded 645 files (64.52 MiB of
the cache's 2.92 GiB), so completeness must be checked; the default `COMMIT` dc05dad lacks
`trailcam/train/device.py` (6.06), so section 3 would fail; a stale duplicate parameter cell
overrode a hand edit; pip kept torch 2.11.0+cu128, torchvision 0.26.0+cu128, Pillow 12.3.0, timm
1.0.29 and PyYAML 6.0.3.

**Done.** `trailcam/data/cache.py` (new): `sidecar_folders(root, max_depth=MAX_DEPTH)` lists
every folder holding a `day_night.json` at most 8 levels below the root (os.walk, symlinked
folders followed, a missing root gives none); `missing_images(meta, image_root)` gives per split
(kept records, of those whose image is missing or empty), one `stat` per image, no decoding;
`check_images(meta, image_root)` returns a line of counts when all are there and otherwise raises
`FileNotFoundError` naming found of expected in all and per split, and the folder, never a file
name, image id or seq id (R03). Only kept records of the five splits are checked: dropped images
(empty, two species, rare) are never read, and the raw annotation lists are not in `Metadata`, so
checking them would mean reading the files a second time for no protection.

Decision: `train_run` calls `check_images` too, before the split is drawn (after the run's first
log lines), so every run is protected, not only notebook runs, including the dry run on its
fixture. The cost is one `stat` per kept image, seconds on the cache; on Kaggle the notebook's
section 3 and then `train_run` each pay it once. A failure there is an exception (traceback with
the counts), not `parser.error`: it needs the metadata loaded, so it sits in `train_run` rather
than in `main`'s preflight, which keeps the existing `main` tests that stub `load_metadata` and
`train_run` unchanged.

`notebooks/kaggle_train.ipynb` (outputs cleared): `DATA_ROOT =
Path("/kaggle/input/datasets/codx888/cct20-512/cct20_512")`; `COMMIT =
e321151aed8aaf0cedc4581685235898796a511d`, main at this session's start; `CONFIG_NAME =
"small_cnn"` and `RUN_SEED = None` kept. How to run, now seven steps: File > Import Notebook with
the GitHub link; Add Input, with the mount path and that another account or dataset name changes
`DATA_ROOT`; keep exactly one parameter cell and edit only `COMMIT` (and `DATA_ROOT` for another
dataset), `COMMIT` being the main commit the coordinator gives at or after this item's merge (the
default predates 6.07 and is refused in section 2); GPU T4 x2 and Internet on; rehearse sections
1 to 3, then stop the session; Save & Run All; paste the row. Section 1 prints the four
parameters in effect. Section 2 refuses a clone without `trailcam/data/cache.py` ("predates item
6.07"), before the pip install. Section 3's cache cell lists `sidecar_folders(Path("/kaggle/input"))`
when `DATA_ROOT` lacks a part, and after the metadata and file count prints
`check_images(meta, needed["images"])`.

Tests: `tests/test_cache.py` (new, 9): a complete fixture cache passes with its count; one image
deleted or emptied in cis_val is refused with "N-1 of N" in all and per split and no id, file name
or seq id in the message; a missing image folder is refused; an extra file changes nothing;
`train_run` refuses an incomplete cache before `make_split` and writes no out dir; the search
finds Kaggle's `datasets/codx888/cct20-512/cct20_512` (depth 4) and nothing else, the depth limit
(depth 8 found, 9 not, `max_depth=0` the root only), a missing root, a symlinked folder.
`tests/test_notebook.py`: `DATA_ROOT` and `CONFIG_NAME` defaults; the default `COMMIT` is a
commit of this repo descending from 17ca508 (dc05dad does not; CI has the full history); exactly
one cell assigns the parameters and the check cell prints them; the clone cell refuses a commit
without `cache.py` before the pip cell; the cache cell uses `sidecar_folders` (no glob depths) and
calls `check_images` after `load_metadata`, in a cell before `train.main`; the How-to-run steps
in order. Mutations checked: `train_run` without the check fails the `train_run` test; an empty
file counted as present fails the empty case. `docs/review/invariants.md`: R03's test cell names
`tests/test_cache.py`.

Steps: `check.sh` green (603 passed); the completeness and search tests above; `uvx --from
nbconvert jupyter nbconvert --clear-output --inplace notebooks/kaggle_train.ipynb` leaves the
notebook unchanged (`cmp`); `uv run python scripts/guard_data.py --all` exits 0 with the changes
staged. Local `main` had not moved at commit time (still e321151); origin could not be reached
from this session.

**Next.** For 6.04: set `COMMIT` to the main commit the coordinator gives at or after this
item's merge, rehearse sections 1 to 3 (section 3 should print all kept images found), stop the
session, then Save & Run All. Review: tier 2, `/review-session 6.07`.

Notes, recorded and not fixed (the user's decision, via the coordinator, no code or notebook
change):

- Section 1 wording: "a second parameter cell would show here" holds only for a duplicate cell
  above section 1; a later one would override the values without showing there. Left as is, to
  avoid another review round on the item that blocks 6.04: How-to-run step 3 says to delete
  duplicates, and section 4's printed command line and section 5's row show the values used.
- Default `COMMIT`: it predates 6.07 and is refused in section 2 by design. A session started
  without changing it leaves the clone at e321151, so the re-run with the right sha asks for a
  Factory reset (the existing clone-reuse behaviour, named in the message); no GPU time is spent.
- Size before content: `check_images` stats each kept image (present, regular, non-empty) and
  does not decode it, so a file Kaggle shows at full size before its content has loaded would
  pass. Whether a partly loaded session can do that was not observed and cannot be checked here.

Review: tier 2, ran `/review-session 6.07` (e321151..532a74f; code-reviewer Opus xhigh,
risk-reviewer Fable high), nothing found, no fix round. Scenarios R01, R03, R05, R06, R07, R14,
R16, R17, R25, R27, R31, R32, R37, R38, R40 to R45 held; no known gaps passed, no spec changed.
Neither reviewer could run the nbconvert step (no network; it was run in the session and left the
notebook unchanged) or check the Kaggle mount and the stat cost there (the rehearsal of 6.04).

## 2026-10-14: session 8, item 8.03, attention-in-box and the animal-removed test (Opus 5.5)

Taken ahead of its 10-18 date by the user's decision in the coordinator session, alongside 8.01.
Main (0270f80, 8.01) was fast-forwarded into this branch before the commit; this branch had no
commit of its own, so `git merge --ff-only main` (the sync tool merges `origin/main`, which does
not have 8.01 yet). My `docs/review/invariants.md` edit was set aside as a patch and reapplied
cleanly (8.01 changed R32 and R33, this item R03, R23, R26 and R38).

**Done.** `trailcam/evaluate/attention.py` and `trailcam/evaluate/gradcam.py`, with
`tests/test_attention.py` (35 tests) and `tests/test_gradcam.py` (7). Rules, stated in the
module docstring:

- Grad-CAM: a small hook-based implementation (Selvaraju et al. 2017) for the top-1 class, on
  `layer4` of ResNet-18, the third conv block of the small CNN and `bn2` of EfficientNet-B0;
  ReLU, then bilinear upsampling to 224 x 224, not normalised in `gradcam.py`. Not the
  `grad-cam` package, which would add OpenCV and scikit-learn to the runtime dependencies.
  Checked once in a throwaway env (`uv run --with grad-cam`, 1.5.7, not added to the project):
  after the package's min-max scaling the maps are identical for ResNet-18 and within 0.007 for
  the small CNN (correlation 1.0 on four random images each).
- Attention-in-box: the map is divided by its total over the whole 224 x 224 input; the score is
  the share inside the box. The box is mapped into the eval transform's frame (shorter side 256,
  centre crop 224) from the loaded image's size, and the geometry is tested against
  `EvalTransform` itself on seven image sizes. A box cut by the crop is clipped to it. Several
  boxes count as their union: each pixel takes the share of its area the boxes cover, the largest
  share where they overlap. Excluded and counted in the table: images without a box, and images
  whose boxes all fall outside the centre crop (2 of 72 on the fixture's val). A map that is zero
  everywhere has no score and is counted; the image stays in the animal-removed test.
- Animal-removed: each box, rounded outwards to whole pixels of the loaded image, is filled with
  the per-channel mean of the whole decoded RGB image (box included), on 0 to 255 values, before
  resize and normalisation; the blanked image goes through the same `EvalTransform`.
- Reporting: one row per split (val and cis_test are cis cameras, trans_val and trans_test
  trans; splits are never pooled) and per subset (all, correct, wrong on the unblanked
  prediction). Attention: images, mean, median, and the mean box share of the frame, which is
  what an even map would score, so the score can be read against it (an addition to the item's
  wording). Animal-removed: accuracy and mean top-1 confidence before and after blanking, and the
  share of predictions kept. Output: `attention.md` and `attention_in_box.png` (one panel per
  split, correct vs wrong as shares of their own images) under `--out`; counts and scores only.
- Test-set isolation: the command line takes `--split val` and/or `trans_val` only (argparse
  refuses anything else with exit 2) and drops the `Split` before the pass. 9.01 calls
  `attention_pass`, `summarise` and `write_report` directly for cis_test and trans_test.
  `attention_pass` refuses records whose `split` differs from the split it is given.
- Real runs: `python -m trailcam.evaluate.attention --config-name <name> --checkpoint <run>.pt
  --data-root <cache> --split val --split trans_val --out <dir>`. The checkpoint must hold the
  config's model and the class map; the table header gives the checkpoint sha256, the split seed
  and the split hash to match against the ledger row. `--dry-run=True` uses the fixture and random
  weights.

Dependency: `matplotlib>=3.8.4` for the histogram, which 8.02 can reuse. 3.8.4 is the oldest
version `tests/test_attention.py` passes on with NumPy 2 (3.8.3 and older declare `numpy<2`; 3.9.0
also passes). It brings contourpy, cycler, fonttools, kiwisolver, pyparsing, python-dateutil and
six, none of them CUDA; locked 3.11.2. `tests/test_dependencies.py` adds the minimum. Kaggle
preinstalls matplotlib, but I could not check its version. The notebook's before-and-after
version check covers torch, torchvision, Pillow, timm and PyYAML, not matplotlib; if Kaggle's copy
were older than 3.8.4, pip would upgrade it (no CUDA involved).

Steps: `uv run pytest tests/test_attention.py` 35 passed. A map inside the box scores 1.0 and one
outside scores 0.0. An even map scores the box's share of the frame, and half the mass inside
scores 0.5. Blanking changes only the box's pixels and fills them with the image mean. The fixture
command line writes the table and the PNG under `--out` and nothing under `docs/figures/`. Each of
these mutations fails a test: normalising after cropping to the box, zero fill, a zero image mean,
`trans_test` added to the CLI splits, no crop offset, a direct 224 resize, overlapping boxes
counted twice, the blank region rounded inwards, correct and wrong swapped, Grad-CAM without its
ReLU, on the lowest class, min-max scaled, or leaving its hook. `check.sh` green (631 passed). In
`docs/review/invariants.md` the test cells of R26 and R23 (the rows naming 8.03) are filled, and
R03 and R38 now mention the new tests. `config.yaml` and `plan.md` are unchanged.

For 8.02: `gradcam.py` exists with the map only. An overlay that draws `gradcam()` shows what
attention-in-box scores. Plan section 7 names the `grad-cam` package for the overlays, which would
be a second implementation; 8.02 decides. The `assets/`-only rule (R04, R35) applies to the overlay
writer, not to this map function.

**Next.** 8.02 (figures, Grad-CAM overlays, the evaluate entry point), then 8.04.

Review: tier 2, `/review-session 8.03`

Review 8.03 fix (the user's decision, via the coordinator): both reviewers found that Grad-CAM is
hand-written while plan section 7, CLAUDE.md's stack line and item 8.02's description name the
`grad-cam` package (low). Kept the hand-written version. Plan section 7 now says Grad-CAM is
hand-written in `trailcam/evaluate/gradcam.py`, with no `grad-cam` package, and that the overlays
draw with the same function. R26 now says the measures run on val and trans_val before the final
scoring, which matches R17 and the command line, and that the overlays use the scored function.
Handed on:

- 8.02 must draw its overlays with `trailcam.evaluate.gradcam.gradcam` (its description, which
  names grad-cam, stays as written because items are not edited).
- 9.01 must call `attention_pass` outside `torch.inference_mode()`: Grad-CAM needs autograd and
  raises inside it.

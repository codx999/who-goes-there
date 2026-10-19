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

Review: tier 2, ran `/review-session 8.03` (0270f80..b28e75a; code-reviewer Opus xhigh,
risk-reviewer Fable high; both resumed once after a network error). One low, made by both: the
Grad-CAM drift fixed above. Scenarios R01, R03, R04, R05, R23, R26, R35, R38, R42 to R45 held;
known gaps passed: R04 and R35 the overlay writer (8.02), R26 the final scoring's calls (9.01), R23
the demo and the Space (10.01), R38 Kaggle's preinstalled matplotlib version. Fix 367f87f, after
merging main at e321151 (one conflict, in this log: 8.01's notes kept with its entry). The CLAUDE.md
Stack line was approved by the user directly in this session.

Fix round: `/review-session 3e8089c..HEAD 8.03` (CLAUDE.md the one high-stakes file). One low
from code-reviewer, fixed in the next commit: R26 asked the 8.02 overlays to draw the same
`gradcam` function but did not name 8.02, so a session that leaves `gradcam.py` alone would not
get the row. R26 now names item 8.02 and gives "none yet for the overlays drawing `gradcam`
(8.02)" in its test column. risk-reviewer found nothing in CLAUDE.md. Scenarios R01, R03, R23,
R26, R42 to R45 held. No third round.

Notes, recorded and not fixed (the user's decision, via the coordinator):

- 11.01: the Space imports `trailcam.evaluate.gradcam` for its optional overlay, not the
  `grad-cam` package. Item 11.01's steps, `docs/setup-playbook.md` and `docs/HANDOFF.md` still
  name the package. plan.md wins, and the item is not edited.
- gradio is in CLAUDE.md's stack but not yet in `pyproject.toml`: the demo item (10.01) adds it.

Merged main at 21faa79 (6.07) before this commit, through the sync tool. Two conflicts: this log
(6.07's entry first, since it merged first) and R03's test cell (6.07's `tests/test_cache.py` and
8.03's `tests/test_attention.py` both kept).

**Next.** 8.02 (overlays draw `trailcam.evaluate.gradcam.gradcam`), 8.04.

## 2026-10-14: session 8, item 8.04, day/night and common/rare slices and the feature map (Opus 5.5)

Taken ahead of its 10-18 date by the user's decision in the coordinator session, to start the
evaluation code early. Built on 8.03's branch (merged locally at 367f87f, its fix round
included) for matplotlib, `CAMERA_GROUPS`, `CLI_SPLITS` and `load_checkpoint`; no second copy of
the dependency. 8.03 must reach main before this branch does.

**Done.** `trailcam/evaluate/slices.py`:

- `predict_split(model, records, split, ...) -> SplitPredictions`: one eval-mode pass that keeps
  the label, prediction, night flag and location per image and, unless `keep_features=False`, the
  backbone features (the head's input, flattened: 512 for ResNet-18). Refuses records from another
  split or without a night flag (the sidecar must be loaded).
- `slice_rows`: per split (one camera group each) and per cut, the whole split and its two slices,
  day/night and common/rare, with images, classes present, accuracy and macro-F1 from
  `metrics.py` (macro-F1 over the classes present in the slice, R32).
- `mix_rows`: for val with trans_val and cis_test with trans_test, the trans accuracy re-weighted
  to the cis night share, `(1 - p) * day + p * night`, and the gap left at that common mix. This
  separates the cis-to-trans gap from trans_test's larger night share (0.688 vs 0.604 at 6.03).
  Only accuracy is re-weighted, since macro-F1 is not a weighted mean of its slices.
- `render_report` / `write_report`: `slices.md`, one section per checkpoint with a day/night
  table and a common/rare table (split, camera group, counts, share, accuracy and macro-F1 for
  all and each slice, classes present), plus the same-mix table.
- `pca_2d` (centred, `torch.linalg.svd` in float64, each component's sign fixed by its largest
  loading, explained variance share), `project` (one PCA per checkpoint over all splits given, so
  cis and trans share a map), `write_feature_maps`: `features_by_species.png` and
  `features_by_camera.png`, each with one block of small panels per checkpoint side by side
  (arm A beside arm B), one panel per species or per camera location with its images highlighted
  over the rest in grey (blue for cis cameras, orange for trans). I used small panels because 12
  species or up to 20 cameras in one scatter cannot be told apart by colour.
- Command line: `--checkpoint CONFIG_NAME PATH` (repeat per arm), `--data-root`, `--split` (val
  and/or trans_val, both by default), `--out`; `--dry-run=True` uses the fixture and two
  random-weight models. One checkpoint per model and arm; the configs must share a split seed.
  The header gives each checkpoint's sha256 and the split hash.

Common vs rare rule, fixed now, before any test scoring (R17), and not tuned on results: a class
is rare when it holds fewer than 4% of the seed-42 train images (`docs/data-card.md`, 13,587).
Rare: bird (509, 3.7%), rodent (440, 3.2%) and skunk (232, 1.7%), the same three the research
report calls rare. The next class, car (603, 4.4%), is common. `RARE_CLASSES` is a constant, and a
test re-derives it from the data card's train column.

Test-set isolation: the command line accepts val and trans_val only (argparse choices plus a
second check) and drops the `Split` before the pass. 9.01 produces the slices and feature maps
of the test images by calling `predict_split`, `write_report`, `project` and
`write_feature_maps` directly.

Steps: `uv run pytest tests/test_slices.py` 20 passed. Slice scores are hand-computed. Day + night
and common + rare add up to the split, both on the data and in the dry run's written table. An
empty slice shows n/a. The same-mix row is hand-computed. PCA recovers known components with
explained shares 0.9 and 0.1. `predict_split` matches the model's argmax and the head input, and
removes its hook. The dry run writes `slices.md` and two PNGs under `--out` and nothing under
`docs/figures/`. A real run on the fixture laid out as the cache, with a small CNN and a ResNet-18
arm A checkpoint, names both sha256s. The command line refuses cis_test, trans_test and train
(exit 2, nothing written) and reads only the splits given. No image id, sequence id or file name
appears in the table or the printout. Each of these mutations fails a test: car added to the rare
classes, re-weighting by the trans share, no sign fix, no centring, trans_test added to the CLI
splits, the whole split's macro-F1 for every slice, truncated features, the hook left on, and every
image counted as rare. `check.sh` green (651 passed).

`docs/review/invariants.md`: no row names 8.04, so no test cell names it. `tests/test_slices.py` is
added to the test cells of R03 (no ids), R04 (PNGs under `--out` only), R17 (CLI refuses test
splits; the rare rule comes from train counts) and R32 (accuracy beside macro-F1 per slice). Not
changed, for the user to decide: the "where" columns of R17 and R32 do not list
`trailcam/evaluate/slices.py` or 8.04, so `/review-session` selects those rows only through
`trailcam/**` (R03) and `trailcam/evaluate/**` (R05).

For 9.01: call `predict_split` once per checkpoint and test split, and pass the results to both
`write_report` and `project`. A small-CNN pass on both test splits with features kept holds about
36k x 25,088 floats, so pass `keep_features=False` there; the feature map is for the ResNet-18
arms.

**Next.** 8.02 (figures, Grad-CAM overlays drawn with `trailcam.evaluate.gradcam.gradcam`, the
evaluate entry point).

Review: tier 2, ran `/review-session 8.04` on 37ff282..138dd07, the 8.04 commit only (the user
chose to leave out 8.03's commits, which were merged here and reviewed in 8.03's session).
code-reviewer ran on Opus at xhigh and risk-reviewer on Fable at high. Two lows, both from
code-reviewer, both fixed in d8e06e6 (the user's decision, via the coordinator):

- Invariants map: no "where" column routed `trailcam/evaluate/slices.py` to R04, R17 or R32.
  Those three rows now list the path and item 8.04, and `tests/test_risk_files.py` stays green.
- Mix table: the cis-trans comparison at the same day/night mix reported accuracy only. It now has
  an accuracy row and a macro-F1 row per pair. Each trans image is weighted `p / q` at night and
  `(1 - p) / (1 - q)` by day, with p the cis and q the trans night share. Both scores come from
  the weighted counts, and macro-F1 averages over the classes with a positive weighted support.
  The weighted accuracy equals the earlier `(1 - p) * day + p * night`. The module docstring now
  says macro-F1 is re-weighted inside the counts, since it is not a weighted mean of its slices.
  New functions: `weighted_scores` and `mix_weights`; `MixRow` now has `metric`, `cis` and `trans`.
  Tests: hand-computed one-class and two-class cases; unit weights reproduce `metrics.py` on 200
  random cases; the weights average 1 and put the cis share at night; bad input is refused. Seven
  mutations of the new code each fail a test. `tests/test_slices.py`: 24 passed.

risk-reviewer found nothing. Scenarios R01, R03, R04, R05, R17, R32, R42 to R45 held, except R32
(partial for the mix table, the second low) and R44 (the map gap, the first low). Known gaps
passed: R17 and R32 the final scoring (9.01), R04 the figure and Grad-CAM writers (8.02).

Fix round: `/review-session 138dd07..d8e06e6 8.04`, same reviewers, models and effort. Nothing
found by either; every scenario held. No third round.

Notes, recorded and not fixed (the user's decision, via the coordinator):

- 9.01: `predict_split` keeps the backbone features by default. A small-CNN pass on cis_test plus
  trans_test (14,478 + 21,473 images, 25,088 features each) would need well over 20 GB for the
  float64 copy and the SVD in `pca_2d`. 9.01 passes `keep_features=False` for the small CNN and
  projects the ResNet-18 arms only, which is what plan.md section 7 asks for.
- matplotlib floor: `pyproject.toml` declares `matplotlib>=3.8.4`, a floor checked with
  `tests/test_attention.py` only. `slices.py` also uses `Figure.subfigures(squeeze=False)` and
  constrained layout with subfigures, and its tests ran on the locked 3.11.2 only.
- Comment slip in `tests/test_slices.py`: the macro-F1 comment in
  `test_the_trans_scores_are_reweighted_to_the_cis_night_share` reads "FN 0.5 + 2 x 0.5 = 2.5";
  the day miss weighs 1.5, so it should read "1.5 + 2 x 0.5". The asserted value, 3 / 5.5, is
  right. Left as it is, since a test edit would need a third review round.

Merged main at 0968e1a (6.07 and 8.03) through the sync tool. Two conflicts: this log (8.03's
review lines kept with its entry, then this entry) and R03's test cell (main's
`tests/test_cache.py` and this item's `tests/test_slices.py` both kept). `uv.lock` merged without
a conflict. `check.sh` green after the merge (669 passed).

## 2026-10-15: session 6, item 6.08, the cache as one archive, unpacked to local disk (Opus 5.5)

New item, approved by the user in the coordinator session on 2026-10-15 and added to
`docs/progress.json` as 6.08 (session 6, owner claude, due 2026-10-16). Evidence: the 6.04
small-CNN version (Save & Run All, GPU T4 x2, commit 21faa79, `cct20-512` attached) showed
"Starting..." for about 48 minutes, "Running for 2892.9s", an empty Logs tab and 0 B output,
and the GPU quota fell by about that much, so a committed run is billed while it starts. Leading
explanation: a committed run copies every input file before the first cell, and the cache is
57,864 small files (2.92 GiB). This item blocks every GPU run.

**Done.**

- Archive: `scripts/kaggle_pack_cache.py` (new; in the mypy list of `pyproject.toml` and
  `check.sh`) packs the attached `cct20-512` dataset into `/kaggle/working/cct20_512.tar`, an
  uncompressed tar with `annotations/`, `images/` and `day_night.json` under `cct20_512/`. Before
  writing anything it checks that the cache is whole: the five annotation files, the sidecar with
  one entry per annotated image, and in `images/` exactly the 57,864 annotated JPEGs, none empty.
  Members are written in sorted order with zeroed owners and times, so the same cache gives the
  same archive, and its SHA-256 is printed. It writes to a `.part` file and renames it at the end.
  Output is counts only (R03).
- Decision: packing, not a `--archive` option on `kaggle_prepare_cct20.py`. Packing keeps the
  exact bytes the sidecar's day/night flags were measured on and the rehearsals checked. A
  rebuild would mean the 6.5 GB download and 57,864 decodes and re-encodes again, possibly under
  a newer Pillow, giving a different cache. A committed CPU run copies every input before its
  first cell, so the lazy mount cannot hand the packer a partial cache, and that wait costs no
  GPU quota. If the copy is short anyway, the completeness check refuses with counts.
- Unpack: `trailcam/data/cache.py` gains `unpack_cache(archive, dest)`. It checks every member
  before writing: regular files and folders only (symlinks, hard links and special files are
  refused), no absolute name, no `..`, nothing outside `cct20_512/`. It also refuses a `dest`
  under `/kaggle/working` (would be saved as output), an existing `dest/cct20_512` and too little
  free space. It extracts with tarfile's `data` filter and returns `Unpacked(root, files, size,
  seconds)`. A new `archive_files(root)` lists the `.tar` files under a root (any depth up to 8)
  when the path is wrong. Refusals name kinds and counts, never a member name.
- Notebook (outputs cleared): new parameters `CACHE_ARCHIVE =
  Path("/kaggle/input/datasets/codx888/cct20-512-tar/cct20_512.tar")` and `LOCAL_CACHE =
  Path("/tmp/trailcam-cache")`. `/tmp` was chosen because 6.03 held 6.5 GB there; `/kaggle/temp`
  is not verified. A new cell after the GPU check and before the cache cell unpacks the archive,
  prints the counts and seconds, and sets `DATA_ROOT = unpacked.root`; a wrong path lists the
  `.tar` files under `/kaggle/input`. The cell sits after the import guard (the first `trailcam`
  import must stay guarded) and after `check_gpu` (seconds, so a wrong session fails before the
  unpacking). The loose-file path is kept: `CACHE_ARCHIVE = None` reads `DATA_ROOT` as before.
  Section 2 now refuses a commit whose `cache.py` lacks `unpack_cache` ("predates item 6.08").
  The default `COMMIT` is 0968e1a, main at this session's start, which is refused there. How to
  run: attach `cct20-512-tar`, not the loose dataset; edit only `COMMIT` (and `CACHE_ARCHIVE`);
  the Logs tab shows the unpacking and one line per epoch.
- Visible progress: `trailcam/train/train.py` logs through `log_line`, which prints with
  `flush=True` (the default `log` of `fit` and `train_run`). Each epoch's line now ends with
  `; <s> s, <n> train images/s`: the seconds of train plus val, and the training images per
  second. The ledger row is unchanged.
- `docs/plan.md` section 2 (Preprocessing) records the archive and why. In
  `docs/review/invariants.md`, R03's test cell names the new tests; no row names 6.08.

Verification: `bash scripts/check.sh` green (671 passed, 26 of them new). New tests:
`tests/test_cache.py` (a normal tar unpacked with counts; a traversal (two forms), an absolute
name, a member outside the top folder, a relative and an absolute symlink, and a hard link each
refused before anything is written; a destination under `/kaggle/working`, an existing tree and
an archive without the top folder refused; the archive finder at the Kaggle mount depth).
`tests/test_kaggle_pack.py` (the fixture packed and unpacked gives the same files and bytes and
passes `check_images`; the same cache gives the same SHA-256 after a change of mtimes; a missing,
empty or extra image, a short sidecar, a wrong count or a missing annotation file is refused with
counts only and no file is left; `main` refuses outside Kaggle and an existing output).
`tests/test_train.py` (the epoch line's format with seconds and images/s; `log_line` flushes and
is the default). `tests/test_notebook.py` (the new parameters, one reassignment of `DATA_ROOT`,
the 6.08 refusal, the cell order, and the unpacking cell executed on a tiny tar for the archive,
`None` and a missing path). `uvx --from nbconvert jupyter nbconvert --clear-output --inplace
notebooks/kaggle_train.ipynb` leaves the notebook unchanged. `uv run python
scripts/guard_data.py --all` passes. `/dryrun` passes: epoch 1/1 train loss 2.6539, val loss
2.4667, logits (2, 12), "16.1 s, 28.7 train images/s", the same 16-column row, ledger untouched.

Not verified (Kaggle only): whether "New Dataset" from a notebook output keeps a `.tar` as one
file (the steps below say to stop if the Data tab shows folders); the exact mount path of the
archive (the finder lists it if the default is wrong); how long a GPU committed run takes to
start with one 2.9 GiB input, and how long the unpacking takes (both printed).

For the user, once, to make the archive (also in `scripts/kaggle_pack_cache.py`):

1. New Notebook; Accelerator None (CPU), Internet on.
2. Add Input: the private `cct20-512` dataset.
3. One code cell, `<COMMIT>` the merge commit of this item:
   `!git clone --quiet https://github.com/codx999/who-goes-there /tmp/who-goes-there`,
   `!git -C /tmp/who-goes-there checkout --quiet --detach <COMMIT>`,
   `!pip install --quiet /tmp/who-goes-there`,
   `!python /tmp/who-goes-there/scripts/kaggle_pack_cache.py`. The clone goes to `/tmp`, so
   the output is the one tar.
4. Save Version > Save & Run All; wait for it to finish. The log ends with the counts, size,
   seconds and SHA-256.
5. Output > New Dataset, slug `cct20-512-tar`, Private. The Data tab must show one file,
   `cct20_512.tar`, about 2.9 GiB. Note the SHA-256 and size here.

Then every GPU run (6.04, 6.05, 7.02 and on) attaches `cct20-512-tar` and sets `COMMIT` to this
item's merge commit, which the coordinator gives. The training COMMIT for every GPU run is now
that merge commit.

For the learning journal (`docs/learning-journal.md` is not on main yet; Kaggle topic):

- Lesson: on Kaggle a committed run copies every input file before its first cell, so thousands
  of small files make it slow to start, and the start is billed to the GPU quota.
- Noticed by: the 6.04 version sat at "Starting..." for about 48 minutes with an empty Logs tab
  and 0 B output, and the quota page fell by about that much.
- Fix: ship the cache as one uncompressed tar (`scripts/kaggle_pack_cache.py`, on a CPU session)
  and unpack it to `/tmp` at the start of the run (`trailcam.data.cache.unpack_cache`, the
  notebook's unpacking cell). Every log line is flushed, with seconds per epoch, so a running
  version shows its progress.

Main had moved to c303041 (8.04) during this session; merged after the item commit (e1fe83f)
through the sync tool, since the tool refuses uncommitted changes. Two conflicts: this log (8.04's
entry first, since it merged first, then this one) and R03's test cell (8.04's
`tests/test_slices.py` and this item's `tests/test_cache.py` and `tests/test_kaggle_pack.py` all
kept). `check.sh` green after the merge (695 passed).

**Next.** The user packs the archive (steps above), then 6.04 on `cct20-512-tar` at this item's
merge commit.

Review: tier 2, `/review-session 6.08`

Review: tier 2, ran `/review-session 6.08` (c303041..ef9100d; code-reviewer Opus xhigh,
risk-reviewer Fable high). Two lows, both from code-reviewer; risk-reviewer found nothing. All of
R01, R03, R05 to R07, R14, R16, R17, R25, R27, R31, R32, R37, R38 and R40 to R45 held. The user
merged e1fe83f and ef9100d into main as they stood, so the pack script reached GitHub and the
packing run could start; the fixes are new commits on top, per the user's decision relayed by the
coordinator.

- Low: nothing caught the loose `cct20-512` dataset still being attached next to the archive.
  Fix 67b61ce: the unpacking cell refuses while a folder holding a `day_night.json` is under
  `/kaggle/input` and `CACHE_ARCHIVE` is set, and How-to-run step 2 says to remove that input.
  The input folder is now the constant `INPUT` in the parameter cell (not a parameter), so the
  tests can run the cell on a temporary folder.
- Low: `unpack_cache`'s free-space check ignored block rounding (about 100 to 150 MB over 57,864
  files), so a disk-full error could stop it partway and name one image path (R03). Fix 67b61ce:
  `space_needed` rounds each file up to the disk's block size, counts one block per folder and
  adds a 64 MiB margin (`UNPACK_MARGIN`). The check now runs on the nearest existing folder,
  before `dest` is created. A test sets the free space just above the files' bytes (refused,
  nothing written), one byte short of the blocks and margin (refused), and exactly that (enough).

Fix round: `/review-session ef9100d..HEAD 6.08` (67b61ce; files: the notebook, `cache.py`,
`tests/test_cache.py`, `tests/test_notebook.py`). One low from code-reviewer; risk-reviewer found
nothing and noted the same limitation. All of R01, R03, R06, R07, R16, R17, R27, R37, R38 and R41
to R45 held.

- Low: the refusal also fired in a committed run, whose start is already paid for, so it threw
  away the run and saved nothing. Fixed in 18f7e6a, as the playbook asks for fix-round findings:
  the cell refuses only when `KAGGLE_KERNEL_RUN_TYPE` is `Interactive` (the rehearsal); any other
  value, `Batch` or unset, warns on stderr and unpacks. The cell prints the run type. The
  variable's name and values come from memory: I could not confirm them online. A wrong guess
  only means the check warns instead of refusing, so a committed run is never lost. The next
  rehearsal's printed "run type" line confirms it. No third round, as none was requested.
- Note (risk-reviewer): mypy strict run directly on `tests/test_cache.py` flags
  `cache_module.shutil` as not explicitly exported. Tests are outside the mypy `files` list, so
  `check.sh` and CI are unaffected.

New evidence (the user, via the coordinator, 2026-10-15): the user's CPU packing run (Save & Run
All, no accelerator, the same 57,864-file `cct20-512` attached, commit ef9100d) printed its first
log line about 5 minutes after submission. So copying the input files does not by itself explain
the 48-minute "Starting..." of the 6.04 GPU run. The cause is uncertain, and the likelier one is
the GPU session itself (waiting for a T4, or a GPU container stuck starting), billed throughout.
Plan.md section 2 (not a high-stakes file) now says the cause is not known and gives the practice
that protects the quota whatever the cause: about 10 minutes after Save & Run All, open the Logs
tab; if it is still empty and the version still shows "Starting...", cancel it and resubmit.
Deferred to a later item, by the coordinator's correction: the same change in high-stakes files
would need another review round after the fix round. These still state the input copy as the
cause: How-to-run step 2 of `notebooks/kaggle_train.ipynb`, the module docstrings of
`trailcam/data/cache.py` and `scripts/kaggle_pack_cache.py`, and the reason in the unpacking
cell's message. The 10-minute practice is not yet in How-to-run step 6. This log commit changes
only `docs/plan.md` and this log. `check.sh` green.

The user's packing run (the coordinator, 2026-10-15). CPU, Save & Run All, commit ef9100d, 17
minutes in all, the first log line about 5 minutes after submission. It printed:

- "Packing 57870 files (2.95 GiB) from /kaggle/input/datasets/codx888/cct20-512/cct20_512".
- Progress every 10,000 files, at about 40 s each.
- "Archive /kaggle/working/cct20_512.tar: 57870 files (annotations 5, images 57864,
  day_night.json 1) under cct20_512/".
- "3213506560 bytes (2.99 GiB), 377 s".
- "sha256 724ec52cd59ecd96b752347449dd94932eb984355c700e16c8f8234a481081ed".

The user is creating the private dataset `cct20-512-tar` from that output. Its Data tab should
show the one file, `cct20_512.tar`, at 3,213,506,560 bytes.

For the learning journal (`docs/learning-journal.md` is not on main yet; Kaggle topic; these
replace the lines in the 6.08 entry above):

- Lesson: on Kaggle a committed version is billed from submission, including any time it sits at
  "Starting..."; why it can sit there is not always clear.
- Evidence: the 6.04 GPU version (57,864 small input files) showed "Starting..." for about 48
  minutes with an empty Logs tab, and the quota fell by about that much. A CPU version with the
  same files logged its first line in about 5 minutes, so the input copy alone does not explain
  it; a slow GPU session start (a T4 queue or a stuck container) is likelier.
- The archive still helps: one 2.9 GiB tar instead of 57,864 files starts faster, and training
  reads local disk (`scripts/kaggle_pack_cache.py`, `trailcam.data.cache.unpack_cache`).
- Practice: about 10 minutes after Save & Run All, check the Logs tab. If it is still empty and
  the version still shows "Starting...", cancel and resubmit. Every training log line is flushed,
  with seconds per epoch, so an empty log does mean that no cell has started. The practice is
  not yet in the notebook's How-to-run step 6; a later item adds it.

The training COMMIT for every GPU run is the merge of this final state into main; the
coordinator gives it.

**Next.** The user creates `cct20-512-tar` from the packing run's output (the steps in
`scripts/kaggle_pack_cache.py`), then 6.04 on it at the coordinator's COMMIT.

## 2026-10-15: session 8, item 8.02, the evaluation figures, the Grad-CAM overlays and the entry point (Opus 5.5)

Taken ahead of its 10-18 date by the user's decision in the coordinator session. Main at 0968e1a
(8.03 merged); 8.04 (`slices.py`) is reviewed on its branch but not on main, so it is not imported
here (see "8.04" below).

**Done.** `trailcam/evaluate/figures.py`, `trailcam/evaluate/evaluate.py`, the overlays added to
`trailcam/evaluate/gradcam.py`, `per_camera_scores` added to `trailcam/evaluate/metrics.py`;
`tests/test_figures.py` (21 tests) and two tests in `tests/test_metrics.py`. Rules, stated in the
module docstrings:

- `figures.py` takes scores, never images (R03, R04): the confusion matrix (rows true, columns
  predicted, `CLASSES` order; counts in the cells, shaded by share of the row so rare classes
  read; an absent class's row is empty and labelled "absent"), per-class bars (precision, recall,
  F1 for the classes present, support under each name) and per-camera bars. The per-camera bars
  show accuracy beside macro-F1 (R32), worst accuracy first, with the split's accuracy as a dashed
  line; the item says "accuracy bars", and R32 asks for macro-F1 wherever a score is shown.
  Colours from the reference data-viz palette: one blue ramp for the matrix, categorical slots 1
  to 3 for the bars.
- `per_camera_scores` (metrics.py): accuracy and macro-F1 per camera location, each location
  scored on its own images, so macro-F1 averages over the classes present there.
- Overlays (`gradcam.py`): `overlay` and `write_overlay` draw `trailcam.evaluate.gradcam.gradcam`,
  the function attention-in-box scores (R26), on the photo's eval-transform output; the frame
  shown is that output with the normalisation undone, so it is the 224 x 224 the model saw. For
  display the map is divided by its maximum (attention divides by its sum: same shape) and
  blended in orange with opacity up to 0.6; zero-map pixels are untouched. The PNG is the frame
  beside the overlay, titled with the prediction, its confidence and, when a class is given,
  correct or wrong. `assets_image` refuses any photo not under a folder named `assets` (the repo's
  `assets/` by default), including one reached through `..` or a symbolic link, and its message
  does not echo the path (R03, R04, R35). The repo's `assets/` was not created (8.05).
- `evaluate.py`: `python -m trailcam.evaluate.evaluate --config <name> [--checkpoint <run>.pt
  --data-root <cache>] --split val [--split trans_val] --out <dir> [--overlay
  assets/<photo>[=<class>] ...] [--assets <dir>] [--config-file <config.yaml>]`. One eval pass
  per split; `metrics.md` (per split: accuracy, macro-F1, classes present; per-class, per-camera
  and confusion-matrix tables) and `<split>_confusion_matrix.png`, `<split>_per_class.png`,
  `<split>_per_camera.png` under `--out`; overlays as `gradcam_overlay_<n>.png`, numbered, so no
  photo file name (possibly an image id) reaches the output. A `fixture: true` config (dryrun)
  uses the synthetic fixture and random weights; any other needs `--checkpoint` (loaded with
  8.03's `load_checkpoint`, which refuses another model or class map) and `--data-root`. The
  header gives the checkpoint sha256, split seed and split hash, as attention.md does.
- Test-set isolation (R17): `--split` takes `val` and `trans_val` only (exit 2 otherwise); the
  split object is deleted once the given splits' records are taken. 9.01 calls `predict`,
  `write_report` and `write_overlays` directly.

Decision to review: the item's step spells the config as `--config dryrun`, so here `--config`
is the config name (`--config-name` is accepted as an alias) and the YAML path is
`--config-file`. In `train.py`, `attention.py` and `slices.py`, `--config` is the YAML path. A path
given to `--config` here fails with "no config named ...", exit 2, so a mix-up cannot run the
wrong config; but the flag means two things across the four entry points.

8.04: `evaluate.py` does not run attention (8.03) or the slices (8.04). Each has its own entry
point with its own `--out`; slices takes arm A and arm B together, which a one-checkpoint entry
point cannot. 8.04's `SplitPredictions` carries labels, predictions and locations, so a caller
holding one scores it here with `SplitResult(run.split, run.labels, run.predictions,
run.locations)`, no second pass (the hook is stated in the docstring).

Steps: `uv run python -m trailcam.evaluate.evaluate --config dryrun --split val --out <scratch>`
exit 0, writes `metrics.md`, `val_confusion_matrix.png`, `val_per_class.png`,
`val_per_camera.png`, and `docs/figures/` does not exist before or after. `uv run pytest
tests/test_figures.py` 21 passed. The figures were looked at on the fixture and on a made-up
1,500-image trans_val score set (scratchpad only, not committed). Each of these mutations fails a
test: no assets containment check, any root name accepted, the overlay map squared or taken on
a flipped image, the frame denormalised without the mean, cameras ordered by id, an absent row
divided by its zero total, the overlay paths checked only after evaluating, `cis_test` added to
`CLI_SPLITS`, trans_val read when only val is asked for, and per-camera macro-F1 computed over
the whole split. The last two first survived and the tests were tightened. `check.sh` green
(668 passed). Invariants: R04 and R26 (the rows naming 8.02) have their test cells filled; R03
and R35 now cite `tests/test_figures.py`.

Process note: during the mutation checks a helper restored files with `git checkout`, which
reset the uncommitted `gradcam.py` to HEAD; it was restored from a copy taken just before, and
the two mutations it left in untracked files were undone by hand. The final diff was re-read and
the full suite rerun afterwards.

**For 8.06: an evaluation notebook is needed (not built; needs your OK).**
`notebooks/kaggle_train.ipynb` cannot take it: `tests/test_notebook.py` (R17) forbids an
evaluation call in any of its code cells. A separate `notebooks/kaggle_evaluate.ipynb` would:

1. install `trailcam` from the public repo at a pinned commit, and clone the repo at the same
   commit for `config.yaml` and `assets/` (8.05 must be merged first; a wheel ships neither);
2. attach the 512 px cache dataset and the checkpoints (the training runs' outputs as a Kaggle
   dataset), and check each checkpoint's sha256 against its ledger row;
3. per checkpoint: `python -m trailcam.evaluate.evaluate --config <name> --config-file
   <clone>/config.yaml --checkpoint <run>.pt --data-root <cache> --split val --split trans_val
   --out /kaggle/working/eval/<run>`, plus `--overlay <clone>/assets/<photo>=<class>` four times
   and `--assets <clone>/assets` on the checkpoint the overlays come from; then
   `python -m trailcam.evaluate.attention` with the same checkpoint;
4. once per seed: `python -m trailcam.evaluate.slices` with the arm A and arm B checkpoints;
5. zip `/kaggle/working/eval` for download. You then copy the chosen PNGs into `docs/figures/`
   (40 files, 20 MB in all; one checkpoint's val set is 3 PNGs + 4 overlays + the attention
   histogram + 2 feature maps).

It would come with a test like `tests/test_notebook.py`: no code cell names cis_test or
trans_test, outputs cleared.

Notes, not fixed:

- 11.01: `gradcam.py` now imports matplotlib (for `write_overlay`) and torchvision's eval
  transform. If the Space imports it for its optional overlay, `space/requirements.txt` needs
  matplotlib; `overlay()` returns PIL images, which Gradio can show without a PNG.
- 9.01 must call `write_overlays` (like `attention_pass`) outside `torch.inference_mode()`.

**Next.** Review 8.02. Then 8.04 merge, 8.05 (your photos into `assets/`), and the 8.06
notebook if you approve it.

Review: tier 2, `/review-session 8.02`

Review: tier 2, ran `/review-session 8.02` (0968e1a..d3ee3ca; code-reviewer Opus xhigh,
risk-reviewer Fable high). risk-reviewer found nothing. code-reviewer found one low: R17's
"where" column did not name `evaluate.py` or `attention.py`, so a later diff to either alone would
not bring the test-isolation scenarios to a reviewer. It also raised a note for 8.06: the
overlay's left panel was the unblended eval crop of the `assets/` photo, which would have put a
raw dataset image into `docs/figures/`. Scenarios R01, R03, R04, R05, R17, R26, R32, R33, R35
and R42 to R45 held. Known gaps passed: R04 and R35 the real photos and the `docs/figures/` copy
(8.05, 8.06), R26 and R32 the final scoring's calls (9.01), R35 the demo and the Space.

Fix 79b9594 (the user's decisions, via the coordinator):

- R17's "where" column now names `evaluate.py`, `attention.py` and item 8.02, and R32's
  `evaluate.py`. The user approved the rule-map change.
- The overlay PNG is now one panel, the blended image, and `Overlay` no longer keeps the
  unblended frame. A test captures the saved figure and checks it has one axes holding the
  blended image. This supersedes "the eval frame beside the overlay" above.
- 8.04 merged first (c303041). I did not fold its slices into the val report: slices takes arm A
  and arm B together, and `evaluate.py` works on one checkpoint. A test shows that 8.04's
  `SplitPredictions` scores in `evaluate.py` without a second pass. That is the hook for 8.06.

Before the fix I merged main twice through the sync tool: c303041 (8.04) and ef9100d (6.08).
Both times the conflicts were in this log and in the test cells of R03 and R04. Both sides were
kept, and the other items' entries come first.

Fix round: `/review-session cb258d4..79b9594 8.02` (the gradcam.py and test_figures.py the
high-stakes files). Both reviewers found the same low independently. A map that is zero
everywhere left the photo unblended, so the one-panel PNG would have been the raw photo.
code-reviewer found a second low: R17's and R32's test cells did not name the tests guarding the
newly listed modules. Both were fixed in c8a9a08, after merging main at 61f1ecc (6.08's review;
the one conflict was in this log). The fixes:

- `overlay` raises `ZeroMapError`, with no path in the message, and no PNG is written. The test
  fails if the refusal is removed.
- R17's test cell now names the split refusals in `tests/test_figures.py` and
  `tests/test_attention.py`. R32's names `per_camera_scores` and `test_render_metrics_by_hand`.

Scenarios R01, R03, R04, R05, R17, R26, R35 and R42 to R45 held. No third round. `check.sh`
green (724 passed).

Notes, recorded and not fixed:

- mypy on tests: `tests/` is outside `check.sh`'s mypy scope. Run there, it reports 4 `arg-type`
  errors at `tests/test_metrics.py:180-181` (`pytest.approx` in a `CameraScores`) and 3 at the
  `Figure.savefig` wrapper in `tests/test_figures.py`. The tests are correct at runtime. These
  matter only if mypy is widened to `tests/`.
- 8.06 needs `notebooks/kaggle_evaluate.ipynb` (described above). It still needs the user's OK.
- 11.01: `gradcam.py` imports matplotlib. `overlay()` returns the blended PIL image, which Gradio
  shows without a PNG.

For the learning journal (`docs/learning-journal.md` is not on main yet):

- A figure can leak data through the panel beside the one you look at: the overlay's raw left
  panel would have put a dataset photo into `docs/figures/` with every test still green.
- "Never the raw photo" needs the degenerate input too: a zero heat map blends to nothing, so
  the guard has to refuse it, not just draw it.
- A review map only works if the rows name the files: a new module that enforces a rule must be
  added to that rule's "where" and test columns, or later diffs to it skip the scenario.
- Restore mutation tests by copying files, never with `git checkout`: it silently reset
  uncommitted work, which a backup taken just before recovered.

**Next.** The user merges this branch. Then 8.05 (photos into `assets/`) and the 8.06 evaluation
notebook, if approved.

## 2026-10-15: session 8, item 8.07, the data charts (Opus 5.5)

New item, approved by the user in the coordinator session on 2026-10-15 and added to
`docs/progress.json` as 8.07 (session 8, owner claude, due 2026-10-17). Why: the data was
inspected and cleaned in numbers and tests (items 3.01 to 3.05) but no chart existed; the user
wants the exploration visible, for learning and for the presentation's data slide.

**Done.** `trailcam/data/datavis.py` (new). Placed in `trailcam/data/`, not `trailcam/evaluate/`:
it reads the metadata and the split only, no model, prediction or image, so it belongs with the
data code and keeps imports one way (`evaluate` uses `data`, never the reverse); `.claude/rules/data.md`
covers both folders. `python -m trailcam.data.datavis --root <dir> --seed 42 --out <dir>
[--sidecar day_night.json]` writes five PNGs and prints counts only:

1. `class_counts.png`: each class's share of each split with its count (opossum 20% of train;
   raccoon 25% of trans_test against 7% of train; trans_val lacks squirrel, car and rodent).
2. `camera_locations.png`: images per camera stacked by split, cis vs trans. Cameras are labelled
   C1 to C10 and T1 to T10 by image count; location ids are never drawn (R03). One cis camera
   holds 9,579 kept images, the smallest 891; trans_val is a single camera.
3. `day_night.png`: night share of the kept images per split and kept images per capture hour by
   day and night, hour 12 annotated. At first no `day_night.json` was in the main checkout's
   `data/`, so the chart drew the 6.03 cache build's numbers (all 57,864 annotated images before
   the drop rules, per official file). Then the coordinator reported that the user had copied the
   6.03 sidecar to `data/day_night.json` (gitignored, never committed) and asked for kept-image
   shares, with both sets in the data card. The committed chart is now drawn with `--sidecar`.
   Kept images, seed 42: train 8246/13587 = 0.607, val 1436/2394 = 0.600, trans_val 1014/1603 =
   0.633, cis_test 9066/14478 = 0.626, trans_test 15384/21473 = 0.716, all 35146/53535 = 0.657;
   hour 12 has 2,990 night frames of 4,525. These match the coordinator's counts. Without a
   sidecar the chart falls back to the 6.03 numbers and its subtitle says so.
4. `box_fill.png`: box width x height as fractions of the image, log bins, cis vs trans, arm B's
   0.2 to 1.0 range shaded. 93% of the 12,177 train boxes cover less than 20% of the image (median
   near 2 to 3%), so arm B's crop zooms in on almost every boxed image it takes; a second bump
   near the whole frame.
5. `burst_length.png`: sequences and kept images by `seq_num_frames` (4,787 single frames,
   15,969 bursts of 3, 196 of 5); 48,701 of the 53,535 kept images (91%) share their burst with
   another kept image, and 222 bursts have frames in both official train and cis_val.

Every chart carries the Beery et al. 2018 citation and the licence in a footer. Colours: the
dataviz reference palette, one colour per split across all charts, validated with its checker
(adjacent pairs pass; yellow is below 3:1 on white, so every yellow bar has a label or legend).
`ARM_B_FILL` is a copy of `boxcrop.FILL` because only the training transform may import
`boxcrop` (R24); a test checks the two agree. The 6.03 numbers are constants
(`CACHE_BUILD_NIGHT_BY_FILE`, `CACHE_BUILD_NIGHT_BY_HOUR`).

The five PNGs are committed in `docs/figures/` (384 KB of the 20 MB budget, 5 of 40 files),
generated from the real metadata and the sidecar with seed 42. `docs/data-card.md`: "Day and
night" now gives, first, the kept images per split and per hour (sidecar, seed 42) and, second,
the 6.03 build's numbers per official file and per hour, labelled as all annotated images before
the drop rules; the channel-spread gap and the 551 single-band images; and the hour-12 note. A
"Figures" section lists the charts and the command. Item 6.03 stays the user's and is not
flipped.

`notebooks/kaggle_look_at_data.ipynb` (new, outputs cleared): a separate notebook rather than a
section of `kaggle_train.ipynb`, because the training notebook is a Save & Run All batch run whose
tests forbid definitions and extra calls, and a photo grid there would run and be stored with every
training version. It installs `trailcam` from the public repo at `COMMIT`, loads the metadata with
the sidecar, shows a grid of train photos per class, day vs night (titles: class and day or night
only), then runs the datavis CLI with the sidecar into `/kaggle/working/figures` for the
kept-image night shares. Its markdown says never to commit it with outputs (the guard refuses
them anyway, R41) and that photos never go into `docs/figures/`. Only the train split is read for
photos (R17). Smoke-run locally on the synthetic fixture with the install stubbed: every cell ran.

`tests/test_datavis.py` (new, 18 tests): the CLI on the fixture (with and without the sidecar)
writes exactly the five PNGs under `--out` and nothing under `docs/figures/`, from a fixture with
no image file; no drawn text or printed line holds an image id, seq id or file name, and on a
fixture with locations shifted to distinctive numbers no location id is drawn (CCT20's own
location ids are small numbers that collide with counts, so they are checked there); camera tick
labels are C1, C2, T1; every chart has the citation; class counts, camera totals and ranking,
night counts from the sidecar, the fallback to the 6.03 numbers, box fill = w x h, burst counts;
the 6.03 numbers sum to 57,864 and 36,776 and hour 12 is (3120, 5077); on the real metadata the
summary gives 53,535 images, 10 + 10 cameras, 47,729 boxes, 20,952 sequences, 222 shared, 48,701
with a sibling, and the raw annotation files hold the 6.03 build's images per file and per hour;
the look notebook has no outputs, says never to commit them, cites the data and reads the train
split only. Mutations that each fail a test: location ids as labels, no citation, big boxes
dropped, singletons counted as siblings, cameras unranked, a typo in hour 12, the sidecar ignored.

`docs/review/invariants.md`: `trailcam/data/datavis.py` added to the "where" of R03 and R04 (and
8.07 to R04's items); test cells of R03, R04, R06, R17 and R41 name `tests/test_datavis.py`.

Steps: the CLI wrote the five PNGs into a tmp `--out` on the fixture (tests) and into
`docs/figures/` on the real metadata; `uv run pytest tests/test_datavis.py` 18 passed (the 18th
pins the kept-image night counts from the sidecar in `data/`, and fails locally if it is missing);
`uv run python scripts/guard_data.py --all` passed with the PNGs staged; `check.sh` green.
`tests/test_notebook.py` unchanged and passing. Main had not moved, so no sync was needed.

For the learning journal (`docs/learning-journal.md` does not exist on main yet):

- Data charts (8.07): five PNGs in `docs/figures/`, from metadata only; regenerate with
  `uv run python -m trailcam.data.datavis --root <data dir> --seed 42 --sidecar <data
  dir>/day_night.json --out docs/figures`.
- `class_counts.png` and `camera_locations.png`: the classes and the cameras are both imbalanced,
  and the trans cameras see a different mix (raccoon, bobcat), so accuracy alone flatters.
- `day_night.png`: 60 to 72% of kept frames per split are infrared night shots (trans_test
  the most), and hour 12 holds 2,990 night frames: the camera clock is wrong on some cameras, so
  day/night is read from the pixels.
- `box_fill.png`: most animals cover 1 to 5% of the frame, which is why arm B crops in on the box.
- `burst_length.png`: 91% of images have a near-identical sibling in their burst, so splitting by
  image would leak; the split goes by sequence.

**Next.** 8.02, 8.04 (the day/night slices measure accuracy by night and day).

Review: tier 2, `/review-session 8.07`

Coordinator request after the item commit (from the user, 2026-10-15): the user copied the 6.03
sidecar to the main checkout's `data/day_night.json` (gitignored). `day_night.png` was redrawn
with it (kept images per split, seed 42), the data card gives both sets of numbers (kept images
per split first, all annotated images per official file second), and a real-data test pins the
kept-image counts. Folded into the item commit (0bce6a5) before any review.

Review: tier 2, ran `/review-session 8.07` (0968e1a..0bce6a5; code-reviewer Opus xhigh,
risk-reviewer Fable high). risk-reviewer found nothing. code-reviewer found two lows, fixed in
fade23f with the user's go-ahead via the coordinator:

- R03 listed "a notebook cell displays a batch of training images" as a failure while the look
  notebook does exactly that. R03 now records the photo grid as the one exception, with its
  conditions: Kaggle only, train split only, titles with the class and day or night (no file name
  or id), outputs never committed (the guard refuses them).
- The 6.03 night counts were never checked against the sidecar, and a docstring said the sum test
  checked them. A real-data test now recomputes night per official file and per capture hour from
  the sidecar and the raw annotation files, and requires every annotated image to have a flag;
  both constants match. Swapping two hours' night counts fails it. The docstring is corrected.
- Not a finding, fixed with the user's approval: `docs/plan.md` section 2 said "About half the
  images are night shots"; it now gives the measured shares (65.7% of kept images, 63.6% of all
  annotated images; trans_test 71.6% against cis_test 62.6% of kept images). Spec changed: plan
  §2. The rows citing it (R01, R05, R06, R15, R21, R23, R32) were re-read; none states a night
  share, so none changes.

Scenarios R01, R03, R04, R06, R07, R16, R17, R27, R37, R38, R41 to R45 held for both reviewers;
known gaps passed: R04 the figure and Grad-CAM writers (8.02), R06 the Space and the slides
(11.01, 12.02), R17 the scoring script (9.01). Main was merged before the fix at c303041 (8.04;
conflicts in this log and in R03, R04 and R17, both sides kept) and after it at ef9100d (6.08;
conflicts in this log and R03).

Fix round: `/review-session 054f7be..fade23f 8.07` (`tests/test_datavis.py` the one high-stakes
file). Fixed in e1f67cf, after merging main at 61f1ecc (conflict in this log only):

- Both reviewers, low: the photo-title test searched only the source text of title calls, so a
  file name passed through a variable, a label, `ax.text` or a `print` slipped by. The test now
  walks every code cell's syntax tree: no image id, seq id or location as an attribute, a name or
  a string key, and `file_name` only inside the one `Image.open(...)` call. Each of those four
  bypasses fails it. R03's test cell says so.
- code-reviewer, low: plan section 5's augmentation row said "half the data is infrared". With the
  user's approval it now reads "about two thirds of the kept images are infrared (section 2)".
  Spec changed: plan §5. The rows citing it (R17, R20, R22, R27, R31, R37, R38, R40) were
  re-read; R20 gives the grayscale probability without a share, so none changes.
- code-reviewer, outside the range: the data card's "About six in ten kept images" became two in
  three (0.657), matching its table.

Scenarios R01, R03, R04, R42 to R45 held for both. No third round (the user's decision).

Notes, recorded and not fixed:

- R45: code-reviewer reported one stray no-op command (`uv run --quiet python -c "pass"`)
  outside its allowed list in the fix round; it read nothing. The harness flagged risk-reviewer's
  fix-round report for a settings-shaped pattern; the only match was its mention of
  `.claude/settings.json` in the R43 result, with no instruction in it.
- The look notebook reads the loose 6.03 dataset (`cct20-512`), which still exists and is the
  training notebook's fallback; since 6.08, training reads the archive dataset (`cct20-512-tar`).
  Moving the look notebook to the archive is the user's call.
- `uv run mypy tests/test_datavis.py` reports two `no-untyped-call` errors on
  `FigureCanvasAgg.draw`; `check.sh` type-checks `trailcam/` and the scripts only, not `tests/`.

Also approved by the user via the coordinator, for item 6.03 (the user's): `docs/plan.md`
section 11, open question 3 (tarball layout) is closed with the 6.03 cache build's output (1
directory, 57,864 `.jpg` files, all at depth 1 in one top-level folder). With the data card's
"Day and night" section, it is what 6.03 still needed; the user flips 6.03 after the merge. Spec changed: plan §11; no row cites it.

`check.sh` green at e1f67cf: 718 passed. `docs/learning-journal.md` is still not on main, so the
learning-journal lines stay above, under "For the learning journal".

**Next.** 8.02, then 9.01.

## 2026-10-15: session 6, item 6.09, a ledger test for the life of the ledger, the small CNN row, the 6.08 wording (Opus 5.5)

New item, approved by the user in the coordinator session on 2026-10-15 and added to
`docs/progress.json` as 6.09 (session 6, owner claude, due 2026-10-16). What happened: the user
committed the small CNN run's ledger row (93b9769, item 6.04) and CI failed on
`tests/test_train.py::test_the_ledger_file_holds_the_header_and_no_rows`, a 5.03 test written
before any run existed. The row itself passed the ledger checker. 93b9769 was reverted by
f414912 (main green, 698 passed); the commit had been made without running `check.sh` first.

**Done.**

- Test: `test_the_ledger_file_holds_the_header_and_no_rows` is replaced by
  `test_the_ledger_file_is_its_text_then_the_header_then_rows_that_pass_the_check`: the
  explanatory text (non-empty, no table line), the header of `LEDGER_COLUMNS` (both lines of
  `ledger_header`), then zero or more rows. It calls `trailcam.train.ledger.check_ledger` for the
  rest (header exactly once, every line after it a well-formed row, the cross-row checks), with
  no problems and one good row per line after the header; nothing of the checker is copied.
  5.03's other guarantee, that the dry run never appends, is
  `test_the_dry_run_trains_on_its_own_fixture_and_never_appends` (the ledger file is unchanged
  byte for byte), and the checker refuses a `dryrun` row (`tests/test_ledger.py`). R40's test
  cell in `docs/review/invariants.md` now says this.
- Ledger row, re-added byte for byte from 93b9769, and checked against the values of the brief
  through `ledger_row` before it was appended:

  `| small_cnn-seed0-20261015T223717Z | small_cnn | A | small_cnn | f40ed6e3960104f10a78e3d33a8f530cde82e23a555def76f694e19ab09a3e95 | 42 | 0 | c78a9c1048f63aa52f34578ca17a6391315afdc2e478b4dbfd8a3aa954d1b009 | 15 | 14 | cosine warmup 1 | 0.6625 | 0.6216 | 0.1965 | 0.0775 | 18529bf80a299551e41970e380f36cb2d1d217ddb838322b56b596829c47573c |`

  `uv run python -m trailcam.train.ledger --check docs/ledger.md` exits 0: "ledger: 1 rows, every
  check passed", small_cnn A, split seed 42, run seed 0. No leakage alarm: trans_val accuracy
  (0.1965) is far below val (0.6625). Item 6.04 is the user's and stays `passes: false`; the user
  flips it after the merge.
- Notebook (text and defaults only; outputs cleared): How-to-run step 2 attaches the packing
  notebook's output with Add Input > Notebook, not a dataset, since Kaggle extracts a `.tar` when
  a notebook's output becomes a dataset (`cct20-512-tar` came out as folders); it says to remove
  both `cct20-512` and `cct20-512-tar`. Default `CACHE_ARCHIVE =
  Path("/kaggle/input/notebooks/codx888/notebook2b5a4bf6ad/cct20_512.tar")` (verified
  2026-10-15: one file of 3,213,506,560 bytes, unpacked in about 35 s; step 5 gives the 35 s).
  Step 6 adds the practice: about 10 minutes after Save & Run All, open the Logs tab and, if it
  is empty and the version still shows "Starting...", cancel and submit again; it says why the
  2026-10-15 stall happened is not known. The unpacking cell's message no longer says a committed
  run copies the 57,864 files; section 3's text no longer says the loose files slow the start.
  Default `COMMIT` is left as it was (0968e1a, refused in section 2); the user sets the pinned
  training commit 61f1eccf24a3e33cf876e4ceb9dd5e5af8aed020, which this item leaves valid: no
  training code changes here (the only `trailcam` change is the docstring of
  `trailcam/data/cache.py`).
- Docstrings: `trailcam/data/cache.py` and `scripts/kaggle_pack_cache.py` no longer name the input
  copy as the cause of the stall (cause not known; a CPU run with the same input logged its first
  line in about 5 minutes). The packing script's step 5 now says to keep the version and attach
  its output as a notebook input, not to make it a dataset, and its note on Save & Run All no
  longer says the run copies every input file.
- `tests/test_notebook.py`: the new default and How-to-run strings (step 2, the 10-minute
  practice in step 6, in order), and a new test that neither the notebook nor the two docstrings
  name the input copy as the cause.
- Not changed, for the coordinator: `docs/plan.md` line 102 still names the private dataset
  `cct20-512-tar` as where the archive comes from; plan.md is edited by other sessions, so it is
  left for one of them or a follow-up item.

Checks: `bash scripts/check.sh` green; the ledger checker exits 0 with the row; `uvx --from
nbconvert jupyter nbconvert --clear-output --inplace notebooks/kaggle_train.ipynb` leaves the
notebook byte for byte unchanged; `uv run python scripts/guard_data.py --all` passes.

For the learning journal (`docs/learning-journal.md` is not on main yet; testing topic):

- Lesson: a test written for the starting state (an empty ledger) broke on the first real row,
  because it pinned a moment, not the rule. Test the invariant that holds for the file's whole
  life, and reuse the checker the project already has instead of a second copy of its rules.
- Practice: run `bash scripts/check.sh` before every commit, even one that only touches docs or
  data rows: the docs are read by the tests, so a docs-only commit can turn CI red.
- Evidence: 93b9769 added one ledger row, passed the ledger checker, and failed CI on the 5.03
  test; it was reverted (f414912) and re-landed here with the test replaced.

**Next.** Review tier 2. After the merge, the user flips 6.04; the next GPU run uses the
notebook's new `CACHE_ARCHIVE` default at the commit the coordinator gives.

## 2026-10-15: review 6.09: one low fixed, main merged (Opus 5.5)

Tier 2 (`/review-session` on f414912..6f53b6c): code-reviewer (Opus 5.5 xhigh) on the range,
risk-reviewer (Fable 5.1 high) on the notebook, `scripts/kaggle_pack_cache.py`,
`trailcam/data/cache.py`, `tests/test_notebook.py` and `tests/test_train.py`. Scenarios R01, R03,
R06, R07, R16, R17, R25, R27, R37, R38, R40, R41 and R42 hold for both. Between them they
confirmed the row's config hash (`config_hash(load_config("small_cnn"))` at HEAD) and split hash
(`REAL_HASH` of `tests/test_split.py`). Risk-reviewer found nothing.

Code-reviewer, low: `docs/plan.md` section 2 still said the archive is "saved as a second
private dataset, `cct20-512-tar`", the route the notebook and the packing script now say fails,
and the notebook's missing-archive message asked "is the archive dataset attached?". The user's
decision (via the coordinator): fix it here; the plan.md edit is approved.

**Done.**

- Main merged (3a4fcf3: 8.02, 8.07 and the 6.03 flip) with the app's sync. Two conflicts, both
  sides kept: `docs/progress-log.md` (main's 8.02 and 8.07 entries, then 6.09) and R40/R41 of
  `docs/review/invariants.md` (6.09's R40 test cell, 8.07's R41 test cell).
- `docs/plan.md` section 2: the tar is attached as the packing notebook's output (Add Input >
  Notebook, at `/kaggle/input/notebooks/<user>/<notebook>/cct20_512.tar`), not as a dataset,
  since Kaggle extracts a `.tar` that becomes a dataset. Spec changed: plan section 2.
- Notebook, unpacking cell: a missing archive now says "is the output of the packing notebook
  attached?". The first wording, with "notebook's", broke the cell (an apostrophe inside the
  f-string's single-quoted literal); `tests/test_notebook.py` caught it when it parsed the cells.
  `test_the_unpack_cell_sets_data_root_or_keeps_the_loose_dataset` now pins the message.
- Outputs cleared; nbconvert `--clear-output` leaves the notebook unchanged. `check.sh` green.

The fix (ee80471) touches `docs/plan.md` and the notebook (high-stakes), so one fix round ran on
the fix range.

Fix round (`/review-session fa72fd9..HEAD 6.09`, the user's start): code-reviewer (Opus 5.5
xhigh) and risk-reviewer (Fable 5.1 high), nothing found by either. R01, R03, R06, R07, R16, R17,
R27, R37, R38, R41 and R42 hold; the new assertion makes the test stricter. Spec changed: plan
section 2, cited by R01, R05, R06, R15, R21, R23 and R32; the edit is only how the archive is
attached, which none of them concern. `check.sh` green, 745 passed.

Left for a later item (code-reviewer, optional, outside the range): the cache-check cell (cell-11)
still ends its error with "none (is the dataset attached?)". It shows only for a tar without the
annotations or the sidecar, which `kaggle_pack_cache.py` refuses to write; the wording predates
6.09.

**Next.** The user merges and pushes; then the user flips 6.04.

## 2026-10-15: session 9, item 9.01, the final scoring script (Opus 5.5, xhigh)

Main at 9c63bac. Built and tested on the synthetic fixture only: no real checkpoint was loaded
and no real cis_test or trans_test was read (that is 9.02, the user's run on Kaggle).

**Decisions (the user's, asked mid-session).** The item wants one ledger row per checkpoint with
the test scores and who approved, but `LEDGER_COLUMNS` has 16 training columns and `ledger.py`
requires that table to end `docs/ledger.md`. The user chose: (1) a separate
`docs/test-ledger.md` with its own columns, so `docs/ledger.md`, `train.py` and `ledger.py` are
unchanged; (2) a checkpoint that already has a test row is refused, so each is scored once.
Recorded in plan.md section 7 (Protocol) and in R40's rule.

**Done.** `trailcam/evaluate/score_test.py`, `docs/test-ledger.md` (header, no rows),
`tests/test_score_test.py` (37 tests).

- `python -m trailcam.evaluate.score_test --approved-by <name> --checkpoint <run>.pt ...
  --data-root <cache> --out <new dir> [--config --ledger --test-ledger --overlay --assets]`.
  The config of each checkpoint is the one its ledger row names, with the row's run seed, so a
  checkpoint cannot be scored under another recipe.
- Refusals, exit 1, all before any checkpoint is loaded or any test image read: no or empty
  `--approved-by` (or one holding `|`); a non-empty `--out`; an overlay outside `assets/`; a
  ledger that fails `ledger.py`'s check or a malformed test ledger; a missing checkpoint file, one
  given twice, one with no ledger row ("unknown checkpoint"), one already in the test ledger; a
  ledger row whose checkpoint is not given; a config that no longer hashes to its row; more than
  one split hash; a file of another model (`load_checkpoint`). After the metadata is read: a
  cache whose split hash is not the checkpoints' (still before any prediction).
- One pass: `predict_split` once per checkpoint and test split; its predictions feed the
  headline, `evaluate.write_report` (per-class, per-camera, confusion matrix, three PNGs per
  split) and `slices.write_report`. `keep_features` only for the ResNet-18 (`FEATURE_MODELS`).
  `attention_pass` once per checkpoint and split, and `write_overlays`, outside inference mode.
  The overlays are drawn first, from `assets/` alone, so a failing one stops the run before any
  test split is scored.
- Output under `--out`: `headline.md` (per checkpoint: cis_test and trans_test accuracy and
  macro-F1, both gaps, cis minus trans; a mean row per model and arm with more than one seed;
  rows in ledger order, never by score; the leakage alarms of `train.py` applied to the test
  scores; the Beery et al. citation), `slices.md`, the two feature maps (ResNet-18 only), and per
  run id `metrics.md`, six PNGs, `attention.md`, `attention_in_box.png` and the overlays.
- Test ledger rows: run id, model, arm, config, split hash, checkpoint sha256, the four scores,
  both gaps, scored at (UTC), approved by. Printed with the alarms before the append, as
  `train.py` does; exit `ALARM_EXIT` (3) when an alarm fired, else 0.
- `evaluate.py`'s docstring now says 9.01 uses `predict_split`'s pass, not `predict`.
- invariants.md: R17, R26 and R32 test cells name `tests/test_score_test.py`; R32's "where" adds
  `score_test.py` and item 9.01; R40's rule says where the final scoring's rows go, and its test
  cell names the new tests.

Decision to review: every run in `docs/ledger.md` must be given in the one run (a row whose
checkpoint is missing exits 1). With the re-scoring refusal this makes the scoring all at once and
once only, and it is R17's "a listed checkpoint is missing" gap. If a ledger run should not be
reported, this blocks 9.02 until the user decides; dropping the check is one `if`.

Steps: `uv run pytest tests/test_score_test.py` 37 passed: the headline has both splits, both
metrics and both gaps per checkpoint (by hand from the spied predictions); the test ledger rows
have the files' sha256 and the approver; without `--approved-by` exits 1 (also via `python -m`);
an unknown checkpoint exits 1. One run on the fixture (scratchpad) printed the table and rows and
exited 3: random weights put trans within two points of cis, so the alarm fired, as intended.

Mutation checks, each restored by copying the file back, each failing a test: the approver check
removed (2 ways), the unknown-checkpoint check, the missing-ledger-row check, the already-scored
check, the non-empty `--out` check, the config-hash check, the split-hash check; a test split by
another route (`CLI_SPLITS` widened to cis_test and trans_test in `attention.py`; val added to the
records returned; `TEST_SPLITS` swapped for val and trans_val); a second `predict_split` pass;
features kept for the small CNN; the attention pass inside `torch.inference_mode()`; the overlays
removed, and moved after the test passes. 17 of 17 killed.

`check.sh` green (761 passed). Then main moved to 3a4fcf3 (8.07, 6.03) and I merged it with the
sync tool. The conflicts were in this log and in R17's and R41's rows of invariants.md (8.07
added `tests/test_datavis.py` to both). Both sides were kept, with the other items' entries
first. `check.sh` green after the merge (781 passed).

Notes, not fixed:

- 9.02's steps say "ledger rows ... pasted into docs/ledger.md"; with the user's decision they go
  into `docs/test-ledger.md`. Items are not edited by Claude; the user may want to reword 9.02.
- On Kaggle, pass `--config`, `--ledger`, `--test-ledger` and `--assets` from the clone (the wheel
  ships no `docs/`), and a new `--out` under `/kaggle/working`. The clone's `docs/ledger.md` must
  hold every training row (6.09 adds the small CNN's).
- One mutation run took 11 minutes of wall time while pytest reported 42 s; the unmutated file
  runs in 45 s and exits cleanly. Not reproduced.

For the learning journal (`docs/learning-journal.md` is not on main yet), evaluation:

- A test set is a one-shot measurement: once you have seen its score, every later choice is
  tuned on it a little, so `score_test.py` scores every checkpoint together and refuses one that
  already has a row in `docs/test-ledger.md`.
- The script refuses to run without `--approved-by` because "the user's go-ahead" is a rule only
  if it leaves a trace; the approver's name goes into every row.
- Refusals are checked before anything is loaded, and the tests prove it by replacing the
  loaders with functions that fail: a refusal after the first prediction would already be a look.
- Where to look: `trailcam/evaluate/score_test.py` (docstring), `docs/test-ledger.md`,
  `tests/test_score_test.py`, plan.md section 7 (Protocol).

**Next.** Review 9.01 (tier 3). Then 9.02, the user's single run on Kaggle once every arm and
seed has finished training.

Review: tier 3, ran `/review-session 9.01` (3a4fcf3..bc61d94; code-reviewer Opus xhigh,
risk-reviewer Fable high). Both found the same low independently. Findings:

- High (code-reviewer): R17's "scored while a run is still training" was open. A run in training
  has no ledger row, so a ledger with the small CNN and arm A seed 0 passed while arm B seed 0
  trained. After arm B's row was added, nothing could be scored without a second test ledger.
- Low (both): the test rows were printed only after the slice tables, feature maps and headline.
  A crash there, or in a later checkpoint's pass, left test scores on disk with no row.
- Low (risk-reviewer; a hardening note from code-reviewer): two training rows with one checkpoint
  sha256 were not refused. One file would be scored as two rows, and the test ledger would then
  refuse itself.
- Low (code-reviewer): 9.02's steps still say the rows go into `docs/ledger.md`. This is the
  user's edit after the merge; items are not touched here.
- Risk-reviewer's design question: two rows of one config and run seed (a rerun) could never be
  scored.

Scenarios R01, R03, R05, R26, R32, R42 to R45 held for both. R40 held for the row content and
order. R17 broke on scenario 2 (the high).

Fix 5612c2d (the user's decisions, via the coordinator):

- `check_complete`: refuses a ledger with no small CNN row, with no ResNet-18 row, or whose arm A
  and arm B rows differ in run seeds. The message names each missing arm and seed.
- `check_unique`: refuses two rows of one checkpoint sha256, and two rows of one config and run
  seed (the message says the user must pick one).
- `log_row`: each checkpoint's test row and alarms are printed, then the row is appended, right
  after that checkpoint is scored. This comes before its tables, the next checkpoint, the slices,
  the feature maps and the headline.
- Decision to review: with rows appended per checkpoint, a run that stops part way would have
  left the rest unscoreable. The "already scored" refusal and the "every row given" check
  together refuse every rerun. So a ledger row now counts as present if it is given or already
  in the test ledger, and a second run with the checkpoints left finishes the scoring. Given
  again, the scored ones are still refused. That second run's slices, feature maps and headline
  hold its own checkpoints only.
- Plan section 7, the docstring, and R17's and R40's test cells state the rules.
- Tests (46 in the file, 9 new): five incomplete ledgers, including the reviewer's two-row case
  and arm A {0, 1} with arm B {0}; a complete paired ledger with seeds 0 and 1 passes; one
  checkpoint in two rows; a rerun of arm A seed 0; a run whose second checkpoint fails has
  printed and appended the first row, a full rerun is refused, and a run with the other two
  finishes the ledger.
- Mutation checks, each restored by copying and each failing a test, 8 of 8: the small CNN
  check, the seed pairing, the no-ResNet-18 check, `check_complete` not called, the duplicate
  hash, the duplicate config and seed, the rows logged at the end, and no resume.

`check.sh` green (790 passed).

**Next.** Fix round: `/review-session bc61d94..5612c2d 9.01` (`score_test.py` and its tests are
high-stakes files). Then the user's tier-3 option, `/code-review ultra`. The user rewords 9.02's
steps after the merge.

Fix round: `/review-session bc61d94..5612c2d 9.01` (code-reviewer Opus xhigh, risk-reviewer
Fable high). Findings:

- High (code-reviewer; risk-reviewer found a narrower version, low): the resume path counted an
  already-scored row as present. So runs trained after the test scores were seen could be scored
  later: arm A and arm B seed 1, or a rerun of arm B seed 0 after its old row was removed. Before
  the fix this was impossible. The resume was my decision, flagged in the fix entry above.
- Low (both): a failure between a row's append and its tables (or the slices and feature maps)
  lost those outputs for good, since the checkpoint was then "already scored". A resumed run's
  headline and alarms covered only its own checkpoints.

Scenarios R01, R03, R05, R26, R32, R42 to R45 held for both. R17's named scenario held
(`check_complete`); the high is another route around it. R40 held, with these findings left over.
Code-reviewer also noted that plan section 7 credited the resume rule to the user; it was mine.

Fix 60b2185 (the user's decision: drop the resume):

- A test ledger that already holds a row is refused, whichever checkpoints are given: the final
  scoring runs once, for every checkpoint together. Runs added after it can never be scored.
- `print_row` prints each checkpoint's row and alarms as soon as it is scored; the rows are
  appended together after every checkpoint's tables, the headline, the slices and the feature
  maps. A run that stops part way appends nothing and is run again from the start with a new
  `--out` (the checkpoints are pinned by sha256, so the scores are the same).
- `read_test_ledger` also refuses one run id in two rows.
- Plan section 7 credits each rule to the decision it came from; R17's and R40's test cells
  updated.
- Tests (48 in the file): the part-way test now fails the second checkpoint and, in a second run,
  the feature maps; both print the rows scored and append none, then a full run appends all
  three. The rescoring test also gives seeds 1 of both arms added to the ledger later: refused.
  The malformed test ledger cases check each reason on the reader, including one run id in two
  rows.
- Mutation checks, each restored by copying and each failing a test, 7 of 7: the run-once
  refusal, the rows appended per checkpoint, the rows appended before the figures, the run-id
  check, the sha256 check, and the rows not printed. The run-id mutation first survived, because
  `main` refuses any test ledger with rows before the reader's check is reached. The test now
  checks the reader's message.

`check.sh` green (792 passed). No third round unless the user asks.

**Next.** The user's tier-3 option, `/code-review ultra` on the branch, with time for one fix
round before 9.02. The user rewords 9.02's steps after the merge, then merges this branch.

Ultra review (2026-10-16): `/code-review ultra` on the branch (the user's tier-3 option; a
first attempt was stopped before it finished). Four findings, all graded nit, all verified real:

- `check_complete` saw a missing arm only when the other arm already had that seed. With seed 0
  of both arms in the ledger and seed 1 still training it passed, and the run-once rule would
  then lock seed 1 out. The docstring and plan section 7 claimed more than the code did. The
  ledger cannot see a run with no row, so completeness has to be stated.
- `append_rows` re-read the test ledger but did not re-apply the run-once check. A second
  scoring, or rows pasted in during the run, would leave duplicate rows that every later read
  refuses.
- `docs/test-ledger.md` still described the first version (refuses "a checkpoint that already
  has a row"; "pastes the printed rows here" without saying which copy).
- `--approved-by` with U+2028, U+0085 or another line break `splitlines()` splits on passed the
  `\n` and `\r` check and would be written as a broken row.

Fix f5b23e4 (the user's decisions):

- `--run-seeds` is required (exit 1 without it). `check_complete` refuses a ledger without a
  small CNN row, without an arm A and an arm B row for each seed given, or with a ResNet-18 row
  of another seed. The headline records the seeds.
- `append_rows` refuses a test ledger that gained a row during the run; `main` exits 1 and the
  rows are already printed.
- `breaks_table`: a `|` or any non-printable character, which covers every `splitlines()`
  boundary. Used by `check_approver` and `ledger_line`.
- `docs/test-ledger.md` rewritten: the script appends to the clone's copy, the user pastes the
  printed rows once into the repo's file, and every refusal is listed. Plan section 7 and R17's
  test cell state `--run-seeds`.
- Tests (57 in the file): the ultra case (seed 0 of both arms, `--run-seeds 0 1`) and a ledger
  seed not given are refused; no `--run-seeds` exits 1; approvers with U+2028, U+0085, a
  vertical tab and a tab are refused, and a cell with U+2028 or U+001C is refused by
  `ledger_line` while accented names pass; an append to a test ledger that gained a row is
  refused and leaves it unchanged.
- Mutation checks, 5 of 5 killed, each restored by copying: no `--run-seeds` check, a missing
  seed not seen, an extra seed not seen, the cell check back to `\n` and `\r` only, and no
  re-check at the append.
- Not tested through `main`: the `except Refusal` around the append (exit 1, not a traceback).
  It is covered by reading only; a test would need another full scoring run.

`check.sh` green (801 passed). Reviewing these fixes would be a third round, which needs the
user's request.

Merged main at 914e5fe (6.09 and the 6.04 flip; the small CNN row is now in `docs/ledger.md`) with
the sync tool. The conflicts were in this log and in R40's test cell (6.09 rewrote "holds the
header and no rows"); both sides were kept, the other item's entries first. `check.sh` green
after the merge (802 passed).

**Next.** On Kaggle, 9.02 now needs `--run-seeds` (for example `--run-seeds 0 1 2`, or `0` if
seeds 1 and 2 are not trained). The user rewords 9.02's steps after the merge, then merges this
branch.

## 2026-10-15: session 12, item 12.07, the learning journal (Opus 5.5)

New item, approved by the user in the coordinator session on 2026-10-15 and added to
`docs/progress.json` as 12.07 (session 12, owner claude, due 2026-10-17). Why: the user wants a
record written for them, to go back and learn what was done and why; this log is written for
reviewers and later sessions. The journal also feeds the slides (12.06, the presentation plan,
comes next).

**Done.** `docs/learning-journal.md`, in plain language, by topic: the problem and the dataset
choice; inspecting and cleaning the data; splitting without leakage; Dataset, DataLoader and
transforms; arm A vs arm B; the models; the training loop and the ledger; macro-F1 next to
accuracy; test-set isolation and leakage alarms; the evaluation pieces built so far (8.02 and 8.04
named as still to come); running on Kaggle; how the project was run. Each topic has What we did,
Why it mattered here, Where to look and Try it yourself. Then a dated "Surprises and lessons"
timeline (eight entries, from the paper vs files counts to the partial file count settled by
`check_images`), "Kaggle steps you did" (seven steps, training commit 21faa79b0aed), and "How to keep
this journal". Sources: this log, the plan, the data card, the data inspection, the research
report, the code and tests, and the Kaggle facts the user gave in the item (6.03's 22-minute build,
night share 63.6% overall and 68.8% in trans_test, hour 12 at 61.5% night, the quota and phone
verification, the dataset page error). Counts only: no image id, file name or data row (R03); the
Beery et al. 2018 citation sits in the introduction.

Steps: the file exists; a scratch script checked all 92 links (relative file links and the
in-page topic anchors): every file link resolves to a tracked file and every anchor to a heading.
All 13 "Try it yourself" commands, extracted from the file and run as written, exit 0 on the Mac
(0 to 10 s each, the dry run 22 s; the dry run printed its row and left `docs/ledger.md`
unchanged). `check.sh` green (645 passed). `passes` is flipped.

Noted, not changed (outside this item): `docs/data-card.md` still says the day/night share is
pending 6.03, and plan section 2 says about half the images are night shots; the measured share
(63.6%) is in the journal only. Recording it in the data card is 6.03's step (owner user).

**Next.** 12.06, the presentation plan, which can draw on the journal; each later item appends 3 to
5 lines to the journal under its topic.

Review: tier 0 (docs only), nothing to run; no invariant row's spec document changed.

Update after main moved (requested by the coordinator session): main at c303041 (8.04) merged in
through the sync tool. One conflict, in this log: 8.04's entry kept first (it merged first), then
this one; `docs/progress.json`, `docs/review/invariants.md` and 8.04's files merged cleanly. The
journal's evaluation topic now covers 8.04 as merged: the day/night and common/rare slices (the
under-4% rare rule fixed on train counts), the trans scores re-weighted to the cis night share,
and the feature map, with `slices.py` and `tests/test_slices.py` linked and a second command (the
slices dry run). Night shares over the kept, re-split images, given by the coordinator and checked
here with `load_metadata` and `make_split` on the sidecar copy the user downloaded into `data/`
(counts only): train 0.607, val 0.600, trans_val 0.633, cis_test 0.626, trans_test 0.716, all kept
0.657 (35,146 of 53,535). The timeline now says which photos each share is over: 63.6% and 68.8%
are over all 57,864 by official file (6.03), the shares above over the 53,535 kept. The test count
in topic 12 is now 669. Steps re-run: 94 links resolve; all 14 commands exit 0 (the slices dry run
8 s); `check.sh` green (669 passed).

Review: tier 0 (docs only), nothing to run.

Second update after main moved (requested by the coordinator session): main at 61f1ecc (6.08)
merged in through the sync tool, one conflict in this log (6.08's entry kept first, then this
one). The journal's Kaggle topic and timeline now cover, from 6.08's entry and the coordinator's
message: the cache as one tar packed on a CPU notebook and unpacked to `/tmp` (about 35 s, with
`unpack_cache`'s member checks); the 48-minute "Starting..." stall of the first 6.04 version,
with its cause not known, and the 10-minute check-and-resubmit practice; Kaggle extracting a
`.tar` when a notebook output becomes a dataset, so the packing notebook's output is attached
instead (`/kaggle/input/notebooks/<user>/<slug>/cct20_512.tar`, 3,213,506,560 bytes); and the
first real run (6.04, small CNN, commit 61f1ecc): 15 epochs at about 44 s, about 350 train
images/s, 12.8 minutes, about 13 minutes of quota, val accuracy 0.6625 and macro-F1 0.6216
against trans_val 0.1965 and 0.0775, still improving at the 15-epoch cap, recipe not tuned on it.
"Kaggle steps you did" now runs to 11 steps, with the training COMMIT 61f1ecc after the first
21faa79 attempt. New links: `scripts/kaggle_pack_cache.py`, `tests/test_kaggle_pack.py`; a
second command for the archive tests. Steps re-run: 96 links resolve; all 15 commands exit 0;
`check.sh` green (698 passed).

Noted for the coordinator, not changed here (outside a docs item): the notebook's
`CACHE_ARCHIVE` default on main still points at the `cct20-512-tar` dataset path, which Kaggle
unpacks; runs set it by hand to the notebook-output path, as the journal's step 9 says.

Review: tier 0 (docs only), nothing to run.

Third update (requested by the coordinator session): a timeline entry for the 6.04 ledger row
that turned main red at 93b9769 (`tests/test_train.py::test_the_ledger_file_holds_the_header_and_no_rows`,
committed without running `check.sh`), reverted by f414912; the test change belongs to item 6.09
and is not touched here. This branch was never on 93b9769 (it sits on 61f1ecc, whose files equal
f414912's), so no sync. Links (96) and all 15 commands re-checked; `check.sh` green.

Review: tier 0 (docs only), nothing to run.

## 2026-10-16: item 12.07, journal entries for 8.02, 8.07, 6.09 and 9.01 (Opus 5.5)

Requested by the coordinator session. Main at f706326 merged in through the sync tool; one
conflict, in this log (the new 2026-10-15 entries first, as merged, then 12.07's). The journal
gains a short entry per item, folding in the "For the learning journal" lines of each log entry:
8.07 under data cleaning (the five charts and what each shows), 8.02 under evaluation (figures,
overlays from `assets/` only, `evaluate.py`; the raw-panel and zero-map lessons; restore
mutations by copying), 9.01 under test-set isolation (scored once, `--approved-by`,
`--run-seeds`, refusals before loading, `docs/test-ledger.md`), 6.09 under how the project was
run (test the rule, not the moment; check.sh before every commit). Topic 10 no longer lists 8.02
as still to come, the Kaggle step 9 says `CACHE_ARCHIVE` now defaults to the notebook output,
and the test count is 802. Steps re-run: 109 links resolve, all 19 commands exit 0, `check.sh`
green (802 passed).

Review: tier 0 (docs only), nothing to run.

## 2026-10-16: session 10, item 10.01, the demo's prediction and its Gradio app (Opus 5.5)

**Done.**

- `trailcam/demo/predict.py`: `load_model(checkpoint, config_name)` loads with 8.03's
  `load_checkpoint` (no second loader), which refuses another model or class map, and returns the
  model in eval mode on the CPU. `predict(model, photo)` returns a `Prediction`: the top class and
  two runners-up as (label, softmax probability) pairs, most probable first, and the forward
  pass's seconds. Labels come from `CLASSES`; the photo goes through `EvalTransform` after
  `convert("RGB")` (a night photo is grayscale).
- Timer (the 8.01 carry-over): `trailcam/evaluate/metrics.py` gains `timed_forward(model, image)
  -> (logits, seconds)`, the same warm-up plus timed pass, returning the timed pass's detached
  logits; `time_one_prediction` is now `timed_forward(...)[1]`, so its tests run unchanged. The
  demo decodes from those logits, so a request costs the warm-up and one timed pass (two
  forwards), not three. The transform, the move to the device and the softmax stay outside the
  timed region; `tests/test_demo.py` checks the order transform, forward, clock, forward, clock.
- Grad-CAM in memory: `trailcam/evaluate/gradcam.py` gains `overlay_photo(model, photo)`, the
  overlay of a PIL image already in memory, returned and never written. `overlay(model, path)`
  keeps `assets_image` and now delegates to it after opening the file; `write_overlay`, the only
  writer, still goes through `overlay`, so the R04 containment for files is unchanged. The app
  calls it outside `torch.inference_mode()`.
- `trailcam/demo/app.py`: Gradio Blocks with upload, a Grad-CAM checkbox, a Classify button,
  `gr.Label` with the top class and two runners-up, the seconds ("0.042 s (forward pass only)"),
  the overlay image, the Beery et al. 2018 citation and CDLA-Permissive 1.0. Examples are the
  photos directly under `assets/` (`example_photos`, each through `assets_image`, so a link out
  of the folder is refused); no `assets/` folder or an empty one means no examples. `--check`
  builds the interface and exits 0 without launching; without `--checkpoint` it uses the
  config's model with random weights (default config `resnet18_arm_b`; both arms are a
  ResNet-18, open question 7 picks the checkpoint). A launch without `--checkpoint` is refused.
- gradio (6.30.0 locked, `>=6.30.0`): the user's decision in this session, a `demo` dependency
  group with `[tool.uv] default-groups = ["dev", "demo"]`, not `uv add gradio` as a runtime
  dependency. `uv sync`, `check.sh` and CI install it as they are; the wheel's Requires-Dist and
  pip on Kaggle never see it, so the notebook's install does not pull gradio's ~30 packages
  (R38). `uv.lock` committed.
- `docs/review/invariants.md`: test cells of R04 (the in-memory overlay writes nothing; the path
  overlay still refuses a photo outside `assets/`), R33 (`timed_forward`, the demo's two passes,
  transform before the clock) and R35 (the demo's examples) filled; the Space and the sort script
  stay open (11.01, 10.02).

Steps: `uv run pytest tests/test_demo.py` 13 passed: predict on a synthetic-fixture JPEG through
a random-weight small CNN loaded from a checkpoint file returns three distinct (label,
probability) pairs in `CLASSES`, descending, summing to at most 1, and a positive float of
seconds; and `python -m trailcam.demo.app --check` (run as a subprocess too) builds the interface
and exits 0. `tests/test_metrics.py` adds the `timed_forward` test. `check.sh` green.

Notes for 11.01 (the Space):

- `space/requirements.txt`: `uv export --no-dev --no-hashes --group demo` (the `--no-dev` export
  alone leaves gradio out), trimmed to torch, torchvision, gradio, Pillow, NumPy, matplotlib
  (`gradcam.py` imports it at module level) and PyYAML (`trailcam.train.config`); timm only if
  the Space serves EfficientNet-B0. `predict.py` imports `load_checkpoint` from
  `trailcam/evaluate/attention.py`, which brings in more of `trailcam` (split, train) but no other
  third-party package.
- `load_model` needs the root `config.yaml` (only to name the model): a Space that pip-installs
  `trailcam` has no checkout, so it uploads `config.yaml` and passes `config_path`, or 11.01
  proposes another way.

For the learning journal (docs/learning-journal.md is not on main yet; topic: demo):

- Top-3 with probabilities: softmax turns the 12 logits into probabilities that sum to 1; the
  demo shows the three largest, so they sum to at most 1, and a low top value says the model is
  unsure. See `trailcam/demo/predict.py`.
- Why the timer excludes decoding: the seconds should measure the model, not JPEG decoding or
  resizing, which depend on the photo's size; and the timed pass's own logits are reused so a
  request runs the model twice, not three times. See `timed_forward` in
  `trailcam/evaluate/metrics.py`.
- What Grad-CAM shows: which part of the 224 x 224 frame pushed the top class's score up (the
  last conv block's channels weighted by their gradients); on camera traps, a map on the animal
  is good, a map on the background suggests the model learned the scene. See
  `trailcam/evaluate/gradcam.py`, `overlay_photo`.

**Next.** 10.02 (the sort script). The user merges and pushes.

## 2026-10-16: review 10.01: two lows fixed, `--share` added, a fix round next (Opus 5.5)

Review (`/review-session 10.01`, the user's start, range f706326..HEAD): code-reviewer (Opus 5.5
xhigh) on the range, risk-reviewer (Fable 5.1 high) on the nine high-stakes files. Scenarios R01,
R03, R04, R05, R20, R21, R23, R26, R32 to R35, R38 and R42 to R45 hold; skipped as known gaps:
the Space parts of R33 and R35 (11.01), the sort script (10.02), the `assets/` photos (8.05).
Risk-reviewer found nothing. Code-reviewer, three lows:

- Fixed: the default-examples test passed on an empty `assets/` whatever the default was. It now
  asserts the defaults of `example_photos` and of `--assets` are the repo's `assets/`.
- Fixed: R20, R21, R23 and R34 still read "none yet (10.01)". New test: every forward of a
  request (warm-up, timed pass, Grad-CAM) takes one argument, equal to `EvalTransform` of the
  photo, and the handler and `predict` take the photo (and the Grad-CAM switch) only. Another
  checks that the UI says "model confidence" and never "accuracy"; the label now reads "Top class
  and two runners-up (model confidence)". The four test cells are filled.
- Decided, the user's go via the coordinator, confirmed by the user in this session:
  `trailcam/demo/**` and `space/**` join the paths of `.claude/rules/data.md`; plan.md's open line
  for sessions 10 and 11 (section 10, "Sessions 10 and 11") now records the decision. Spec
  changed: plan.md, that line only.

Outside the criteria, the user's go via the coordinator: every `gr.Blocks` build made Gradio's
version check and an analytics request to `api.gradio.app` (tests, CI, the live demo). The app
is now built with `analytics_enabled=False`. In gradio 6.30 the version check runs only when the
Blocks has analytics on (`blocks.py`, the `analytics_enabled` branch), so no environment variable
is needed; the inner Blocks of `gr.Examples` is built only when caching examples, which the app
does not. Test: the built Blocks has analytics off and the version check is never called.

Added at the user's go-ahead, via the coordinator: `--share` launches with Gradio's
`share=True`, the demo-day backup if the Space is down. The default stays local only, and
`--check --share` builds without opening a tunnel. A share link is temporary (it expires after
a time set by Gradio) and is served from the Mac, so it works only while the laptop stays on,
awake and online, with the app running. Test: `--share` passes `share=True` to a mocked
`launch`, the default `share=False`, `--check --share` never launches.

Main merged twice through the sync tool (9.01/9.02, then 12.07), both sides of each log
conflict kept. With 12.07 on main, the demo lines of "For the learning journal" above are now
topic 13 of `docs/learning-journal.md`.

`check.sh` green. The fix touches `trailcam/demo/app.py` and `tests/test_demo.py` (high-stakes,
test lines modified), so one fix round runs on the fix range.

Fix round (`/review-session d3e8cf1..HEAD 10.01`, the user's start): code-reviewer (Opus 5.5
xhigh) on the range, risk-reviewer (Fable 5.1 high) on `.claude/rules/data.md`,
`tests/test_demo.py` and `trailcam/demo/app.py`. Scenarios R01, R03, R04, R05, R20, R21, R23,
R33, R34, R35 and R42 to R45 hold (Space and sort-script parts skipped as known gaps). Spec
changed: `.claude/rules/data.md` (its paths only; R03, R20, R23, R24 cite it) and plan.md section
10 (no row cites it). Two lows, both from `--share`, fixed at the user's go:

- Both reviewers: `app.py`'s docstring said no launch calls `api.gradio.app`; with `--share`,
  Gradio asks it for a tunnel server (`gradio/networking.py`). The docstring now says a local
  build or launch never calls it and `--share` does, by design. Analytics stay off in every mode.
  The 5c03add commit message carries the old claim; commits are not rewritten.
- Code-reviewer: `--share` writes `.gradio/certificate.pem` (Gradio's public root CA, not a
  secret) into the working directory, which nothing ignored. `.gradio/` is now in `.gitignore`,
  and a test checks `git check-ignore` covers it. The guard is not changed: the file is no secret
  and `tests/test_guard_data.py` only requires the guard's patterns to be in `.gitignore`.

No third round (the user's request would be needed). `check.sh` green.

## 2026-10-16: session 10, item 10.02, the live sort (Opus 5.5)

**Done.**

- `trailcam/demo/sort.py`: `python -m trailcam.demo.sort --src <photos> --dst <new folder>
  [--checkpoint <run>.pt]` copies (`shutil.copy2`, never a move) every `.jpg`, `.jpeg` or `.png`
  under `--src`, at any depth and skipping hidden files, into `--dst/<class>/`, one folder per
  class of `CLASSES`, all twelve made up front. The model comes from 10.01's `load_model` (8.03's
  `load_checkpoint`), loaded once and warmed up once on a blank 224 x 224 input; then each photo
  is one forward pass through `EvalTransform`, its argmax the class (the class `predict` ranks
  first). `predict` itself is not called per photo: it warms up on every call, two forwards per
  photo. Without `--checkpoint` the config's model is built with random weights and the script
  says so (the item's step command has no checkpoint).
- Output: counts and seconds only. One line per photo (its number, the running seconds, the class
  and that class's count so far), then the count per class, the total, the seconds and photos per
  second. No file name or path (R03), no accuracy: the script reads no labels or metadata, so it
  cannot score anything (R17); the docstring says so. An unreadable photo is counted as
  `unreadable` and skipped; two photos with the same name get `_1`, `_2` suffixes.
- Refusals, checked before the model loads and again in `sort_photos`: `--dst` inside a git
  repository, this one included (`REPO_ROOT` or any ancestor holding `.git`, so a worktree and
  the main checkout are both covered; R01); `--dst` equal to or inside `--src`; an existing
  `--dst` that is a file or a non-empty folder (no override flag: a rehearsal deletes its folder);
  a `--src` that is not a folder.
- `DEFAULT_CONFIG` and `PHOTO_SUFFIXES` moved from `app.py` to `predict.py`, so the sort imports
  them without importing gradio; `app.py` imports them back, unchanged in value.
- `tests/test_sort.py` (14 tests, small CNN with random weights, synthetic fixture photos): the
  CLI sorts every fixture photo into a class folder with the same bytes and leaves `--src`
  unchanged; the no-checkpoint path says random weights; the output holds no photo stem, no
  folder path and no "accuracy", one line per photo; one warm-up plus one forward per photo; with
  the fixture's metadata beside the photos in `--src`, `open` is only ever called on photos; the
  module imports no metadata, split, json, ledger or metrics code; each refusal; the CLI refuses
  before loading a model; unreadable photos, hidden files and name clashes.
- Mutation checks, each caught by `tests/test_sort.py`: a file name printed; the git refusal off;
  only `REPO_ROOT` refused; the `--src` refusal off; only `--dst == --src` refused; the non-empty
  refusal off; the metadata JSON opened; a move instead of a copy; a warm-up per photo.
- Step: `uv run python -m trailcam.demo.sort --src <fixture> --dst <scratch>` sorted all 684
  synthetic fixture photos (ResNet-18 arm B config, random weights, CPU) in 6.1 s, printing counts
  and seconds only (no `.jpg` or scratch path in the output); a `--dst` in the worktree and a
  rerun into the full `--dst` both exited 2 with the refusal.
- Learning journal: topic 13 gains the live sort and why Gradio; a new topic 14 "Deployment"
  (what Spaces is, why people share models, Spaces vs AWS and the free clouds, the PRO and
  ZeroGPU decision); topic 12 gains the shared usage budget and `git merge --no-edit`; three dated
  lines in "Surprises and lessons" (the Spaces free tier, vim on merge, the usage budget), from
  the coordinator's notes.

`check.sh` green.

**Next.** Review: tier 2 (`trailcam/demo/**` is high-stakes). Then 11.01 (the Space's `app.py`,
ZeroGPU-ready).

## 2026-10-16: review 10.02: four lows fixed, a fix round next (Opus 5.5)

Review (`/review-session 6de1b54..HEAD 10.02`, the user's start), tier 2: code-reviewer (Opus 5.5
xhigh) on the range, risk-reviewer (Fable 5.1 high) on `tests/test_sort.py`,
`trailcam/demo/app.py`, `trailcam/demo/predict.py` and `trailcam/demo/sort.py`. Scenarios R01,
R03 to R05, R17, R20, R21, R23, R33, R34, R42 to R45 hold (Space parts skipped as known gaps);
R35 broke on one clause. No spec changed. Four lows, all fixed at the user's go:

- Both reviewers: R35 says the sort reads a folder outside the repo, but only `--dst` was
  checked, so `--src` could be the dataset folder. `check_source` now refuses a `--src` inside this
  repo or any git repository (the same parent walk as `--dst`, `_in_git_repository`), in `main`
  before the model loads and again in `sort_photos`; `SourceError` beside `DestinationError`.
  Three new tests; turning the check off fails three, skipping it in `main` fails one.
- Code-reviewer: R35's test cell still said "none yet for the sort script". It now names
  `tests/test_sort.py`, and so do R03 (no names printed) and R17 (no labels read).
- Code-reviewer: the journal recorded the hosting decision (PRO, then ZeroGPU) that plan.md did
  not. plan.md section 8 now records it, dated, and says item 11.02's "free CPU tier" text predates
  it and the section wins; item 11.02 itself is not edited (items are never edited). Row R36,
  which repeated "free CPU tier", now cites the hardware section 8 names.
- Risk-reviewer: the no-labels test spied on `builtins.open` only, while `Path.read_text` and
  friends call `io.open`. The spy now covers both; a planted `read_text` of the fixture's JSON
  passes the old test and fails the new one.

Spec changed in this fix: plan.md section 8 (R36 cites it, updated in the same commit) and
`docs/review/invariants.md`. `check.sh` green. The fix touches `trailcam/demo/sort.py` and
modifies lines of `tests/test_sort.py`, so one fix round runs on the fix range.

## 2026-10-16: fix round 10.02: four lows fixed, no third round (Opus 5.5)

Fix round (`/review-session 46434c2..HEAD 10.02`, the user's start): code-reviewer (Opus 5.5
xhigh) on the range, risk-reviewer (Fable 5.1 high) on `tests/test_sort.py` and
`trailcam/demo/sort.py`. Scenarios R01 to R05, R07, R17, R20, R21, R23, R33 to R35, R42 to R45
hold (Space parts skipped); R36's scenario conflicted with the new section 8. Spec changed: plan.md
section 8 (rows R02, R04, R07, R33 to R36 cite it; all agree after this fix). Four lows, fixed at
the user's go:

- Code-reviewer: `check_source` checked `--src` and its parents only, so a folder above the repo
  (or a link into `data/`) reached the dataset. It now also refuses a `--src` holding any photo
  that, links followed, lies inside a git repository, before the model loads. Two new tests;
  turning the check off fails two, not following links fails one.
- Both reviewers (code from the first commit): a failed `shutil.copy2` (full disk, permission)
  raised an `OSError` whose traceback names the photo's path (R03). The copy is now caught, the
  photo counted as `not copied` without its name, and the sort goes on. New test with a copy that
  raises an `OSError` carrying the path; catching the wrong error or keeping the class label fails
  it.
- Code-reviewer: R36 called a CUDA torch a defect, but the ZeroGPU phase needs one. Section 8 and
  R36 now say the CPU Basic phase uses the lock's CPU torch and the ZeroGPU phase a CUDA torch
  and `spaces`, settled in 11.01.
- Code-reviewer: plan.md sections 4 and 9 still said "free" CPU Space; both now say CPU Basic.

Not fixed, noted: the reviewer could not confirm the Hugging Face tier facts (it recalled ZeroGPU
hosting needing PRO); check them before the switch around 11-15. The imports test would miss the
`import trailcam.data.metadata` form (not used today). R03 and R35's test cells name the new
tests. Per the playbook this round's findings are fixed and listed, with no third round unless the
user asks. `check.sh` green.

## 2026-10-16: 10.02 follow-up: the import check parses `sort.py` (Opus 5.5)

The fix round noted the import test missed `import trailcam.data.metadata`. Checked at the user's
request: it was a text search, so it also missed `from json import loads` and imports by string
name; and the sort's process loads `metadata`, `split`, `json` and `train` anyway, through the
dataset and predict imports, so the test could only ever guard `sort.py`'s own source. A planted
`from json import load` read of the fixture's JSON passed it, and the open spy caught it.

At the user's go (option 2: a stricter test, `sort.py` unchanged, no third round):
`test_sort_py_names_no_metadata_split_json_ledger_or_metrics_module` parses `sort.py` with `ast`,
collects every dotted part of every import and every name imported from one, and refuses
`metadata`, `split`, `json`, `ledger`, `metrics` and `importlib`, and any `__import__`; a
parametrised test checks the collector sees eight import forms. Planting `import
trailcam.data.metadata`, `from json import load as ...`, `import importlib` or `__import__('json')`
into `sort.py` fails it. R17's test cell now says the check guards the file, and the open spy the
behaviour. `check.sh` green.

## 2026-10-16: item 11.01: the Space's files and the deploy script (Opus 5.5)

Done (11.01, Opus 5.5 high): `space/app.py`, `space/README.md`, `space/requirements.txt`,
`scripts/deploy_space.py`, `tests/test_space.py` (12 tests), `tests/test_deploy_space.py` (28).
The user's decisions of 2026-10-16 (through the coordinator) are in plan.md section 8: hosting
(PRO and CPU Basic from about 10-20, free ZeroGPU from about 11-15), the ZeroGPU-ready app, Grad-CAM
in the Space and off by default (open question 5 closed), and `--weights <path>` with no arm named
(open question 7 noted). progress.json items are not edited; the user rewords 11.02.

- **Same UI.** `build_app` of `trailcam/demo/app.py` gained an optional `handler` (the only change
  to 10.01 code); the Space passes its own and otherwise builds the identical interface, analytics
  off. `app.py` holds no prediction code: `classify` calls `predict`; a test refuses a class list,
  transform, timer or softmax in it. 10.01 never set `GRADIO_ANALYTICS_ENABLED`, so neither does
  the Space.
- **ZeroGPU-ready.** `import spaces` in a `try`, falling back to the bare handler; the model goes
  to `cuda` at module level when CUDA is available, else the CPU; the button's handler (prediction
  and Grad-CAM) is wrapped in `spaces.GPU`, takes the photo and the switch only, and reads the
  model from the module, so a ZeroGPU call never passes the model.
- **Torch wheel: CPU now.** The requirements carry the lock's `+cpu` wheels with the PyTorch CPU
  index line: a much smaller download and build, the wheels CI tests. ZeroGPU needs CUDA torch, so
  the switch is: drop the index line and the `+cpu` suffixes, put torch at a version ZeroGPU lists
  (its docs listed 2.8.0 to 2.13.0 today; the lock has 2.14.1, so 2.13.0 and torchvision 0.28.0
  unless 2.14 is listed by then), set the hardware to ZeroGPU, redeploy. `spaces` comes from
  Hugging Face (not in the lock); check it imports there, and that a Grad-CAM `gr.Warning` still
  shows from inside a GPU call.
- **How the Space gets `trailcam`: uploaded, not a `git+https` pin.** The deploy uploads the
  package's tracked modules at their repo paths. Why: the Space always runs the code of the commit
  it was deployed from (the deploy refuses uncommitted or untracked changes under `space/`,
  `trailcam/`, `config.yaml` and `assets/`, and names HEAD in the Space's commit message), with no
  pin to bump after every fix and no dependency on what is pushed; the requirements file is then
  the whole environment; and `CONFIG_PATH` and `ASSETS_DIR` find `config.yaml` and `assets/` at the
  Space's root unchanged. Cost: the dry run lists 33 package modules, training code included.
- **Requirements**: `uv export --no-dev --no-hashes --group demo` trimmed to torch, torchvision,
  gradio, Pillow, NumPy, matplotlib and PyYAML; timm is out (imported only for EfficientNet, add it
  if the fallback is ever used). A test follows every module the app loads at startup and checks the
  file names exactly those distributions, each line an export line of the lock.
- **README header**: `sdk: gradio`, `sdk_version: 6.30.0` and `python_version: "3.12"` (tested
  against the lock and `.python-version`), `suggested_hardware: cpu-basic`, citation and licences.
- **Deploy script**: an allowed list of Space paths (`app.py`, `requirements.txt`, `README.md`,
  `config.yaml`, `model.pt`, photos directly under `assets/`, `trailcam/**/*.py`); refuses a file
  under this checkout's, the main checkout's or `TRAILCAM_DATA_ROOT`'s `data/` (links followed,
  before any file is loaded), a non-weights file outside the repo, weights that are not a `.pt`
  checkpoint of `DEFAULT_CONFIG`'s model, and untracked examples. Token from `HF_TOKEN` only (no
  `--token`), passed to `HfApi` and never printed; `--dry-run` reads no token, never imports
  `huggingface_hub` and opens no socket. One commit with `create_commit`; files no longer listed
  stay in the Space. `huggingface_hub` is in a new `deploy` group (default groups, not the wheel,
  not the `demo` export).
- `config.yaml` is uploaded (the user's OK) and found by `CONFIG_PATH` at the Space's root.
- Mypy lists: `scripts/deploy_space.py` and `space/app.py` in pyproject.toml and check.sh;
  `tests/test_control_plane.py` updated. Invariants R02, R05 to R07, R20, R21, R23, R33 to R36,
  R38: test cells filled. Learning journal section 14: the 11.01 lines.

Checks: planted faults (data check off, commit check off, a partial token print, dry run
uploading, weights unchecked, non-`.py` package files, no `@gpu`, CPU-only device, the Space's
handler dropped, a gradio pin off the lock, timm added, matplotlib dropped) each fail a test; the
partial-print case first passed, so the token test now looks for any six characters of it.
`space/app.py` built locally with a random ResNet-18 at `space/model.pt` (removed after): CPU,
three scores, a 224 x 224 Grad-CAM. `git check-ignore space/model.pt` prints it. Dry run on the
real repo before the commit: refused (4 uncommitted files), as designed; after the commit (with a random ResNet-18 `.pt`, no `HF_TOKEN`): exit 0, 38 files: `README.md`,
`app.py`, `config.yaml`, `model.pt`, `requirements.txt` and 33 `trailcam/**/*.py` modules (the
only `data/` in the list is the `trailcam/data/` package); no examples, since `assets/` has none
yet (the user picks them, item 8.05).
`check.sh` green (889 passed).

Next: review (tier 2), then 11.02 and 11.03 (the user).

## 2026-10-16: review 11.01: four lows fixed, plus `spaces` first (Opus 5.5)

Review (`/review-session eceba54..HEAD 11.01`, the user's start), tier 2: code-reviewer (Opus 5.5
xhigh) on the range, risk-reviewer (Fable 5.1 high) on the 11 high-stakes files. Scenarios R01 to
R07, R20, R21, R23, R33 to R36, R38, R42 to R45 hold in both reports; R36's "deployed before the
scoring" holds by process only (your step, not checked by code). Spec changed: plan.md sections 8,
10 and 11 (rows R02, R04, R07, R33 to R36); both reviewers traced them, no conflict. No finding
above low; all fixed at the user's go:

- Code-reviewer: the card's `short_description` was 64 characters; the reviewer recalls a Hub limit
  of 60 (not verified), which would fail the first deploy's commit. Now 54, with a length test.
- Code-reviewer: `HF_HUB_DISABLE_TELEMETRY` was set after gradio had already imported
  `huggingface_hub`, which reads it once at import, so the upload's user agent still carried the
  torch version; and the docstrings said a dry run never loads `huggingface_hub`, which gradio does.
  The upload now sets `huggingface_hub.constants.HF_HUB_DISABLE_TELEMETRY`, which the library reads
  at each request (`utils/_headers.py`); the test checks that constant; the docstrings now say the
  library is loaded but a dry run builds no client, reads no token and makes no network call.
- Code-reviewer (R35): the deploy only added files, so a photo taken out of `assets/` stayed public
  as an example. The upload now lists the Space's files and the same commit deletes those under
  `assets/` and `trailcam/` that the plan no longer holds; nothing else in the Space is touched.
  The dry run says so (it cannot look without the network). Two new tests.
- Code-reviewer: `ALLOWED` matched photo suffixes case-sensitively while `example_photos` and the
  guard do not, so `assets/X.JPG` blocked the deploy. Now case-insensitive, with a test.
- Risk-reviewer (could not verify, low at most): `import spaces` came after torch, while Hugging Face
  asks for it first. Moved before torch, with a test of the order.

Each fix was checked by planting its fault back: all six fail a test. Not fixed, noted: whether
gradio's import makes a network call in a dry run (sockets are blocked only after import in the
test); Hugging Face's side (the 60-character limit, pip honouring `--extra-index-url`, `spaces`
provided on the Space) is checked at 11.03 and the switch. The fix touches `scripts/deploy_space.py`,
`space/` and modifies test lines, so one fix round runs on the fix range. `check.sh` green.

## 2026-10-17: fix round 11.01: clean, no third round (Opus 5.5)

Fix round (`/review-session c74777c..HEAD 11.01`, the user's start), tier 2: code-reviewer (Opus 5.5
xhigh) on the fix range, risk-reviewer (Fable 5.1 high) on its five high-stakes files. No finding
from either; scenarios R01 to R03, R05 to R07, R20, R21, R23, R33 to R36, R38, R42 to R45 hold
(R36's "deployed before the scoring" by process only). Spec changed: plan.md section 8 and R35's
test cell, both matching the code. Noted, not fixed: the deploy lists the Space's files and then
commits with no `parent_commit`, so a Hub-side edit in between could be lost (one user, unlikely);
a direct mypy run on `tests/test_space.py` flags `module.loaded = loaded` (tests are outside
check.sh's mypy scope); the Hub-side checks stay with 11.03. Merged `main` (8aa83bf, 13ecf46):
plan.md open question 5 kept from this branch, 6 from main. Review: tier 2, ran both reviewers;
fix round ran, clean.

## 2026-10-17: session 13, item 13.01, the phase-2 plan (Opus 5.5)

Items 13.01 and 13.02 added to `docs/progress.json` (session 13, owner claude; due 2026-10-24 and
2026-10-28), the user's OK of 2026-10-17 through the coordinator. 13.02 (the phase-2
infrastructure) is not implemented here. Docs only: no code, config or notebook changed.

**Done.** `docs/phase2-plan.md`, "Phase 2: improving generalisation to new cameras", covering the
ten agreed points: the goal and baseline (arm A, 3 seeds); the phase-1 results as trans_val means
(small CNN 0.6625 / 0.6216 val, 0.1965 / 0.0775 trans_val; arm A 0.8661 / 0.8557 and 0.3783 /
0.3101; arm B 0.8537 / 0.8428 and 0.3337 / 0.3026, accuracy / macro-F1) and the noise (arm A
seeds: val macro-F1 range 0.0027, trans_val accuracy 0.3525 to 0.4099, macro-F1 range 0.0285);
screen, confirm and keep with numeric thresholds; the rules carried over with the honesty note;
the hypothesis line; the eight-step ladder with change, hypothesis, needs and tier per step; the
budget; the timing; 13.02's infrastructure; future work. `docs/plan.md` section 9 points to it
(a paragraph, not a numbered open question, so it cannot collide with 8.08 or 9.03). Journal
topic 15, "What we'd improve next, and how".

Mid-item, the user's decision through the coordinator: a literature review of improvement methods
for new-location shift will be done separately, as an external research pass, and later become
Part 7 of `docs/research-report.md`. So the ladder (section 6) is marked provisional, to be
re-ranked from that review's evidence, and step 4 (class-balanced loss) is marked low expected
value, citing WILDS in the report's Part 3 (plain training not meaningfully beaten by CORAL, IRM
or Group DRO) and Part 4 (reweighting did not help). No research done here.

Found while writing, recorded in the plan (no code touched):

- Step 1 (resolution) is not config-only: 224 px is a constant in the transforms, the attention
  frame, Grad-CAM, the demo and the Space, so it needs a tier-2 code item. That makes the
  pre-presentation option cost a code item plus 13.02 in the week of 9.02.
- Adding any field to `Config` changes every phase-1 config hash (`config_hash` hashes all fields
  but `arm` and `run_seed`), and `score_test.py` refuses a config whose hash no longer matches its
  ledger row. Each code step that adds a setting must leave the phase-1 hashes unchanged, with a
  test pinning them. Not in 13.02 as agreed; flagged for the user.
- The phase-1 ledger checker refuses one model under two config hashes, a second reason for the
  separate ledger; 13.02's checker has to allow it.
- config.yaml is in the risk regex, so even config-only steps are tier 2 by the script.
- Arm B's seeds spread about three times wider than arm A's on trans_val macro-F1 (range 0.088 vs
  0.029), so the noise bands are per recipe.
- Open questions for the user in the plan's section 11: the screen's val condition (as agreed it is
  stricter than the keep rule's 1 point); the screen's trans_val condition (as agreed it passes
  most no-effect changes, so the 5-hour cap, not the screen, likely ends the ladder); step 5 as one
  change; the Space after phase 2; CLAUDE.md's and R17's "every choice on val".

Checks: every path and relative link in the plan and the new journal topic resolves; every
phase-1 number re-derived from `docs/ledger.md` (means, ranges, per-seed differences, standard
deviations) and the data card (2,394 val images, about 212 batches per epoch); the journal's
command (`uv run python -m trailcam.train.ledger --check docs/ledger.md`) exits 0; `check.sh`
green.

**Next.** 13.02 after the presentation; the open questions in `docs/phase2-plan.md` section 11 are
the user's.

Review: tier 0 (docs only), nothing to run. Spec changed: plan.md section 9 only, cited by no
invariant row.

## 2026-10-17: item 13.01 revised: the ladder from the literature review (Opus 5.5)

The user's "go" through the coordinator, on top of 79c1a10 and before the review: the ladder in
`docs/phase2-plan.md` is rebuilt from an external literature review (three AI research tools run
by the user), replacing the provisional one. Only the facts the coordinator checked against the
original pages are marked verified (the WILDS leaderboard, Gao et al. 2023, Mesnage et al. 2025,
Connect Later); everything else is marked "to verify in 13.03". Item 13.03 (owner claude, due
2026-10-24, the user's OK) added: Part 7 of `docs/research-report.md` from that review, each
citation verified, then the ladder re-checked. No research done here.

The ladder now: 0 a soup and an ensemble of arm A's three seeds (evaluation only); 1 copy-paste,
arm C; 2 SWA; 3 a longer head-only phase; 4 resolution 320; 5 ResNet-50; 6 infrared-aware
augmentation; a "Dropped, with reasons" list (reweighting losses, MixUp, IRM and Group DRO,
wildlife-pretrained backbones, test-time augmentation and ensembles for the demo). Section 3 adds
resolvability: gains under about 4 points cannot be confirmed with three seeds, so steps 2 to 6
confirm on five. Journal topic 15 rewritten to match.

Checked against the repo, and corrected or flagged in the plan:

- trans_val is 1,603 kept images (1,725 released; the data card's filter table). Plan.md section 3
  quotes the released 1,725; not changed here (a phase-1 spec section).
- The cache and the sidecar hold every annotated image, empties included, so arm C needs no cache
  rebuild. But "empties from training-camera locations" would admit cis_test's 1,180, which are
  test images: the pool is the 995 empties of the official cis_val file only, minus any whose
  sequence has a frame in val, with tests. Arm C's tier: 2 at xhigh, tier 3 recommended.
- Resolution (step 4) and the infrared-aware augmentation (step 6) need code, not config: the
  input size and the augmentation strengths are constants in `trailcam/data/dataset.py`.
- Step 3: head-only epochs count inside `max_epochs`, so `max_epochs` rises with them to keep the
  body's 9 epochs (one change, the head-only phase).
- A per-camera breakdown cannot stand in for extra seeds: trans_val is one camera.
- The "tens of milliseconds" on the CPU is not measured in this repo; it is marked expected, to be
  measured at 11.03.
- Arm B on the final recipe (first draft's step 8) is not in the revised ladder: open question 4.
  ABSGD scores 33.0 on the leaderboard, 2.2 above ERM; whether it is a reweighting method is left
  to 13.03, since it bears on dropping the reweighting losses.

Checks: every path and relative link in the plan and in journal topic 15 resolves; the journal's
command exits 0; `check.sh` green.

Review: tier 0 (docs only), nothing to run. Spec changed: plan.md section 9 only (in 79c1a10),
cited by no invariant row.

## 2026-10-17: session 8, item 8.08, the evaluation notebook and the arm B crop check (Opus 5.5)

The user approved item 8.08 on 2026-10-17 through the coordinator, and it was added to
`docs/progress.json` (session 8, owner claude, due 10-18) with the description and steps given.
Main was at b64c486 (item 7.03, the 7 ledger rows), so that is the notebook's default `COMMIT`. It
needs no code from this branch: the notebook calls only entry points that are already on main.

**Done.**

- `notebooks/kaggle_evaluate.ipynb`, committed with its outputs cleared: 8 sections and 22 cells.
  - Parameters, one cell: `COMMIT`, `CACHE_ARCHIVE` (the training notebook's default),
    `CHECKPOINTS_ROOT` (`/kaggle/input/datasets/codx888/trailcam-checkpoints`), and
    `ASSETS_OVERLAY_RUN` (default `None`). The derived paths sit in the same cell. The clone goes to
    `/tmp/who-goes-there`, not to the version's output.
  - Sections 1 to 3 are the training notebook's patterns: the parameter check and the GPU pin, the
    clone at `COMMIT`, the pip install that keeps Kaggle's torch, the import-once guard, the
    byte-for-byte module check, and `check_gpu` with the workers warning. The clone cell also
    refuses a `COMMIT` that lacks `unpack_cache`, any of the three entry points, or the demo's
    `PHOTO_SUFFIXES`.
  - Section 4 runs before anything is unpacked. It checks the ledger, finds the `.pt` files at any
    depth under `CHECKPOINTS_ROOT`, and matches them to the ledger rows by SHA-256. It refuses
    unknown files (counted only), rows with no file (named by run id), and two rows with one hash.
    If the folder holds no `.pt` file, it lists the folders under `/kaggle/input` that do, with
    counts. A file attached twice counts as one checkpoint.
  - Section 5 is the archive unpack and the cache completeness check, as in training.
  - Section 6 calls `evaluate.main` and `attention.main` for each row, in ledger order. Section 7
    calls `slices.main` once per run seed, with arm A and then arm B. Section 8 writes
    `/kaggle/working/eval.zip` and prints the summary.
- Decisions to review:
  - Beyond the item's letter, section 4 checks each row's config at `COMMIT`: config hash, model,
    arm and split seed, as `score_test`'s `_config_of` does. Section 5 checks that the cache's split
    hash equals the rows' (R14). Without these checks, a changed `config.yaml` or another cache
    would evaluate on a different val with no error.
  - The entry points are called in-process (`evaluate.main(argv)` and so on), as the training
    notebook calls `train.main`. Each call is inside `try`/`except SystemExit`, and a non-zero exit
    fails the cell.
  - A non-empty `EVAL_DIR` is refused, so the outputs of two runs in one session never mix.
  - Overlays are made only for the run `ASSETS_OVERLAY_RUN` names, and only if `assets/` holds
    photos at `COMMIT`. Each photo is checked with `assets_image` in section 4. The printed command
    line counts the photos and does not name them (a name may be an image id, R03). The overlays
    take no `=<class>`: no machine-readable class list for the 8.05 photos exists, so their titles
    show the prediction and its confidence.
  - The summary puts each run's val and trans_val accuracy and macro-F1 beside the ledger's. A
    difference above 0.01 prints a CHECK line to stderr, and the cell does not fail. Training scored
    in mixed precision and this notebook scores in float32, so small differences are expected. I
    have not verified that Kaggle keeps a failed version's output, so the notebook warns after the
    zip rather than raising. The SHA-256 pins each checkpoint; the score comparison catches a
    changed pipeline.
  - The small CNN never reaches slices. No `keep_features=False` is needed: the slices command line
    has no such switch, and it keeps the 512 ResNet-18 features the feature maps need.
  - How to run, step 9, is the copy list for `docs/figures/`: seed 0's arm A and arm B and the small
    CNN (val confusion matrix, per-class plot, attention histogram), seed 0's two feature maps, four
    overlays, and seed 0's `slices.md` as `slices_val_seed0.md`. That is 15 PNGs. With the five 8.07
    charts that makes 20 of the guard's 40 files, which leaves 20 for 9.02.
- `notebooks/kaggle_look_at_data.ipynb`, new section 4, an eye check of arm B crops. It shows one
  boxed train photo per class, day and night by turns, with every box outlined. Beside each photo
  are 3 crops from `TrainTransform("B").sampler.box_crop` (the box path; in training only half the
  boxed images take it) and 2 from `.control` (arm A's sampler), each titled with its fill. A line
  prints how many arm B crops hold a whole box at 20% to 100% fill. The intro and the markdown note
  say the outputs are for the user's eyes, cleared before any commit and never put into
  `docs/figures/` (R03, R04). `ARM_B_CROPS` (3) and `ARM_A_CROPS` (2) are in the parameter cell. No
  training code changed. On the synthetic fixture (scratchpad only), all 36 arm B crops held their
  box whole, at fills of 22% to 100%. It has not been run on real photos: that is the user's look
  on Kaggle.
- Tests:
  - `tests/test_evaluate_notebook.py` (47 tests, about 30 s). The notebook's shape: outputs
    cleared; both notebooks pass the guard; no code cell names a test split or `score_test`; every
    `--split` is val or trans_val; the parameter cell is the only one that binds its names; the
    default `COMMIT` holds the 7 rows and the needed code; the one pip install; the GPU pin; the
    order checkpoints, then unpack, then cache, then any evaluation.
  - Same file: the checkpoint cell run as written on made-up files, covering every refusal. The
    evaluation cells run with recorders, each argv parsed by its own module's parser, for empty and
    present `assets/` with and without `ASSETS_OVERLAY_RUN`.
  - Same file: every cell from section 4 to the summary, run as written on a cache-shaped fixture
    with five random-weight checkpoints and the real entry points. The tests check the outputs, the
    zip, R03 on everything printed, the split-hash refusal, and the summary's tolerance.
  - `tests/test_datavis.py` (3 new tests). The crop cell uses the real sampler (static check). It
    is run on the fixture with the samplers spied on, not replaced, and every panel is the crop
    they returned. The markdown note is checked too.
- `docs/review/invariants.md`. R03's rule text now lists the crop check as a second exception
  beside 8.07's photo grid, same terms: train photos, Kaggle only, no file name, outputs never
  committed. This rule-map change was approved by the user, via the coordinator, 2026-10-17. The
  test cells of R03, R04, R06, R07, R14, R17, R24, R38 and R41 now name the new tests. Items or
  paths were added to the where columns of R04, R14, R17, R24, R38 and R41.
- `docs/learning-journal.md`, topic 10: one entry for 8.08.

Steps: `uv run pytest tests/test_evaluate_notebook.py` 47 passed. `uv run pytest tests/test_datavis.py
-k "look or crop"` 7 passed. Ruff check and format are clean on both notebooks. Mutation checks:
22 faults planted in the two notebooks, each restored by copying from a backup, never with
`git checkout`, and each makes a test fail. The faults:

- the evaluation notebook: cis_test in the split list; an unknown file, or a missing one,
  accepted; overlays on every run, or without photos; the small CNN paired for slices, or its file
  passed to slices; photo names printed; the CHECK, split-hash, config, `EVAL_DIR`, unpaired-seed,
  overlay-run, `assets_image` or attention-exit check removed; the zip of one folder only;
- the look notebook: a hand-made arm B crop; arm A from a new sampler; no day/night alternation;
  val photos.

One small-CNN fault first survived because I had written it as an equivalent change; rewritten,
it fails. `check.sh` green (943 passed).

**For 8.06: its steps can now be run** with this notebook (How to run, steps 1 to 9, the
`trailcam-checkpoints` dataset first). 8.05's photos are optional for the run: without them the
notebook makes no overlays and says so.

Notes, not fixed:

- The time of a full Kaggle run is not measured. Each step prints its seconds.
- Not verified on Kaggle: the mount path of the new dataset (the search covers any depth), and
  whether a committed version that fails keeps its output.
- The look notebook still reads the loose 6.03 dataset (`DATA_ROOT`), as in 8.07, not the archive.

**Next.** Review 8.08. Then you merge it, then 8.05 (photos) and 8.06 on Kaggle.

Review: tier 2, `/review-session 8.08`

Review: tier 2, ran `/review-session 8.08` (b64c486..4c4c594; code-reviewer Opus 5.5 xhigh on the
range, risk-reviewer Fable 5.1 high on the two notebooks and two test files). Nothing found by
either, so no fix round. Scenarios R01, R03, R04, R06, R07, R14, R16, R17, R24, R27 (not
applicable, nothing trains), R37, R38 and R41 to R45 held. Spec changed: none; R03's rule text in
`docs/review/invariants.md` changed, and the user approved its wording via the coordinator on
2026-10-17. Noted by the reviewers, not fixed:

- an arm B crop the sampler cannot draw would be titled "no usable box" and lower the printed
  count (none on the fixture);
- mypy run directly on the two test files reports 11 errors (tests are outside check.sh's mypy
  scope);
- Kaggle behaviour, the dataset mount path and whether GitHub's `main` holds b64c486 could not be
  verified offline.

Merged `main` at c4b1e56 (13.01) through the sync tool: one conflict, in this log; main's two
13.01 entries kept first, then this one. `check.sh` green after the merge (943 passed).

## 2026-10-17: session 9, item 9.03, the final scoring's notebook (Opus 5.5, xhigh)

New item, approved by the user via the coordinator on 2026-10-17 and added to
`docs/progress.json` as 9.03 (session 9, owner claude, due 2026-10-19). Main at b64c486. Built and
tested on the synthetic fixture only: no real checkpoint was loaded and no real cis_test or
trans_test was scored (that is 9.02, the user's single run on Kaggle).

**Done.** `notebooks/kaggle_score_test.ipynb` (22 cells, outputs cleared) and
`tests/test_score_notebook.py` (44 tests).

- Parameter cell: `COMMIT` (default b64c486, main at the session start, whose ledger holds the 7
  rows), `CACHE_ARCHIVE` (the training notebook's default), `CHECKPOINTS_ROOT = None`,
  `APPROVED_BY = ""`, `RUN_SEEDS = (0, 1, 2)`, and the fixed paths: the clone's `config.yaml`,
  `docs/ledger.md`, `docs/test-ledger.md` and `assets/`, and `OUT = /kaggle/working/score`.
- Sections 1 to 3 follow the training notebook: the parameter check and `CUDA_VISIBLE_DEVICES=0`,
  the clone at `COMMIT`, the one `pip install` keeping Kaggle's torch, the stale-import guard, the
  installed modules compared with the clone byte for byte, `check_gpu`, `unpack_cache` (the loose
  dataset refused in an interactive session), and the cache check with `check_images`.
- Section 4 finds the `trailcam-checkpoints` folder under `/kaggle/input` (any depth, links
  followed, two mounts of one folder counted once; none or several refused with the folders that
  hold `.pt` files) and calls `score_test.py`'s own checks, the ones its `prepare` runs first:
  `resolve_checkpoints` with `read_ledger`, `read_test_ledger` and `check_run_seeds`. So a missing
  or changed checkpoint, an incomplete ledger or a test ledger with a row fails the rehearsal,
  before any weights are read. Nothing is duplicated; the scoring repeats every check.
- Section 5, the go-ahead, is the code cell right before the scoring: `check_approver(APPROVED_BY)`,
  re-raised as "the final scoring needs the user's go-ahead ... nothing has been scored". A Run All
  with `APPROVED_BY` empty is therefore a full rehearsal that loads nothing.
- Section 6: one `subprocess.Popen` of `python -m trailcam.evaluate.score_test` with
  `--approved-by=<APPROVED_BY>` (the `=` form, so a name starting with a dash parses),
  `--run-seeds`, every checked checkpoint, `--data-root`, `--config`, `--ledger`,
  `--test-ledger`, `--out`, `--assets` and one `--overlay` per photo directly under the clone's
  `assets/`. Output streamed line by line (`PYTHONUNBUFFERED`, `errors="replace"`, so a decoding
  error cannot close the pipe mid-run). Exit 0 or 3 goes on; any other exit raises, saying what
  to do.
- Section 7: the clone's test ledger read back with `read_test_ledger`, checked one row per
  checkpoint by sha256, and printed under `scoring_header()` for pasting; then `OUT` zipped to
  `/kaggle/working/score.zip`.
- Markdown: this runs once; `APPROVED_BY` starts empty (the rehearsal); never Save & Run All with
  it filled in until the user means it (said at the top and over the go-ahead cell); set it back
  to `""` right after the run; if it stops before `appended ... rows`, rerun from the start with a
  new `--out` (a new version starts with an empty `/kaggle/working`); if it stopped after, never
  run it again. Licence and citation.
- plan.md section 7 (Protocol): one sentence naming the notebook and the empty `APPROVED_BY`.
  invariants.md: R06, R07, R16, R17 (and 9.03 in its items), R38, R40 and R41 test cells name
  `tests/test_score_notebook.py`. Learning journal topic 9: five lines.

Decisions to review:

- The clone may differ from `COMMIT` in `docs/test-ledger.md` only (the training notebook lets
  `docs/ledger.md` differ). A session that has scored then reaches the run-once refusal in
  section 4, rather than being told to factory-reset into a fresh clone whose test ledger is
  empty again.
- Exit 3 (a leakage alarm on the test scores) does not raise, unlike the training notebook. The
  rows are appended by then and the scoring cannot run again, and I have not verified that Kaggle
  keeps a failed version's output. So the cell reports the alarm on stderr and again after the
  zip, and the outputs are kept. R16's test cell says so.
- Overlays carry no class (`--overlay <photo>` only), so their titles show the predicted class
  without a correct or wrong verdict. `assets/` is empty until 8.05, so there are none yet. A true
  class per photo would need another parameter; that is the user's call.

Not covered here (limits):

- A fresh clone at the same `COMMIT` cannot know that the scoring ran until the rows are committed
  to `docs/test-ledger.md`. Until then only `APPROVED_BY` guards a second run, hence the "set it
  back" step. A possible follow-up, the user's call: the notebook could also refuse when
  `origin/main`'s `docs/test-ledger.md` holds a row.
- Kaggle-side facts not checked: where the `trailcam-checkpoints` dataset mounts (the search
  covers any depth), the scoring's run time on a T4, and Kaggle's handling of a failed version's
  output.

Steps:

- `uv run pytest tests/test_score_notebook.py`: 44 passed, covering outputs cleared and the guard
  (`content_problems`, `path_problems`); `APPROVED_BY` defaults to `""`, and the go-ahead cell
  refuses `""`, blanks, `None`, `|` and line breaks, and comes right before the scoring cell;
  exactly one code cell runs score_test (the module named once, one `Popen`, not in a loop, no
  other process but git and pip); the exact names imported from `trailcam` (from `score_test.py`
  no `main`, `prepare`, `load_models` or `score_checkpoint`), no dynamic code, and `trailcam.` used
  for `__file__` only; no cell imports, names or calls evaluate, attention, slices or train, or
  any `main`; only the parameter cell binds the parameters and paths; `RUN_SEEDS` (0 1 2, 0, 2 0)
  reaches `--run-seeds`, as score_test's own parser reads the command; the default COMMIT's
  ledger, test ledger and configs pass score_test's checks for 0 1 2.
- The cells as written on the fixture of `tests/test_score_test.py` (random weights): the
  rehearsal passes every check and stops at the go-ahead, with `torch.load`, `Image.open` and
  `subprocess.Popen` replaced by functions that fail; a missing or changed checkpoint, a test
  ledger with a row and a missing seed are refused in section 4; a full run through
  `python -m trailcam.evaluate.score_test` exits 3 (random weights fire the alarm, as intended),
  appends 3 rows in ledger order with the approver, prints them under the header, zips 34 files,
  and a second run is refused. The clone cell, run on a local clone of this repo, lets a changed
  test ledger through and refuses a changed ledger.
- Mutation checks, each restored by copying the notebook back, each failing a test: 23 of 23
  killed. The refusal paths: `APPROVED_BY` defaulted to a name; the check removed; the check
  bypassed (`APPROVED_BY or "nobody"`); the refusal printed instead of raised; the go-ahead cell
  moved after the scoring; deleted. A second scoring: the scoring cell duplicated; `main(argv)`
  imported from score_test; `subprocess.run(command)`; the `Popen` in a loop;
  `trailcam.evaluate.score_test.main` by attribute. An evaluate call: `-m trailcam.evaluate.evaluate`;
  `predict` imported from evaluate; `attention_pass` through score_test; `train.main`. Others: the
  seeds hard-coded; the approver hard-coded; `RUN_SEEDS` rebound in the scoring cell; the clone
  check excluding `docs/ledger.md`; a failed exit not raised; the pre-check given no test ledger;
  an output left in a cell. One survived at first (the scoring cell falling back to a stale
  `approver` when `APPROVED_BY` is empty); a new test now checks that the command carries
  `APPROVED_BY` itself, which score_test refuses.
- `uvx --from nbconvert jupyter nbconvert --clear-output --inplace notebooks/kaggle_score_test.ipynb`
  leaves it unchanged (`cmp`); `uv run python scripts/guard_data.py --all` passes; ruff lints and
  formats the notebook cleanly.

`check.sh` green (937 passed).

**Next.** Review, tier 3 (the final scoring): `/review-session b64c486..HEAD 9.03`. Its tier rule
names 9.01 and `score_test.py` but not this notebook, so it reports tier 2 by itself; the
tier-3 option, `/code-review ultra`, is the user's. Then 9.02, the user's single run with this
notebook.

## 2026-10-17: review 9.03: one low fixed, `main`'s test ledger read too (Opus 5.5, xhigh)

Review (`/review-session b64c486..HEAD 9.03`, the user's start): code-reviewer (Opus 5.5 xhigh)
on the range, risk-reviewer (Fable 5.1 high) on its two high-stakes files, the notebook and its
tests. Tier 2 by the command's rule (it names 9.01 and `score_test.py`), tier 3 by CLAUDE.md (the
final scoring). Scenarios R03, R06, R07, R16, R17, R27, R37, R38, R40, R41 and R42 hold in both
reports. Spec changed: plan.md section 7 and the test cells of R06, R07, R16, R17, R38, R40 and
R41; both reviewers traced them, no conflict. One finding:

- Low (risk-reviewer; code-reviewer listed it under "could not verify" as a design limit): the
  run-once refusal never fires on Kaggle, since a fresh clone at `COMMIT` holds that commit's empty
  test ledger. Once the rows are on `main`, an `APPROVED_BY` left filled in would let a later Save
  & Run All score every checkpoint again.

Notes, not findings: mypy run directly on the test file reports 3 type errors (tests are outside
`check.sh`'s mypy scope); the markdown says to set `COMMIT` at or after 9.03's merge while the
default is earlier (harmless: `trailcam/` is the same across the range).

The user's decisions, via the coordinator on 2026-10-17: (1) fix the low; (2) no ultra review for
9.03: it is a thin wrapper over `score_test.py`, which had its own ultra review, so this item gets
the two standard reviewers and the fix round; (3) bring the branch up to date with `main`.

Merged `main` at c4b1e56 (13.01, the phase-2 plan) with the sync tool, as 014c19d. The conflict was
in this log only; both sides kept, 13.01's entries first. `check.sh` green after the merge (937
passed).

Fix:

- A new code cell right after the imports (section 3) fetches `origin/main` into the clone, writes
  its `docs/test-ledger.md` to `MAIN_TEST_LEDGER` (under `/tmp`, a new constant of the parameter
  cell) and reads it with `score_test.py`'s `read_test_ledger`. Any row refuses: "the final
  scoring has already been recorded on main, and it runs once". It comes before the GPU check and
  the unpacking, so it fails in seconds; every existing refusal stays.
- Markdown: "This runs once", How-to-run steps 5 and 7 (commit the rows to `main` right after the
  run), and section 3. plan.md section 7's run-once wording (the user's OK); R17's and R40's test
  cells; the journal's 9.03 lines.
- Tests (45 in the file). The new one uses a local stand-in for the public repo whose `main` is the
  default `COMMIT`: the cell passes. After a row is committed to that `main`, the reused clone
  fetches it, and the cells run in notebook order stop at the new cell, before the unpacking and the
  scoring cell. The order test includes the new cell. The fixtures' clone now gets the repo's test
  ledger cut at its header (`_empty_test_ledger`), so these tests still pass once 9.02's rows are
  in `docs/test-ledger.md`.
- Mutation checks, each restored by copying: the check disabled, the cell deleted, the fetch
  removed, `COMMIT`'s ledger read instead of `main`'s, and the cell moved after the scoring: 5 of 5
  killed. The earlier 23 rerun on the fixed notebook: 28 of 28 killed.
- `nbconvert --clear-output` leaves the notebook unchanged; ruff clean.

Found, not fixed (outside this item): the fixture of `tests/test_score_test.py` copies the repo's
`docs/test-ledger.md` as its empty test ledger (`_build_world` and `fresh`). Once 9.02's rows are
pasted there, its scoring tests see 7 rows and fail, so `check.sh` goes red when the rows are
committed. The same cut-at-the-header fix applies; it needs the user's OK (9.01's tests, a
high-stakes file) before 9.02's rows are committed.

The fix touches a high-stakes file and modifies test lines, so one fix round runs on the fix range.
`check.sh` green (938 passed).

**Next.** Fix round: `/review-session 014c19d..HEAD 9.03`. Then 9.02, the user's single run.

## 2026-10-17: fix round 9.03: one high-by-rule and one low fixed, and 9.01's fixture trap (Opus 5.5, xhigh)

Merged `main` twice more with the sync tool, as the coordinator asked: dc19821 (8.08) as 8fbfb8e,
with conflicts in this log and in rows R06, R07, R17, R38 and R41 of invariants.md (both items had
added a test-file clause; `main`'s rows kept and each 9.03 clause added whole, so every row names
`tests/test_evaluate_notebook.py` and `tests/test_score_notebook.py`, and R17's items read "8.08,
... 9.03"); then 05cd17a (8.06) as a0bbbaf, with a conflict in plan.md section 7 (the run-once
sentences kept, 8.06's "Val results" bullet after them). `check.sh` green after each (988
passed).

Fix round (`/review-session 9.03`, the user's start; range dc19821..HEAD, the whole item against
today's `main`, its first commit reviewed before): code-reviewer (Opus 5.5 xhigh), risk-reviewer
(Fable 5.1 high) on the notebook and its tests. Scenarios R06, R07, R16, R17, R27, R37, R38, R40,
R41 and R42 hold in both reports; R03 breaks for the code-reviewer (finding 1). Findings:

- High by R03's rule, nothing leaked yet (code-reviewer; the risk-reviewer saw the same print and
  judged it allowed, since `assets/` photos are not from `data/`): the scoring cell printed
  `shlex.join(command)` with every `--overlay <clone>/assets/<photo>` path. R03, as approved in
  8.08, counts an `assets/` photo's name in printed output as a breach (a photo's name may be an
  image id), and the evaluation notebook counts its photos instead. `assets/` is empty until 8.05.
- Low (code-reviewer): "committed to `main`" in the notebook and plan.md section 7; the check
  fetches `main` from GitHub, so a local commit blocks nothing until it is pushed.

The user's decisions ("go", via the coordinator, 2026-10-17): fix both, and fix the trap reported
in the review 9.03 entry above (9.01's tests copying `docs/test-ledger.md`). These are the fix
round's fixes: no third round unless the user asks.

Fixes:

- A. The scoring cell prints its command without the `--overlay` arguments, plus " and N --overlay
  photos of assets/"; the overlays reach `Popen` only (`[*command, *overlays]`). Section 6's
  markdown says why. Tests: the cell alone with a stand-in `Popen` prints the count and not the
  photo's name; the cells on the fixture, with the real scoring and a photo named like an image id,
  print the count and the name nowhere in stdout or stderr. R03's test cell names them.
- B. "committed and pushed to `main` on GitHub" in the notebook's "This runs once" section,
  How-to-run step 7 and section 3, and in plan.md section 7 ("Between the run and that push").
  R40's test cell too.
- C. `tests/test_score_test.py`: `_empty_test_ledger` writes the header of `TEST_LEDGER_COLUMNS`
  (`scoring_header`) and no row; `_build_world` and `fresh` use it, never a copy of
  `docs/test-ledger.md`, and `tests/test_score_notebook.py` imports it in place of its own.
  `test_the_repo_test_ledger_reads` already accepts rows (docstring says so). New test: a copy of
  the repo's test ledger with a valid row reads through the reader, and with `TEST_LEDGER_PATH`
  pointed at it the fixtures still start with no row; only the two tests that check the repo's
  file name it. Proved by hand: with a row added to the repo's `docs/test-ledger.md`, the whole
  suite passed except the new test itself, which had assumed the repo file had no row (fixed to
  read the appended row last); both files then passed with the row in place (103 passed). The file
  was restored by copying and checked with `cmp`. R40's test cell names the rule.
- Mutation checks, each restored by copying: the photo names printed again (killed by the cell's
  test); `_build_world` and `fresh` copying the repo's test ledger again (each killed by the new
  test): 3 of 3, rerun after the test's fix.
- Journal: a timeline entry, "the 6.04 trap, a second time".
- `nbconvert --clear-output` leaves the notebook unchanged; ruff clean.

Not verified (both reviewers): Kaggle-side behaviour (dataset mounts, whether Factory reset clears
`/kaggle/working`, the live fetch from GitHub) and the run time: whether 7 checkpoints with the
Grad-CAM attention pass over both test splits fit in one 12-hour GPU session. 8.06's val run is
the nearest timing.

`check.sh` green (989 passed).

**Next.** Ready to merge (this merges first, before 13.02 and 13.04). Then 9.02 on 2026-10-19:
the user's single run with this notebook.

## 2026-10-17: session 13, item 13.02, the phase-2 infrastructure and config-only variants (Opus 5.5)

The user's decisions of 2026-10-17, through the coordinator: phase 2 runs this week, before the
presentation (10-18 infrastructure and arm C's code, 10-19 screen, 10-20 confirm, 10-21 stack,
10-22 freeze); exactly one phase-2 test scoring, on 10-22, of the frozen final recipe only, written
into the plan before any phase-2 result; 13.02's description and due date (now 2026-10-18)
updated in `docs/progress.json` with the user's approval. Base `c4b1e56`.

**Done.**

- `docs/ledger-phase2.md`: its own text, then the 16-column header of `docs/ledger.md`, no rows.
  `trailcam/train/ledger.py` checks a ledger named `ledger-phase2.md` as phase 2: every cell
  check, a `p2_` config on every row, several config hashes per model but one per config, one
  split seed for the whole ledger, and a summary grouped by config first. Any other ledger
  refuses a `p2_` row, so one pasted into `docs/ledger.md` fails `check.sh` and the final
  scoring's read.
- `trailcam/train/train.py`: `check_ledger_phase` refuses a `p2_` config with any ledger but
  `docs/ledger-phase2.md`, and a phase-1 config with that one, before training (with `preflight`)
  and again in `append_row`; `PHASE2_LEDGER_NAME`, `PHASE2_LEDGER_PATH`. The config's
  `input_size` reaches `TrainTransform(arm, size=...)` and `EvalTransform(size)`. The loop, the
  optimizer, the schedule, early stopping and the AMP path (its scheduler warning included) are
  unchanged.
- `notebooks/kaggle_train.ipynb`: parameter `LEDGER = "docs/ledger.md"` (or
  `"docs/ledger-phase2.md"`), passed to `--ledger` as `CLONE / LEDGER`; the first check cell
  refuses a `LEDGER` other than the two and a config of the other phase, before the clone; phase-1
  names stay an explicit list, `p2_` names match `p2_[a-z0-9_]+` and an unknown one fails at
  `load_config` after the install, before the cache is unpacked. Outputs cleared.
- `trailcam/train/config.py`: field `input_size` (at least 32; 224 for `small_cnn`, whose head is
  sized for it); `HASH_NEUTRAL = {"input_size": 224}`, left out of `config_hash` at that value, so
  the phase-1 hashes are unchanged; `PHASE2_PREFIX = "p2_"` and `is_phase2`.
- `trailcam/data/dataset.py`: both transforms take a `size`, 224 by default; `eval_resize(size)`
  keeps 256/224's margin (366 for 320). `trailcam/data/loaders.py`: `make_loaders` takes the eval
  transform and refuses a train transform of another size.
- `trailcam/model/`: `resnet50`, torchvision's `IMAGENET1K_V1` weights, head `fc` (2048 to 12),
  Grad-CAM layer `layer4`.
- `trailcam/evaluate/attention.py`: `load_checkpoint`, which every evaluation tool, the demo and
  the Space use, refuses a config whose `input_size` is not 224 before reading the file, so a
  320 px model is never scored at 224. Phase-1 configs are all 224, so nothing they do changes.
- `config.yaml`: `input_size: 224` in `common`; `p2_baseline` (arm A exactly, config hash
  `483aa055…`, arm A's), `p2_head3` (`head_only_epochs: 3`, body trains 7 of 10; `a21b04f0…`),
  `p2_res320` (`input_size: 320`; `5a20cf07…`), `p2_r50` (`model: resnet50`; `05a685a9…`).
- Tests: `tests/test_config.py` (`P2_CHANGES`: each `p2_` config is `resnet18_arm_a` with exactly
  its named change; R31's recipe check narrowed to the phase-1 configs, now with 224 px; the
  phase-1 hashes pinned to every `docs/ledger.md` row; `input_size` hash-neutral at 224 only; the
  phase is the name); `tests/test_ledger.py` (the real phase-2 ledger passes, from the shell too; a
  `p2_` row pasted into a copy of `docs/ledger.md` fails; a phase-1 row in the phase-2 ledger
  fails; several hashes per model pass, two per config and two split seeds fail; the shared checks
  hold; the summary groups by config); `tests/test_train.py` (the mix-ups stop before training
  and `append_row` refuses them; a phase-2 run appends to the phase-2 ledger; the input size
  reaches both transforms; the real phase-2 ledger's shape); `tests/test_notebook.py` (`LEDGER`
  default; the check cell run as written accepts each phase's pair and refuses 8 mix-ups);
  `tests/test_models.py` (ResNet-50's `/review-shapes` table, 23,532,620 parameters, V1 weights by
  spy, both ResNets at 320 px); `tests/test_dataset.py`, `tests/test_loaders.py`,
  `tests/test_gradcam.py`, `tests/test_attention.py`, `tests/test_boxcrop.py` (the arms still
  differ in the sampler alone, at one size).
- Docs: `docs/phase2-plan.md` (this week's timeline; the control re-run; screen in parallel,
  confirm, then stack; a pinned commit per batch; the pre-declared test scoring and 13.06; the
  phase-2 transform exception, CLAUDE.md and `.claude/rules/` unchanged for phase 1; steps 3 to 5
  now config-only; the budget; section 9 as built; open question 7 answered, 8 and 9 new);
  `docs/plan.md` section 9's phase-2 paragraph (it said phase 2 waits for the presentation);
  phase-2 lines in R17, R20, R25, R30, R31 and R40; journal topic 15 (five lines).

**Training-path files changed since `61f1ecc`**, phase 1's training commit: the modules
`trailcam.train.train` imports (arm B's sampler included), plus `config.yaml`, the notebook,
`pyproject.toml` and `uv.lock`, diffed with git.

- Before this item (`61f1ecc..c4b1e56`): `trailcam/data/cache.py` (docstring only),
  `trailcam/evaluate/metrics.py` (`timed_forward` and `per_camera_scores` added; `accuracy` and
  `macro_f1` unchanged), `notebooks/kaggle_train.ipynb` (the archive's default path and two
  messages), `pyproject.toml` (the demo and deploy dependency groups, outside the wheel),
  `uv.lock`.
- This item: `config.yaml`, `notebooks/kaggle_train.ipynb`, `trailcam/data/dataset.py`,
  `trailcam/data/loaders.py`, `trailcam/model/backbone.py`, `trailcam/model/factory.py`,
  `trailcam/train/config.py`, `trailcam/train/train.py`.
- Unchanged: `trailcam/data/boxcrop.py`, `classes.py`, `metadata.py`, `split.py`, `synthetic.py`,
  `trailcam/model/small_cnn.py` and the packages' `__init__.py` files. `p2_baseline` seed 0 is
  the GPU check that none of it moved arm A (the plan's section 3).

**Choices made here, for the user to see.**

- ResNet-50's weights are `IMAGENET1K_V1` (76.1% ImageNet top-1), not the default `V2` (80.9%,
  a newer, heavier-augmented recipe): ResNet-18 has only V1, so V2 would be a second change.
- The evaluation tools refuse a non-224 config rather than read the size from the checkpoint:
  `load_checkpoint` refuses any checkpoint key beyond the three `train.py` writes, so the format
  stays. This touches the final scoring's code path (`score_test.py` calls it) with a refusal no
  phase-1 config can meet.
- Every pair is with `p2_baseline` at the same commit, its seeds 1 and 2 re-run on 10-20 (open
  question 8); the control check uses section 2's bands around phase 1's arm A seed 0.
- A stacked config will need a named exception in `test_config.py`'s one-change rule.
- `p2_head3` keeps one warmup epoch, so the body joins in epoch 4 at about 0.88 of its peak rate.
- Item 13.06 (`score_test.py`'s phase-2 mode) is named in the plan but not added to
  `docs/progress.json`: a new item needs the user's OK.
- This week's worst case is about 8 GPU hours against the plan's 5-hour cap (open question 9).

Checks: `uv run python -m trailcam.train.ledger --check docs/ledger-phase2.md` exits 0 ("phase-2
ledger: 0 rows, every check passed"), and on `docs/ledger.md` 0 (7 rows); `/dryrun`'s command
(resnet18 arm B, fixture) trained one epoch, logits (2, 12), train loss 2.6539, val loss 2.4667,
row printed and not appended; `docs/ledger.md` unchanged; the phase-1 config blocks unchanged
(`common` gained `input_size: 224`, the value phase 1 hard-coded; hashes pinned by the test);
`bash scripts/check.sh` green.

Merged `main` at dc19821 (8.08) through the sync tool: conflicts in this log (8.08's entry kept
first, then this one) and in `docs/review/invariants.md`, merged cell by cell against c4b1e56;
only R17 had a cell changed on both sides (its "where" now lists items 8.08 and 13.02, its test
cell 8.08's notebook tests and then 13.02's). `check.sh` green after the merge.

**Next.** Merge, then 13.04's `p2_arm_c` adds its `P2_CHANGES` line; before the 10-19 screen,
one commit with every variant's hypothesis line, pinned as the notebook's `COMMIT`. Open
questions 8 and 9 and item 13.06 are the user's.

Review: tier 2 (config, notebook, ledger checker and `train.py` are high-stakes; the
`load_checkpoint` refusal is on the final scoring's path, see above). Command: `/review-session
13.02`. Spec changed: `docs/plan.md` section 9 (R31), `docs/phase2-plan.md` (now cited by R17,
R20, R25, R30, R31, R40).

## 2026-10-17: item 13.02, the plan's trans_val caveat from 8.06 (Opus 5.5)

The coordinator's request (2026-10-17), from 8.06's evaluation run, applied to
`docs/phase2-plan.md` on top of 2cddcb9; docs only.

- Section 2: trans_val is one camera (location 125) and a bobcat-heavy mix (734 of 1,603, 46%;
  the per-class counts are the data card's), phase 1's trans_val spread is mostly bobcat recall
  there (0.19 to 0.40, missed bobcats mostly called cat), and macro-F1 averages in rabbit (9) and
  bird (7); so a trans_val gain is weak evidence of a general new-camera gain.
- Section 3: val macro-F1 must hold (the keep rule's condition 3 and the screen's val condition
  are requirements, never traded for trans_val); trans_val accuracy and macro-F1 are reported
  together with bobcat recall and the per-class table as context; no selection metric added.
- Section 4: the pre-declared scoring on trans_test (21,473 images, 9 cameras, the data card) is
  the real check of a phase-2 gain.
- Section 6, step 1: 8.06's animal-removed test leaves val accuracy about 0.57 with the boxes
  blanked (chance 1 in 12, 0.08), evidence of background reliance, which arm C targets.

Checked against the repo: the counts, the one location and trans_test's 21,473 images on 9
locations match `docs/data-card.md`. Location 125, the bobcat recall range and the 0.57 are 8.06's,
as the coordinator reported them, and are marked so in the plan until 8.06's entry lands.

Found: the per-class context has no path for phase-2 checkpoints yet. The evaluation notebook
(8.08) reads `docs/ledger.md` only and the evaluation tools refuse a non-224 config, so
`p2_res320` gets none; the plan says a small follow-up item is needed (the user's OK).

Review: docs only, inside 13.02's range; `/review-session 13.02` covers it. Spec changed:
`docs/phase2-plan.md` sections 2, 3, 4 and 6 (cited by R17, R20, R25, R30, R31, R40).

## 2026-10-17: review 13.02: one low finding, fixed in the docs; main merged at 05cd17a (Opus 5.5)

The user ran `/review-session 13.02` on `dc19821..38b07c9` (tier 2): code-reviewer (Opus 5.5,
xhigh) on the range, risk-reviewer (Fable 5.1, high) on the 20 high-stakes files. All 24
scenarios held for both (R03, R05, R06, R07, R12, R14, R16, R17, R20 to R27, R30 to R32, R37,
R38, R40 to R42). The code-reviewer ran `check.sh` (1001 passed, none skipped); the
risk-reviewer the ten test files (399 passed), mypy and ruff.

- **Low, both reviewers** (the risk-reviewer as a residual exposure): the 224 px refusal in
  `load_checkpoint` checks the config the caller names, not the checkpoint, which records only
  model, classes and weights. So a `p2_res320` checkpoint loaded under a 224 config (the eval
  command lines with `resnet18_arm_a`, the demo, the Space, `deploy_space.check_weights`, which
  loads with `DEFAULT_CONFIG`) is scored or served at 224 with no error, and R20's phase-2 line
  promised more. Verified from the code: `scripts/deploy_space.py:138-145`,
  `trailcam/evaluate/attention.py:539-559`; ResNet-18's weights are the same at 320 px
  (`test_the_resnets_take_320_px`). Fixed by the user's choice (option a): the limit is stated
  in R20's phase-2 line, `docs/phase2-plan.md` section 4, step 4 and open question 6. Options b
  (the checkpoint carries its input size, a checkpoint-format change) and c (the deploy matches
  the weights' sha256 to a ledger row of the named config) stay the user's, due before a phase-2
  checkpoint is deployed or test-scored (13.06). `tests/test_attention.py`'s docstring is left as
  it is: it describes the config path, which the guard does refuse.
- **Note, risk-reviewer:** `score_test.py` does not refuse `docs/ledger-phase2.md` by name;
  already left to 13.06 in the plan (a phase-2 ledger has no small-CNN row, so it cannot be
  scored today). No change.
- **For the user to confirm, code-reviewer:** the approvals the docs cite as "the user's decision
  of 2026-10-17 via the coordinator": R17's phase-2 line (step decisions on trans_val, one more
  test scoring on 10-22) and the edit of 13.02's description and due date in `progress.json`.

The coordinator's request, from 13.04's review, for `train.py` to refuse a phase mix-up from a
shell run: already in d8c7be1 (13.04 was reviewed against `main`, which lacks 13.02). Checked on
a scratch copy of HEAD, the worktree untouched: removing the call in `main()` fails 3 tests, and
removing the one in `append_row` fails 3; from the shell, both mix-ups exit 2 with "before
training: ..." and leave the ledger copy unchanged. No change.

Merged `main` at 05cd17a (8.06's figures and results) through the sync tool, no conflicts. The
plan now cites plan.md section 7 for camera 125 and the bobcat recall, and for the animal-removed
test its range, val accuracy from about 0.90 to 0.50 to 0.61 (the coordinator had sent about
0.57).

No fix round: the fix touches `docs/phase2-plan.md`, `docs/review/invariants.md` and this log
only, no high-stakes file. `check.sh` green.

Review: tier 2, ran both reviewers; one low finding, fixed (docs); no fix round.

## 2026-10-17: review 13.02, the low finding fixed in code: the checkpoint carries its input size (Opus 5.5)

The user's decision ("go", 2026-10-17, through the coordinator): fix the review's low finding in
13.02, since `p2_res320` trains on 10-19. On top of a94b482; `main` is still 05cd17a, already
merged.

- `trailcam/train/train.py`: every new checkpoint holds `input_size` beside `model`, `classes`
  and `state_dict`. Not training logic: the loop, the optimizer, the schedule and the AMP path are
  unchanged.
- `trailcam/evaluate/attention.py`, `load_checkpoint`: accepts `CHECKPOINT_KEYS` (with the size)
  or `PHASE1_CHECKPOINT_KEYS` (without it, phase 1's seven checkpoints, which count as 224 and
  load as before); refuses a size that is not a whole number, and a size other than the config's
  ("<file> was trained at 320 px, but the config's input size is 224 px"). New keyword
  `eval_size` (default 224), the side of the caller's eval transform: the refusal of a config of
  another size, before the file is read, now compares with it. Every caller passes none, so their
  behaviour is the old one plus the checkpoint check.
- The one loader: the only `torch.load` in `trailcam/`, `space/` and `scripts/` is in
  `load_checkpoint`. It is called by `evaluate.py`, `attention.py`, `slices.py`, `score_test.py`,
  and `predict.load_model`, which the demo app, the sort, `space/app.py` and
  `scripts/deploy_space.py` use. No notebook loads a checkpoint. `score_test.py` is not changed;
  its phase-1 checkpoints (three keys) load as before.
- `trailcam/demo/predict.py`: `load_model`'s docstring names the size check.
- Tests: `tests/test_attention.py` (a phase-1-format checkpoint loads as 224 and is refused under
  a 320 config; a 320 px checkpoint loads under `p2_res320` with `eval_size=320`, is refused under
  `resnet18_arm_a` and, at the tools' 224, under `p2_res320`; a size of `True`, `"320"` or
  `224.0`, or an extra key, is refused); `tests/test_train.py` (`train_run`'s real checkpoint,
  training and scoring stubbed, holds `CHECKPOINT_KEYS` with 224 for `resnet18_arm_a` and 320 for
  `p2_res320`, and loads under its own config only); `tests/test_deploy_space.py` (a 320 px
  ResNet-18 checkpoint is refused by the deploy).
- Mutation check on a scratch copy of the tree, the 13 checkpoint and size tests: saving no size
  in `train.py` fails 2; skipping the loader's size comparison fails 4; refusing the three-key
  format fails 2; reading a missing size as 320 fails 2; skipping the `eval_size` check fails 2.
  Unmutated, all 13 pass.
- Docs: R20 (rule's phase-2 line, test cell, and its "where" now lists `trailcam/train/train.py`
  and `scripts/deploy_space.py`), R26's test cell; `docs/phase2-plan.md` sections 4, 6 (step 4)
  and 9, and open question 6.

`check.sh` green.

The fix round (`train.py` and the loader are high-stakes), started by the user:
`/review-session a94b482..HEAD 13.02`, code-reviewer (Opus 5.5, xhigh) on the range and
risk-reviewer (Fable 5.1, high) on its six high-stakes files. Both found nothing; all 21
scenarios held (R03, R04, R05, R14, R16, R17, R20, R21, R23, R25 to R27, R30 to R35, R37, R40,
R42). `check.sh` 1009 passed (code-reviewer); 296 passed over every test file that builds
checkpoints (risk-reviewer). Not findings, left as they are: three docstrings still name only
"model or class map" as refusals (`score_test.py` lines 31-32 and 431, tier-3 code;
`scripts/deploy_space.py:139`), incomplete rather than wrong. Neither reviewer could open the
seven real phase-1 checkpoints or the Space's `model.pt`; both are three-key files by the old
save line and the old exact-key loader, which loaded them in 8.06.

Merged `main` at 364e851 (9.03) through the sync tool: conflicts in this log (9.03's three
entries kept first, then 13.02's four, all dated 2026-10-17) and in `docs/review/invariants.md`,
merged cell by cell against 05cd17a. R17's "where" and test cells and R40's test cell changed on
both sides; 13.02 had only appended to each, so each is 9.03's text followed by 13.02's addition.
9.03's edits to R03, R06, R07, R16, R38 and R41 are kept as they are.

The merged tree then failed one test, 1 of 1055:
`tests/test_score_notebook.py::test_the_default_commit_holds_the_ledger_the_scoring_accepts`. It
reads the scoring notebook's default `COMMIT` (b64c486, older than 13.02) config.yaml with today's
code, which requires `input_size`: "dryrun: missing fields ['input_size']". On Kaggle the notebook
installs the code of the same commit as its config.yaml, so the scoring itself is not affected.
The user chose a test-only fix over defaulting the field in `config.py`: the test reads an old
config.yaml as today's code does, with `input_size: 224` added to `common` (`_read_as_today`),
what its code hard-coded and what `HASH_NEUTRAL` leaves out of the hash. Prototyped first: all 7
phase-1 rows of that commit hash to their ledger rows. Mutation check: with the bridge but
`batch_size` changed too, the test fails on "no longer matches its ledger row (config hash)", so
the hash check still bites. `check.sh` green after the merge.

The fix modifies an existing test in a high-stakes file, so by the playbook one fix round was
due on the merge's own changes (`git show --remerge-diff ddec877`: this helper and the two
conflict resolutions). The user waived it (2026-10-17): the coordinator checked the test change,
`_read_as_today`, which is guarded by its assertions (only a config without any `input_size`
is bridged, and only with `HASH_NEUTRAL`'s 224).

**Next.** Ready to merge: 13.02 merges after 9.03 (in, at 364e851) and before 13.04.

Review: tier 2, ran both reviewers; fix round ran (a94b482..10e53b1), both reviewers clean; the
sync's fix round (ddec877) waived by the user.

## 2026-10-17: session 13, item 13.04, arm C: the copy-paste sampler (Opus 5.5)

The user approved item 13.04 on 2026-10-17 through the coordinator. It was added to
`docs/progress.json` (session 13, owner claude, due 2026-10-19) with the description and steps
given. Effort xhigh: the background pool is a leakage boundary, treated like split code. Main
moved to dc19821 (8.08) mid-item: synced through the sync tool (a fast-forward; the work was set
aside in a tagged stash, restored by its SHA and the entry dropped).

**Done.**

- `trailcam/data/metadata.py`: an image of the `empty` rule is also kept as an `EmptyFrame`
  (image id, file name, official file, location, sequence id, the sidecar's night flag; the repr
  shows no id, and `__eq__` is written out as `Record`'s is, so a failed `==` in pytest shows the
  repr, review 3.05), per file in `Metadata.empties`, which defaults to empty. Records, drop counts, the
  split and its hash are unchanged (`c78a9c10…`, the pinned test passes). One behaviour change: an
  empty image with a non-integer location now fails loading, as a kept image's does (none in
  CCT20).
- `trailcam/data/copypaste.py`, new. `background_pool(meta, split)` keeps an empty frame only if
  it comes from a pooled file (official train or cis_val), sits at a train location and its
  sequence has no image in val, trans_val, cis_test or trans_test. `check_pool` re-checks the
  result independently (every rule, no image id of any split, no repeat, a night flag on each) and
  refuses with counts only. An edited split (records no longer matching its hash) is refused. The
  pool's repr and `summary` give counts per day and night and its SHA-256 only. `CopyPaste` is the
  paste step: with probability `p`, for a boxed image, one box drawn uniformly, its rectangle
  rounded out to whole pixels, pasted onto a background of the image's night flag at another
  location (drawn uniformly), at the same width and height fractions of the frame, at a uniform
  position wholly inside the frame with its bottom edge in the lower half. It returns the
  composite and the pasted box, or the image and its own boxes. Torch's generator only; with
  `p=0` no coin is drawn. Every background must be on disk when the step is built (counts only).
- `trailcam/data/dataset.py`: `ARMS` gains "C". `TrainTransform("C", paste=...)` is arm A's
  sampler and steps with the paste step first; arms A and B refuse a paste step and arm C needs
  one. The dataset hands the training transform each record's boxes, night flag and location.
- `trailcam/train/config.py`: `paste_p`, the first phase-2 setting, is one of `PHASE2_FIELDS`:
  None when a config leaves it out, and then out of the hash. Arm C needs it, the other arms
  refuse it, and it must lie in 0 to 1. `trailcam/train/train.py`: `train_transform` builds the
  arm's transform; for arm C it imports `copypaste` (there only), builds the pool from the run's
  metadata and split, and logs its summary line before training.
- `config.yaml`: `p2_arm_c`, resnet18_arm_a's recipe with `arm: C` and `paste_p: 0.5`.
- `notebooks/kaggle_look_at_data.ipynb`, new section 5, outputs cleared: 6 boxed train photos,
  day and night by turns. Each row shows the photo, the composite the real step returned with the
  pasted box outlined, and the 224 px training input. Pool and step are built as training builds
  them (`p=1.0` here); the dataset hands the step each photo's fields, so no cell names a location
  or an id. The cell prints the pool's summary line.
- Tests: `tests/test_copypaste.py`, new, 49 tests, about 20 s:
  - pool rules on the synthetic split with 13 planted frames, each breaking one rule and passing
    the others; `check_pool`'s refusals; the real pool pinned;
  - day/night, another location, the fallback, the probability, unboxed images, several boxes;
  - geometry over eleven cases; torch only; pickling; the arms; bad arguments;
  - no evaluation path loads `copypaste` (static and in a fresh interpreter).
- Other tests: `test_metadata.py` (3 new: the empty frames read back from the files, the location
  check, the default; the R03 leak test now also compares two empty frames, and failed before
  `EmptyFrame` wrote out its `__eq__`); `test_config.py` (4 new: arm C against arm A, the phase-1 hashes against
  every `docs/ledger.md` row, the hash rule, the optional field; 9 new refusals);
  `test_train.py` (arm C through `train_run`); `test_datavis.py` (3 new: the paste cell's static
  check, its note, the cell run on the fixture).
- Adjusted tests: `test_boxcrop.py` (the transform now has a `paste` attribute; "C" is an arm, so
  "D" is the bad one); `test_ledger.py` (likewise for the malformed arm cell); `test_notebook.py`
  (the training notebook's check cell keeps phase 1's configs and so refuses `p2_arm_c`, see
  below).
- `docs/review/invariants.md`: new R28 (the pool and the paste step); R23 and R25 extended to arm
  C; R03's test cell extended, and its rule text given a third exception (below); P2 added to the
  sources, R20 to R28 in the id ranges. `docs/phase2-plan.md`: what step 1 built (section 6) and the
  hash form (section 9). `docs/learning-journal.md`, topic 15: one entry.

**Numbers** (real metadata, split seed 42, cache sidecar; counts only):

- Empty frames per file: train 0, cis_val 995, trans_val 61, cis_test 1,180, trans_test 1,778.
  Every one has a night flag.
- The pool is all 995 of cis_val's, day 531 and night 464, at all 10 train locations, hash
  `87c5d379bfad28f7c5ede2d5ab8093dc7299fd6143782d2616db223e86c1d8bb`. None of the 995 shares a
  burst with val, train or any evaluation file, so the sequence rule removes none today; it stays
  as the guard.
- 11,633 boxed train images (7,048 night): each has at least 350 backgrounds of its flag at other
  cameras, so arm C never falls back for want of one on the real data.
- Frame sizes: boxed train images 2048 x 1494 (10,444), 2048 x 1536 (914), 2560 x 1920 (275); the
  pool 975, 2 and 18. Keeping the width and height fractions stretches a paste by 3% at most.

**Decisions to review.**

1. The pool rules: the intersection of your three rules and the plan's file rule (section 6: "only
   the 995 empty frames of the official cis_val file"). Your rules alone would admit 1,179 of
   cis_test's 1,180 empties: train cameras, bursts with no kept image. The plan calls them test
   images. Both together give cis_val's 995.
2. The config hash: a phase-2 field is hashed when set, so `p2_arm_c` does not share
   `resnet18_arm_a`'s hash. The alternative: leave `paste_p` out of the hash like `arm`, so arm C
   and arm A share it, as arm A and B do. Then two arm C configs with different probabilities would
   share a hash too. That alternative is a two-line change; 13.02's checker may prefer it for
   pairing.
3. Placement: Mesnage et al. 2025, section 2.4 (checked on the paper's page today): a random x
   and y with the segment's bottom in the bottom half. Their ±5% scale and ±5° rotation are not
   taken: you asked to keep the box's size.
4. The step returns the pasted box with the composite, so the boxes that reach the sampler
   describe the composite. Arm C's sampler is arm A's, which ignores boxes.
5. R03 wording, **for your approval**, as for 8.08: R03's rule now names a third exception, the
   arm C paste check. It also says the composites show the pool's empty frames: cis_val frames
   that are training inputs and in no split, not train photos. The rest is unchanged.
6. Not edited, for you to decide: CLAUDE.md's ground rule ("Bounding boxes are training-time
   augmentation (arm B) and attention scoring only") and `.claude/rules/data.md` (the box sampler
   "is used by the arm B training transform only"). Both are still true of the arm B sampler, but
   arm C now uses boxes in training too. Proposed: "(arm B; in phase 2, arm C)".
7. Do not run `p2_arm_c` before 13.02 merges: `train.py` would append its row to
   `docs/ledger.md`. The training notebook's check cell lists phase 1's configs only, so it
   refuses `p2_arm_c`; `test_notebook.py` now pins that. From the shell, a pasted row would fail
   the phase-1 ledger check (a second ResNet-18 config hash), but nothing stops the append itself.
   With "C" in `ARMS`, the phase-1 checker's arm cell accepts C; 13.02's `p2_` refusal covers
   that.

**Checks.**

- `uv run pytest tests/test_copypaste.py` 49 passed. `uv run pytest tests/test_datavis.py -k "look
  or crop or paste"` 10 passed, also with DeprecationWarnings as errors.
- Mutation checks (each source copied, mutated, tested and restored by a script; never `git
  checkout`), 18 planted faults, each fails a test:
  - no sequence exclusion, a trans location allowed, no file rule: each in the filter alone, and
    in the filter and `check_pool` together;
  - no day/night matching; no other-location rule; no lower-half bias; no scaling between frame
    sizes; `copypaste` imported at module level in `train.py`;
  - each of `check_pool`'s five rules alone; no split-hash check;
  - a phase-2 field hashed when None.
- Arm C through `train_run` on the synthetic fixture: the dry-run config with `arm: C`,
  `paste_p: 0.5`, one epoch, once without workers and once with two spawned workers. It trained,
  logged the pool's summary line and wrote its checkpoint (temporary directory, removed). The
  alarm is the dry run's, reported only.
- The split hash is unchanged, and every `docs/ledger.md` row's config hash still matches its
  config.
- `check.sh` green (1,013 passed, mypy, ruff and the data guard clean).

**Next.** Review 13.04. Merge it after 13.02, which adds the phase-2 ledger. Then the step's
pinned commit with the hypothesis line, and your eye check of section 5 on Kaggle. Open for you:
the R03 wording (decision 5), CLAUDE.md and `data.md` (decision 6), and the hash form (decision
2).

Review: tier 2 by the playbook (no split code). Tier 3 is recommended, because the pool is a
leakage boundary like the split (`docs/phase2-plan.md` section 6, open question 5): `/review-session
13.04`, then `/code-review ultra`. Spec changed: `docs/review/invariants.md` (R03's rule text,
which needs your approval; R23 and R25 extended; R28 new; sources and id ranges);
`docs/phase2-plan.md` sections 6 and 9, now cited as P2 by R23, R25 and R28.

## 2026-10-17: review 13.04: two lows, one fixed; the user's decisions, R03 wording approved (Opus 5.5)

Review: tier 2, ran `/review-session 13.04` on dc19821..6c91d35 (code-reviewer Opus 5.5 xhigh on
the range, risk-reviewer Fable 5.1 high on the 15 high-stakes files). Neither found anything high,
and all 27 scenarios held for both: R01, R03, R05, R06, R07, R10, R14, R15, R16, R17, R20, R21,
R23, R24, R25, R27, R28, R31, R32, R37, R38, R40, R41, R42, R43, R44, R45. The code-reviewer's own
`check.sh` was green (1,013 passed). Tier 3 (`/code-review ultra`) was recommended and skipped:
the user's decision. Spec changed: `docs/phase2-plan.md` sections 6 and 9 (P2, the source of
R23, R25 and R28); the rule text of R03, R23 and R25, and the new R28.

The two lows:

- **Both reviewers.** Before 13.02, a command-line run of `p2_arm_c` would append its row to
  `docs/ledger.md`: `train.py` refuses no `p2_` config, and `append_row` checks only the header.
  Both confirmed the guards that hold today: the training notebook refuses `p2_arm_c`, and a
  pasted row fails `check.sh`. Not fixed here: the user assigned the refusal to 13.02
  (`check_ledger_phase` in `train.py`).
- **Code-reviewer.** Three sources of R23 and R25 still describe arm B alone: CLAUDE.md's ground
  rule, `.claude/rules/data.md` ("(arm B)") and `.claude/rules/training.md` (the arm "selects the
  training crop sampler and nothing else"). **Fixed** with the minimal wording the user approved,
  relayed by the coordinator and confirmed by the user in this session (CLAUDE.md and `.claude/`
  are the user's files). CLAUDE.md's ground rule and `data.md` now read "(arm B; in phase 2, arm
  C)". `training.md` adds "in phase 2, arm C also adds a paste step before arm A's crop".

The user's decisions on 13.04's list ("go", 2026-10-17, relayed by the coordinator):

1. The stricter pool: the 995 empty frames of the official cis_val file only, never a cis_test
   file empty. As built.
2. `paste_p` is hashed, so `p2_arm_c` has its own config hash. As built.
3. The placement: Mesnage's random x with the segment's bottom in the lower half, no scale or
   rotation. As built.
4. Boxes travel with the composite. As built.
5. R03's wording, the third eye-check exception (arm C's composites), approved by the user via
   the coordinator on 2026-10-17. `docs/review/invariants.md` now reads "approved for items 8.07,
   8.08 and 13.04".
6. The wording of CLAUDE.md, `data.md` and `training.md`: done, see the second low above.
7. The `p2_` refusal: in 13.02.

Merged `main` at 05cd17a (8.06) through the sync tool as e44bb33: a clean merge, since 8.06
touched only `docs/figures/` and `docs/plan.md`. `check.sh` was green after it (1,013 passed).
This commit is the fix: the wording edit, the R03 wording and this entry; no code or test changed.
It touches CLAUDE.md and `.claude/rules/`, which are high-stakes paths, so the playbook calls for
one fix round on this commit's range. The user starts it.

**Next.** The fix round, if the user starts it. Then the merge, after 9.03 and 13.02, syncing again
if main has moved.

## 2026-10-17: fix round 13.04: two lows and a miscount fixed, no third round (Opus 5.5)

Review: tier 2, ran `/review-session e44bb33..HEAD 13.04` on 634b760, the review fix
(code-reviewer Opus 5.5 xhigh on the range, risk-reviewer Fable 5.1 high on CLAUDE.md,
`.claude/rules/data.md` and `.claude/rules/training.md`). Neither found anything high. All 9
scenarios held for both: R01, R03, R23, R25, R28, R42, R43, R44 and R45. The new wording matches
the code. Spec changed: CL, DR and TR (the box wording only), and R03's approval phrase. Fixed in
this commit, with the coordinator's go and no third round, per the playbook:

- **Code-reviewer, low.** `docs/phase2-plan.md` section 6 still said arm C is compared with arm A
  "on identical configs apart from `arm`", where R25, `config.yaml` and `config.py` say `arm` and
  `paste_p`. Its line "the metadata loader drops empties today" was stale too. Both sentences
  predate 13.04 (3bfcf63). Now: "apart from `arm` and `paste_p`", and the loader "dropped empties",
  which item 13.04 keeps aside as `Metadata.empties`.
- **Risk-reviewer, low.** `trailcam/data/boxcrop.py`'s docstring said no other module of the package
  imports it, but `copypaste.py` imports `to_pixels`. Now: no other module imports the sampler, arm
  C's `copypaste.py` imports only `to_pixels`, and the eval transform never loads the module. A
  docstring only; no code changed.
- **Code-reviewer, a record.** The review entry above said 28 scenarios for a list of 27; the
  selection gave those 27 and none was left out. Corrected to 27 above. The review commit's
  message (634b760) still says 28; it stays as it is, since that commit is not rewritten.

Both reviewers flagged that they could not verify the approvals. The user approved the CL, DR and
TR wording directly in this session; the R03 wording came relayed by the coordinator.

**Next.** Ready to merge, after 9.03 and 13.02; sync again if main moves.

## 2026-10-18: main merged at 89e649a (9.03, 13.02): arm C on 13.02's conventions (Opus 5.5)

13.02 merged first, and the coordinator asked for this sync at xhigh care, integrating rather than
only keeping both sides. Merged through the sync tool. 11 files conflicted: `config.yaml`,
`trailcam/data/dataset.py`, `trailcam/train/config.py`, `trailcam/train/train.py`, the tests of
`boxcrop`, `config`, `ledger` and `train`, `docs/phase2-plan.md`, this log and
`docs/review/invariants.md`.

**Resolved.**

- **One hash rule.** 13.02's `HASH_NEUTRAL` is now `{input_size: 224, paste_p: None}`, and a field
  at its phase-1 value stays out of the hash. My `PHASE2_FIELDS` is gone. `OPTIONAL_FIELDS`
  (`paste_p`) now says only that a config may leave the field out.
  - Both pins of the phase-1 hashes hold: 13.02's literal hashes and its per-row run seeds.
  - A set `paste_p` is still hashed (decision 2), so `p2_arm_c` has a config hash of its own.
- **`config.yaml`.** `p2_arm_c` now sits in 13.02's phase-2 block, after `p2_baseline`, and takes
  `input_size: 224` from `common`. `P2_CHANGES` gains `p2_arm_c: {arm: C, paste_p: 0.5}`. The
  one-change rule counts arm C's own setting as part of the arm's change.
- **The transform.** `TrainTransform(arm, *, sampler, paste, size)` runs the paste step on the full
  image, then arm A's crop resized to `size`. Arms A and B are unchanged. `train_transform` passes
  `config.input_size` to every arm; the eval transform is 13.02's `EvalTransform(input_size)`.
- **Ledger and checkpoint.** `p2_arm_c` goes through 13.02's `check_ledger_phase` and is refused
  with `docs/ledger.md`. It appends to `docs/ledger-phase2.md`, and its checkpoint carries
  `input_size`.
- **13.02's text corrected.** It said arm C would share arm A's config hash, in `ledger.py`'s
  docstring, R25, a `test_ledger.py` docstring and `phase2-plan.md` section 6. With `paste_p`
  hashed, arm C's rows carry their own hash and are paired by config and run seed.
- **Superseded.** My `test_notebook.py` change, which kept the training notebook to phase 1:
  13.02's notebook takes `p2_` configs, but only with the phase-2 ledger, so 13.02's test stands
  as it is. My copy of the ledger-hash pin is removed too, since 13.02's is stronger.
- **9.03's test.** `test_score_notebook.py` bridges a pre-13.02 `config.yaml` and guards
  `HASH_NEUTRAL`'s contents. It now names `paste_p`, which needs no bridge: left out, it is None.
- **Unchanged: the pool and every leakage rule.** The merge did not touch `copypaste.py`. Its
  tests and the real pool pin (995 frames, day 531 and night 464, `87c5d379…`) pass.

**New tests.**

- `test_train.py`:
  - `test_arm_c_trains_into_the_phase2_ledger_only_and_its_checkpoint_carries_the_size`. Through
    `main`, `p2_arm_c` against a copy of `docs/ledger.md` stops before training. Through the
    phase-2 ledger it runs `train_run` with the paste step and logs the pool's summary line. It
    appends one row that passes the phase-2 check, and writes a checkpoint carrying input size 224,
    which `load_checkpoint` accepts. Training and scoring are replaced; the rest is the real run.
  - `PHASE_MIXUPS` gains `(p2_arm_c, ledger.md)`.
  - The arm/size test covers `p2_arm_c` at 224.
- `test_copypaste.py`: `test_arm_c_crops_the_composite_at_the_configs_input_size` checks 320 px
  items, and that with the paste off arm C gives arm A's 320 px tensor draw for draw.
- `test_ledger.py`: the phase-2 ledger with a hashed arm C row passes.

**Docs.**

- `docs/review/invariants.md`:
  - R03 keeps the third exception and adds 9.03's scoring-notebook test;
  - R25 merges 13.02's phase-2 line with arm C's `paste_p` and its own hash;
  - R23 and R28 are kept;
  - the header names 13.04's rows.
- `docs/phase2-plan.md`, sections 6 and 9.
- This log: main's entries first, then 13.04's. The 13.04 entry above still says `PHASE2_FIELDS`;
  that is now `HASH_NEUTRAL` plus `OPTIONAL_FIELDS`.

**Review.** The resolution changes high-stakes code: `config.py`, `dataset.py`, `train.py`,
`ledger.py` and tests. The playbook calls for a fix round, which the user starts: `/review-session
89e649a..<this merge> 13.04`. That range is 13.04 as it lands on main; 9.03's and 13.02's own
changes are not in it. `git show --remerge-diff` on the merge shows the resolutions alone.

## 2026-10-18: fix round of the 89e649a merge: clean, no further round (Opus 5.5)

Review: tier 2, ran `/review-session 89e649a..cae22d7 13.04` (code-reviewer Opus 5.5 xhigh on the
range, risk-reviewer Fable 5.1 high on its 20 high-stakes files). The range is 13.04 as it lands
on main after 9.03 and 13.02, with the merge's integration as the focus. Nothing found by either.
All 27 scenarios held: R01, R03, R05, R06, R07, R10, R14, R15, R16, R17, R20, R21, R23, R24,
R25, R27, R28, R31, R32, R37, R38, R40, R41, R42, R43, R44 and R45. The code-reviewer checked
R45; the risk-reviewer could not, since no agent file was in its list. The code-reviewer's own
`check.sh` was green (1,129 passed).

Both reviewers confirmed:

- the phase-1 config hashes are unchanged;
- arms A and B and the eval transform are unchanged;
- the pool and leakage rules are unchanged: no diff since 995624b in `copypaste.py`,
  `metadata.py`, `split.py` or `boxcrop.py`;
- `p2_arm_c` follows 13.02's ledger and checkpoint conventions.

The risk-reviewer read the merge's own resolutions in full: every conflict kept both sides' code
and assertions.

Two near-misses were checked and found sound:

- The dropped 13.04 ledger-hash test. 13.02's pin still checks every ledger row, and "only arm
  C sets `paste_p`" is still checked across all configs.
- "C" in the phase-1 ledger's arm column. An arm C row in `docs/ledger.md` is still refused, by
  the `p2_` prefix and its own config hash.

Noted, outside the range: `.claude/rules/training.md` still says every training run appends to
`docs/ledger.md`. That is 13.02's wording; it goes to item 13.05, with the user's OK there.
Could not be verified offline: that the Kaggle cache holds the pool's 995 frames.
`CopyPaste` refuses before training if any is missing.

No fix, so no further round. This commit changes this log only.

**Next.** Ready to merge.

## 2026-10-18: item 12.02, README results section (Opus 5.5)

**Done.** `README.md` gains a Results section, from `docs/test-ledger.md` and `docs/plan.md`
section 7 (test and val results) only:

- the headline table, copied from plan.md section 7 (small CNN; arm A and arm B means of 3
  seeds; cis_test and trans_test accuracy and macro-F1; both gaps), with the scoring's approver
  (codx999) and time (20261018T193117Z);
- the experiment claim in two sentences, wording approved by the user in this session, and a
  per-seed trans_test table for arms A and B;
- four figures from `docs/figures/` by relative link: the arm B trans_test confusion matrix, the
  per-camera chart, the attention histogram and the features-by-camera map, with the data line;
- honest notes: trans_val is one camera and pointed the other way; cis_test sits far below val
  (time shift); the test sets were scored once; the day/night re-weighting;
- a Demo section with the Space link as a placeholder (11.03 deploys it) and the local commands;
- a "What's next" line pointing to `docs/phase2-plan.md`; no phase-2 results;
- the Status section updated, and a link to `docs/data-card.md` in the Dataset section.

Every decimal in the README was checked against the test ledger, the training ledger,
`headline_test.md` and plan.md section 7 by script; every figure link resolves. The learning
journal gains 5 lines under topic 5.

**Not flipped.** 12.02 stays `passes: false`: its step needs the Space URL in the README, which
exists only after 11.03 (the user's decision in this session). Flip it once 11.03 records the URL.

**For the user.** Item 9.02's last step ("pasted into ... README.md") is met once this merges;
the user can then flip 9.02.

**Review.** Tier 0, docs only (`README.md`, the journal and this log).

## 2026-10-18: session 13, item 13.08, the first three screens and arm C's pre-declared follow-up (Opus 5.5)

The user approved item 13.08 on 2026-10-18 through the coordinator. It was added to
`docs/progress.json` (session 13, owner claude, due 2026-10-19).

**Done.**

- `docs/ledger-phase2.md`: the user's Kaggle rows, pasted verbatim (seed 0, commit `69dfe93`):
  `p2_baseline-seed0-20261018T211046Z` and `p2_arm_c-seed0-20261018T214210Z`, then
  `p2_res320-seed0-20261018T215623Z`. The third came through the coordinator mid-item, after the
  item's text was written. The checker
  (`uv run python -m trailcam.train.ledger --check docs/ledger-phase2.md`) exits 0: "phase-2
  ledger: 3 rows, every check passed". The three config hashes match today's `config.yaml`
  (`483aa055…`, `5c7c51c2…`, `5a20cf07…`), and every split hash matches phase 1's (`c78a9c10…`).
- `config.yaml`: `p2_arm_c_p25`, after `p2_arm_c`. It is `p2_arm_c` with `paste_p: 0.25` and
  takes `input_size` from `common`. `paste_p` is hashed, so it has its own config hash,
  `42d47bfda80f99a821b0257a1f1c74543d93587cf2f5966a735899df54410fb9`.
- Tests:
  - `tests/test_config.py`: `P2_CHANGES` gains `p2_arm_c_p25: {arm: C, paste_p: 0.25}`, so the
    one-change and unique-hash tests cover it. A new test checks that it loads as a phase-2
    config, differs from `p2_baseline` only in `arm` and `paste_p` and from `p2_arm_c` only in
    `paste_p`, takes 224 from `common`, and hashes apart from both. The "only arm C names
    `paste_p`" check now reads `P2_CHANGES` instead of naming `p2_arm_c`.
  - `tests/test_train.py`: `PHASE_MIXUPS` gains `(p2_arm_c_p25, ledger.md)`. A run against a
    copy of `docs/ledger.md` stops before training, and `append_row` refuses it too. The arm/size
    test covers it at 224 with its paste step.
  - `tests/test_notebook.py`: the training notebook's check cell refuses it with
    `docs/ledger.md`.
  - Adjusted: arm C's command-line test now appends to the phase-2 ledger's text and header
    without its rows. The real rows carry the real split hash, and the synthetic split's row
    would disagree with them ("split seed 42's rows disagree on split hash"). This is the only
    test the real rows broke.
  - Mutation check: with `paste_p: 0.5` in `p2_arm_c_p25`, three `test_config.py` tests fail
    (the one-change test, the unique hashes and the new test).
- `docs/phase2-plan.md`, step 1: the screen's result, and the follow-up with its hypothesis line,
  dated before any run. Step 4: `p2_res320`'s screen. Section 9: the new config's hash. Open question 1: a note that arm C
  fails under either version, so the question stays open. No rule changed.
- `docs/learning-journal.md`, topic 15: five lines and a command.

**Findings, seed 0.**

- **The control passes.** `p2_baseline` reproduced phase 1's arm A seed 0 bit for bit. Every
  epoch's losses and scores are identical, from the Kaggle log as the user reported it, and so is
  the config hash. Every ledger cell matches phase 1's row except the run id and the checkpoint
  sha256. A differing sha256 is expected: since review 13.02 the checkpoint also records its input
  size, so the file differs even if the weights are the same. So section 3's baseline check
  passes, well inside both bands.
- **`p2_arm_c` is screened out.** Val macro-F1 0.8238 against 0.8542 (-0.0304). Trans_val
  macro-F1 0.3886 against 0.2915 (+0.0971). Final train loss 0.47 against 0.11. The pool on
  Kaggle was 995 frames (day 531, night 464), hash `87c5d379…`, as 13.04 measured on the Mac.
  - Against its hypothesis (trans_val 0 to +0.06, val flat to -0.005): trans_val came out above
    its range and val far below it. None of section 5's four words fits both halves.
  - **Screened out by the pre-declared rule**, the user's decision of 2026-10-18. The val drop is
    beyond the screen's 0.003, and also beyond open question 1's 1-point alternative. Val is never
    traded for trans_val, and trans_val is one camera, which misled in phase 1 (9.02). The row
    stays in the ledger, and the baseline is unchanged.
  - Recorded as a finding: on one seed, copy-paste traded familiar-camera for new-camera
    performance. The higher train loss fits that: composites are harder to fit.
- **`p2_res320` passes the screen.** Val macro-F1 0.8705 against 0.8542 (+0.0163, so no drop).
  Trans_val macro-F1 0.3186 against 0.2915 (+0.0271, not worse). Trans_val accuracy 0.3618
  against 0.3724 (-0.0106). Both conditions hold, so it goes on to the confirm on seeds 1 and 2.
  - Run time, as the coordinator reported it: about 60 s per epoch against the baseline's 42 s
    (11.6 minutes against 8.5, 1.4x).
  - Against its hypothesis (trans_val 0 to +0.03, val +0.5 point, about 2x the run time):
    trans_val matched, near the top of its range. Val rose about three times as much as
    expected, and the run cost less than expected.
  - Trans_val's macro-F1 and accuracy moved in opposite directions. On one camera that is the
    noise section 3 warns about, which is what the confirm is for.
  - Before it can be kept, step 4 also needs the CPU latency check (section 3). Its per-class
    and bobcat context needs the 320 px follow-up item section 3 names, since the evaluation
    tools run at 224 only.
  - Open question 8 (whether `p2_baseline` seeds 1 and 2 are re-run for the confirm) is the
    user's. For it: `p2_baseline` seed 0 reproduced phase 1's arm A seed 0 bit for bit.

**The pre-declared follow-up**, the user's decision of 2026-10-18: `p2_arm_c_p25`, one change
from `p2_baseline` (the arm, with its setting). It is judged by the same screen, confirm and keep
rule. Its hypothesis line, in `docs/phase2-plan.md` step 1 and here, before any run:

```text
Step 1 follow-up, copy-paste arm C at paste probability 0.25 (p2_arm_c_p25). Expect: about half
of arm C's val cost, most of its trans_val gain. Because: half as many composites leave more
training frames on their own scene, which val's familiar cameras reward, while a quarter of the
boxed images still breaks the link between scene and species.
```

**For the user.**

- Taken literally, the hypothesis predicts a screen failure. Half of arm C's val cost is about
  -0.015, which is beyond the screen's 0.003 and the keep rule's 0.01. So the variant passes only
  if its val cost comes out well under its own hypothesis. The plan says so beside the line.
- It adds one screen run, about 18 minutes, to section 7's budget (not edited).
- Its rows need the phase-2 notebook settings: `CONFIG_NAME = "p2_arm_c_p25"` and
  `LEDGER = "docs/ledger-phase2.md"`, at a commit that contains this item.

**Checks.** The ledger checker exits 0 on both ledgers. `docs/ledger.md` and the phase-1 config
blocks are unchanged, and the phase-1 hashes are pinned by the test. `uv run pytest
tests/test_config.py tests/test_notebook.py tests/test_ledger.py tests/test_train.py` passed.
`bash scripts/check.sh` green (1,135 passed; mypy, ruff and the data guard clean).

**Next.** Review 13.08. Then the user runs `p2_arm_c_p25` seed 0 on Kaggle, with the commit
pinned, and its row is screened against `p2_baseline` seed 0.

Review: tier 2 (`config.yaml` is high-stakes in `scripts/risk_files.sh`). Command:
`/review-session 13.08`. Spec changed: `docs/phase2-plan.md` step 1, section 9 and open question
1 (notes only, no rule).

## 2026-10-18: review 13.08: two lows, one fixed in a new test, one for the user (Opus 5.5)

`/review-session 13.08`, started by the user: tier 2, range `60189a7..339d021`. The
code-reviewer ran on Opus 5.5 at xhigh, the risk-reviewer on Fable 5.1 at high. Neither found
anything high or medium. Every scenario holds (R25, R28, R31, R40, R42, R01, R03, R43, R44,
R45), and both reviewers' test runs were green. The user's go for the fix came through the
coordinator.

1. **Low, both reviewers: no test tied the phase-2 ledger's config hashes to `config.yaml`.**
   The phase-2 checker compares rows with each other only. Fixed in `tests/test_config.py`, by
   added lines only:
   - `PHASE2_HASHES` pins every `p2_` config's hash. That covers the configs with rows and the
     pre-declared ones (`p2_arm_c_p25`, `p2_head3`, `p2_r50`), so a config edited before its run
     fails too.
   - `test_the_phase2_config_hashes_are_the_ledgers`: each `docs/ledger-phase2.md` row's config,
     at the row's run seed, hashes to the row's config hash and to the pin. `p2_baseline`'s rows
     carry arm A's phase-1 hash, because `input_size` 224 and `paste_p` None are hash-neutral.
     Arm C's rows do not, because a set `paste_p` is hashed.
   - Mutation checks, each restored from a copy afterwards, each failing the new test:
     `max_epochs` 11 in `p2_res320`; `paste_p` dropped from `HASH_NEUTRAL`; `input_size` dropped
     from it; one character of `p2_arm_c`'s row hash changed; `p2_arm_c_p25`'s `paste_p` 0.3.
   - The reviewers could not compute the hashes. They match: each row's hash equals today's
     `config_hash`, and `p2_arm_c_p25`'s is `42d47bfd…`, as the plan says.
2. **Low, code-reviewer: item 13.08's text names two rows and two screens.** The commit holds
   three, and the coordinator's later rows make four. Not changed: CLAUDE.md says items are never
   edited. **For the user:** reword 13.08 (it was created in this unmerged branch), or move the
   `p2_res320` rows to an item of their own.

Raised by the code-reviewer, for the user:
- Which `p2_baseline` run `p2_arm_c_p25` pairs with. It will run at a commit after `69dfe93`.
  Between the two, only `config.yaml` changed on the training path (it gained this block), and
  `p2_baseline`'s own block is unchanged.
- The bit-for-bit epoch match and the train losses come from the Kaggle output, not the repo.

The fix only adds lines to a test file (a new test, a new constant and a new import line, with no
line modified or deleted), so by the playbook no fix round is needed.

## 2026-10-18: item 13.08, `p2_res320` confirm seed 1 of 2 (Opus 5.5)

The coordinator relayed the row (seed 1, commit `69dfe93`). It is pasted verbatim into
`docs/ledger-phase2.md` as `p2_res320-seed1-20261018T221422Z`. The checker exits 0 ("phase-2
ledger: 4 rows, every check passed"), and the new hash test holds for it.

- **Pairing.** The baseline is phase 1's `resnet18_arm_a` seed 1 in `docs/ledger.md`, not a
  `p2_baseline` re-run. That is the user's decision on open question 8, as the coordinator
  relayed it, made because `p2_baseline` seed 0 reproduced phase 1's arm A seed 0 bit for bit.
  `docs/phase2-plan.md` still lists open question 8 as open; recording the answer there is the
  user's call.
- **Result.** Val macro-F1 0.8678 against 0.8559 (+0.0119). Trans_val macro-F1 0.3275 against
  0.3187 (+0.0088). Trans_val accuracy 0.3568 against 0.3525 (+0.0043).
- **So far.** Trans_val macro-F1 is up on both seeds (+0.0271 on seed 0, +0.0088 on seed 1), a
  mean of +0.018 over two seeds against keep condition 1's 0.03. Val is up on both.
- **Next.** The keep decision waits for seed 2 and the CPU latency check (section 3).

## 2026-10-18: session 13, item 13.05, phase-2 evaluation support (Opus 5.5)

The user approved item 13.05 on 2026-10-17 through the coordinator. It was added to
`docs/progress.json` (session 13, owner claude, due 2026-10-20) with the description and steps
given. Base `03fee36`; main has not moved. Val and trans_val only: no code path added here reads a
test split, and `score_test.py` is not changed.

**Done.**

- **The eval size comes from the config.** `evaluate.py`, `attention.py` and `slices.py` build
  their eval transform at `config.input_size` and pass the same side to `load_checkpoint` as
  `eval_size`. So `p2_res320` is scored at 320 (shorter side 366, centre crop 320), and a
  checkpoint whose recorded size is not its config's is refused before any image is read.
  - New keywords, each defaulting to 224: `size` on `evaluate.predict`, `write_overlays`,
    `attention_pass`, `AttentionDataset`, `eval_geometry`, `box_in_frame`, `box_mask`,
    `slices.predict_split`, and Grad-CAM's `overlay`, `overlay_photo` and `write_overlay`.
  - `SplitRun.input_size`, also 224 by default. `render_report` refuses splits scored at two
    sizes.
  - The final scoring, the demo, the Space and its deploy pass none of these, so they stay at 224.
    `load_checkpoint`'s message now names them as the 224-only callers.
- **Attention-in-box at 320.** The box is mapped through `eval_resize(size)` and the `size` crop.
  The mask, the Grad-CAM map and the box share are all on the `size` x `size` frame.
- **Headers.** A table of a non-224 config says `input size <n> px` (`attention.size_note`), and
  the attention text names the frame (`320 x 320`). At 224 the text is exactly as before.
- **Slices: the pair is the user's.** A phase-2 checkpoint is labelled by its config name
  (`slices.model_label`), since `p2_baseline`, `p2_head3` and `p2_res320` are all ResNet-18 arm A.
  A phase-1 one keeps `checkpoint_label`, which the final scoring also uses. A run that mixes
  phase-1 and phase-2 checkpoints is refused: phase 2 compares with `p2_baseline` only.
- **`notebooks/kaggle_evaluate.ipynb`** (outputs cleared):
  - `LEDGER = "docs/ledger.md"` (or `"docs/ledger-phase2.md"`), with `LEDGER_PATH = CLONE /
    LEDGER`. The check cell refuses any other value. The attached `.pt` files are matched by
    SHA-256 to that ledger's rows, as before.
  - With the phase-2 ledger, the clone cell also refuses a `COMMIT` without 13.05's code.
  - `SLICE_PAIRS = None` keeps phase 1's arm A beside arm B per seed, in `slices-seed<n>`. In
    phase 2 it pairs each config with `p2_baseline` at the same run seed, in
    `slices-p2_baseline-<config>-seed<n>`.
  - A named list of config pairs replaces the default; `[]` means no slices.
  - Refused before the cache is unpacked: a named config with no row, the small CNN, a named pair
    that shares no seed, and two rows of one config and seed (a re-run), named by run id. A
    default pair that shares no seed is skipped, and the notebook says so.
  - How to run: a second checkpoint dataset for phase 2 (`trailcam-checkpoints-p2`), and nothing
    of phase 2 copied into `docs/figures/`. The batch's entry here quotes the summary and
    trans_val's per-class rows.
- **`.claude/rules/training.md`**: the ledger sentence now reads "Every training run appends one
  row to its phase's ledger: phase-1 runs to `docs/ledger.md`, phase-2 (`p2_`) runs to
  `docs/ledger-phase2.md`". The user approved this exact wording in this session (2026-10-18),
  and nothing else in the file changed.
- **Tests.**
  - `tests/test_phase2_eval.py`, new, 35 tests, about 60 s, on a cache-shaped fixture:
    - a 320 px checkpoint evaluated end to end, with every batch 320 x 320 in all three tools
      and the overlay;
    - a phase-1 checkpoint at 224 with the old headers;
    - each mismatched pair refused by each tool;
    - the phase mix refused;
    - the labels;
    - the 320 geometry against `EvalTransform(320)` for seven image sizes;
    - the 320 mask and box share;
    - the 224 defaults.
  - `tests/test_evaluate_notebook.py`, 13 new tests:
    - the `LEDGER` and `SLICE_PAIRS` checks;
    - the ledger the files are matched to, and both phase mix-ups refused;
    - the default and named pairs, in both phases;
    - the refusals;
    - the clone cell run with git replaced;
    - the full phase-2 run on the fixture (224 and 320 px checkpoints): every table, both slice
      folders, the summary, and no test split.
  - Existing tests adjusted for the parameter's new form only: `PARAMETERS`,
    `test_the_parameters`, and `_write_ledger` now writes to `LEDGER_PATH`. Every other existing
    test is unchanged, including the ones that forbid a test split or the final scoring in the
    notebook.
- **Docs.**
  - `docs/phase2-plan.md`: section 3's follow-up is now built, and says why trans_val is read per
    class. Sections 4 and 6 (step 4) say where 224 still holds; section 9 has 13.05's paragraph.
  - `docs/review/invariants.md`: R17, R20 and R26 (rule text, where, tests) and the header.
  - Journal topic 15 gains 13.05's lines.

**Checks.**

- **Phase 1 unchanged, byte for byte.** HEAD's code (`git archive` into a scratch folder) and this
  tree ran the same evaluate and attention runs (ResNet-18 arm A and the small CNN, with an
  overlay), the arm A and B slices, and the three dry runs. All on one fixture, one config and the
  same phase-1-format checkpoints. All 31 output files are identical: tables, every PNG and both
  overlays.
- **Mutation checks**, on a scratch copy of the tree (the worktree untouched), 31 planted faults,
  each caught by a test:
  - 20 in the code: each place the size is dropped or replaced by 224 (predict, the overlays, both
    `run_splits`, each tool's `load_checkpoint` call, the geometry, the mask, the attention
    dataset, the box share, `predict_split`, `overlay_photo`), the mixed-size report, both
    `load_checkpoint` size checks, the phase-2 label, the phase-mix refusal;
  - 11 in the notebook: `LEDGER_PATH` not following `LEDGER`, no `LEDGER` or `SLICE_PAIRS` check,
    the check reading `docs/ledger.md` always, unknown or missing checkpoints not caught, no 13.05
    code check, phase 2 taking phase 1's pairing, re-runs allowed, a named pair with no seed or no
    row allowed.
- The journal's command runs green (20 passed).
- `bash scripts/check.sh` green: 1,177 passed (1,129 before, 48 new), mypy, ruff and the data guard
  clean.

**For the user.**

- The phase-2 checkpoints go into a separate private Kaggle dataset; the How-to-run step suggests
  `trailcam-checkpoints-p2`, but any slug works through `CHECKPOINTS_ROOT`.
- A re-run of `p2_baseline` at a later commit (plan section 4) gives two rows of one config and
  seed. Its default slices are then refused by run id, so the evaluation stops in seconds; set
  `SLICE_PAIRS` to pairs without it, or `[]`. Evaluate and attention are unaffected.

Review: tier 2 by the playbook (evaluation code, the notebook, `.claude/rules/` and tests are
high-stakes; no split code, no final scoring). Command: `/review-session 13.05`. Spec changed:
`.claude/rules/training.md` (TR), `docs/phase2-plan.md` sections 3, 4, 6 and 9 (P2, cited by R17,
R20, R26), `docs/review/invariants.md`.

**Next.** The review, which the user starts. Then the merge; sync first if main has moved (9.02's
test-ledger rows today).

## 2026-10-18: review 13.05: two lows, one fixed, one deferred; main merged at 60189a7 (Opus 5.5)

Review: tier 2. The user ran `/review-session 13.05` on `03fee36..462b69a`: code-reviewer (Opus
5.5, xhigh) on the range, risk-reviewer (Fable 5.1, high) on its 8 high-stakes files. Neither found
anything high. All 19 scenarios held for both: R03, R04, R05, R06, R07, R14, R16, R17, R20, R23,
R26, R27, R32, R35, R37, R38, R40, R41 and R42. The risk-reviewer marked R27 and R32 as not
touched by the range. The code-reviewer's `check.sh` was green (1,177 passed, none skipped). It
re-read the spec-changed rows (R17, R20, R23, R25, R26, R28, R30, R31, R37, R40, R41) against the
code and found them matching.

The user's decisions (2026-10-18, through the coordinator):

- **Fixed: low, both reviewers.** `slices.py` loaded each checkpoint inside the scoring loop. So
  with two checkpoints, a mismatched second one was refused only after the first had scored val
  and trans_val. Nothing was written, but GPU time was spent, and R20's new 13.05 line ("refused
  by each of the three before any image is read") did not hold for `slices.py`. Now `main` loads,
  and so checks, every checkpoint before the metadata or any image is read; R20's line holds as
  written. New test
  `test_slices_refuses_a_mismatched_second_checkpoint_before_scoring_the_first`: four cases, a
  good first checkpoint and a mismatched second, with `predict_split` and `load_metadata`
  replaced by a refusal, so nothing is scored or read. Against 462b69a's `slices.py` (scratch
  copy), all 4 cases fail; with the fix they pass.
- **Deferred: low, code-reviewer.** A re-run `p2_baseline` seed (two rows of one config and seed)
  stops the default slices in section 4, and `SLICE_PAIRS` cannot name run ids. It fails safe and
  nothing is mis-paired. The user answered open question 8 today: `p2_baseline` seeds 1 and 2 are
  not re-run, since seed 0 reproduced phase 1's arm A seed 0 bit for bit, and phase 1's arm A
  seeds 1 and 2 serve as the baseline's. `docs/phase2-plan.md` records that answer (section 3's
  confirm, open question 8) and section 9 records two known limitations: this one, and that
  default phase-2 slices pair at seed 0 only, since the baseline's seeds 1 and 2 are phase-1
  rows.
- **Left as it is (code-reviewer, a note).** `ASSETS_OVERLAY_RUN` may name a phase-2 run; it stays
  off by default.

Not verifiable by the reviewers: the approvals. The `training.md` wording was approved by the user
directly in this session; the item itself came through the coordinator. Not checked: GPU memory
at 320 px on a T4. The risk-reviewer noted mypy errors in the two test files; `check.sh` does not
type-check `tests/`, and most predate the range.

Merged `main` at 60189a7 (9.02's test results, 12.02's README) through the sync tool as daa97f9.
One conflict, this log: both sides appended an entry, so 12.02's entry comes first and 13.05's
after. No code file changed on both sides.

The fix touches `trailcam/evaluate/slices.py`, a high-stakes file, so by the playbook one fix
round is due on the fix commit's range. The user starts it. `check.sh` green after the merge and
the fix: 1,181 passed.

**Next.** The fix round. Then sync again before reporting ready to merge, since main moves when
13.08 merges.

## 2026-10-18: fix round 13.05: one low fixed in the plan; main merged at af9ceb5 (Opus 5.5)

Review: tier 2. The user ran `/review-session daa97f9..73fbf2a 13.05`: code-reviewer (Opus 5.5,
xhigh) on the fix commit's range, risk-reviewer (Fable 5.1, high) on `trailcam/evaluate/slices.py`
and `tests/test_phase2_eval.py`. Neither found anything high. All 8 scenarios held for both: R03,
R04, R05, R17, R20, R26, R32 and R42. Both confirmed the fix: every checkpoint is checked before
the metadata or any image is read, and the new test fails on the pre-fix `slices.py`. The
code-reviewer's `check.sh` was green (1,181 passed, none skipped).

- **Fixed: low, code-reviewer.** `docs/phase2-plan.md` gave two pairing rules. The confirm bullet,
  section 9 and open question 8 recorded the user's answer (the baseline's seeds 1 and 2 are phase
  1's arm A runs). But section 1, section 3's opening and its stack rule, section 4's pinned-commit
  rule and section 8 still described a same-commit `p2_baseline` re-run. Applied as written,
  section 4 would have required re-running the seeds the user decided not to re-run. Now all of
  them agree. Section 4 names the exception: seeds 1 and 2 pair across commits with phase 1's arm
  A at `61f1ecc`, because `p2_baseline` seed 0 reproduced arm A seed 0 bit for bit, and the rule
  holds as written for seed 0. The stack is judged against `p2_baseline` seed 0 and arm A seeds 1
  and 2. Section 7's budget drops the baseline's two confirm runs. Docs only; no code or test
  depends on this text.
- Noted, not checked by the reviewer: "bit for bit" is 13.08's finding; its ledger row's scores
  match arm A seed 0's to four decimals.

Merged `main` at af9ceb5 (item 13.08: four phase-2 ledger rows, `p2_arm_c_p25`, its tests and plan
notes) through the sync tool as 43dc928. Three conflicts, each two appended blocks, all kept:
- this log: 13.08's three entries first, then 13.05's;
- the journal's topic 15: 13.08's lines, then 13.05's;
- `docs/progress.json`: 13.05, then 13.08.

`docs/phase2-plan.md` merged without conflict; 13.08's screen notes under steps 1 and 4 and every
13.05 passage are both in it.

No further round: the fix touches `docs/phase2-plan.md` only, which is not high-stakes. `check.sh`
green after the merge and the fix: 1,188 passed.

**Next.** Ready to merge; sync again first if main has moved.

## 2026-10-18: item 8.05, Space and demo tests independent of assets/ (Opus 5.5)

A test fix the user approved on 2026-10-18 through the coordinator, logged under 8.05 with no new
item. 8.05 stays the user's and is not flipped here.

**Found.** With the user's 10 trans_test photos in `assets/` (uncommitted, copied into this
worktree only), check.sh failed 2 tests in `tests/test_space.py`: both took the app's click
handlers as a list and expected only the button's, but with photos present Gradio adds the
examples' loader (`Examples.create.<locals>.load_example`). The Space was right; the tests
assumed an empty `assets/`, the trap of 6.04 and 9.01 again. Reproduced before the fix: exactly
those 2 failed.

**Done.**

- `button_handler` (in `tests/test_demo.py`, imported by `tests/test_space.py`) finds the function
  the Classify button's click runs by its trigger, the button's id and `click`, then the tests
  check it by identity (`is space.classify_on_device`). `example_names` lists the photos the
  app's examples show.
- `load_space` takes a required `assets` folder and points `gradcam.ASSETS_DIR` at it before
  loading `space/app.py`, so no Space test reads the repo's `assets/`. The existing tests use a
  missing folder and now also check that there are no examples.
- New: the Space with 3 synthetic photos in a tmp `assets/`, with and without `spaces`, shows
  those 3 as examples and the button still runs `classify_on_device` (wrapped by `spaces.GPU`
  when installed); the local demo with photos runs the handler it is given, or `classify` by
  default. Synthetic fixture images only, never the real photos.
- `tests/test_deploy_space.py` already builds its own repo in a tmp folder; the other demo tests
  that read the repo's `assets/` pass with the photos.

**Verified.** check.sh green with the 10 photos in `assets/` (1,132 passed) and without them
(the commit). Mutation checks, each restored: the helper back to "the app's only handler", 3
tests fail; `load_space` ignoring its folder, 3 fail (with the photos present); `build_app`
dropping the examples, 3 fail. The deploy dry-run, in a throwaway clone with the photos
committed (untracked photos are refused, by design) and random ResNet-18 weights, lists 49 files,
the 10 photos among them (577 kB in all), and nothing from `data/`; the guard passes there.
The photos are removed from this worktree and never staged. The learning journal gains 5 lines
under "Surprises and lessons".

**For the user.** Commit the photos in main as planned; check.sh should stay green.

**Review.** Tier 2 (`tests/` is high-stakes): `/review-session`.

## 2026-10-18: item 8.05, review fixes for the assets/ test fix (Opus 5.5)

The user's go on both low findings of the tier-2 review of `3c347ff`, relayed by the coordinator.

**Fixed.**

- *code-reviewer (Opus 5.5 xhigh), low: the Space tests no longer caught a second handler that
  runs the model.* The tests without photos assert again that the app has exactly one handler
  (the Space's two, and the local demo's build without examples). The tests with photos assert
  that every handler besides the button's is Gradio's (`handlers_of_our_own(other_handlers(...))`
  is empty). Regression test: `test_a_second_handler_of_our_own_is_caught` adds `classify` on
  upload to a built app, with and without examples, and the checks find it.
- *risk-reviewer (Fable 5.1 high), low: no test checked that building with examples writes
  nothing into the repo.* The tests with photos (the Space, with and without `spaces`, and the
  local demo) run in an empty working folder with `GRADIO_CACHE_EXAMPLES=true`, as on Hugging
  Face, record every `gr.Examples` built, and assert caching is off and the folder is still
  empty. Regression test: `test_the_caching_check_sees_examples_that_get_a_handler` shows the
  recorder sees caching once examples get a handler and outputs (Gradio writes that cache at
  launch, not at the build, so the caching flag is the check that catches it).

**Verified.** Mutation checks on `trailcam/demo/app.py`, each restored, each run against the new
tests and against the committed ones (`3c347ff`): an extra `photo.upload(handle, ...)`, 7 new
tests fail, the old ones pass; the examples given `fn` and `outputs`, 3 fail, the old ones pass;
the build copying an example into the working folder, 3 fail. With the 10 photos back in
`assets/`, `test_space.py`, `test_demo.py` and `test_deploy_space.py` pass (68), and the photos
are removed again. check.sh green without them (the commit).

**Review.** Fix round on the fix range only, as the playbook asks: the fix changes existing test
lines in `tests/test_space.py` and `tests/test_demo.py`.

## 2026-10-18: session 13, item 13.09, phase 2's verdicts and the freeze (Opus 5.5)

The user approved item 13.09 on 2026-10-19 through the coordinator (that date is the
coordinator's; it is 2026-10-18 here). It was added to `docs/progress.json` (session 13, owner
claude, due 2026-10-20). Docs only: no code, config or notebook change.

**Done.**

- `docs/phase2-plan.md`:
  - A new section 12, "Results and the freeze": a table of every variant's paired deltas (val
    macro-F1, trans_val macro-F1 and accuracy, per seed) and verdict; the freeze; future work; a
    summary paragraph for the slides.
  - The opening paragraph points to section 12. Section 4: the pre-declared phase-2 test scoring
    is marked not run, and 13.06 not needed. Section 6: the results of `p2_arm_c_p25`,
    `p2_head3`, `p2_res320`'s confirm and `p2_r50`, each under its step, with the hypothesis
    word for the follow-up (*smaller*). Section 8: as run. Section 10: the four future-work
    items. Section 11: open questions 1, 2 and 8 marked answered, and a note on what remains
    open (3 open; 4, 6 and 9 do not arise; 5 settled by review 13.04).
- `docs/learning-journal.md`, topic 15: eight lines on screening, confirming and keeping, and
  why a pre-declared rule that rejects everything is still a result, with a command.

**The verdicts**, the user's decisions of 2026-10-18 and 19. Each delta was recomputed from
`docs/ledger-phase2.md` and `docs/ledger.md` with a scratch script (not committed), pairing seed 0
with `p2_baseline` seed 0 and seeds 1 and 2 with phase 1's arm A seeds 1 and 2 (open question 8).
The script also checks that `p2_baseline` seed 0's scores equal arm A seed 0's. Every number the
coordinator relayed matched.

- `p2_arm_c`: screened out (val macro-F1 -0.0304; trans_val +0.0971).
- `p2_arm_c_p25`: screened out (val -0.0151; trans_val +0.0493). The val cost halved and so did
  the trans_val gain, about 1:3 at both paste rates.
- `p2_res320`: confirmed over 3 seeds; not kept (mean trans_val +0.0158 < 0.03, though up on all 3
  seeds; mean val +0.0100).
- `p2_head3`: screened out (val -0.0078; best epoch 10 of 10, read as undertrained).
- `p2_r50`: passed the screen (val +0.0128, trans_val +0.0197); not kept (trans_val -0.0219 on
  seed 1). Seed 2 not run, by the user's decision.

**Found while checking, recorded in section 12.** Trans_val accuracy and macro-F1 move in
opposite directions on 4 of the 8 variant runs: `p2_arm_c_p25` (accuracy -0.0499), `p2_head3`
(-0.0755), `p2_res320` seed 0 (-0.0106) and `p2_r50` seed 1 (accuracy +0.0361 while macro-F1
fell). So copy-paste's 1:3 trade-off is a macro-F1 result: at paste probability 0.25, trans_val
accuracy fell. No verdict changes, since the rule is on macro-F1.

**For the user.**

- The answers to open questions 1 and 2 are my reading of the verdicts: the screen's conditions
  were applied as written (val 0.003; trans_val not more than 0.03 worse). Under question 1's
  1-point alternative, `p2_head3` would have gone on to the confirm. Question 2's alternative
  would change no verdict.
- Arm C's "val still rising" is the user's reading of the Kaggle log; the ledger shows its best
  epoch as 9 of 10. The plan says so.

**Checks.** The ledger checker exits 0 on `docs/ledger-phase2.md` with every row. Every relative
link in the new text resolves. The journal command runs green. No file outside `docs/` changed.
Synced to main at `4157bf9` (the coordinator's commit of `p2_r50` seed 1, a fast-forward), and
the deltas recomputed from the committed row. `bash scripts/check.sh` green (1,193 passed). The
first run after the sync had 3 errors in `tests/test_dependencies.py`'s `uv build` fixture; the
build and the file passed alone at once (6 passed), and the full re-run was green, so it was
transient, most likely another suite running in parallel.

**Next.** Item 12.01's slides can take section 12's summary paragraph.

Review: tier 0 (docs only; `scripts/risk_files.sh` prints nothing). Nothing to run. Spec changed:
`docs/phase2-plan.md` sections 4, 6, 8, 10, 11 and the new 12, notes and results only, no rule
changed. Rows citing P2 §4 or §6: R17, R20, R23, R25, R26, R28, R30, R31, R40.

## 2026-10-19: items 11.02, 11.03 and 12.02, the Space is live (coordinator, Opus 5.5)

**Done.** The user created the Space `Codx999/who-goes-there` (Gradio, CPU Basic, public; PRO,
since a Gradio Space now needs it, plan section 8) and a fine-grained token with write access to
this Space only, held in `HF_TOKEN` for the one command and unset after. The weights were
downloaded again from the Kaggle dataset `trailcam-checkpoints`; sha256 `a0a54889…24e9` matches
the ledger's arm B seed 2 row. `scripts/deploy_space.py --dry-run` listed 49 files;
the deploy from `c3ea399` uploaded them as Space commit `7d58559`, 0 stale files deleted. The
first attempt got a 401 at the commit step (the token's write permission was not saved yet);
the retry, after a `whoami` check, passed.

**Verified on the public Space** (built in about 4 minutes, RUNNING on cpu-basic): all 10
`assets/` examples are listed. `trans_bobcat_day_01` with Grad-CAM on: cat 79%, bobcat 20%,
coyote 0%, forward pass 0.033 s, overlay returned. `trans_coyote_night_05`: coyote 100%. Both
match the 8.06 overlays of the same model. The URL is in `README.md` (Demo) and plan section 8.

**For the user.** Flip 11.02 and 11.03 (yours). 11.02's "free CPU tier" wording predates
the PRO rule; plan section 8 already says so and wins. 12.02 flipped: the README has the
headline table, the experiment, the Space link, the local run and the data card link.

**Next.** Phase 3 (approved 2026-10-19: presentation moved to 2026-11-04, keep rule on new
cameras) is planned as one item after the 2026-10-21 usage reset.

## 2026-10-19: item 13.10, phase 3's plan (Opus 5.5, high)

**Done.** `docs/phase3-plan.md`, "Phase 3: a stronger backbone for new cameras", from the user's
decisions of 2026-10-19 (through the coordinator): the goal (trans_val macro-F1); the keep rule,
fixed before any run (mean trans_val macro-F1 up more than 0.03 over the baseline, up on every
one of 3 seeds, mean val macro-F1 down no more than 0.03); the suggested screen (seed 0: trans_val
at least +0.03 and val at most -0.03, so the screen asks of one seed what the keep rule asks of
the mean) and the confirm (seeds 1 and 2); a finalist rule if both candidates are kept. Baseline:
phase 1's `resnet18_arm_a` seeds 0 to 2, as in phase 2, plus one control re-run `p3_baseline`
seed 0 (justified: 13.11 changes training-path files, phase 2's own re-run rule, and it tests the
new ledger cheaply); every candidate also shown against arm B. Candidates: `p3_convnext`
(ConvNeXt-Tiny, pinned to `convnext_tiny.fb_in1k`: the locked timm 1.0.30 resolves the bare name
to an ImageNet-12k tag) and `p3_convnext_cp` (plus arm C at 0.5), with hypothesis lines; BioCLIP
and MegaDetector out, as future work. The honest limits, the test-set rule (one finalist, once,
`APPROVED_BY`, its own `docs/test-ledger-phase3.md`, both phases reported and the second use
disclosed; none if nothing is kept), the ledger (a new `docs/ledger-phase3.md`), the budget (about
2 to 3 GPU hours, cap 3) and the timeline to 2026-11-04. The `p3_` configs are left to 13.11:
`load_configs` refuses a model not in `MODEL_NAMES`, and a `p3_` name routes to `docs/ledger.md`
today. The Space needs more than a config for a ConvNeXt (timm is not in
`space/requirements.txt`, `tests/test_space.py` asserts so, and `DEFAULT_CONFIG` is fixed in
`trailcam/demo/predict.py`), so that is in 13.11 too.

The presentation moved to 2026-11-04: CLAUDE.md (the user confirmed in this session), README.md,
plan.md section 10 (rows for phases 2 and 3, session 12 re-dated) and a dated note at the top of
`docs/phase2-plan.md` (its 10-26 lines kept as the record of what was planned then). Not edited,
as history: `docs/HANDOFF.md`, `docs/research-prompt.md`, `docs/research-report.md`. plan.md
section 9 points at the new plan. Items 13.10 and 13.11 added to `docs/progress.json` (the user's
OK via the coordinator); 13.11's steps also name the phase-3 ledger routing and the notebooks'
`LEDGER` value, which the plan's section 8 needs. Five journal lines on pre-registration (topic 15).

**For the user.** Your brief said trans_val is "125 images from one camera"; it is 1,603 images
from one camera, location 125 (`docs/phase2-plan.md` section 2). The plan says the latter.
Open questions for you (plan section 12): the finalist rule, the screen, (b) after a broken (a),
and the test-scoring code item, which is not in `docs/progress.json` and needs your OK.

**Checks.** Every score in the plan recomputed from `docs/ledger.md` and `docs/ledger-phase2.md`
(the means, the keep and screen bars 0.3401, 0.8257, 0.3215 and 0.8242, and the hindsight
screen of phase 2's five seed-0 runs). Both ledger checks exit 0. The relative links resolve. No
10-26 deadline left in CLAUDE.md, README.md or plan.md's schedule. `bash scripts/check.sh` green.

**Next.** Item 13.11 (code, due 2026-10-22).

Review: tier 0 (docs only). `scripts/risk_files.sh` prints `CLAUDE.md`, for a one-date change in
its first paragraph; as in session 1, you decide whether that line gets `/code-review xhigh
--max-findings all <base>..HEAD`. Spec changed: plan.md sections 9 and 10, phase2-plan.md's
header; no rule changed. Rows citing PL §9: R31 (the training recipe, unchanged). 13.11 adds
the phase-3 lines of R17, R25, R30 and R40 (`docs/phase3-plan.md` section 9).

## 2026-10-19: item 13.10, the user's answers to the open questions (Opus 5.5, high)

**Done.** The user's answers to `docs/phase3-plan.md` section 12, given 2026-10-19 in the
coordinator session and relayed here, recorded there as "decided 2026-10-19, the user": the
finalist rule and the screen as written; both screens run regardless, and (b) is held only if
(a)'s training is broken; the test-scoring item approved. Sections 1, 7 and 11 now name it.
Added to `docs/progress.json` as **13.12** (owner claude, tier 3, due 2026-10-27, passes false):
`score_test.py`'s phase-3 mode for the finalist's 3 seeds into `docs/test-ledger-phase3.md`,
gated by `APPROVED_BY`, refusing a recipe that was not kept; built only after a screen passes,
recorded as skipped if none does. Id 13.12, not the unused 13.06, because `docs/phase2-plan.md`
uses 13.06 for phase 2's never-built scoring. The user keeps 13.11's wider steps and skips the
optional review of 13.10.

**Checks.** `bash scripts/check.sh` green. No file outside `docs/` changed.

Review: none (tier 0, the user's decision).

## 2026-10-19: item 12.01, the slide outline and speaker notes (Opus 5.5, high)

**Done.** `docs/slides.md`: 17 slides plus backups, each with what goes on it, its figure and
speaker notes. The order follows the item: problem; the classifier; the live demo (the Space,
backups the `--share` link and the 12.04 video, then the 12.05 sort, and a note to pick the
demo photos in advance knowing which are right); the headline table, mean of 3 seeds; val 0.87
against cis_test 0.67 on its own slide; the experiment; arm B vs arm A in plan section 7's
words ("small but consistent", the macro-F1 gap barely moves); attention-in-box and the
animal-removed test, read as mixed under plan section 6's rule; the four Grad-CAM overlays; two
errors from the trans_test confusion matrix (bobcat as cat, raccoon as opossum); day vs night;
the feature map (optional); phase 2 on one slide; a marked phase-3 placeholder; limitations and
next steps (detect then classify, BioCLIP, SWA, sequence voting); "the idea is not new, the
measurement is". The Beery et al. 2018 and CDLA-Permissive 1.0 citation is on every slide that
shows a photo or a chart of the data. A journal topic 16 (an honest results story) was added.

**Checks.** Every number in the slides was matched by script against `docs/ledger.md` plus plan
section 7. Every phase-1 result number is there. Every `docs/figures/` path in the slides exists
(21 files). `uv run python -m trailcam.train.ledger --check docs/ledger.md` exits 0.

**Flagged, not edited (the item's step 2).**

- The phase-2 slide (14) carries 12 numbers only in `docs/phase2-plan.md` section 12 and
  `docs/ledger-phase2.md` (val and trans_val macro-F1 changes, and the 0.03 keep bar); plan
  section 7 does not carry them. They are cited to section 12 on the slide.
- Slide 7 has arm B's design range (box fills 20% to 100% of the crop), from plan section 6.
- The demo model's own test scores (arm B seed 2) are only in `docs/test-ledger.md` and
  `docs/figures/headline_test.md`, so they were left off; slide 4 uses "5 of 10" from section 7.
- The confusion-matrix counts behind slide 11 are only in the figure; the slide circles the
  cells and the notes say to read them off it.

So 12.01 is left at `passes: false`: step 2 holds for phase 1 but not for slide 14. It can be
flipped if the user accepts phase2-plan section 12 as a source for that one slide.

**Phase 3.** At first `docs/phase3-plan.md` was not on `main`, so slide 15 was drafted from the
brief. After the sync with `main` at `6e83efb` (13.10 merged; the conflicts in this log and the
journal kept both sides), slide 15 was rewritten from the plan: the two candidates (ConvNeXt-Tiny,
and ConvNeXt-Tiny with copy-paste at p 0.5), the keep rule, the screen, one finalist at most, and
section 7's disclosure if a finalist is scored on test. Still no phase-3 result. Its rule
thresholds (0.03) and "seven recipes" are from `docs/phase3-plan.md` sections 1 and 6, not plan
section 7: flagged like slide 14. No 10-26 date was in the slides; the header already says
2026-11-04. 12.01 stays at `passes: false` by the coordinator's request, until the user decides
on slide 14's source.

**Not known yet.** Which of the 10 `assets/` photos the Space gets right, beyond the four
overlays (bobcat and opossum wrong, both coyotes right); slide 4's note says to write them
down at the rehearsal.

**Next.** 12.04 and 12.05 (the user's); slide 15's verdicts once phase 3 freezes.

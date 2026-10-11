# Who Goes There: design document

Written 2026-10-10 (session 1), revised 2026-10-11 after the CCT20 literature review
(`docs/research-report.md`). This is the source of truth for the project: when another document
disagrees with it, this file wins. It replaces `docs/HANDOFF.md`, which is kept for history. Facts
about the data come from `docs/data-inspection.md`; the class requirements from
`docs/course-guideline.md`; the gaps and accepted defaults from `docs/setup-review.md`; the
literature and the experiment from `docs/research-report.md`.

## 1. Problem statement

Who Goes There is a **classifier**. The input is one camera-trap photograph (a JPEG from a fixed,
motion-triggered camera, day or infrared night shot). The output is one of **12 classes**: 11
animal classes plus `car`. The core challenge is **generalising to cameras the model never saw in
training**: every camera has a fixed background, so a model can learn "this background means
coyote" and look good on familiar cameras while failing on new ones. The headline result is
therefore the gap between accuracy on **cis** test images (cameras also present in training) and
**trans** test images (cameras held out entirely), as defined by Beery et al. (2018).

The project has one research question on top of the classifier (section 6): can the dataset's
bounding boxes, used **only at training time** as a crop augmentation, make a whole-image
classifier attend to the animal and shrink that gap, with nothing extra deployed at inference?

Out of scope: detection and MegaDetector, bounding boxes at inference, sequence-level (burst)
classification, the `empty` class, backbones larger than ResNet-18, the full 243k-image Caltech
Camera Traps release, and any second training-time intervention.

## 2. Dataset and data decisions

**Source.** Caltech Camera Traps, CCT20 benchmark subset, from LILA BC. 57,864 images (the paper
says 57,868; the four-image discrepancy is in the paper itself) from 20 camera locations in the
American Southwest; COCO-style JSON metadata with `location`, `seq_id`, `frame_num`,
`date_captured` (a full timestamp on every image), `width`, `height`, and a bounding box on about
90% of animal annotations (1,454 of 14,071 train annotations lack one). The benchmark tarball
(`eccv_18_all_images_sm.tar.gz`, 6.5 GB) holds the images resized to at most 1024 px per side.
Licence CDLA-Permissive 1.0; attribution and citation in the README and `docs/data-card.md`. The
images and the metadata never enter git: the data guard, hooks and CI refuse them, and the
metadata (3 MB, tens of thousands of records) stays under `data/` locally and on Kaggle.

**Classes.** The metadata has 16 categories with non-contiguous ids. The task keeps 12, listed here
in training-count order with the CCT category id:

| index | class | CCT id | train (official) | trans test |
| --- | --- | --- | --- | --- |
| 0 | opossum | 1 | 2726 | 5505 |
| 1 | rabbit | 10 | 2443 | 846 |
| 2 | coyote | 9 | 1409 | 2249 |
| 3 | cat | 16 | 1287 | 1418 |
| 4 | squirrel | 5 | 1250 | 1051 |
| 5 | raccoon | 3 | 1027 | 5469 |
| 6 | dog | 8 | 820 | 888 |
| 7 | bobcat | 6 | 786 | 2338 |
| 8 | car | 33 | 668 | 1113 |
| 9 | bird | 11 | 484 | 224 |
| 10 | rodent | 99 | 375 | 62 |
| 11 | skunk | 7 | 229 | 328 |

The class-to-index map is written once, in `trailcam/data/classes.py`, and imported by training,
evaluation, the demo and the Space. Nothing redefines it inline.

**Dropped.**

- `empty` (id 30): zero images in the official train split, so it cannot be learned from the
  files as given; the 4,014 empties in val and test are filtered out. The task is species-only.
  The research report argues for moving the 995 cis_val empties into training as a 13th class;
  decided against for the headline, because `empty` is defined by background alone and comes from
  only 10 cameras, so it is the class most likely to widen the trans gap and blur the experiment.
  It is an "add if early" run (section 7).
- `badger` (9 train images), `fox` (5) and `deer` (42 train, 0 in trans test): too rare to learn
  or to score (Schneider et al. 2020 put the practical floor near 1,000 images per class; Beery
  et al. 2020 needed synthetic data to learn deer from CCT20). Alternative considered: an `other`
  class; rejected because it would be a three-way mix of unrelated animals with 56 pool images.
- The 78 images that carry two different species (62 are bobcat + rabbit): dropped from every
  split. Images with several annotations of the same species (407 in train) keep that one label.

**Kept per image, beyond the label.** `location`, `seq_id`, every bounding box (training-time
augmentation and the attention score, never a model input), the capture hour from `date_captured`,
and a day/night flag computed during the Kaggle resize from the pixels (infrared frames are
grayscale JPEGs whose three channels are near-identical; the clock alone is unreliable, hour 12
holds about twice as many images as its neighbours). About half the images are night shots.

Boxes are stored once, in `trailcam/data/metadata.py`, as fractions (x, y, width, height) of the
image's original `width` and `height`, so the same box holds on the 1024 px tarball, the 512 px
cache and every crop (the resizes keep the aspect ratio); nothing downstream converts pixels
again. Boxes that run past the frame are clipped to it and counted (100 of 47,729 kept boxes).
The day/night flag lives in a sidecar, `day_night.json`, written next to the image cache by
`scripts/kaggle_prepare_cct20.py`: one JSON object mapping each image id to
`{"night": <bool>, "hour": <int>}` (extra keys such as the grayscale score are allowed). Training
passes its path explicitly (`--sidecar`), since the cache and the annotation files may sit in
different folders; without it the flag is absent.

**Imbalance.** opossum alone is 20% of train; opossum + raccoon are 47% of trans test. Accuracy
alone would reward predicting opossum, so macro-F1 sits next to accuracy everywhere (section 7).

**Preprocessing.** The 1024 px images are resized once on Kaggle so the longer side is 512 px
(`scripts/kaggle_prepare_cct20.py`, about 2 to 3 GB) and saved as a private Kaggle dataset;
training reads the cache. Models take 224 x 224 inputs; 512 leaves room for a 320 px run if there
is time. Normalisation uses ImageNet mean and standard deviation for both models so they share
one pipeline.

## 3. Split design and leakage tests

The official **location split** is kept: 10 cis locations (train, cis_val, cis_test) and 10 trans
locations (trans_val, trans_test), no overlap. cis_test is a different-days hold-out on the same
cameras, so the cis-to-trans gap is a location effect, not a time effect. Two published splits
need repair:

- The official `train` and `cis_val` files **share 224 sequences** (bursts of three frames), which
  breaks the invariant that every burst sits in one split (the paper says they share none; the
  released files do). So the two files are **pooled (17,037 images before filtering) and re-split
  by `seq_id`**, about 85% train and 15% val, from a fixed seed, stratified by the sequence's
  majority class and by location. The split is a deterministic function of the metadata and the
  seed, and its hash is recorded in the ledger. It is not committed as a file: the data guard
  refuses record dumps, and a seed plus a hash reproduces it exactly.
- `cis_test`, `trans_val` and `trans_test` are clean (zero shared sequences with anything) and stay
  as published, minus the class filters above.

| split | role | source |
| --- | --- | --- |
| train | fitting | pooled train + cis_val, re-split by sequence |
| val (cis) | model selection, early stopping, every hyperparameter choice | same pool |
| trans_val | sanity check only (1,725 images, one camera, 9 of the 12 classes); reported with macro-F1 over the classes present, never selected on | official |
| cis test | scored once at the end | official |
| trans test | scored once at the end; the headline | official |

**Leakage and coverage tests** (`tests/test_split.py`, tier 3 code, implemented at xhigh):

1. No location appears in both train and any trans split.
2. No `seq_id` appears in two splits.
3. No image id appears in two splits.
4. Every cis location appears in both train and val; every class has at least 30 val images.
5. Split sizes, per-class counts and the split hash match pinned values for the given seed.

**Leakage alarms** during training: trans accuracy close to cis accuracy, val accuracy far above
what the published benchmark suggests, or any test id seen by the training loader.

## 4. Model plan

Two models, trained and scored the same way, so the slides can show what pretraining buys:

1. **Small CNN from scratch** (`trailcam/model/small_cnn.py`): the chapter 04 TinyVGG pattern, three
   conv blocks (conv, ReLU, conv, ReLU, max-pool) and a linear head, at 224 px. It satisfies the
   course's two-layer minimum on its own and is the warm-up: one run, one paragraph.
2. **Fine-tuned pretrained backbone** (`trailcam/model/backbone.py`): **ResNet-18** with
   ImageNet-1k weights (torchvision), final linear layer replaced by a 12-way head, all layers
   fine-tuned with a lower learning rate on the body than on the head. 11.7M parameters, about
   1.8 GFLOPs per image, about 45 MB of weights; well under a second per image on the CPU-only
   free tier of Hugging Face Spaces, and the standard Grad-CAM subject (target layer `layer4`).
   ImageNet contains no CCT images, so pretraining is not leakage. Fallback if a Space prediction
   takes over about one second: EfficientNet-B0 (timm), 5.3M parameters, about 21 MB.

`/review-shapes` prints the layer, input shape, output shape and parameter table for both models
before any training run, and checks the head size against the class map.

## 5. Training plan

| Setting | Decision |
| --- | --- |
| Loss | Cross-entropy, unweighted, the same in both experiment arms |
| Optimizer | AdamW; backbone at 1e-4, head at 1e-3 (two parameter groups); weight decay 1e-2 |
| Schedule | Cosine annealing with a one-epoch linear warmup, **and** early stopping on val macro-F1 with patience 3; the best checkpoint is restored |
| Epochs | Up to 15 for the small CNN, up to 10 for the backbone (one head-only epoch first); the ledger records epochs run and the best epoch |
| Batch and input | 64 images at 224 x 224; mixed precision on the GPU |
| Augmentation, control arm | Training only: random resized crop (scale 0.65 to 1.0, mirroring the paper's crops), horizontal flip, mild colour jitter, random grayscale with probability 0.1 (half the data is infrared). Val and test: resize so the shorter side is 256, centre crop 224, normalise |
| Augmentation, treatment arm | Identical, except the crop sampler described in section 6 |
| Seeds | One seed for the split, one per run; both set (`torch`, `numpy`, `random`) and logged |
| DataLoader | Shuffle on the training loader only; val and test loaders never shuffle |
| Compute | A Kaggle notebook (`notebooks/`, outputs cleared) attaches the 512 px cache built once by `scripts/kaggle_prepare_cct20.py`, installs `trailcam` from the public repo with pip (the one place pip is allowed; the package needs a `[build-system]` table, added in session 5) and runs the training script. The Mac runs code, tests and the dry-run only |
| Dry run | `/dryrun` runs `train.py --max-epochs=1 --batch-size=2 --dry-run=True`; `--dry-run` writes the synthetic fixture (`trailcam/data/synthetic.py`, also used by the tests) into a temporary directory itself, since the guard blocks fixture images in git, runs one epoch on a tiny batch, prints its ledger row without appending it, and stops |
| Ledger | Every training run (not the dry run) appends one row to `docs/ledger.md` (created with its header row in session 5, same column order as the row `train.py` prints): run id, model, arm, config, split seed, run seed, split hash, epochs run, best epoch, scheduler, val accuracy, val macro-F1, trans_val accuracy, checkpoint hash. The row is what gets reviewed; Kaggle runs are started by the user, who pastes the row |

GPU budget (Kaggle, 30 hours a week, T4 or P100): small CNN about 0.5 h; each ResNet-18 run about
1 h over roughly 13,600 cached training images; two arms times up to three seeds is 6 h; the
evaluation passes about 1 h. Total under 10 h.

## 6. Research question and experiment

**Question.** Does training the whole-image classifier with **box-guided crop augmentation**,
bounding boxes used only at training time, shift its attention onto the animal and shrink the
cis-minus-trans gap on CCT20, compared with the identical model trained with standard random
resized crops?

**What is established** (details and numbers in `docs/research-report.md`). Beery et al. 2018:
a full-image classifier loses about 22 points of top-1 accuracy on unseen cameras (19.1% cis error
vs 41.0% trans); classifying ground-truth box crops more than halves both errors; more cis data does
not close the gap. Koh et al. 2021 (WILDS iWildCam): unseen-camera shift is a standard benchmark,
macro-F1 the accepted metric, and off-the-shelf robustness methods do not beat plain training.
Schneider et al. 2020: the gap replicates on other data. Beery, Morris, Yang 2019: the deployed
answer is detect, crop, then classify, which needs a detector at inference. Geirhos et al. 2020
names the mechanism: shortcut learning on background cues.

**What this adds.** The prior work compares whole image at test against crop at test. This project
spends the free boxes at training time only, as an augmentation that teaches a whole-image model
where to look, so that deployment stays a single classifier with no detector. The report found no
published controlled comparison of this on CCT20 with a quantified attention measure (confidence
moderate: a workshop paper may exist). The honest framing is a controlled measurement of a known
idea in a setting where its effect is unknown, not a new method. The slide says: "The idea is not
new. The measurement is."

**Design.**

- **Arm A, control**: the training transform in section 5.
- **Arm B, treatment**: for each training image that has a box, with probability 0.5 the crop is
  sampled so that it contains the box and the box fills between 20% and 100% of the crop, with
  random jitter of the crop position; otherwise, and for every image without a box, the arm A
  sampler is used. The sampler lives in `trailcam/data/boxcrop.py` with unit tests that the crop
  always contains the box and that evaluation transforms never touch boxes.
- **Controls**: same backbone, initialisation seed, epochs, optimizer, schedule, resolution, batch
  size, loss and evaluation transform. The only difference is the training crop sampler. Both arms
  see the same whole image, resized and centre-cropped, at validation, test and in the demo.
- **Seeds**: seed 0 of both arms first. Seeds 1 and 2 of both arms only if seed 0 of arm B is done
  by 2026-10-16; otherwise one seed per arm is reported, and said so.
- **Checkpoint selection**: val macro-F1, for both arms, as in section 5.

**Outcomes measured on cis test and trans test, once, at the end, per arm and seed.**

1. Accuracy and macro-F1, and the cis-minus-trans gap in both.
2. **Attention-in-box**: for every box-annotated test image, the fraction of normalised Grad-CAM
   mass inside the ground-truth box; its distribution on cis vs trans and for correct vs wrong
   predictions. The box is used only to score attention, never as input.
3. **Animal-removed test**: the same box-annotated test images with the box region blanked to the
   image mean; accuracy and mean top-1 confidence per arm and camera group. A model that learned
   animals should fall toward chance; a model that learned scenery keeps its prediction.

**Reading the result.** Positive: arm B beats arm A on trans macro-F1 across all seed pairings and
the attention-in-box median rises. Attention rises but the gap stays: the gap is not only about
background, the animals' appearance shifts too, which is what Koh et al. found for robustness
methods. Mixed outcomes are reported as mixed. A negative result is a result.

## 7. Evaluation

- **Headline table**: accuracy and macro-F1, side by side, on cis test and trans test, for the
  small CNN and for both ResNet-18 arms. The cis-minus-trans gap is the number the slides lead
  with.
- **Per-class** precision, recall and F1 on trans test, and the **confusion matrix** (both as
  figures under `docs/figures/`, which has a 40-file, 20 MB guard budget).
- **Experiment outcomes** from section 6: attention-in-box distributions and the animal-removed
  table, arm A vs arm B.
- **Grad-CAM** (the `grad-cam` package, `layer4` of ResNet-18) overlays on correct and wrong cases
  among the `assets/` photos, which are trans-test samples and the only dataset images allowed in
  the repo (the data rule keeps raw dataset images out of `docs/figures/`).
- **Slices of the same test predictions**, no extra training: day vs infrared night, per camera
  group and per class; common vs rare species; accuracy per unseen camera as a bar chart (bars
  only in `docs/figures/`: the worst camera is known only after the final scoring, too late to
  pick its photo for `assets/`; the slide may show that camera's photo only if one is already in
  `assets/`, with the Beery et al. citation).
- **Feature map**: backbone features of the test photos projected to two dimensions, coloured
  once by species and once by camera, for arm A and arm B side by side.
- **Inference timer**: `time.perf_counter()` around the forward pass for one image, after one
  warm-up call, reported in seconds; the same function serves the Gradio UI.
- **Add if early, zero GPU**: temperature scaling fit on val, with expected calibration error and
  a reliability diagram on cis vs trans; a 13-class run with `empty` moved from cis_val.
- **Protocol**: cis test and trans test are scored **once**, at the end, with the user's go-ahead,
  by a script that writes the ledger rows and the results tables. That scoring is tier 3. No
  test score is looked at before every arm and seed that will be reported has finished training.

## 8. Demo and deployment

**Gradio app** (`trailcam/demo/app.py`): upload an image, get the top class with its confidence, the
two runners-up, the inference time in seconds, and an optional Grad-CAM overlay. The model
outputs 12 softmax probabilities; the prediction is the argmax; confidences are the model's own
estimate and are called that on the slide (calibrated only if the add-if-early item runs). Example
images come from `assets/` (6 to 12 trans-test photos including infrared night frames, 24-file and
12 MB budget, locations listed in the data card). A second, local-only script
(`trailcam/demo/sort.py`) sorts a folder of about 200 trans-test photos into class folders with a
visible timer and per-class counts for the live part of the presentation, with a recorded backup
video. Optional for the talk: a human baseline, the room votes on 20 unseen-camera photos before
the model answers.

**Hugging Face Space** (`space/`): `app.py` (imports the same predict function), `requirements.txt`
exported from uv (`uv export --no-dev --no-hashes`, trimmed to what the app imports) and
`README.md` with the Space header. The weights are **not** in this repo (the guard blocks `.pt`):
`scripts/deploy_space.py` uploads `app.py`, the requirements, the examples and the weights with
`huggingface_hub`, reading the token from the environment. The Space runs on the free CPU tier.
Account and Space creation are the user's steps. The Space serves arm B if it wins, else arm A.

## 9. Decisions and why (quotable)

**Architecture.** A pretrained ResNet-18, fine-tuned end to end, against a small CNN trained from
scratch. The dataset has about 13,600 training images across 12 imbalanced classes; that is too
few to learn good low-level features from zero, which is exactly what ImageNet pretraining
supplies. ResNet-18 is small enough for CPU inference on a free Space, trains in about an hour on
a T4, and is the best-understood Grad-CAM subject. The scratch CNN is kept as the baseline that
shows what pretraining buys.

**Loss under imbalance.** Cross-entropy, because the task is single-label multi-class. Imbalance is
handled by measuring macro-F1 next to accuracy and by stratifying the split, not by reweighting:
a weighted loss would be a second variable in the experiment, and WILDS found such methods do not
move unseen-camera scores. Oversampling lengthens every epoch and overfits the rare classes; label
smoothing is a calibration tool, handled by the optional temperature-scaling item instead.

**Optimizer.** AdamW: adaptive step sizes make it robust to a learning rate that is only roughly
tuned, and decoupled weight decay regularises the fine-tuned backbone. SGD with momentum can match
it but needs a tuning budget the schedule does not have. Two parameter groups keep the pretrained
weights on a ten-times smaller learning rate than the new head.

**Epochs.** Not fixed in advance. A cosine schedule shrinks the learning rate towards zero, and
early stopping on val macro-F1 (patience 3) ends the run when it stops improving; the best
checkpoint is restored. Both course options are therefore in use, and the ledger records how many
epochs actually ran.

**Headline metric.** Accuracy because the course asks for standard metrics and it is what the
audience expects; macro-F1 because opossum is a fifth of training and predicting it blindly would
score well on accuracy. The result that matters is the cis vs trans gap on both.

**The experiment.** One intervention, one variable, controls on everything else, measured on
unseen cameras and with a quantified attention score plus a behavioural check. Not a new method:
a controlled measurement the dataset paper did not run.

## 10. Schedule

The presentation is Monday 2026-10-26. One session per item group; items live in
`docs/progress.json`. Tier 3 work (the split, the final scoring) is placed so one fix round still
fits before the deadline. The research-question work is guarded: seeds 1 and 2 run only if arm B
seed 0 is done by 10-16.

| session | date | work | review |
| --- | --- | --- | --- |
| 0 | 10-10 | Bootstrap (done) | tier 1, done |
| 1 | 10-10 to 10-11 | This design document, CLAUDE.md, progress.json; literature review and revision (done) | tier 0 docs; CLAUDE.md control plane, reviewed |
| 2 | 10-11 | Review machinery (Part B of the reusable playbook): invariants with the 12-class rows, `risk_files.sh` with `^trailcam/`, `^space/`, `\.ipynb$`, both reviewer agents, `/review-session`; first review run by the user | tier 2 |
| 3 | 10-12 | Data: class map, metadata loader with boxes and hour, filters, re-split by sequence with the leakage and coverage tests, data card | tier 3 |
| 4 | 10-13 | Dataset, transforms, DataLoaders, synthetic fixture generator, box-guided crop sampler with tests | tier 2 |
| 5 | 10-14 | Dependencies and mypy overrides, small CNN, ResNet-18, training loop with scheduler, early stopping, seeds, arm flag, ledger file, dry-run config, packaging for the Kaggle install; `/review-shapes`, `/dryrun` | tier 2 |
| 6 | 10-15 | Kaggle: prepare script written, 512 px cache with the day/night flag as a Kaggle dataset, notebook; small CNN run and arm A seed 0 (user) | run ledger |
| 7 | 10-16 to 10-17 | Arm B seed 0 (10-16); seeds 1 and 2 of both arms if on schedule (10-17); ledger checker | run ledger |
| 8 | 10-18 | Evaluation: metrics, per-class, confusion matrix, attention-in-box, animal-removed test, slices, per-camera chart, feature map, Grad-CAM, inference timer; example photos into `assets/` (user); val figures | tier 2 |
| 9 | 10-19 | The single final scoring on cis and trans test, every reported arm and seed (user go-ahead) | tier 3 |
| 10 | 10-20 | Gradio demo, local sort script with timer | tier 2 |
| 11 | 10-21 | Space: `space/`, requirements export, deploy script; account, Space and deployment (user) | tier 2 |
| 12 | 10-22 to 10-23 | Slides, README results and Space link, backup video, live-sort rehearsal; calibration if early | tier 0 |
| buffer | 10-24 to 10-25 | Fix rounds from the tier 3 review, rehearsal | |
| | 10-26 | Presentation | |

Slides: problem, the classifier and live demo, the honest test (cis vs trans), the experiment
(arm A vs arm B, attention-in-box, animal-removed), two errors from the confusion matrix, day vs
night, limitations and next steps.

### Setup maintenance

The review machinery and the tooling were written before the code (sessions 0 to 2). Each
session below also updates the setup it touches, in the same commit as the code:

- **Every session**: fill the "test" cell of the `docs/review/invariants.md` rows the item builds
  (the rows say "none yet" and name the item). A new high-stakes module or config file goes into
  the `scripts/risk_files.sh` regex, its test and the map in the same commit (row R44); the root
  `config.yaml` of item 5.02 is the first known case.
- **Session 5** (dependencies): the mypy overrides for torch, torchvision and timm; a CPU-only
  torch index in the lock so CI and the Mac do not download CUDA builds; the CI run (15-minute
  timeout, `uv sync --locked`) checked once with the new wheels. Confirm `/dryrun` and
  `/review-shapes` still match the module paths and flags 5.03 implements.
- **Sessions 6 and 11** (new scripts): `scripts/check.sh` runs mypy on `trailcam/` and
  `scripts/guard_data.py` by explicit path; `kaggle_prepare_cct20.py` and `deploy_space.py` are
  added there (and to the mypy `files` list) when written.
- **Sessions 10 and 11** (demo and Space): `.claude/rules/data.md` covers `trailcam/data/**` and
  `trailcam/evaluate/**` only; decide whether `trailcam/demo/**` and `space/**` join its paths so
  the "assets only, never `data/`" rules apply there.
- **After the first review** (item 2.04): add to the allow list in `.claude/settings.json`
  whatever read-only git commands the reviewers prompted for (`git show`, `git rev-parse` are
  the likely ones).

## 11. Open questions

1. **Course dataset rule.** Closed 2026-10-11: the user decided to proceed with CCT20 without
   waiting on the TA. It is not in `torchvision.datasets` and is loaded through a custom `Dataset`
   as chapter 04 teaches; if asked, cite the location-split motivation. Food-101 remains the named
   fallback only if the course later rejects the dataset outright.
2. **Submission format.** The Space (app.py, weights, requirements.txt) is known; whether a report,
   notebook or video is also required is not.
3. **Tarball layout.** The folder structure inside `eccv_18_all_images_sm.tar.gz` is unknown until
   it is opened on Kaggle (session 6); the prepare script prints it.
4. **`car` as a class.** Kept because it has 668 training and 1,113 trans-test images; drop to 11
   classes if the audience finds a non-animal class confusing.
5. **Grad-CAM in the Space.** It doubles the CPU work per prediction; it may stay local-only, with
   static heatmaps in the slides.
6. **Seeds.** Whether seeds 1 and 2 ran, decided on 2026-10-16 and recorded here.
7. **Which arm the Space serves.** Decided after the final scoring (session 9).

## 12. Sources

- Sara Beery, Grant Van Horn, Pietro Perona. "Recognition in Terra Incognita." Proceedings of the
  European Conference on Computer Vision (ECCV), 2018. https://arxiv.org/abs/1807.04975
- Caltech Camera Traps on LILA BC: https://lila.science/datasets/caltech-camera-traps
  (licence CDLA-Permissive 1.0: https://cdla.dev/permissive-1-0/)
- Koh et al. "WILDS: A Benchmark of in-the-Wild Distribution Shifts." ICML 2021.
  https://arxiv.org/abs/2012.07421
- Schneider, Greenberg, Taylor, Kremer. "Three critical factors affecting automated image species
  recognition performance for camera traps." Ecology and Evolution, 2020.
  https://doi.org/10.1002/ece3.6147
- Beery, Morris, Yang. "Efficient Pipeline for Camera Trap Image Review." 2019.
  https://arxiv.org/abs/1907.06772
- Norouzzadeh et al. "Automatically identifying, counting, and describing wild animals in
  camera-trap images with deep learning." PNAS, 2018. https://doi.org/10.1073/pnas.1719367115
- Beery et al. "Synthetic Examples Improve Generalization for Rare Classes." WACV 2020.
  https://arxiv.org/abs/1904.05916
- Geirhos et al. "Shortcut Learning in Deep Neural Networks." Nature Machine Intelligence, 2020.
  https://arxiv.org/abs/2004.07780
- Selvaraju et al. "Grad-CAM." ICCV 2017. https://arxiv.org/abs/1610.02391
- Guo et al. "On Calibration of Modern Neural Networks." ICML 2017. https://arxiv.org/abs/1706.04599
- Full literature review with numbers and limitations: `docs/research-report.md`
- Course material: https://github.com/skillspire-github/pytorch-deep-learning, chapters 04
  (custom datasets), 06 (transfer learning), 09 (deployment); class guideline in
  `docs/course-guideline.md`
- torchvision models: https://docs.pytorch.org/vision/stable/models.html; timm:
  https://github.com/huggingface/pytorch-image-models
- Gradio: https://www.gradio.app/docs; Hugging Face Spaces: https://huggingface.co/docs/hub/spaces
- pytorch-grad-cam: https://github.com/jacobgil/pytorch-grad-cam
- Working method: `docs/references.md` and `docs/setup-playbook.md`

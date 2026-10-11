# Who Goes There: design document

Written 2026-10-10 (session 1). This is the source of truth for the project: when another document
disagrees with it, this file wins. It replaces `docs/HANDOFF.md`, which is kept for history. Facts
about the data come from `docs/data-inspection.md`; the class requirements from
`docs/course-guideline.md`; the gaps and accepted defaults from `docs/setup-review.md`.

## 1. Problem statement

Who Goes There is a **classifier**. The input is one camera-trap photograph (a JPEG from a fixed,
motion-triggered camera, day or infrared night shot). The output is one of **12 classes**: 11
animal classes plus `car`. The core challenge is **generalising to cameras the model never saw in
training**: every camera has a fixed background, so a model can learn "this background means
coyote" and look good on familiar cameras while failing on new ones. The headline result is
therefore the gap between accuracy on **cis** test images (cameras also present in training) and
**trans** test images (cameras held out entirely), as defined by Beery et al. (2018).

Out of scope: detection and bounding boxes, sequence-level (burst) classification, the `empty`
class, and anything that needs more than a Kaggle notebook's free GPU.

## 2. Dataset and data decisions

**Source.** Caltech Camera Traps, CCT20 benchmark subset, from LILA BC. 57,868 images from 20
camera locations in the American Southwest; COCO-style JSON metadata with `location`, `seq_id`,
`frame_num`, `date_captured`, `width`, `height`. The benchmark tarball
(`eccv_18_all_images_sm.tar.gz`, about 6.5 GB) holds the images resized to at most 1024 px per
side. Licence CDLA-Permissive 1.0; attribution and citation in the README and `docs/data-card.md`.
The images and the metadata never enter git: the data guard, hooks and CI refuse them, and the
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

- `empty` (id 30): zero images in the official train split, so it cannot be learned; the 4,014
  empties in val and test are filtered out. The task is species-only.
- `badger` (9 train images), `fox` (5) and `deer` (42 train, 0 in trans test): too rare to learn
  or to score. Alternative considered: an `other` class; rejected because it would be a three-way
  mix of unrelated animals with 56 training images.
- The 78 images that carry two different species (62 are bobcat + rabbit): dropped from every
  split. Images with several annotations of the same species (414 in train) keep that one label.

**Imbalance.** opossum alone is 20% of train; opossum + raccoon are 47% of trans test. Accuracy
alone would reward predicting opossum, so macro-F1 sits next to accuracy everywhere (section 6),
and a class-weighted loss is one of the planned experiments (section 5).

**Preprocessing.** The 1024 px images are resized once on Kaggle to 256 px on the shorter side and
cached; training reads the cache. Models take 224 x 224 crops. Normalisation uses the pretrained
backbone's published ImageNet mean and standard deviation (the small CNN uses the same values so
the two models share one pipeline).

## 3. Split design and leakage tests

The official **location split** is kept: 10 cis locations (train, cis_val, cis_test) and 10 trans
locations (trans_val, trans_test), no overlap. Two published splits need repair:

- The official `train` and `cis_val` files **share 224 sequences** (bursts of three frames), which
  breaks the invariant that every burst sits in one split. So the two files are **pooled (17,037
  images before filtering) and re-split by `seq_id`**, stratified by class, about 85% train and
  15% val, from a fixed seed. The split is a deterministic function of the metadata and the seed,
  and its hash is recorded in the ledger. It is not committed as a file: the data guard refuses
  record dumps, and a seed plus a hash reproduces it exactly.
- `cis_test`, `trans_val` and `trans_test` are clean (zero shared sequences with anything) and stay
  as published, minus the class filters above.

| split | role | source |
| --- | --- | --- |
| train | fitting | pooled train + cis_val, re-split by sequence |
| val (cis) | model selection, early stopping, every hyperparameter choice | same pool |
| trans_val | sanity check only (1,725 images, one camera, 10 classes); reported, never selected on | official |
| cis test | scored once at the end | official |
| trans test | scored once at the end; the headline | official |

**Leakage tests** (`tests/test_split.py`, tier 3 code, implemented at xhigh):

1. No location appears in both train and any trans split.
2. No `seq_id` appears in two splits.
3. No image id appears in two splits.
4. Split sizes, per-class counts and the split hash match pinned values for the given seed.

**Leakage alarms** during training: trans accuracy close to cis accuracy, val accuracy far above
what the published benchmark suggests, or any test id seen by the training loader.

## 4. Model plan

Two models, trained and scored the same way, so the slides can show what pretraining buys:

1. **Small CNN from scratch** (`trailcam/model/small_cnn.py`): the chapter 04 TinyVGG pattern, three
   conv blocks (conv, ReLU, conv, ReLU, max-pool) and a linear head. It satisfies the course's
   two-layer minimum on its own and is the baseline.
2. **Fine-tuned pretrained backbone** (`trailcam/model/backbone.py`): EfficientNet-B0 from `timm`
   with ImageNet weights, classifier replaced by a 12-way linear head. 5.3M parameters, about 0.4
   GFLOPs per image, 16 MB of weights: it runs in well under a second on the CPU-only free tier of
   Hugging Face Spaces. ImageNet contains no CCT images, so pretraining is not leakage.
   Fallback if a Space prediction takes over about one second: MobileNetV3-Large (timm).

`/review-shapes` prints the layer, input shape, output shape and parameter table for both models
before any training run, and checks the head size against the class map.

## 5. Training plan

| Setting | Decision |
| --- | --- |
| Loss | Cross-entropy. One experiment with class weights (inverse frequency, square-root damped); the variant with the better val macro-F1 is the final one |
| Optimizer | AdamW; backbone at 1e-4, head at 1e-3 (two parameter groups); weight decay 1e-2 |
| Schedule | Cosine annealing with a one-epoch linear warmup, **and** early stopping on val macro-F1 with patience 3; the best checkpoint is restored |
| Epochs | Up to 15 for the small CNN, up to 10 for the backbone (one head-only epoch first); the ledger records epochs run and the best epoch |
| Batch and input | 64 images at 224 x 224 |
| Augmentation | Training only: random resized crop (scale 0.6 to 1.0), horizontal flip, mild colour jitter. Val and test: resize to 256, centre crop 224, normalise |
| Seeds | One seed for the split, one for the run; both set (`torch`, `numpy`, `random`) and logged |
| DataLoader | Shuffle on the training loader only; val and test loaders never shuffle |
| Compute | A Kaggle notebook (`notebooks/`, outputs cleared) downloads the annotations and the tarball, builds the 256 px cache once and saves it as a private Kaggle dataset, installs `trailcam` from the public repo and runs the training script. The Mac runs code, tests and the dry-run only |
| Dry run | `/dryrun` runs one epoch on a tiny batch of synthetic fixture images generated at test time (the guard blocks fixture images in git) and stops |
| Ledger | Every run appends one row (`docs/ledger.md`): run id, model, config, split seed, run seed, split hash, epochs run, best epoch, scheduler, val accuracy, val macro-F1, trans_val accuracy, checkpoint hash. The row is what gets reviewed; Kaggle runs are started by the user, who pastes the row |

Runs happen in this order: small CNN, backbone with plain cross-entropy, backbone with class
weights. Everything is chosen on val. Kaggle's weekly GPU budget (30 hours) is far above the need:
one backbone epoch over about 14,000 cached images takes minutes on a T4.

## 6. Evaluation

- **Headline table**: accuracy and macro-F1, side by side, on cis test and trans test, for both
  models. The cis-minus-trans gap is the number the slides lead with.
- **Per-class** precision, recall and F1 on trans test, and the **confusion matrix** (both as
  figures under `docs/figures/`, which has a 40-file, 20 MB guard budget).
- **Grad-CAM** (the `grad-cam` package, last conv block of the backbone) on correct and wrong
  trans-test cases, to show whether the model looks at the animal or the background.
- **Inference timer**: `time.perf_counter()` around the forward pass for one image, after one
  warm-up call, reported in seconds; the same function serves the Gradio UI.
- **Protocol**: cis test and trans test are scored **once**, at the end, with the user's go-ahead,
  by a script that writes the ledger row and the results table. That scoring is tier 3.

## 7. Demo and deployment

**Gradio app** (`trailcam/demo/app.py`): upload an image, get the top-3 classes with confidence,
the inference time in seconds, and an optional Grad-CAM overlay. Example images come from
`assets/` (a few trans-test photos, 24-file and 12 MB budget, locations listed in the data card).
A second, local-only script (`trailcam/demo/sort.py`) sorts a folder of about 200 trans-test photos
into class folders for the live part of the presentation, with a recorded backup video.

**Hugging Face Space** (`space/`): `app.py` (imports the same predict function), `requirements.txt`
exported from uv (`uv export --no-dev --no-hashes`, trimmed to what the app imports) and
`README.md` with the Space header. The weights are **not** in this repo (the guard blocks `.pt`):
`scripts/deploy_space.py` uploads `app.py`, the requirements, the examples and the weights with
`huggingface_hub`, reading the token from the environment. The Space runs on the free CPU tier,
which is why the backbone is small. Account and Space creation are the user's steps.

## 8. Decisions and why (quotable)

**Architecture.** A pretrained EfficientNet-B0, fine-tuned end to end, against a small CNN trained
from scratch. The dataset has about 14,000 training images across 12 imbalanced classes; that is
too few to learn good low-level features from zero, which is exactly what ImageNet pretraining
supplies. EfficientNet-B0 is the course's chapter 06 backbone and small enough for CPU inference
on a free Space. The scratch CNN is kept as the baseline that shows what pretraining buys.

**Loss under imbalance.** Cross-entropy, because the task is single-label multi-class. Imbalance is
handled by measuring macro-F1 next to accuracy and by one controlled experiment with
inverse-frequency class weights (square-root damped so the 229-image skunk class does not dominate
the gradient). Oversampling was rejected because it lengthens every epoch and overfits the rare
classes; label smoothing is a calibration tool, not an imbalance fix, and stays off by default.

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

## 9. Schedule

Today is Saturday 2026-10-10; the presentation is Monday 2026-10-26. One session per item group;
items live in `docs/progress.json`. Tier 3 work (the split, the final scoring) is placed so one
fix round still fits before the deadline.

| session | date | work | review |
| --- | --- | --- | --- |
| 0 | 10-10 | Bootstrap (done) | tier 1, done |
| 1 | 10-10 | This design document, CLAUDE.md, progress.json (done) | tier 0 docs; CLAUDE.md control plane |
| 2 | 10-11 | Review machinery (Part B of the reusable playbook): invariants with the 12-class rows, `risk_files.sh` with `^trailcam/`, `^space/`, `\.ipynb$`, both reviewer agents, `/review-session`; first review run by the user | tier 2 |
| 3 | 10-12 | Data: class map, metadata loader and filters, re-split by sequence with the leakage tests, data card | tier 3 |
| 4 | 10-13 | Dataset, transforms, DataLoaders, synthetic fixture generator | tier 2 |
| 5 | 10-14 | Small CNN, backbone, training loop with scheduler, early stopping, seeds, ledger, dry-run config; `/review-shapes`, `/dryrun` | tier 2 |
| 6 | 10-15 | Kaggle notebook: download, 256 px cache as a Kaggle dataset, small CNN run (user) | run ledger |
| 7 | 10-16 to 10-17 | Backbone runs: plain and class-weighted cross-entropy; selection on val; ledger checker | run ledger |
| 8 | 10-18 | Evaluation: metrics, per-class, confusion matrix, Grad-CAM, figures, inference timer; val figures | tier 2 |
| 9 | 10-19 | The single final scoring on cis and trans test (user go-ahead) | tier 3 |
| 10 | 10-20 | Gradio demo, example images in `assets/`, local sort script | tier 2 |
| 11 | 10-21 | Space: `space/`, requirements export, deploy script; account, Space and deployment (user) | tier 2 |
| 12 | 10-22 to 10-23 | Slides, README results and Space link, backup video, live-sort rehearsal | tier 0 |
| buffer | 10-24 to 10-25 | Fix rounds from the tier 3 review, rehearsal | |
| | 10-26 | Presentation | |

Slides: problem, live demo, cis vs trans result, confusion matrix with two errors, next steps.

## 10. Open questions

1. **Course dataset rule.** CCT20 is not in `torchvision.datasets`; it is loaded through a custom
   `Dataset` as chapter 04 teaches. Confirm the TA's reading by 2026-10-13, mentioning the
   location-split motivation.
2. **Submission format.** The Space (app.py, weights, requirements.txt) is known; whether a report,
   notebook or video is also required is not.
3. **Tarball layout.** The folder structure inside `eccv_18_all_images_sm.tar.gz` is unknown until
   it is opened on Kaggle (session 6).
4. **`car` as a class.** Kept because it has 668 training and 1,113 trans-test images; drop to 11
   classes if the audience finds a non-animal class confusing.
5. **Grad-CAM in the Space.** It doubles the CPU work per prediction; it may stay local-only, with
   static heatmaps in the slides.
6. **Loss variant.** Plain or class-weighted cross-entropy is decided on val in session 7 and
   recorded here.

## 11. Sources

- Sara Beery, Grant Van Horn, Pietro Perona. "Recognition in Terra Incognita." Proceedings of the
  European Conference on Computer Vision (ECCV), 2018. https://arxiv.org/abs/1807.04975
- Caltech Camera Traps on LILA BC: https://lila.science/datasets/caltech-camera-traps
  (licence CDLA-Permissive 1.0: https://cdla.dev/permissive-1-0/)
- Course material: https://github.com/skillspire-github/pytorch-deep-learning, chapters 04
  (custom datasets), 06 (transfer learning), 09 (deployment); class guideline in
  `docs/course-guideline.md`
- timm: https://github.com/huggingface/pytorch-image-models
- Gradio: https://www.gradio.app/docs; Hugging Face Spaces: https://huggingface.co/docs/hub/spaces
- pytorch-grad-cam: https://github.com/jacobgil/pytorch-grad-cam
- Working method: `docs/references.md` and `docs/setup-playbook.md`

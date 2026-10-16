# Learning journal

A plain-language record of what this project did and why, written for you (the student) to come
back to. It covers sessions 0 to 6, the Kaggle work and the evaluation items taken early (8.01,
8.03, 8.04), up to 2026-10-15. `docs/progress-log.md` holds the full detail for reviewers and
later sessions; this file holds the ideas behind it.

It is organised by **topic**, not by date. Each topic has four parts:

- **What we did**: the work, in plain words.
- **Why it mattered here**: the reasoning, and what would have gone wrong without it.
- **Where to look**: the files, tests, docs and plan sections (links are relative to this folder).
- **Try it yourself**: one or two commands that are safe and quick to run on the Mac from the
  repository root (most take seconds; the training dry run and the two evaluation dry runs take
  under a minute).

After the topics come a dated **Surprises and lessons** timeline, the **Kaggle steps you did**,
and **How to keep this journal**.

The data is the Caltech Camera Traps CCT20 benchmark (CDLA-Permissive 1.0), from Sara Beery,
Grant Van Horn and Pietro Perona, "Recognition in Terra Incognita", ECCV 2018
(https://arxiv.org/abs/1807.04975). Everything below gives counts and shapes only: no image, file
name or data row.

## Topics

1. [The problem and the dataset choice](#1-the-problem-and-the-dataset-choice)
2. [Inspecting and cleaning the data](#2-inspecting-and-cleaning-the-data)
3. [Splitting without leakage](#3-splitting-without-leakage)
4. [Dataset, DataLoader and transforms](#4-dataset-dataloader-and-transforms)
5. [Arm A vs arm B: the research question](#5-arm-a-vs-arm-b-the-research-question)
6. [The models: a small CNN vs ResNet-18](#6-the-models-a-small-cnn-vs-resnet-18)
7. [The training loop](#7-the-training-loop)
8. [Why macro-F1 sits next to accuracy](#8-why-macro-f1-sits-next-to-accuracy)
9. [Test-set isolation and leakage alarms](#9-test-set-isolation-and-leakage-alarms)
10. [Evaluation pieces built so far](#10-evaluation-pieces-built-so-far)
11. [Running on Kaggle](#11-running-on-kaggle)
12. [How the project was run](#12-how-the-project-was-run)
13. [The demo](#13-the-demo)
14. [Deployment](#14-deployment)

---

## 1. The problem and the dataset choice

**What we did.** We defined the task as a classifier: one camera-trap photo in, one of 12
classes out (11 animals plus `car`). The data is CCT20: 57,864 photos from 20 fixed,
motion-triggered cameras in the American Southwest (Beery et al. 2018). The cameras are split
into two groups. **cis** cameras appear in training (and in a test set taken on other days).
**trans** cameras never appear in training. The headline result is the gap between accuracy on
cis test photos and on trans test photos.

**Why it mattered here.** A fixed camera always sees the same background. A model can learn
"this bush means coyote" and score well on cameras it knows while failing on new ones. Beery et
al. measured this: about 19% error on known cameras against 41% on new ones, roughly a 22-point
drop. So the honest test is on cameras the model never saw, and that is what the trans test is.

The course asks for a dataset from `torchvision.datasets`. CCT20 is not in it. On 2026-10-11 you
decided to keep CCT20 anyway: it is loaded through a custom `Dataset` exactly as the course's
chapter 04 teaches, everything downstream is the same, and the location split is what makes the
project interesting. Food-101 is the named fallback if the course rejects the dataset outright.

**Where to look.** [plan.md](plan.md) sections 1 and 2 (problem, dataset) and section 11, open
question 1 (the dataset rule); [course-guideline.md](course-guideline.md), including its note on
the dataset rule; [research-report.md](research-report.md) Part 6 (the decision report) and Part 3
(the literature); [data-card.md](data-card.md) for licence and citation.

**Try it yourself.** The 12 classes and their order are pinned by a test that reads them back from
the plan's table:

```bash
uv run pytest -q tests/test_classes.py
```

## 2. Inspecting and cleaning the data

**What we did.** Before writing any model code, session 0 read the five annotation files (train,
cis_val, trans_val, cis_test, trans_test) and recorded what they actually contain. Then the
metadata loader applied four drop rules, in this order:

1. **two_species**: 78 photos carry two different animals (62 of them bobcat + rabbit). A photo
   can have only one label here, so they go.
2. **empty**: 4,014 photos with no animal. Official train holds none, so the model could never
   learn the class from the files as given.
3. **rare**: badger, fox and deer (9, 5 and 42 training photos; deer has none in trans test). Too
   few to learn or to score.
4. **unannotated**: photos with no annotation (none exist).

53,535 of the 57,864 photos are kept. Bounding boxes are stored as **fractions** of the original
image size (x, y, width, height), so the same box fits the 1024 px tarball, the 512 px cache and
any crop. 100 of the 47,729 kept boxes ran past the image edge and were clipped to it.

**Why it mattered here.** Every number later (class balance, split sizes, what "accuracy" means)
depends on these rules, so they are written once in code and counted. Checking the files against
the paper also found two disagreements (see the timeline): the paper says 57,868 photos where the
files hold 57,864, and the paper says no burst of photos is shared between train and cis_val
where the files share 224. The second one changed the split design (topic 3). The capture clock
also looked odd: hour 12 holds 4,525 photos against 1,769 at hour 11 and 1,124 at hour 13, so
the clock cannot be trusted to say day or night (topic 11).

**Where to look.** [data-inspection.md](data-inspection.md) (raw facts, "Findings that change the
plan"); [data-card.md](data-card.md) (drop table, counts per split, boxes, the hour histogram, the
paper vs files section); [plan.md](plan.md) section 2;
[metadata.py](../trailcam/data/metadata.py) and [classes.py](../trailcam/data/classes.py);
[test_metadata.py](../tests/test_metadata.py).

**Try it yourself.** Load the real metadata and print the counts per split and per drop rule (it
prints counts only, never an id):

```bash
uv run python -m trailcam.data.metadata --root "$(bash scripts/data_root.sh)"
```

**Added by 8.07, the data charts.** Five charts in `docs/figures/`, drawn from the metadata only
(no photo): class counts, camera locations, day/night, box fill and burst length. They show what
the tables said: classes and cameras are imbalanced; 60 to 72% of kept frames per split are night
(hour 12 holds 2,990 night frames, so the clock is wrong); most animals cover 1 to 5% of the frame,
which is why arm B crops in; 91% of photos have a near-twin in their burst, which is why the split
goes by burst. Look at [datavis.py](../trailcam/data/datavis.py),
[day_night.png](figures/day_night.png), [test_datavis.py](../tests/test_datavis.py). Try:

```bash
TRAILCAM_DATA_ROOT="$(bash scripts/data_root.sh)" uv run pytest -q tests/test_datavis.py -k night
```

## 3. Splitting without leakage

**What we did.** Camera traps fire in bursts, usually three frames a second apart, sharing a
`seq_id`. Because the official train and cis_val files share 224 bursts, we pooled the two files
(15,981 kept photos, 6,274 bursts) and re-split them **by burst**: about 85% train, 15% val,
balanced by each burst's majority class and by camera. The other three files (trans_val, cis_test,
trans_test) were already clean and pass through as published.

The split uses no random number generator. Within each (class, camera) group, the val bursts are
the ones with the smallest SHA-256 hash of (seed, seq_id). With seed 42 it gives train 13,587, val
2,394, trans_val 1,603, cis_test 14,478 and trans_test 21,473 photos. A **split hash** (SHA-256 over
every photo's split and label) fingerprints the result: `c78a9c10...b009`. The split is never
saved as a file; seed 42 plus the hash reproduces and verifies it.

**Why it mattered here.** Three frames of one burst are almost the same picture. If one frame is
in train and its neighbour in val, val accuracy measures memory, not learning. Splitting by burst
prevents that. Using hashes instead of `random` means the Mac and Kaggle produce exactly the same
split, whatever the Python version or record order; the hash proves it. The leakage tests check
the rules on both synthetic and real data: no camera in both train and a trans split, no burst in
two splits, no photo in two splits, every cis camera in both train and val, at least 30 val
photos per class (the smallest is skunk, 40).

This was tier 3 code (the highest review level) and got a **mutation check**: 15 bugs were planted
on purpose (splitting per photo, using the global `random`, hashing a set, and so on) and each one
made at least one test fail. That is how you know tests can actually catch mistakes.

**Where to look.** [split.py](../trailcam/data/split.py); [test_split.py](../tests/test_split.py);
[plan.md](plan.md) section 3 (split design, the five leakage tests); [data-card.md](data-card.md)
("Counts per split", "Split seed and hash").

**Try it yourself.** Rebuild the split from the seed and print its sizes and hash, then run only
the leakage tests on the real metadata:

```bash
uv run python -m trailcam.data.split --root "$(bash scripts/data_root.sh)" --seed 42
```

```bash
TRAILCAM_DATA_ROOT="$(bash scripts/data_root.sh)" uv run pytest -q tests/test_split.py -k "seq_id or location or image_id"
```

## 4. Dataset, DataLoader and transforms

**What we did.** `CCT20Dataset` follows the course's chapter 04 pattern: `classes`,
`class_to_idx`, `load_image`, `__len__` and `__getitem__`, returning a `(3, 224, 224)` float tensor
and a class index. A night photo (one grey channel) loads as three equal channels. The DataLoaders
shuffle **only** the training set; val and test keep their order and every photo.

Two transforms:

- **Training** (augmentation): a random crop covering 65% to 100% of the image (aspect ratio 3:4 to
  4:3), resized to 224; a horizontal flip half the time; mild colour jitter (brightness, contrast
  and saturation up to 20%); grayscale 10% of the time, since much of the data is infrared.
- **Evaluation** (val, test, the demo): resize so the shorter side is 256, take the centre 224 x
  224, nothing random.

Both end with ImageNet normalisation (subtract the ImageNet mean, divide by its standard
deviation per channel), the statistics the pretrained ResNet-18 was trained with.

The tests need images, but the data guard keeps every real photo out of git. So
`trailcam/data/synthetic.py` writes a small fake dataset (684 tiny JPEGs, coloured rectangles where
the boxes are, grey ones for "night") into a temporary folder whenever a test or the dry run needs
one.

**Why it mattered here.** Augmentation is a training trick: it shows the model varied versions of
each photo so it learns the animal, not the exact pixels. Applied to val or test it would make the
scores random and unfair, so the evaluation transform is fixed and tested to give the same tensor
twice. Shuffling only the training set keeps evaluation reproducible. The training loader also
uses `drop_last=True` (your decision): a final batch of one photo would break batch norm.

**Where to look.** [dataset.py](../trailcam/data/dataset.py) (`TrainTransform`,
`EvalTransform`, `CCT20Dataset`); [loaders.py](../trailcam/data/loaders.py);
[synthetic.py](../trailcam/data/synthetic.py); [test_dataset.py](../tests/test_dataset.py),
[test_loaders.py](../tests/test_loaders.py); [plan.md](plan.md) section 5 (augmentation rows);
course chapter 04 (custom datasets).

**Try it yourself.** The chapter 04 shape of the dataset and the fixed evaluation transform:

```bash
uv run pytest -q tests/test_dataset.py -k "eval_transform or chapter_04"
```

## 5. Arm A vs arm B: the research question

**What we did.** On top of the classifier, the project asks one question (plan section 6): if the
bounding boxes are used **only at training time**, to steer the random crop toward the animal, does
the model look at the animal more and does the cis-minus-trans gap shrink? Two otherwise identical
training runs:

- **Arm A (control)**: the training transform of topic 4.
- **Arm B (treatment)**: for a training photo with a box, half the time the crop is chosen so it
  contains the whole box and the box fills 20% to 100% of the crop, at a random position. Otherwise
  (and for photos without a box) it uses arm A's crop.

Everything else is the same: model, seed, epochs, optimizer, learning rates, evaluation. At test
time and in the demo both arms see the whole photo, with no box.

**Why it mattered here.** Earlier work (Beery et al. 2018; Beery, Morris and Yang 2019) shows
classifying a crop of the animal works much better, but that needs a detector at prediction time.
This project spends the free boxes at training time only, so the deployed model stays one simple
classifier. The idea is not new; the controlled measurement on CCT20 is ("The idea is not new.
The measurement is."). Because only the crop sampler differs, any difference in results can be
put down to it. Some sampler details were decisions: one box per crop when a photo has several
(not their union), the fill drawn evenly from 20% to 100%, and crops kept within arm A's 3:4 to 4:3
shape so arm B never stretches the animal more than arm A (a first draft did; it was changed).

**Where to look.** [plan.md](plan.md) section 6; [research-report.md](research-report.md) Parts 4
and 5; [boxcrop.py](../trailcam/data/boxcrop.py) (its docstring lists the decisions);
[test_boxcrop.py](../tests/test_boxcrop.py); [config.yaml](../config.yaml) (`resnet18_arm_a` and
`resnet18_arm_b` differ in `arm` only).

**Try it yourself.** The crop always holds the whole box at a fill of 20% to 100%, the arms differ
only in the sampler, and the evaluation code never even loads the box sampler:

```bash
uv run pytest -q tests/test_boxcrop.py -k "hold_the_whole_box or differ_only or never_loads"
```

## 6. The models: a small CNN vs ResNet-18

**What we did.** Two models, trained and scored the same way:

1. **Small CNN from scratch** (the chapter 04 TinyVGG pattern): three blocks of (conv, ReLU, conv,
   ReLU, max-pool) and one linear layer at the end. 348,204 parameters.
2. **ResNet-18 with ImageNet weights** (course chapter 06, transfer learning): the final layer
   replaced by a new 12-way layer, and every layer fine-tuned. 11,182,668 parameters, of which the
   new head is 6,156.

`parameter_groups` splits each model into a **body** and a **head**, so the optimizer can give
them different learning rates: body 1e-4 and head 1e-3 for ResNet-18 (the pretrained body moves
ten times more slowly than the new head). The small CNN has nothing pretrained to protect, so both
groups use 1e-3 (your decision in review 5.02). EfficientNet-B0 is kept as a fallback in case the
Space is too slow.

**Why it mattered here.** About 13,600 training photos is too few to learn good low-level
features (edges, textures) from zero; ImageNet pretraining supplies them. The small CNN is the
baseline that shows what pretraining buys. A curious fact you can see below: in the small CNN, the
final linear layer alone holds about 86% of the parameters, because it connects every value of the
last feature map (25,088 numbers) to each of the 12 classes. ImageNet holds no CCT photos, so
pretraining is not leakage.

**Where to look.** [small_cnn.py](../trailcam/model/small_cnn.py),
[backbone.py](../trailcam/model/backbone.py), [factory.py](../trailcam/model/factory.py)
(`build_model`, `parameter_groups`); [test_models.py](../tests/test_models.py); [plan.md](plan.md)
sections 4 and 9 ("Architecture", "Optimizer"); the `/review-shapes` command
([review-shapes.md](../.claude/commands/review-shapes.md)) prints each layer's shapes.

**Try it yourself.** Print the body and head parameter counts of both models (random weights, no
download):

```bash
uv run python -c "from trailcam.model.factory import build_model, parameter_groups as g; [print(n, *(sum(p.numel() for p in grp) for grp in g(build_model(n, pretrained=False)))) for n in ('small_cnn', 'resnet18')]"
```

## 7. The training loop

**What we did.** `trailcam/train/train.py` trains one config and writes one ledger row. Its
pieces:

- **Loss**: plain cross-entropy (one label per photo, no class weights).
- **Optimizer**: AdamW with weight decay 1e-2 and the two parameter groups of topic 6.
- **Schedule**: the learning rate warms up linearly over the first epoch, then follows a cosine
  curve down toward zero, updated every optimizer step.
- **Head-only first epoch** (ResNet-18): the body is frozen and only the new head learns, so a
  random head does not wreck the pretrained weights. This epoch counts inside the 10.
- **Early stopping**: after each epoch, val macro-F1 is measured. If it does not beat the best so
  far for 3 epochs in a row (a tie does not count), training stops and the weights of the best
  epoch are put back.
- **Mixed precision**: on the GPU the forward pass runs in float16 (faster, less memory) with a
  gradient scaler, so small gradients do not round to zero.
- **Seeds**: one seed sets Python's `random`, NumPy, torch and the shuffle order, so a run can be
  repeated.
- **The ledger**: every real run appends one row to `docs/ledger.md` with 16 columns, including
  three hashes: the **config hash** (of the settings, leaving out `arm` and the run seed, so arm A
  and arm B of one seed must match), the **split hash** (topic 3) and the **checkpoint hash** (of the
  saved weights file).
- **The dry run** (`/dryrun`): one epoch on two synthetic photos per batch, on the Mac, in about 25
  seconds. It prints the row but never appends it.

**Why it mattered here.** The course asks you to justify the optimizer and the number of epochs.
Here the number of epochs is not guessed: cosine plus early stopping decides it, and the ledger
records how many actually ran. AdamW forgives a roughly tuned learning rate. The hashes make the
experiment checkable: if arm A and arm B ever ran on different settings or a different split, the
ledger checker refuses the rows. The dry run catches pipeline bugs in seconds instead of after an
hour of GPU time.

**Where to look.** [train.py](../trailcam/train/train.py) (`lr_factor`, `EarlyStopping`, `fit`,
`train_run`); [config.py](../trailcam/train/config.py) and [config.yaml](../config.yaml);
[ledger.py](../trailcam/train/ledger.py) and [ledger.md](ledger.md);
[test_train.py](../tests/test_train.py), [test_config.py](../tests/test_config.py),
[test_ledger.py](../tests/test_ledger.py); [plan.md](plan.md) sections 5 and 9;
[dryrun.md](../.claude/commands/dryrun.md).

**Try it yourself.** The dry run (the same as `/dryrun`): watch the losses, the `(2, 12)` logits
shape and the ledger row it prints. The leakage alarm at the end is expected, because random weights
on synthetic images score the same everywhere.

```bash
uv run python -m trailcam.train.train --max-epochs=1 --batch-size=2 --dry-run=True
```

## 8. Why macro-F1 sits next to accuracy

**What we did.** Every score is reported twice: **accuracy** (share of photos labelled correctly)
and **macro-F1** (compute the F1 score of each class separately, then average them, so every class
counts equally). Early stopping picks checkpoints by val macro-F1. When a split lacks some classes
(trans_val has 9 of the 12), macro-F1 averages over the classes present.

**Why it mattered here.** The classes are very unequal. Opossum is 20% of training; opossum and
raccoon together are 51% of trans test. A model that mostly says "opossum" can post a decent
accuracy while being useless on skunks and rodents. Macro-F1 exposes that. Accuracy stays because
the course asks for standard metrics and an audience expects it. The imbalance is measured, not
reweighted: a weighted loss would be a second difference between the arms, and the literature
(WILDS) found such methods do not help on unseen cameras.

**Where to look.** [metrics.py](../trailcam/evaluate/metrics.py) (`accuracy`, `macro_f1`,
`per_class_scores`); [test_metrics.py](../tests/test_metrics.py) (hand-computed cases);
[plan.md](plan.md) section 9 ("Headline metric", "Loss under imbalance");
[data-card.md](data-card.md) (counts per class).

**Try it yourself.** Macro-F1 checked against hand-worked examples, including a split with absent
classes:

```bash
uv run pytest -q tests/test_metrics.py -k macro_f1 -v
```

## 9. Test-set isolation and leakage alarms

**What we did.** Every choice (learning rates, epochs, checkpoint, which arm wins) is made on val.
cis test and trans test are scored **once**, at the very end, after every reported run has
trained, with your go-ahead, and logged in the ledger. The code enforces this rather than trusting
memory: the training script drops the test loaders as soon as they are built, the notebook has no
cell that touches a test split, and the attention command line refuses any split but val and
trans_val. trans_val (one camera) is reported but never used to choose anything.

Leakage alarms stop a run that looks too good to be true: a training photo id found in val or
trans_val stops it before training; after training, trans_val accuracy within 0.02 of val accuracy
(or above it), or val accuracy above 0.95, makes it exit with code 3 after writing its row.

**Why it mattered here.** If you look at test scores and then change something, the test set
quietly becomes a second val set and the final number is optimistic. On this dataset, leakage
is easy (shared bursts, shared cameras) and its signature is clear: new-camera accuracy as good as
known-camera accuracy, which the literature says should not happen. So that is exactly what the
alarm watches for.

**Where to look.** [plan.md](plan.md) sections 3 ("Leakage alarms") and 7 ("Protocol");
[train.py](../trailcam/train/train.py) (`check_loaders`, `score_alarms`);
[test_train.py](../tests/test_train.py), [test_attention.py](../tests/test_attention.py),
[test_notebook.py](../tests/test_notebook.py); the invariant rows on splits and test-set isolation
in [invariants.md](review/invariants.md); [ledger.md](ledger.md).

**Try it yourself.** The score alarms, the attention command line's refusal of test splits and the
notebook's "no test split" rule:

```bash
uv run pytest -q tests/test_train.py tests/test_attention.py tests/test_notebook.py -k "score_alarms or refuses_every_split or no_test_split"
```

**Added by 9.01, the final scoring script.** A test set is a one-shot measurement: once you have
seen its score, every later choice is tuned on it a little. So `score_test.py` scores every
checkpoint together and refuses a test ledger that already holds a row. It refuses to run without
`--approved-by` (the go-ahead is a rule only if it leaves a trace; the name goes into every row)
and without `--run-seeds`, which states the seeds reported, since the ledger cannot see a run still
training. Every refusal comes before anything is loaded: a refusal after the first prediction
would already be a look. Rows go to [test-ledger.md](test-ledger.md); look at
[score_test.py](../trailcam/evaluate/score_test.py) (docstring),
[test_score_test.py](../tests/test_score_test.py), [plan.md](plan.md) section 7. Try:

```bash
uv run pytest -q tests/test_score_test.py -k "approver or scored_again or run_seeds"
```

## 10. Evaluation pieces built so far

**What we did.** By 2026-10-15 these are built and tested (none run on real checkpoints yet):

- **Metrics**: accuracy, macro-F1, per-class precision, recall and F1, and a 12 x 12 **confusion
  matrix** (rows the true class, columns the predicted one), written by hand without scikit-learn.
- **Inference timer**: one untimed warm-up call, then `time.perf_counter()` around a second forward
  pass, in seconds. The first call to a model is slow (memory set-up), so timing it would mislead.
- **Grad-CAM**: a heat map of which parts of the photo pushed the model toward its answer. It
  takes the gradients of the predicted class with respect to the last convolutional feature maps
  (`layer4` of ResNet-18), weights the maps by them and keeps the positive part. Written by hand
  (about 80 lines) rather than adding the `grad-cam` package (which would bring OpenCV and
  scikit-learn); checked once against that package and found identical after scaling.
- **Attention-in-box**: the share of the Grad-CAM heat that falls inside the animal's box. 1.0
  means the model looked only at the animal; an even map would score the box's share of the frame,
  which the report prints alongside as the "chance" level.
- **Animal-removed test**: paint the box with the photo's average colour and predict again. A model
  that learned animals should lose its answer; a model that learned the scenery keeps it.
- **Slices** (item 8.04): the same predictions cut two ways, with no extra training. *Day vs
  night* uses the pixel-based night flag of topic 11. *Common vs rare* uses a rule fixed on the
  train counts before any test scoring: a class is rare when it holds under 4% of the training
  photos, which makes bird, rodent and skunk rare (car, at 4.4%, is common). Each slice gets
  accuracy and macro-F1, and the two slices of a cut always add up to the whole split.
- **Same day/night mix**: the splits differ in how many night photos they hold. Over the kept,
  re-split photos the night share is 0.607 in train, 0.600 in val, 0.633 in trans_val, 0.626 in
  cis_test and 0.716 in trans_test (0.657 overall). So part of a cis-to-trans gap could come from
  trans_test simply having more night photos, not from the new cameras. The report re-weights the
  trans photos (night ones by p / q, day ones by (1 - p) / (1 - q), p the cis night share and q
  the trans one) so trans is scored as if it had the cis mix; the gap that is left is the part the
  day/night mix does not explain.
- **Feature map**: the backbone's features for each photo (512 numbers for ResNet-18), squashed to
  two dimensions with PCA and plotted, once coloured by species and once by camera, arm A beside
  arm B. If photos cluster by camera more than by species, the model has learned the scenery.

- **Figures and overlays** (item 8.02): the confusion matrix, per-class bars and per-camera bars
  as PNGs, and Grad-CAM overlays drawn with the same `gradcam` function attention-in-box scores,
  from `assets/` photos only. `evaluate.py` is the one entry point that scores a checkpoint on val
  or trans_val and writes `metrics.md` and the PNGs.

**Why it mattered here.** Accuracy says how often the model is right, not why. Attention-in-box
and the animal-removed test are the measurements that answer the research question (topic 5): does
arm B look at the animal more, and does it depend less on the background? One Grad-CAM function is
both scored and shown in the slides, so the picture and the number cannot disagree. The slices
guard against a wrong story: without the same-mix comparison, a gap caused partly by more night
photos in trans_test would be blamed entirely on the unseen cameras. Fixing the rare rule before
seeing any test score keeps it from being chosen to make the results look good.

**Where to look.** [metrics.py](../trailcam/evaluate/metrics.py),
[gradcam.py](../trailcam/evaluate/gradcam.py), [attention.py](../trailcam/evaluate/attention.py),
[slices.py](../trailcam/evaluate/slices.py) (its docstring explains the re-weighting);
[test_metrics.py](../tests/test_metrics.py), [test_gradcam.py](../tests/test_gradcam.py),
[test_attention.py](../tests/test_attention.py), [test_slices.py](../tests/test_slices.py);
[plan.md](plan.md) sections 6 (outcomes) and 7.

**Try it yourself.** Run attention-in-box and the animal-removed test on the synthetic fixture with
random weights; it writes a table and a histogram into a temporary folder (the scores mean nothing
with random weights; the point is to see the shape of the report):

```bash
uv run python -m trailcam.evaluate.attention --dry-run=True --split val --out "$(mktemp -d)"
```

The same for the slices: the day/night, common/rare and same-mix tables printed, plus the two
feature-map PNGs in a temporary folder:

```bash
uv run python -m trailcam.evaluate.slices --dry-run=True --out "$(mktemp -d)"
```

**Added by 8.02.** Two lessons from the overlays. A figure can leak data through the panel beside
the one you look at: a raw left panel would have put a dataset photo into `docs/figures/` with
every test green. And a zero heat map blends to the raw photo, so the guard refuses it. Restore a
mutation test by copying the file back, never with `git checkout` (it once reset uncommitted
work). Look at [figures.py](../trailcam/evaluate/figures.py),
[evaluate.py](../trailcam/evaluate/evaluate.py), [test_figures.py](../tests/test_figures.py). Try:

```bash
uv run pytest -q tests/test_figures.py -k assets
```

## 11. Running on Kaggle

**What we did.** The Mac writes code and runs tests; Kaggle's free GPU trains. Getting there took
several pieces:

- **Packaging.** A `[build-system]` table in `pyproject.toml` makes the repo installable with
  `pip install git+https://...` (the one place `pip` is allowed; everywhere else it is `uv`).
- **Keeping Kaggle's torch.** Kaggle ships a GPU build of torch. If our package demanded a newer
  torch than Kaggle has, pip would replace it with a generic build. So the minimum versions were
  lowered to ones actually tested (`torch>=2.3`), and the notebook checks the torch, torchvision,
  Pillow, timm and PyYAML versions are the same before and after the install.
- **The image cache.** The 6.5 GB tarball of 1024 px images is streamed once on a Kaggle CPU
  session, each photo resized so its longer side is 512 px, and saved as a private Kaggle dataset
  (about 2.9 GiB). Training reads the cache, not the tarball.
- **One archive, unpacked to local disk** (item 6.08). The cache is 57,864 small files. It is now
  packed once, on a CPU notebook, into one uncompressed tar (`cct20_512.tar`, 3,213,506,560
  bytes) with `scripts/kaggle_pack_cache.py`. At the start of every run the notebook unpacks it to
  `/tmp` on the session's local disk (about 35 s), and training reads from there. `unpack_cache`
  checks every member before writing anything: no links, no absolute names, no `..`, nothing
  outside the top folder, enough free space. That is the standard defence against a tar file
  that tries to write outside its folder.
- **Attach the notebook output, not a dataset.** When a notebook's output is turned into a Kaggle
  dataset, Kaggle unpacks a `.tar` into folders, which undoes the point of the archive. So the
  training notebook attaches the packing notebook's output directly. It mounts at
  `/kaggle/input/notebooks/<user>/<slug>/cct20_512.tar`, and `CACHE_ARCHIVE` in the parameter cell
  is set to that path.
- **The day/night flag.** Infrared night photos are grey: their three colour channels are almost
  equal. The **channel spread** (largest channel minus smallest, averaged over the pixels) is at
  most 2.0 for night and much higher for day. It is stored per photo in `day_night.json` beside the
  cache, together with the capture hour and the spread itself, so a new threshold would not need a
  rebuild.
- **The notebook.** It clones the repo at a fixed `COMMIT`, installs it, checks every installed
  module matches the clone byte for byte, checks the GPU and the cache, trains, and prints the
  ledger row to paste.
- **GPU quota.** Kaggle gives 30 GPU hours a week. GPU sessions only for training and scoring;
  the cache build on CPU; real runs with "Save & Run All", so they run in the background and no
  GPU time idles; interactive GPU sessions stopped right after use. GPU T4 x2 is billed at
  about its wall time, not double. A committed version is billed from submission, including any
  time it sits at "Starting...". So, about 10 minutes after Save & Run All, open the Logs tab: if
  it is still empty and the version still says "Starting...", cancel it and resubmit. Every
  training log line is flushed, with seconds and images per second per epoch, so an empty log
  really means no cell has started.
- **GPU architectures.** Each GPU generation has a compute capability ("sm" number): the P100 is
  Pascal, sm_60; the T4 is Turing, sm_75. A torch build only contains GPU code for the
  architectures it was compiled for. Kaggle's torch 2.11 (built for CUDA 12.8) includes sm_75 to
  sm_120 but no sm_60, so the P100 cannot run it. We switched to **GPU T4 x2**, training on one T4
  (`CUDA_VISIBLE_DEVICES=0`), and `check_gpu` refuses an unsupported GPU in seconds, before any
  training.
- **The rehearsal.** Before a real run: run sections 1 to 3 (setup, GPU check, cache check), stop
  the session, then "Save & Run All". Mistakes surface in a minute, not after an hour.
- **The first real run** (6.04, the small CNN, commit `61f1ecc`). 15 epochs at about 44 s each,
  about 350 training images per second from local disk, 12.8 minutes in all and about 13 minutes
  of quota. Val accuracy 0.6625 and macro-F1 0.6216, against trans_val accuracy 0.1965 and
  macro-F1 0.0775. That is cameras seen in training against a camera never seen, the gap the
  project is about. trans_val is one camera and small, so its numbers are noisy; the reported
  results come only from the final test scoring. The leakage alarm stayed quiet, as it should:
  trans_val is far below val, and val is below 0.95. Val macro-F1 was still improving when the
  15-epoch cap ended the run. The recipe was fixed in advance, so it stays as it is: tuning the
  small CNN on this run, then carrying that over to the ResNet runs, would quietly turn val into
  something the experiment was fitted to.

**Why it mattered here.** Every one of these pieces came from a real failure mode: a silently
replaced torch, a GPU that cannot run the code, a dataset mounted at an unexpected path, a cache
that loaded only partly, a stale notebook cell, a run stuck starting while billed. The common
lesson is to **check cheaply up front and fail loudly**, before spending GPU quota.

**Where to look.** [plan.md](plan.md) sections 2 (preprocessing, the sidecar) and 5 (compute, GPU
budget, platform); [pyproject.toml](../pyproject.toml);
[kaggle_prepare_cct20.py](../scripts/kaggle_prepare_cct20.py),
[kaggle_pack_cache.py](../scripts/kaggle_pack_cache.py) (its docstring has the packing steps);
[daynight.py](../trailcam/data/daynight.py), [cache.py](../trailcam/data/cache.py),
[device.py](../trailcam/train/device.py); [kaggle_train.ipynb](../notebooks/kaggle_train.ipynb)
(its first cell is the "How to run on Kaggle" list); [test_device.py](../tests/test_device.py),
[test_cache.py](../tests/test_cache.py), [test_daynight.py](../tests/test_daynight.py),
[test_kaggle_prepare.py](../tests/test_kaggle_prepare.py),
[test_kaggle_pack.py](../tests/test_kaggle_pack.py),
[test_dependencies.py](../tests/test_dependencies.py).

**Try it yourself.** The GPU check against Kaggle's real architecture list (a T4 accepted, a P100
refused), shown test by test:

```bash
uv run pytest -q tests/test_device.py -v
```

And the archive: packed and unpacked to the same files, and a tar that tries to escape its folder
(a `..`, an absolute name, a link) refused before anything is written:

```bash
uv run pytest -q tests/test_kaggle_pack.py tests/test_cache.py -k "same or traversal or space"
```

## 12. How the project was run

**What we did.**

- **Plan first.** `docs/plan.md` was written before any model code and wins when documents disagree.
  Work is cut into items in `docs/progress.json`, one item per session, each with steps that must
  pass before it is marked done.
- **Tests and mutation checks.** Each item comes with tests, and the important ones are checked by
  planting bugs on purpose (a "mutation") and confirming a test fails. By 2026-10-16 the suite has
  802 tests. `bash scripts/check.sh` runs them all plus strict type checking (mypy), linting and
  formatting (ruff) and the data guard; it must be green before every commit.
- **The data guard.** `scripts/guard_data.py` runs in the git hooks and in CI and refuses images,
  metadata dumps, weights and keys, so no dataset photo or secret can reach the public repo.
- **Review tiers.** A script (`risk_files.sh`) lists the high-stakes files a change touches and
  picks the tier: 0 for docs only, 1 for code that touches no high-stakes file, 2 for high-stakes
  files (two reviewers on different models: a code reviewer on the whole change and a risk reviewer
  on the risky files), 3 for the split code and the final scoring (tier 2 plus a cloud review).
  Every finding is verified before it is fixed, every fix gets a test that fails on the old code,
  and a fix touching a high-stakes file gets one more "fix round". A table of 38 rules that must
  always hold (`invariants.md`) tells the reviewers what to check.
- **The coordinator.** A pinned coordinator session tracks the schedule and hands out items as
  chips; each item runs in its own worktree session, at most two in parallel when they do not
  depend on each other. You start every review, and you merge and push in the main checkout; the
  sessions never do.

**Why it mattered here.** A model can train, print good numbers and still be wrong, through a
leaked burst, a test set peeked at, or a different setting between the arms. Those mistakes do not
crash; they only show up if something checks for them. The tests, the guard and the two-reviewer
tiers are that something. Keeping merge and push with you means nothing reaches the public repo
without your eye on it.

**Where to look.** [setup-playbook.md](setup-playbook.md) ("Code review playbook", "Session
protocol"); [invariants.md](review/invariants.md); [risk_files.sh](../scripts/risk_files.sh),
[check.sh](../scripts/check.sh), [guard_data.py](../scripts/guard_data.py);
[progress.json](progress.json), [progress-log.md](progress-log.md);
[references.md](references.md); the project rules in [CLAUDE.md](../CLAUDE.md).

**Try it yourself.** List the high-stakes files that item 6.07's commit touched, the list that
decided its review tier:

```bash
bash scripts/risk_files.sh 532a74f~1..532a74f
```

**Added by 6.09.** The test that broke on the first ledger row pinned a moment (an empty ledger),
not the rule. It now checks what holds for the file's whole life, by reusing the ledger checker
instead of a second copy of its rules, and the small CNN's row is in [ledger.md](ledger.md). Run
`check.sh` before every commit, docs-only ones too: the tests read the docs. Look at
[test_train.py](../tests/test_train.py), [ledger.py](../trailcam/train/ledger.py). Try:

```bash
uv run python -m trailcam.train.ledger --check docs/ledger.md
```

**Added by 10.02 (budgets).** Weekly Claude usage is one budget shared across all your
projects. On 2026-10-16 it was 72% used with 5 days to the reset, so the must-have items ran first
and the rest waited; "lean" meant no waste (no extra agents, no whole-file rereads), never fewer
checks. Treat it like the Kaggle GPU quota: know the balance before you plan the week. And
`git merge` without `--no-edit` opens vim for the message (Esc, then `:wq`, then Enter); merges now
use `--no-edit`. Look at [setup-playbook.md](setup-playbook.md) ("Session protocol").

## 13. The demo

**Added by 10.01.** The model outputs a probability for each of the 12 classes (softmax), and they
sum to 1. The demo shows the three largest, so those sum to at most 1; a low top value means the
model is unsure. The time shown is the model's forward pass alone: decoding the JPEG and resizing
depend on the photo, not the model. The timed pass's own scores are reused, so each request runs
the model twice (warm-up and timed), not three times. Grad-CAM shows which part of the 224 x 224
frame pushed the top class up: on the animal is good, on the background means the model learned the
scene. Look at [predict.py](../trailcam/demo/predict.py), [app.py](../trailcam/demo/app.py),
`timed_forward` in [metrics.py](../trailcam/evaluate/metrics.py), `overlay_photo` in
[gradcam.py](../trailcam/evaluate/gradcam.py). Try (builds the app with random weights, no launch):

```bash
uv run python -m trailcam.demo.app --check
```

**Added by 10.02.** The live sort copies a folder of photos into one folder per class and
prints a running timer and the count per class, to show throughput on about 200 unseen-camera
photos. The model is loaded and warmed up once, then each photo costs one forward pass. It prints
counts and seconds only, never a file name, and reads no labels, so it cannot score anything: the
test sets are scored once, in 9.01 and 9.02. It refuses a source or a destination inside a git
repository, so it never reads the dataset folder and sorted photos can never be committed. Look at [sort.py](../trailcam/demo/sort.py),
[test_sort.py](../tests/test_sort.py). Try (sorts the synthetic fixture with random weights):

```bash
uv run python -c "from pathlib import Path; from trailcam.data import synthetic; synthetic.write_synthetic_fixture(Path('/tmp/wgt-fixture'))"
uv run python -m trailcam.demo.sort --src /tmp/wgt-fixture --dst /tmp/wgt-sorted
```

**Why Gradio.** The course requires it ([course-guideline.md](course-guideline.md) section 5).
The alternatives: Streamlit is built for dashboards and reruns the whole script on every click;
FastAPI or Flask give full control but mean writing the page, the upload and the server yourself;
notebook widgets cannot be shared as a link; an on-device app is out of scope. Gradio fits the
shape of this demo exactly, one image in and the top 3 out, and one `predict` function serves the
app, the Space and the inference timer. The live sort is a command-line script on purpose: a web
page shows one photo at a time, a terminal shows photos per second.

## 14. Deployment

**What we did.** Chose Hugging Face Spaces to host the Gradio demo (items 11.01 to 11.03): a
Space is a git repository holding `app.py`, `requirements.txt` and the weights, and Hugging Face
builds it and serves it at a public link. Since mid-2026 a free account cannot create a Gradio
Space on the free CPU Basic hardware; that needs PRO, $9 a month. Free ZeroGPU hosting needs an
account older than 30 days, and yours was created on 2026-10-16. The plan: PRO for one month from
about 2026-10-20, then move to free ZeroGPU around 2026-11-15, with `app.py` made ZeroGPU-ready
in 11.01. The `--share` link of 10.01 is the free, temporary backup.

**Why it mattered here.**

- **Why people share models at all.** For researchers: reputation, the field's norm of
  publishing work others can check, feedback, and a portfolio. Companies share for adoption: a
  model people try is a model people build on. Hugging Face pays for the free hosting to grow the
  community that uses its hub.
- **Spaces vs AWS and the free clouds.** The layers run from IaaS (AWS EC2: a bare machine, you
  manage everything), to PaaS (Google Cloud Run, Render: a Dockerfile in, a URL out), to ML
  platforms (SageMaker, Modal: training and serving jobs), to demo hosting (Spaces: push `app.py`,
  `requirements.txt` and the weights). Spaces fits a graded demo; AWS fits a real product, with
  scaling, monitoring and a bill to match.
- **Check the platform's rules before the schedule depends on them.** The free tier changed under
  the plan; finding out on 10-16 left four days to the due date, enough to pay for one month.

**Where to look.** [course-guideline.md](course-guideline.md) section 5;
[plan.md](plan.md) section 8; items 11.01 to 11.03 in [progress.json](progress.json);
[app.py](../trailcam/demo/app.py) for `--share`.

**Try it yourself.** The packages the Space would need, as `requirements.txt` lists them:

```bash
uv export --no-dev --no-hashes
```

---

## Surprises and lessons

Dated, oldest first. Each: what happened, how it was caught, the lesson.

- **2026-10-10. The paper and the files disagree.** The paper says 57,868 photos; the released
  files hold 57,864. The paper says train and cis_val share no burst; the files share 224 (222
  after filtering). *Caught by* counting the files during the session 0 inspection, before any code
  was written. *Lesson:* check the published numbers against the files you actually have. The
  shared bursts would have leaked near-duplicate frames into val; they are why the pool is
  re-split by burst.
- **2026-10-13. Pillow 10.1's JPEG encoder is not reproducible (item 4.03).** On the oldest Pillow
  the project allows, the same grey image saved as JPEG in two processes sometimes gave different
  bytes. *Caught by* the reviewer's question "does this work on the minimum version?", answered by
  running the tests in a separate environment with the old versions: one test failed in every run.
  The locked Pillow 12.3 is fine. *Lesson:* a minimum version is a promise; test it. It also
  led to a check in the Kaggle cache script: encode the same images in two processes and refuse to
  continue unless the bytes match.
- **2026-10-14. Kaggle's torch has no P100 code (item 6.06).** The plan had chosen the P100.
  Kaggle's torch 2.11 is built for sm_75 and newer; the P100 is sm_60. *Caught by* a review question
  ("does Kaggle's torch still support Pascal?") and your short check in a GPU T4 x2 session that
  printed the build's architecture list. *Lesson:* hardware and library versions must match; check
  in a two-minute session before planning hours of GPU time. Now `check_gpu` refuses a mismatch
  in seconds.
- **2026-10-14. The cache build (item 6.03).** 22 minutes on a CPU session; all 57,864 photos
  written. The channel spread had a clean gap between night and day, so the threshold of 2.0 is
  safe. Over all 57,864 photos (empties and dropped classes included, by official file) the night
  share is 63.6% (the plan had guessed about half); trans_test 68.8% against about 60% in the other
  files; and hour 12 is 61.5% night, which confirms the camera clock is wrong on many photos. Over
  the 53,535 kept photos of the re-split the shares are higher: 65.7% overall, train 60.7%, val
  60.0%, trans_val 63.3%, cis_test 62.6% and trans_test 71.6% (computed on 2026-10-15 from the
  sidecar copy you downloaded into `data/`). *Caught by* the counts the prepare script prints (night
  share per split and per hour, the spread histogram). *Lesson:* measure instead of assuming, and
  say which photos a share is over. The pixels, not the clock, say day or night, and the
  unseen-camera test set is more nocturnal than training, which is why 8.04 compares cis and trans
  at the same day/night mix (topic 10).
- **2026-10-14. Kaggle's quota and account rules.** The quota is 30 GPU hours a week, and a short
  T4 x2 session cost about its wall time (29h 49m left after it), not double as a reviewer had
  feared. GPU and internet access needed phone verification of the Kaggle account first. *Caught
  by* reading the quota page before and after the session; the phone check is a Kaggle account
  step that comes before any GPU or internet session.
  *Lesson:* platform limits are part of the plan; measure the cost of a short session before
  committing to long ones.
- **2026-10-14/15. The dataset mount was one level deeper than expected.** Kaggle mounted the
  dataset at `/kaggle/input/datasets/<user>/<slug>/`, not `/kaggle/input/<slug>/`, and below the
  notebook's three-level search. *Caught* during the rehearsal: the notebook's default path did
  not exist and its search did not reach the mount. *Lesson:* never hard-code a platform path from
  memory; search for a marker file (`day_night.json`) and print what is found. The search now goes 8
  levels deep.
- **2026-10-14/15. A duplicate parameter cell overrode the edited one.** The notebook had a stale
  second copy of the parameter cell, which ran after the edited one and silently put the old values
  back. *Caught* during the rehearsal, when the hand edit did not take effect. *Lesson:*
  configuration should live in exactly one place, and the program should print the values it is
  actually using. This led to item 6.07: one parameter cell (tested), defaults set to the real
  dataset path, the values printed, and a cache completeness check.
- **2026-10-15. Kaggle printed a partial file count.** One session start showed 645 files (about
  65 MiB of the 2.9 GiB cache), as if the dataset had only partly loaded; an earlier session had
  found all 57,864. *Caught by* comparing the count with the expected total. Settled by the new
  completeness check (`check_images`), which checks every kept photo exists and is not empty: all
  53,535 kept images were present. *Lesson:* do not trust a progress display; check completeness
  against the metadata before training on it.
- **2026-10-15. The first GPU run sat at "Starting..." for 48 minutes.** The 6.04 version (57,864
  small input files) showed "Starting..." for about 48 minutes with an empty Logs tab and no
  output, and the quota fell by about that much. *Caught by* the empty Logs tab and the quota
  page. The first guess was that Kaggle copies every input file before the first cell. But a CPU
  version with the same files logged its first line after about 5 minutes, so that does not
  explain it; a slow GPU session start is likelier, and the cause is still not known. *Lesson:*
  when you cannot find the cause, protect yourself from the cost: check the log about 10 minutes
  after submitting and cancel and resubmit if nothing has started. Item 6.08 also packed the cache
  into one archive, which starts faster and trains from local disk.
- **2026-10-15. Kaggle unpacked the archive.** Turning the packing notebook's output into a
  dataset made Kaggle extract the `.tar` back into its 57,864 files. *Caught by* the check that
  6.08's steps asked for: the dataset's Data tab must show the one tar file. *Lesson:* platforms
  "help" in ways you did not ask for; verify the thing you meant to create. The fix was to attach
  the notebook's output itself, which keeps the tar as one file (unpacked to local disk in about
  35 s).
- **2026-10-15. The first real training run (6.04, small CNN).** 12.8 minutes, about 13 minutes of
  quota, val accuracy 0.6625 against trans_val accuracy 0.1965 (macro-F1 0.6216 against 0.0775).
  *Caught by* the ledger row and the per-epoch log lines. *Lesson:* the gap between seen and
  unseen cameras showed up at once, as Beery et al. found, and it is large for a model trained from
  scratch. Val was still improving at the 15-epoch cap, but the recipe was fixed in advance and is
  not tuned on this run, so the later runs are not fitted to what this one showed.
- **2026-10-15. The first real ledger row turned main red.** Committing 6.04's row to
  `docs/ledger.md` broke a test written before any run existed, "the ledger holds the header and
  no rows" (`tests/test_train.py`). The commit was made without running `check.sh` first, because it
  touched docs only. *Caught by* main going red after the commit; the commit was reverted, and a
  later item (6.09) changes the test. *Lesson:* run `check.sh` before every commit, docs-only ones
  included, as CLAUDE.md says. A test can depend on a file that looks like plain documentation.

- **2026-10-16. Free accounts can no longer host a Gradio Space on CPU.** Since mid-2026 the
  free CPU Basic Space needs PRO ($9 a month), and free ZeroGPU needs an account older than 30
  days; the account was created that day. *Caught by* trying to plan item 11.02 against the
  current Hugging Face rules, four days before the due date. *Lesson:* free tiers change; check
  them before the schedule leans on them. Decision: PRO for one month from about 10-20, free
  ZeroGPU from about 11-15, and `--share` as the free backup (topic 14).
- **2026-10-16. `git merge` opened vim.** A merge without `--no-edit` stopped in vim to ask for
  the message. *Caught by* the terminal waiting. *Lesson:* Esc, `:wq`, Enter gets out; merges now
  pass `--no-edit`.
- **2026-10-16. The weekly Claude budget ran low.** 72% used with 5 days left, shared with other
  projects. *Caught by* the usage page. *Lesson:* budget it like the Kaggle GPU quota: must-have
  items first, the rest after the reset, and cut waste, not checks.

---

## Kaggle steps you did

In order:

1. **Build the cache on a CPU notebook**: run `scripts/kaggle_prepare_cct20.py` (internet on, no
   GPU), which downloads the tarball, writes the 512 px cache and `day_night.json` under
   `/kaggle/working`, and prints the counts.
2. **Create the private dataset from the notebook's output.** Kaggle's page showed an error, but
   the dataset was created anyway (check the dataset list before retrying).
3. **Import the training notebook** with File > Import Notebook and the GitHub link to
   `notebooks/kaggle_train.ipynb`.
4. **Attach the dataset** with Add Input; it mounts under `/kaggle/input/datasets/<user>/<slug>/`.
5. **Set `COMMIT`** in the one parameter cell. It was first
   `21faa79b0aed9ff4bb776bc5aa85f65ba73bfbe6`.
6. **Rehearse sections 1 to 3** (setup, install, GPU check, cache check) with GPU T4 x2 and internet
   on, then **stop the session**.
7. **Save & Run All.** This first 6.04 version sat at "Starting..." for about 48 minutes (see the
   timeline), which led to item 6.08.
8. **Pack the cache into one archive on a CPU notebook**: attach `cct20-512`, clone the repo,
   `pip install` it and run `scripts/kaggle_pack_cache.py` with Save & Run All (about 17 minutes;
   it prints the file counts, the size and the SHA-256).
9. **Attach the packing notebook's output** to the training notebook (not a dataset made from it,
   which Kaggle unpacks) and remove the loose `cct20-512` input. `CACHE_ARCHIVE` must be the
   archive's path under `/kaggle/input/notebooks/<user>/<slug>/`; since 6.09 that is the
   notebook's default.
10. **Set `COMMIT`** to `61f1eccf24a3e33cf876e4ceb9dd5e5af8aed020`, the training commit for 6.04
    and the runs after it, so every run and both arms train on identical code.
11. **Save & Run All**, check the Logs tab after about 10 minutes, then paste the printed ledger
    row when the run ends.

---

## How to keep this journal

Every later item appends 3 to 5 lines under its topic here: what the item teaches and where to look
(a new topic heading if none fits). A new surprise goes into the timeline with its date, how it
was caught and the lesson. Keep it in plain words, counts only, and keep every link and command
working.

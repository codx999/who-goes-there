> Input to `docs/plan.md` (2026-10-11). Where the two differ, plan.md wins: the plan keeps 12 classes without `empty`, ResNet-18 from torchvision, a 512 px cache, and scores the test sets once at the end.

# Research report: classification task and research question for Who Goes There

Written 2026-10-11 against `docs/data-inspection.md` (counts treated as ground truth), the course
guideline, and the papers listed under Sources. Established findings are marked **[established]**;
things I expect but cannot cite are marked **[hypothesis]**. Where a source could not be verified
from its primary text, I say so.

---

## Part 6 (first): decision report

### A. Dataset and version

Caltech Camera Traps, CCT20 benchmark subset: `eccv_18_annotations.tar.gz` (3 MB, five COCO-style
JSON files) and `eccv_18_all_images_sm.tar.gz` (6.5 GB, resized to 1024 px max side), from LILA BC
([dataset page](https://lila.science/datasets/caltech-camera-traps)). Defined in Beery, Van Horn,
Perona, "Recognition in Terra Incognita", ECCV 2018 ([arXiv 1807.04975](https://arxiv.org/abs/1807.04975)).
License CDLA-Permissive 1.0. Split sizes as published and verified: train 13,553, cis_val 3,484,
trans_val 1,725, cis_test 15,827, trans_test 23,275 (57,864 total; the paper says 57,868, a
four-image discrepancy that is in the paper itself).

### B. Classification task and exact classes

Whole-image, single-label, 14-class classification: **13 species-or-object classes plus `empty`**
(option C below). Classes: opossum, raccoon, squirrel, bobcat, skunk, dog, coyote, rabbit, bird,
cat, car, rodent, empty. Dropped: badger, fox, deer (too rare; deer has zero trans_test images).
Multi-species images (78 across all splits) are dropped. Empties enter the training pool from
cis_val during the mandatory re-split by sequence.

### C. Baseline model

ImageNet-pretrained ResNet-18 (torchvision weights, or timm), full fine-tune, 224 px input,
cross-entropy, AdamW, cosine schedule, about 10 epochs, checkpoint chosen by macro F1 on the
re-split cis validation set. This is the course's chapter 06 recipe applied to a custom dataset.

### D. From-scratch comparison

Include it as a one-run warm-up (about 30 GPU minutes): a 4-block CNN at 128 px, trained with the
identical loop. It demonstrates the course skill and gives the "why pretrain" slide. It is not part
of the research question and gets one paragraph, not a chapter.

### E. Research question (one sentence, testable)

Does training the whole-image classifier with **animal-centred crop augmentation** (bounding boxes
used only at training time, never at inference) shift Grad-CAM attention onto the animal and shrink
the cis-minus-trans gap on CCT20, compared with an identical model trained with standard random
resized crops?

### F. Justification

**[established]** Beery et al. 2018 showed that a full-image Inception-v3 classifier trained on 10
camera locations loses about 22 points of top-1 accuracy on unseen locations (cis error 19.1%,
trans error 41.0%), and that classifying ground-truth box crops cuts both errors by more than half
(8.1% / 19.6%). Koh et al. 2021 made unseen-camera generalization a standard benchmark (iWildCam:
ERM macro F1 47.0 in-distribution vs 31.0 out-of-distribution). Schneider et al. 2020 reproduced
the effect on a Canadian dataset (95.6% at trained locations, 68.7% at new ones). Beery, Morris,
Yang 2019 turned this into the "detect, crop, then classify" pipeline that practitioners use.

**How this differs.** The prior work compares *whole image at test* against *crop at test*. That
comparison needs a detector at deployment. The question here is whether the free boxes can be
spent at *training time only*, as an augmentation that teaches a whole-image model where to look,
so that deployment stays a single classifier. I did not find a paper that isolates box-guided
augmentation as the single variable on CCT20 with a Grad-CAM attention measurement; closest are the
paper's own "random crops containing at least 65% of the region" (not box-aware) and the
crop-at-test pipeline. Confidence that this exact comparison is unpublished: moderate. It is at
minimum an incremental extension with a clean control.

**What the result contributes.** A measured gap on a modern small backbone, a *quantified* Grad-CAM
measure (fraction of attention mass inside the box) on cis vs trans, and a yes/no answer on whether
a zero-cost-at-inference augmentation moves either number.

### G. Experimental design

- **Split.** Pool = train ∪ cis_val (17,037 images, 10 locations). Drop multi-species and the three
  rare classes. Re-split the pool by `seq_id`, roughly 85/15, stratified by class and location.
  cis_test and trans_test stay as published, with the same class and multi-species filters.
- **Controls.** Same backbone, initialization seed, epochs, optimizer, schedule, resolution, batch
  size and evaluation transform. The only difference is the training crop sampler.
- **Metrics.** Accuracy and macro F1 on cis_test and trans_test; the gap (cis minus trans) in both;
  per-class F1; confusion matrices; attention-in-box score on the box-annotated test images.
- **Procedure.** Run A (standard augmentation) and run B (box-guided) with three seeds each.
  Report means and ranges. Checkpoint selection by cis-val macro F1 for both arms.

### H. Risks

- **Dataset.** Empties are absent from train and must come from cis_val (995 images); the empty
  class is background-only by definition, so it is the most location-specific class. Train and
  cis_val share 224 sequences, contradicting the paper's statement; the re-split fixes it.
  trans_val is one camera and cannot select models.
- **Implementation.** Decoding 1024 px JPEGs will bottleneck a Kaggle T4; pre-resize once to a
  shorter side of 320 px and store as a Kaggle dataset. Grad-CAM libraries can silently use the
  wrong layer. The box-guided sampler must not leak at test time.
- **Experimental.** The intervention may do nothing (the paper's data-scaling result suggests the
  gap is stubborn). Seed variance may be as large as the effect; three seeds per arm is the
  minimum. Grad-CAM is a heuristic, not ground truth for "attention".

### I. Scope

- **Must do.** Split code with leakage tests; ResNet-18 baseline; A vs B with three seeds; cis and
  trans scoring; gap, per-class F1, confusion matrices; Grad-CAM gallery plus attention-in-box
  score; Gradio app with inference timer on Hugging Face Spaces.
- **Leave out.** Object detection; MegaDetector; the full 243k-image release; sequence-level
  aggregation; ResNet-50 or larger; any second intervention.
- **Add if early.** Temperature scaling fit on cis val with expected calibration error (ECE)
  reported on cis vs trans (zero GPU cost); a day-versus-night slice of the same test predictions.

### J. Three capstone titles

1. "Who Goes There? Measuring the Seen-to-Unseen Camera Gap in Trail-Camera Species Classification"
2. "Does the Model Look at the Animal? Box-Guided Augmentation and Grad-CAM on Caltech Camera Traps"
3. "Camera-Trap Species Classification in PyTorch: Generalization to New Locations on CCT20"

---

## Part 1: dataset facts

### Confirmation against the paper, and contradictions

Checked against the ar5iv rendering of arXiv 1807.04975v2.

| Fact from data-inspection.md | Paper | Status |
| --- | --- | --- |
| 16 categories (15 named + empty) | "15 classes, plus an empty label" | Agrees |
| Split sizes 13,553 / 3,484 / 1,725 / 15,827 / 23,275 | Same numbers in Sec. 3.3 | Agrees |
| 10 cis + 10 trans locations, no overlap | Train/cis-val/cis-test on 10 locations; trans-val 1 location, trans-test 9 | Agrees (1 + 9 = 10) |
| Train has zero empties | Paper's full-image experiment "includes empty images as a 16th class" (20.83% cis / 41.08% trans error) | **Contradiction**: the released train split cannot have produced that experiment as described. Either the authors trained on a different file or empties were stripped before release. Unexplained; cite the paper's number with this caveat. |
| Train and cis_val share 224 sequences | "training and validation do not share image sequences" | **Contradiction** in the released files. The re-split by seq_id is mandatory. |
| trans_test has 9 locations | trans test = 9 held-out locations | Agrees |

The paper also states cis_val is 5% of even-day data and cis_test is odd days from the same
cameras. So cis_test is a *time* hold-out at seen cameras, which is what makes the gap a location
effect and not a time effect.

### How the original authors treated empties and boxes

**[established]** They ran two classification settings (Sec. 4.1, Table 1): full images with
Inception-v3 at 299 px, and ground-truth bounding-box crops. For Table 1 empties were removed; a
text variant with empties as a 16th class is reported separately. Box classification excludes
empties by construction. Boxes were crowd-sourced on Mechanical Turk with 3 to 10 annotators each.
Results: full image 19.06% cis / 41.04% trans error; crops 8.14% / 19.56%. Error roughly doubles
from cis to trans in both settings, so cropping does not remove the gap, it removes half the error.

**What this changes for a whole-image classifier.** Expect the full-image numbers, not the crop
numbers: roughly 80% cis and 60% trans top-1 with a strong backbone at 299 px. A ResNet-18 at
224 px will likely sit a few points lower **[hypothesis]**. The boxes are still free supervision:
1,454 of 14,071 train annotations lack a box, so about 90% of training animals have one.

### CCT20 versus the full Caltech Camera Traps release

The LILA page lists the full set as 243,100 images from 140 locations, 21 categories plus empty,
about 66,000 boxes, and about 70% empty frames (the iWildCam 2018 paper describes a 292,732-image,
143-location version; the two numbers are different releases). CCT20 is 20 of those locations with
the 16-label vocabulary and the 1024 px resize. Nothing in the full release is worth pulling in for
this project: the empties problem is solvable from cis_val; adding full-release images from the 10
cis locations would require matching location ids across annotation files and risks pulling in
trans-location images; and the 105 GB tarball does not fit the Kaggle workflow.

### Day versus night

The COCO Camera Traps format carries a `datetime` string with time of day (example in the format
spec: "2022-12-31 09:52:50"); the CCT files use the older `date_captured` key. The inspection did
not record whether the time component is populated, so **verify it in the JSON first**. Even if
present, converting clock time to day/night needs sunrise tables for coded locations. The robust
route is pixel-based: infrared night frames are grayscale, so a per-image test that the three
channels are near-identical (mean absolute R minus B below a small threshold) labels IR frames
directly. Use the pixel test as the label and the timestamp as a cross-check.

---

## Part 2: choosing the classification task

The paper's full-image setting is the reference. At 1024 px some animals are tens of pixels; at
224 px they can be a few pixels, which is why crops help so much. A whole-image classifier has
three options: train at higher resolution (320 to 384 px roughly doubles cost per epoch on a T4 and
is the first thing to try if early), use crop augmentation that zooms toward the animal during
training (the intervention chosen here), or accept the loss. Recommendation: 224 px for every
headline run so the control is clean, and treat resolution as a stated limitation.

| Option | Predicts | Label support | Main risk | Difficulty | Evaluation | Compute |
| --- | --- | --- | --- | --- | --- | --- |
| A. Animal vs empty | Presence | 995 cis_val empties vs 16k animals; 1,180 / 1,778 empties in tests | Trivially easy on cis, pure background learning on trans; thin research story; empties only 7% of test, so accuracy is uninformative | Low | Balanced accuracy, precision/recall on empty | Fits easily |
| B. 13 species, empties dropped | Species | All train labels; rare three dropped | Demo cannot say "nothing here"; Gradio app guesses a species on every blank frame | Low to medium | Acc, macro F1, cis vs trans | Fits |
| **C. 13 species + empty (recommended)** | Species or empty | As B plus ~800 empties moved into train via the re-split | Empty is background-defined, so it may dominate the trans gap; report species-only metrics alongside | Medium | Acc, macro F1 (14-class and 13-class species-only), cis vs trans | Fits |
| D. Hierarchical | Presence, then species | Same as C | Two models to train, deploy and explain; doubles the surface for bugs in two weeks; evaluation needs error propagation | High | Stage-wise plus end-to-end | Doubles runs |
| E. Detection | Boxes | 90% of animals boxed | Out of course scope (not a classifier), heavy compute | High | mAP | Does not fit |

C is recommended because it matches the deployment story (a trail-camera triage tool must be able
to say "empty"), it reproduces the paper's own 16-class full-image setting, and the extra cost over
B is one more class. The known weakness, that `empty` is the most background-tied class, becomes a
feature of the analysis: report the 13-class species-only gap next to the 14-class gap so readers
can see how much of the gap `empty` carries.

---

## Part 3: literature review

Numbers below were read from the primary texts (ar5iv or PMC renderings) unless noted.

**1. Beery, Van Horn, Perona (2018). "Recognition in Terra Incognita." ECCV.**
[arXiv 1807.04975](https://arxiv.org/abs/1807.04975). Uses CCT: yes, defines CCT20.
Question: do detectors and classifiers generalize to unseen camera locations? Method: Inception-v3
(299 px) for classification on full images and on ground-truth crops; Faster R-CNN for detection;
cis/trans location split. Finding: full-image top-1 error 19.06% cis vs 41.04% trans; crops 8.14%
vs 19.56%; detection mAP@0.5 77.1 vs 70.2 (ResNet-101 backbone). Data scaling (Fig. 5): reaching
the cis error level at trans locations needs about 20× the images per class; using 25/50/100% of
training images per location gave 80.6/83.0/83.4% trans accuracy on crops, so more data per
location does not close the gap. Stated limitations: one region, one species set, sequence-level
"oracle" results are optimistic, human frame-level performance unknown. **Already established, do
not repeat as if new:** the gap itself, that crops help, and that more cis data does not fix trans.

**2. Koh et al. (2021). "WILDS: A Benchmark of in-the-Wild Distribution Shifts." ICML.**
[arXiv 2012.07421](https://arxiv.org/abs/2012.07421). Uses CCT: no (iWildCam 2020 v2, 203,029 images,
323 cameras, 182 classes including empty). Question: do methods designed for distribution shift
beat plain training on real shifts? Method: ResNet-50 at 448 px, ERM vs CORAL, IRM, Group DRO;
splits by camera (train 243 cameras, OOD val 32, OOD test 48; ID val/test from training cameras on
different days). Finding: ERM macro F1 47.0 ID vs 31.0 OOD; CORAL 32.8, IRM 15.1, Group DRO 23.9 on
OOD, so none beat ERM meaningfully. Limitations stated: ID hyperparameters tuned on OOD val, so ID
to OOD gaps slightly underestimated. **Established:** unseen-camera shift is a standard benchmark,
macro F1 is the accepted metric, and off-the-shelf robustness methods do not help.

**3. Schneider, Greenberg, Taylor, Kremer (2020). "Three critical factors affecting automated image
species recognition performance for camera traps." Ecology and Evolution 10(7).**
[DOI 10.1002/ece3.6147](https://doi.org/10.1002/ece3.6147), [PMC7141055](https://pmc.ncbi.nlm.nih.gov/articles/PMC7141055/).
Uses CCT: no (47,279 Parks Canada images, 36 locations, 55 classes, confidential). Question: what
drives accuracy for practical-scale projects? Method: six ImageNet-pretrained CNNs, transfer
learning, evaluated at trained and untrained locations. Finding: DenseNet201 95.6% top-1 at trained
locations vs 68.7% at untrained; classes with 1,000+ training images reach mean recall 0.971,
under 500 images 0.750. Limitations: one dataset, unreliable recall for tiny test classes.
**Established:** the gap replicates outside CCT; ~1,000 images per class is the practical floor,
which is why badger, fox and deer are dropped here.

**4. Beery, Morris, Yang (2019). "Efficient Pipeline for Camera Trap Image Review."**
[arXiv 1907.06772](https://arxiv.org/abs/1907.06772). Uses CCT: not in the text (Idaho Fish and
Game data, 4.8M images). Question: how to make classifiers transfer to new regions? Method: a
generic animal detector (what became MegaDetector) followed by a classifier on crops. Finding:
detector average precision 0.885 to 0.988 across regions; about 80% of images removed from manual
review; classifier results described only as preliminary. Limitations stated: false positives on
branches and rocks, mitigated by removing static detections. **Established:** detect-then-classify
is the deployed answer to the gap. This project deliberately does not use a detector; say so.

**5. Norouzzadeh et al. (2018). "Automatically identifying, counting, and describing wild animals
in camera-trap images with deep learning." PNAS 115(25).**
[arXiv 1703.05830](https://arxiv.org/abs/1703.05830), [DOI 10.1073/pnas.1719367115](https://doi.org/10.1073/pnas.1719367115).
Uses CCT: no (Snapshot Serengeti, 3.2M images, 48 species). Method: nine CNNs incl. ResNet-18 to
-152, trained from scratch; split by capture event. Finding: empty vs animal 96.6% (ensemble);
species top-1 94.9% ensemble, 93.8% best single (ResNet-152), top-5 99.1%; matched volunteers on
99.3% of data. Limitations: rare species near 0% before rebalancing, noisy event-level labels,
ImageNet transfer did not help at that scale, no new-location test. **Established:** the
high-accuracy same-camera result that the gap literature is a corrective to.

**6. Beery, Liu, Morris, Piavis, Kapoor, Meister, Joshi, Perona (2020). "Synthetic Examples Improve
Generalization for Rare Classes." WACV.** [arXiv 1904.05916](https://arxiv.org/abs/1904.05916).
Uses CCT: yes, CCT20 plus 16k extra trans images. Question: can rendered animals fix a rare class?
Method: Inception-ResNet-v2 on crops; 44 real deer in train; synthetic deer pasted on empties.
Finding: baseline deer error 29.3% cis vs 77.7% trans+; best synthetic setup cut trans deer error
54%; other classes 10.2% cis vs 31.1% trans+. Limitations: one species, synthetic data biases the
classifier at high volumes. **Established:** deer is unlearnable from CCT20 alone (44 examples),
which supports dropping it, and the rare-class route on this dataset needs external data.

**7. Dussert, Chamaillé-Jammes, Dray, Miele (2024). "Being confident in confidence scores:
calibration in deep learning models for camera trap image sequences." Remote Sensing in Ecology and
Conservation.** [bioRxiv 10.1101/2023.11.10.566512](https://www.biorxiv.org/content/10.1101/2023.11.10.566512v1),
journal DOI 10.1002/rse2.412 (DOI not opened; from the lab listing). Uses CCT: no (European data).
Finding: calibration tracks accuracy across five architectures; averaging logits over a sequence
before softmax is best; temperature scaling helps and the optimum temperature transferred across
their datasets. I could not confirm from the abstract whether their test sets are unseen locations.
ECE numbers were not read. **Established:** temperature scaling is the standard camera-trap fix;
whether it survives a location shift on CCT20 is open.

**8. Geirhos et al. (2020). "Shortcut Learning in Deep Neural Networks." Nature Machine Intelligence.**
[arXiv 2004.07780](https://arxiv.org/abs/2004.07780). Uses CCT: no. A perspective that names the
phenomenon this project measures: decision rules that work on the benchmark and fail under shift.
**Established:** the vocabulary ("shortcut", "background cue") and the recommendation to test
out-of-distribution. Cite it for framing, not for numbers.

Also consulted but not listed: Tabak et al. 2019 (ResNet-18, 3.4M US images, 98% in-sample vs 82%
on Canadian out-of-sample data; abstract-level only), Guo et al. 2017 on temperature scaling
([arXiv 1706.04599](https://arxiv.org/abs/1706.04599)), and the iWildCam 2018 dataset paper.

---

## Part 4: the research question

### Assessment of the working plan

A careful replication with a visual explanation is a legitimate capstone framing for a bootcamp:
the dataset paper's gap is a real, well-cited result, and reproducing it with a different backbone,
resolution and split hygiene is informative. Its weakness is that "the gap exists, here is a
heatmap" is a description, and Grad-CAM galleries are easy to cherry-pick.

Two changes make it more than a replication without adding a second project:

1. **Quantify the Grad-CAM claim.** For every box-annotated test image, compute the fraction of
   normalized Grad-CAM mass that falls inside the ground-truth box (attention-in-box). Report its
   distribution on cis vs trans and for correct vs wrong predictions. This turns "the model looks
   at the background" into a number with a confidence interval. The box is used only to score
   attention, never as model input.
2. **Make the intervention the hypothesis.** Box-guided crop augmentation is the single variable.
   It uses the same free labels, costs nothing at inference, and directly targets the mechanism the
   heatmaps are supposed to show.

The combined question is the one in Part 6E. The failure case is still a result: if attention
moves inside the box but the trans gap does not shrink, that is evidence that the gap is not just
"looking at the wrong place" but also appearance shift of the animals themselves (illumination,
camera type, species mix), which is what Koh et al. found for robustness methods.

### Ranking of alternatives

| Rank | Question | Prior work | Still uncertain | Closest study | Type | Needs | Days | If the hypothesis fails |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Working plan with attention-in-box score and box-guided augmentation | Gap and crop benefit (Beery 2018) | Whether training-time-only box use helps a whole-image model; whether attention correlates with correctness | Beery 2018 Table 1 | Incremental extension with a clean control | Boxes, location split | 4 to 5 | Attention shifts but gap stays: the gap is not only background. Reportable. |
| 2 | Calibration under location shift | Temperature scaling (Guo 2017); camera-trap calibration (Dussert 2024) | Whether a temperature fit on cis val transfers to trans | Dussert 2024 | Useful replication on new data; cheap | Softmax outputs only; no GPU | 1 to 2 | Overconfidence persists on trans: useful warning for the demo's confidence display. |
| 3 | Rare vs common species; class-balanced loss or sampling on trans | Rare classes suffer (Norouzzadeh 2018; Schneider 2020); synthetic data for deer (Beery 2020) | Whether reweighting moves trans macro F1 at this scale | Beery 2020 | Incremental; noisy at CCT20 scale (rare here means skunk 229, rodent 375, bird 484) | Class counts | 2 to 3 | No change in trans macro F1: consistent with WILDS, where reweighting did not help. |
| 4 | Day vs night accuracy per class | Night listed as a nuisance factor (Beery 2018; Norouzzadeh 2018); no per-class CCT numbers found | Size of the night penalty and which classes carry it | None found with a clean table (search confidence: moderate) | New as a slice on CCT20, but descriptive, not an intervention | Day/night label (pixel test) | 1 to 2 | No night penalty: interesting, but there is no hypothesis to reject. |
| 5 | Learning curve vs trans accuracy | Done in the paper (Sec. 7.1, Fig. 5): more cis data barely helps trans | Whether the same holds for whole images and a small backbone | Beery 2018 | Replication | None | 1 (three short extra runs) | Curve flat on trans: exactly the paper's finding. |
| 6 | Smaller or faster backbone at near-equal accuracy | Standard engineering (MobileNet, EfficientNet) | Nothing of research interest | Beery 2019 pipeline | Engineering, not a research question | None | 1 to 2 | Fits the course's inference-timer requirement, nothing more. |

Rank 2 is the natural "add if early" because it needs no GPU time: fit one temperature on the cis
validation logits and report ECE and a reliability diagram on both test sets.

**Confidence in the literature search.** High for the five required papers (primary texts read).
Moderate for the claim that box-guided augmentation as a single controlled variable is unpublished
on CCT20; I searched for it and found only the crop-at-test comparison and the 65% random-crop
augmentation, but I may have missed a workshop paper. Moderate to low for day/night per-class
results: I found studies that mention night as harder, none with a clean per-class table on CCT.

---

## Part 5: experimental design

**1. Objective and classes.** 14-way softmax over opossum, raccoon, squirrel, bobcat, skunk, dog,
coyote, rabbit, bird, cat, car, rodent, empty, with an explicit id-to-index map (category ids 1, 3,
5, 6, 7, 8, 9, 10, 11, 16, 33, 99, 30, 21/34/51 dropped). Report species-only metrics as a
secondary view by masking `empty` rows and columns.

**2. Empties and rare classes.** Empties: the 995 cis_val empties join the pool and are re-split
with everything else, giving roughly 800 train and 200 val empties across 10 locations; test
empties (1,180 cis, 1,778 trans) stay as published. Rare classes: drop badger, fox, deer from every
split (removes 60 pool images, 168 cis_test, 15 trans_test). Do not merge into "other": an "other"
class with 60 examples is unlearnable and muddies the confusion matrix.

**3. From-scratch CNN.** Yes, one run, small: four conv blocks (32 to 256 channels), batch norm,
global average pooling, at 128 px, about 1M parameters, 10 epochs. It exists to show the gap from
pretraining and to satisfy the "architecture you can justify" rubric. It is not seeded three times.

**4. Backbone.** ResNet-18, ImageNet-1k weights, replace the final linear layer, fine-tune all
layers with a lower learning rate on the body than on the head, label smoothing 0.1, AdamW, cosine
decay, mixed precision. It trains in well under an hour on a T4 at 224 px once images are
pre-resized. A stronger option that still fits (ConvNeXt-Tiny or ResNet-50) costs 2 to 3× per run
and would consume the seed budget; use ResNet-18 for all headline runs and, only if early, one
ResNet-50 run as a sanity check on whether the gap is backbone-dependent.

**5. Split.** Pool = train ∪ cis_val after filters. Group by `seq_id`, assign groups to train/val
with stratification on the majority class of the sequence and on location. Invariants, each as a
test: no `seq_id` in two splits; every location appears in both train and val; every class has at
least 30 val images; no image id in two splits; cis_test and trans_test untouched except filters;
zero location overlap between any pool split and trans_test. Record the split as a committed CSV
of image ids with a hash, so the run ledger can reference it.

**6. Metrics.** Accuracy and macro F1 on cis_test and trans_test, the gap in both, per-class F1,
two confusion matrices, attention-in-box score. Checkpoint selection: macro F1 on the re-split cis
val set, evaluated each epoch. trans_val is reported once as a sanity check and never used for
selection, because it is one camera and lacks four classes.

**7. The research experiment.** Two arms, identical except the training crop sampler:

- Arm A (control): random resized crop at 224 px with scale 0.65 to 1.0 and horizontal flip, which
  mirrors the paper's augmentation; color jitter kept mild because night frames are grayscale.
- Arm B (treatment): with probability 0.5 per image that has a box, the crop is sampled so that it
  contains the box and the box occupies between 20% and 100% of the crop area, with random
  jitter of the crop position; otherwise the Arm A sampler is used. Images without a box (all
  empties, 10% of animals) always use the Arm A sampler. At evaluation both arms see the same
  resize-and-center-crop whole image.
- Three seeds per arm; report mean and min/max. Comparison: trans macro F1 and the gap, A vs B;
  attention-in-box distribution A vs B on cis and trans; a paired view (same test images) of where
  B fixes or breaks A's predictions. A result counts as positive if B's trans macro F1 exceeds A's
  across all three seed pairings and the attention-in-box median rises; mixed outcomes are
  reported as mixed.

**8. Day-by-day plan.** GPU estimate about 9 hours total, within one week's quota.

| Date | Work | GPU |
| --- | --- | --- |
| 10-12 Sun | Metadata parser, class map, split code with the invariants as tests, split CSV | 0 |
| 10-13 Mon | Kaggle: download tarball, pre-resize to 320 px shorter side, save as Kaggle dataset; Dataset class; dry run on fixture | 0 (CPU) |
| 10-14 Tue | Training loop, eval, checkpointing; from-scratch CNN run; Arm A seed 0 | 1.5 h |
| 10-15 Wed | Score Arm A on cis/trans test; confusion matrices; fix bugs; implement and unit-test the box-guided sampler | 0 |
| 10-16 Thu | Arm B seed 0; Grad-CAM and attention-in-box on a 500-image sample | 1 h |
| 10-17 Fri | Arm A and B seeds 1 and 2 in one session | 4 h |
| 10-18 Sat | Aggregate results, figures (gap bars, per-class F1, attention histograms, heatmap gallery); optional temperature scaling | 0 |
| 10-19 Sun | Gradio app locally: top-3, confidence, heatmap, inference timer; export weights | 0 |
| 10-20 Mon | Hugging Face Spaces deploy; test with held-out trans images; requirements.txt | 0 |
| 10-21 Tue | Final scoring entries in the run ledger; README and data card; buffer for a rerun | 0 to 2 h |
| 10-22 Wed | Slides draft: problem, demo, gap, A vs B, two errors, limitations | 0 |
| 10-23 Thu | Code freeze; rehearsal; recorded backup video of the demo | 0 |
| 10-24 Fri, 10-25 Sat | Slides polish, rehearsals, buffer | 0 |
| 10-26 Sun | Presentation | 0 |

---

## Sources

- Beery, Van Horn, Perona 2018, Recognition in Terra Incognita: https://arxiv.org/abs/1807.04975 (full text read at https://ar5iv.labs.arxiv.org/html/1807.04975)
- Caltech Camera Traps on LILA BC: https://lila.science/datasets/caltech-camera-traps
- COCO Camera Traps format: https://github.com/agentmorris/MegaDetector/blob/main/megadetector/data_management/README.md
- Koh et al. 2021, WILDS: https://arxiv.org/abs/2012.07421; iWildCam config: https://github.com/p-lambda/wilds/blob/main/examples/configs/datasets.py; dataset page: https://wilds.stanford.edu/datasets/
- Schneider et al. 2020, Three critical factors: https://doi.org/10.1002/ece3.6147 (https://pmc.ncbi.nlm.nih.gov/articles/PMC7141055/)
- Beery, Morris, Yang 2019, Efficient Pipeline for Camera Trap Image Review: https://arxiv.org/abs/1907.06772
- Norouzzadeh et al. 2018, PNAS: https://doi.org/10.1073/pnas.1719367115 (preprint https://arxiv.org/abs/1703.05830)
- Beery et al. 2020, Synthetic Examples Improve Generalization for Rare Classes: https://arxiv.org/abs/1904.05916
- Dussert et al. 2024, calibration for camera trap sequences: https://www.biorxiv.org/content/10.1101/2023.11.10.566512v1 (journal: Remote Sensing in Ecology and Conservation, DOI 10.1002/rse2.412)
- Geirhos et al. 2020, Shortcut Learning in Deep Neural Networks: https://arxiv.org/abs/2004.07780
- Guo et al. 2017, On Calibration of Modern Neural Networks: https://arxiv.org/abs/1706.04599
- Tabak et al. 2019, Machine learning to classify animal species in camera trap images (preprint): https://www.biorxiv.org/content/10.1101/346809v1
- Beery et al. 2019, The iWildCam 2018 Challenge Dataset: https://arxiv.org/abs/1904.05986
- Selvaraju et al. 2017, Grad-CAM: https://arxiv.org/abs/1610.02391
- Kaggle notebook documentation (GPU quota; the page did not render for me, quota taken from the brief): https://www.kaggle.com/docs/notebooks

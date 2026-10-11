# Research prompt: CCT20 literature review and capstone decision

Written 2026-10-11. Paste into a fresh session (or launch it from the chip). The session writes its
answer to `docs/research-report.md`.

---

Act as an AI/ML research advisor and computer vision expert. I am a bootcamp student building my
capstone in PyTorch. Help me make an informed decision about the classification task and the one
research experiment for this project. Do not write implementation code.

## Constraints that decide scope

- Presentation is 2026-10-26. Code must be done by about 2026-10-23.
- The course requires an image **classifier** (not a regressor), trained in PyTorch, with a Gradio
  app that shows the predicted label and inference time, deployed to Hugging Face Spaces with the
  weights file and a `requirements.txt`.
- The course prefers a `torchvision.datasets` dataset. CCT20 is not one. I am asking the TA for an
  exception; the fallback is Food-101. Assume CCT20 for this report.
- Compute: Kaggle notebooks, one T4 or P100, 30 GPU hours a week. A 10-epoch run of a ResNet-class
  backbone at 224 px on about 17k images takes roughly an hour. Budget for 5 to 10 full runs.
- Code lives in a tested Python package in a git repo; the notebook is a thin wrapper.
- I was taught to build an image classifier with PyTorch and I want to demonstrate that. I am
  interested in comparing a small CNN trained from scratch with a pretrained ResNet-18, but I
  expect the from-scratch model to lose badly on this data and I know that result is well known.
  Treat that comparison as an optional warm-up chapter, not as the research question.

## Part 1: dataset facts, already verified from primary data

Read `docs/data-inspection.md` in this repo before anything else. It was generated on 2026-10-10
from the official annotation files (`eccv_18_annotations.tar.gz`, CCT20 benchmark, from LILA).
Treat its counts as ground truth. Do not re-derive them from secondary sources. Confirm them
against the original paper (Beery, Van Horn, Perona, "Recognition in Terra Incognita", ECCV 2018)
and flag any contradiction explicitly.

The facts that shape the task choice:

1. 16 classes, 15 species-or-object labels plus `empty`. Class ids are not contiguous.
2. Official split: train 13,553, cis_val 3,484, trans_val 1,725, cis_test 15,827, trans_test
   23,275 images. 10 cis locations, 10 trans locations, no overlap.
3. **Train contains zero `empty` images.** Empties appear only in val and test sets. Any task that
   predicts "empty" must move empties from cis_val into train or drop the class.
4. **Train and cis_val share 224 sequences (photo bursts).** They must be re-split by `seq_id`.
5. badger (9 train), fox (5 train) and deer (42 train, 0 trans_test) are too rare to learn.
6. trans_val is one camera, 1,725 images, 10 classes. It cannot drive model selection.
7. Heavy imbalance: opossum is 20% of train; opossum plus raccoon are 47% of trans_test.
8. Every image has `location`, `seq_id`, `frame_num`, `date_captured`, and most annotations have a
   bounding box. Night images are infrared grayscale. Images are resized to 1024 px max.
9. License CDLA-Permissive 1.0; the data is never committed to the repo.

What I still want from you on the dataset:

- Confirm from the paper how the original authors treated empties and bounding boxes (they
  classified crops, I believe; I will classify whole images). What does that change?
- Explain the difference between CCT20 and the full Caltech Camera Traps release, and whether
  anything in the full release (more locations, more empties) is worth pulling in.
- Say whether day/night can be derived from metadata or must be inferred from the image.

## Part 2: choose the classification task

Options, given fact 3 above:

- A. Binary animal-versus-empty (requires moving empties into train).
- B. Multiclass species, empties dropped (15 classes, minus the three rare ones).
- C. Multiclass species plus `empty` (requires moving empties into train).
- D. Hierarchical: animal-present, then species.
- E. Something better.

For each: what it predicts, how the labels support it, the main technical risk, difficulty for a
bootcamp student, how to evaluate it, and whether it fits the compute budget. Recommend one.
Do not recommend object detection. Address the fact that at 224 px some labelled animals are a few
pixels wide, which is why the original paper used crops, and say what a whole-image classifier
should do about that (higher resolution, crop augmentation, accept the loss).

## Part 3: literature review, focused

Cover the five to eight most relevant papers, no more. These must be included:

- Beery, Van Horn, Perona 2018, "Recognition in Terra Incognita" (the dataset paper, cis vs trans).
- Koh et al. 2021, WILDS benchmark, specifically the iWildCam task (generalization to unseen
  cameras as a standard benchmark).
- Schneider et al. 2020 on factors affecting automated species recognition in camera traps
  (new-location generalization).
- Beery et al. 2019 on the MegaDetector / efficient pipeline approach (detect first, then classify).
- Norouzzadeh et al. 2018 (Snapshot Serengeti) as the classic deep-learning camera-trap result.

Add up to three more only if they directly address: background shortcut learning on camera traps,
day versus night performance, or confidence calibration on hard images.

For each paper: title, authors, year, a working link (arXiv or DOI), whether it uses CCT
specifically, the question asked, the method, the main finding with the number, the limitations
the authors state, and what it has already established that I should not repeat without saying so.

Do not invent citations or numbers. If you cannot verify a claim, say so. Prefer the original
paper over blog summaries.

## Part 4: the research question

My working plan is to measure the gap between cis-test and trans-test accuracy, show with Grad-CAM
whether the model attends to the animal or the background, then try one intervention (stronger
augmentation, crop-around-box training, or a different backbone) and report whether the gap
shrinks. I know this is a replication of the dataset paper's central finding at bootcamp scale.

Assess that plan honestly: is a careful replication with a visual explanation a good capstone
framing? What would make it more than a replication without adding a second project? Then consider
these alternatives and rank all of them against the working plan:

- Day versus night accuracy, per class.
- Rare versus common species, and whether class-balanced loss or sampling helps on trans.
- Training-set size versus trans accuracy (learning curve across 10%, 25%, 50%, 100%).
- Confidence calibration: is the model overconfident on trans images, and does a simple fix help?
- A smaller or faster backbone at near-equal accuracy.

For each: what prior work established, what is still uncertain, the closest existing study,
whether it is new, a useful replication, or an incremental extension, how to run it, which labels
or metadata it needs, implementation difficulty in days, and what a meaningful result looks like
if the hypothesis fails. State your confidence in the literature search.

## Part 5: experimental design

Recommend:

1. The final classification objective and the exact class list.
2. How to handle empties and the three rare classes.
3. Whether to train the from-scratch CNN at all, and if so, how small it should be.
4. The pretrained backbone (ResNet-18 or a stronger option that still trains in an hour) and how
   to adapt it.
5. The data split: re-split train plus cis_val by `seq_id`, keep cis_test and trans_test as
   published. Say what else must hold (no burst in two splits, class coverage).
6. Metrics: accuracy and macro F1 on both cis_test and trans_test, confusion matrix, and the
   cis-minus-trans gap as the headline. Say which metric picks the checkpoint.
7. The one research experiment, with controlled variables and the comparison procedure.
8. A day-by-day plan from 2026-10-12 to 2026-10-26 that fits the compute budget.

## Part 6: decision report (put this FIRST in the output)

A. Dataset and version, with official sources.
B. Classification task and exact classes.
C. Baseline model.
D. Whether to include the from-scratch comparison.
E. The research question, one sentence, testable.
F. Justification: what prior work established, how this differs, what the result contributes.
G. Experimental design: split, controls, metrics, procedure.
H. Risks: dataset, implementation, experimental.
I. Scope: must do, leave out, add if early.
J. Three accurate, unexaggerated capstone titles.

## Output

- Write the report to `docs/research-report.md` as Markdown. Decision report first, then the
  evidence sections. About 3,000 words. End with a Sources list of links.
- Separate established findings from hypotheses. Be explicit about uncertainty.
- One coherent project: one classification task, one research question.
- Do not modify any other file. Do not write implementation code.

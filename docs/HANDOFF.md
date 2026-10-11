# Who Goes There: handoff from the planning session

> Retired 2026-10-10: the decisions here were folded into `docs/plan.md`, which is the source of truth. Kept for history.

Written 2026-10-10 in the Animal Spirits repo, where the project was chosen. The first session in
this repo reads this file and `docs/setup-playbook.md`, sets up the repo (playbook checklist), then
turns this file into `docs/plan.md`. Facts marked "verify" were not confirmed from primary data.

## The assignment

- Bootcamp capstone. The TA prefers an image classifier and requires a neural network.
- Presentation: 2026-10-26 (16 days from this handoff). The repo is public on GitHub.
- Animal Spirits (the finance-comment classifier) stays a private side project; nothing from it is
  reused except the setup and review machinery.

## Decisions

| Topic | Decision |
| --- | --- |
| Name | Who Goes There; folder and repo `who-goes-there`; package `trailcam` |
| Framework | PyTorch: `torch`, `torchvision`, `timm`; `gradio` and `grad-cam` for the demo |
| Dataset | Caltech Camera Traps, CCT20 benchmark subset (from LILA) |
| Task | One label per photo: the animal species, or empty |
| Key result | Accuracy on seen camera locations (cis) vs unseen locations (trans) |
| Compute | Kaggle notebook (free GPU) downloads the data; the Mac runs code and tests only |
| Workflow | Local Claude Code sessions, one item per session, per the setup playbook. Not Claude Code Projects (beta; parallel cloud threads, auto-fix on PRs, faster usage) |
| Models | Implement on Opus 5.5 at high (xhigh for split code and final scoring); reviews started by the user |

## Why CCT20

- Hard in realistic ways: infrared night shots, tiny or partly hidden animals, motion blur,
  heavy class imbalance, many empty frames.
- Each camera sees a fixed background, so a model can learn "this background means coyote".
  The official location split exposes that. This is the same leakage story as Animal Spirits'
  split by video and author, and it is the project's main point.
- License CDLA-Permissive: fine for a public repo with attribution (data never committed).
- Not part of ImageNet, so an ImageNet-pretrained backbone has not seen the test images.

Alternatives considered: HAM10000 (runner-up; skin lesions, split by `lesion_id`, CC BY-NC),
Oxford-IIIT Pet, Food-101, DeepWeeds, MedMNIST+, RealWaste. Rejected as too easy: MNIST, CIFAR-10,
SVHN, GTSRB, EuroSAT. Avoid Stanford Dogs and Imagewoof: their images come from ImageNet.

## Dataset facts

- 57,868 images from 20 camera locations in the American Southwest (USGS and NPS data).
- 15 animal classes plus empty. From memory, verify against the metadata: coyote, bobcat, deer,
  raccoon, opossum, skunk, rabbit, squirrel, rodent, fox, badger, bird, cat, dog, car.
  Papers give different class counts for different subsets.
- Benchmark images about 6 GB, resized to at most 1024 px on a side; split metadata about 3 MB.
- Published split sizes: 13,553 train, 3,484 val, 15,827 cis test, 23,275 trans test. On LILA,
  "eccv_train" covers the train and cis locations and "eccv_val" the trans locations (verify how
  the metadata marks each split before writing the split code).
- Papers often drop multi-species images; decide and record the rule.
- Cameras shoot bursts: keep every sequence in one split.
- Cite: Beery, Van Horn, Perona, "Recognition in Terra Incognita", ECCV 2018.

## Plan sketch (to refine into docs/plan.md)

1. Setup per the playbook checklist; CI green.
2. Data: metadata parsing, class map, split code with leakage tests (location, sequence), data card.
3. Baselines: a small CNN trained from scratch, then a fine-tuned pretrained backbone (timm).
4. Experiments chosen on val; one final scoring on cis and trans test, logged in the run ledger.
5. Analysis: per-class metrics, confusion matrix, cis vs trans gap, Grad-CAM on right and wrong cases.
6. Demo and slides: Gradio app (top-3 with confidence, preloaded trans-test images, heatmap),
   a live sort of about 200 photos, a recorded backup video.

Slides: problem, live demo, cis vs trans result, confusion matrix with two errors, next steps.

## Open questions

- Exact class list and how empty and multi-species images are handled.
- Metric to headline (macro-F1 vs accuracy) given the imbalance.
- Input resolution and backbone size that fit Kaggle's weekly GPU hours.
- Whether the course requires anything specific in the submission (report, notebook, video).

## Sources

- torchvision datasets: https://docs.pytorch.org/vision/stable/datasets.html
- LILA wildlife catalogue: https://source.coop/agentmorris/lila-wildlife
- Recognition in Terra Incognita: https://arxiv.org/pdf/1807.04975
- iWildCam 2019: https://ar5iv.labs.arxiv.org/html/1907.07617
- Claude Code Projects: https://code.claude.com/docs/en/claude-projects
- Setup playbook (live doc): https://claude.ai/code/artifact/fdd57274-42a8-4dbd-b6d2-d5d33f6a785d

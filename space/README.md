---
title: Who Goes There
emoji: 🦝
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
suggested_hardware: cpu-basic
license: mit
pinned: false
short_description: Camera-trap species classifier on Caltech Camera Traps
---

# Who Goes There

Upload a camera-trap photo and the model names the animal: one of 11 species, or `car`. It shows
the top class with the model's confidence, the two runners-up and the seconds of the forward pass.
Tick the Grad-CAM box to see which part of the photo pushed the top class up.

The model is a ConvNeXt-Tiny (27.8 M parameters, pretrained on ImageNet-1k) fine-tuned on Caltech
Camera Traps (CCT20) with copy-paste training: for about half of the training photos with a box,
the animal's box is pasted onto an empty frame from a training camera (night onto night,
day onto day), so the model has to find the animal instead of learning each camera's background.
It is the project's phase-3 finalist, run seed 2, the seed with the best validation macro-F1.
Code: https://github.com/codx999/who-goes-there (MIT).

## Test results

The model was tested on cameras it saw in training (cis_test, 14,478 photos) and on 9 cameras it
never saw (trans_test, 21,473 photos); the gap between the two is the project's headline result.
Accuracy and macro-F1 over the classes present, 12 classes; the gap is cis minus trans. Beside it
is phase 1's ResNet-18 (box-guided crops, arm B, seed 2), which this Space served before.

| model | cis_test accuracy | cis_test macro-F1 | trans_test accuracy | trans_test macro-F1 | accuracy gap | macro-F1 gap |
| --- | --- | --- | --- | --- | --- | --- |
| ConvNeXt-Tiny with copy-paste (phase 3), seed 2 | 0.7627 | 0.7292 | 0.6540 | 0.5315 | 0.1087 | 0.1977 |
| ResNet-18 arm B (phase 1), seed 2 | 0.6986 | 0.6461 | 0.5399 | 0.4275 | 0.1586 | 0.2187 |

These are the test sets' second use. Phase 1's scores came first (2026-10-18); phase 3's recipe
was then chosen on the validation sets only, one of them a single unseen camera, so it carries a
little selection optimism there, and its test scores chose nothing: not the recipe, not the
seed. Every row is in the repo's `docs/test-ledger.md` and `docs/test-ledger-phase3.md`.

## Data and licence

The training data and the example photos are from Caltech Camera Traps, licensed under the
Community Data License Agreement, Permissive, version 1.0 (CDLA-Permissive 1.0):

> Sara Beery, Grant Van Horn, Pietro Perona. "Recognition in Terra Incognita." European Conference
> on Computer Vision (ECCV), 2018.

The code is MIT-licensed.

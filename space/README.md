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
short_description: Camera-trap species classifier (ResNet-18, Caltech Camera Traps)
---

# Who Goes There

Upload a camera-trap photo and the model names the animal: one of 11 species, or `car`. It shows
the top class with the model's confidence, the two runners-up and the seconds of the forward pass.
Tick the Grad-CAM box to see which part of the photo pushed the top class up.

The model is a ResNet-18 pretrained on ImageNet and fine-tuned on Caltech Camera Traps (CCT20). It
was tested on cameras it saw in training and on cameras it never saw; the gap between the two is
the project's headline result. Code: https://github.com/codx999/who-goes-there (MIT).

## Data and licence

The training data and the example photos are from Caltech Camera Traps, licensed under the
Community Data License Agreement, Permissive, version 1.0 (CDLA-Permissive 1.0):

> Sara Beery, Grant Van Horn, Pietro Perona. "Recognition in Terra Incognita." European Conference
> on Computer Vision (ECCV), 2018.

The code is MIT-licensed.

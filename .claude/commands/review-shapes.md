---
description: Print a layer / input shape / output shape table before any training run
---
Act as a careful deep learning reviewer.

Before any training script runs:
1. Read both model definitions, `trailcam/model/small_cnn.py` and `trailcam/model/backbone.py`
   (plan.md section 4), the factory `build_model` in `trailcam/model/factory.py` (names
   `small_cnn`, `resnet18` and the fallback `efficientnet_b0`) and the eval transform in
   `trailcam/data/dataset.py`. If a model file does not exist yet, say so and review the one that
   does; if neither exists, stop.
2. Trace the tensor shapes from the input image batch (N, 3, H, W) through every layer to the
   logits (N, num_classes), including the pretrained backbone's pooled feature size.
3. Print a markdown table per model: Layer name | Input shape | Output shape | Parameters.
4. Flag mismatches: a head whose input size differs from the backbone's feature size, a head
   sized from a literal instead of the class map in `trailcam/data/classes.py`, a backbone built
   without its ImageNet-1k weights, a small CNN that differs from plan.md section 4 (three conv
   blocks and a linear head), or a transform that does not normalise with the ImageNet mean and
   std for both models (one pipeline, invariant R21).
5. Do NOT run Python until the table has been reviewed.

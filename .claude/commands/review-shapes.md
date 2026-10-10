---
description: Print a layer / input shape / output shape table before any training run
---
Act as a careful deep learning reviewer.

Before any training script runs:
1. Read the model definition in `trailcam/model/`. If none exists yet, say so and stop.
2. Trace the tensor shapes from the input image batch (N, 3, H, W) through every layer to the
   logits (N, num_classes), including the pretrained backbone's pooled feature size.
3. Print a markdown table: Layer name | Input shape | Output shape | Parameters.
4. Flag mismatches: a head whose input size differs from the backbone's feature size, a
   num_classes that differs from the class map, or normalization that differs from the backbone's
   published mean and std.
5. Do NOT run Python until the table has been reviewed.

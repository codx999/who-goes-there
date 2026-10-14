"""Grad-CAM maps for the three models (plan.md sections 6 and 7; R26).

`gradcam(model, images)` runs one forward pass, takes the gradient of each image's top-1 logit
with respect to the target layer's output, and returns one map per image at the input size,
with the logits of the same pass. The method is Selvaraju et al. 2017: the layer's channels are
weighted by their spatially averaged gradients, summed, passed through a ReLU and upsampled
bilinearly (half-pixel centres, `align_corners=False`) to the input's height and width. The map
is not normalised here: it is non-negative, and whoever reads it decides the normalisation
(attention-in-box divides by the whole map's sum, `trailcam/evaluate/attention.py`).

The target layer is the last block whose output the head pools (`target_layer`): `layer4` of
ResNet-18 (plan.md section 4, the standard Grad-CAM subject), the third conv block of the small
CNN, and `bn2` of timm's EfficientNet-B0 (the head convolution after its batch norm and
activation). It is a small hook-based implementation rather than the `grad-cam` package, which
would bring OpenCV and scikit-learn into the runtime dependencies; the overlay figures of item
8.02 can draw the same maps, so the overlays show what attention-in-box scores.

The model only ever receives the images: no box reaches it (R23). Gradients are taken with
`torch.autograd.grad`, so no parameter's `.grad` is touched, and the hook is removed on return.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models import ResNet

from trailcam.model.small_cnn import SmallCNN


def target_layer(model: nn.Module) -> nn.Module:
    """The layer Grad-CAM reads: the last feature map before the head's pooling or flatten."""
    if isinstance(model, ResNet):
        layer: nn.Module = model.layer4
        return layer
    if isinstance(model, SmallCNN):
        return model.conv_block_3
    bn2 = getattr(model, "bn2", None)  # timm's EfficientNet: conv_head, then bn2 with its act
    if isinstance(bn2, nn.Module) and hasattr(model, "conv_head"):
        return bn2
    raise TypeError(f"no Grad-CAM target layer known for {type(model).__name__}")


def gradcam(
    model: nn.Module, images: Tensor, layer: nn.Module | None = None
) -> tuple[Tensor, Tensor]:
    """Grad-CAM maps of shape (N, H, W) for the top-1 class of each image, and the (N, classes)
    logits of the same forward pass, both detached.

    `images` is a normalised (N, 3, H, W) batch as the eval transform makes it; `layer` defaults
    to `target_layer(model)`. Call it with the model in eval mode, so batch norm uses its running
    statistics and each image's map depends on that image alone.
    """
    if images.dim() != 4:
        raise ValueError(f"expected an (N, 3, H, W) batch, got shape {tuple(images.shape)}")
    layer = target_layer(model) if layer is None else layer
    captured: list[Tensor] = []

    def keep(module: nn.Module, inputs: Any, output: Tensor) -> None:
        captured.append(output)

    handle = layer.register_forward_hook(keep)
    try:
        with torch.enable_grad():
            # A leaf that needs a gradient keeps the graph even if every parameter is frozen.
            logits: Tensor = model(images.detach().requires_grad_(True))
            if len(captured) != 1:
                raise RuntimeError(f"the target layer ran {len(captured)} times in one forward")
            activations = captured[0]
            top = logits.argmax(dim=1, keepdim=True)
            # Each image's logit depends on its own activations only (eval mode), so the
            # gradient of the sum is each image's own gradient.
            (gradients,) = torch.autograd.grad(logits.gather(1, top).sum(), activations)
    finally:
        handle.remove()
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * activations).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=tuple(images.shape[-2:]), mode="bilinear", align_corners=False)
    return cam[:, 0].detach(), logits.detach()

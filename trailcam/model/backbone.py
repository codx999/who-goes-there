"""The fine-tuned pretrained backbones (plan.md section 4).

- `build_resnet18`: torchvision's ResNet-18 with its ImageNet-1k weights, the final linear layer
  (`fc`, 512 pooled features) replaced by a fresh `num_classes`-way head. The model of record and
  the Grad-CAM subject (target layer `layer4`).
- `build_efficientnet_b0`: the named fallback if a Space prediction takes over about one second,
  timm's EfficientNet-B0 with its ImageNet-1k weights (`efficientnet_b0.ra_in1k`, 1280 pooled
  features) and a fresh head (`classifier`). timm is imported here only, so the ResNet-18 path
  never needs it.

Both load their ImageNet-1k weights by default, which downloads them once into the torch or
Hugging Face cache; `pretrained=False` builds the same architecture with random weights (the
tests, which have no network). Both pretrained weight sets expect the ImageNet mean and standard
deviation that `trailcam/data/dataset.py` normalises with (R21); `tests/test_models.py` checks it.

ImageNet contains no CCT images, so the pretraining is not leakage.
"""

from __future__ import annotations

from torch import nn
from torchvision.models import ResNet, ResNet18_Weights, resnet18

RESNET18_WEIGHTS = ResNet18_Weights.IMAGENET1K_V1
EFFICIENTNET_B0_TIMM_NAME = "efficientnet_b0.ra_in1k"


def build_resnet18(num_classes: int, *, pretrained: bool = True) -> ResNet:
    """ResNet-18 (ImageNet-1k weights unless `pretrained=False`) with a `num_classes`-way `fc`."""
    model: ResNet = resnet18(weights=RESNET18_WEIGHTS if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_efficientnet_b0(num_classes: int, *, pretrained: bool = True) -> nn.Module:
    """timm EfficientNet-B0 (ImageNet-1k weights unless `pretrained=False`), fresh head."""
    import timm

    return timm.create_model(
        EFFICIENTNET_B0_TIMM_NAME, pretrained=pretrained, num_classes=num_classes
    )

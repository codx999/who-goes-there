"""The fine-tuned pretrained backbones (plan.md section 4).

- `build_resnet18`: torchvision's ResNet-18 with its ImageNet-1k weights, the final linear layer
  (`fc`, 512 pooled features) replaced by a fresh `num_classes`-way head. The model of record and
  the Grad-CAM subject (target layer `layer4`).
- `build_efficientnet_b0`: the named fallback if a Space prediction takes over about one second,
  timm's EfficientNet-B0 with its ImageNet-1k weights (`efficientnet_b0.ra_in1k`, 1280 pooled
  features) and a fresh head (`classifier`). timm is imported here only, so the ResNet-18 path
  never needs it.
- `build_resnet50`: phase 2's larger backbone (`docs/phase2-plan.md` step 5, config `p2_r50`),
  torchvision's ResNet-50 with its first ImageNet-1k weights (`IMAGENET1K_V1`, the original
  recipe, as ResNet-18's only weights are; V2's newer recipe would add a second change), `fc`
  (2048 pooled features) replaced the same way; Grad-CAM's target layer is `layer4` as well.
- `build_convnext_tiny`: phase 3's backbone (`docs/phase3-plan.md` section 3, configs
  `p3_convnext` and `p3_convnext_cp`), timm's ConvNeXt-Tiny with its ImageNet-1k weights, pinned
  by tag (`convnext_tiny.fb_in1k`): in the locked timm (1.0.30) the bare name `convnext_tiny`
  resolves to `convnext_tiny.in12k_ft_in1k`, pretrained on ImageNet-12k first, which would change
  the pretraining data as well as the architecture. timm builds the head (`head.fc`, 768 pooled
  and layer-normed features) at `num_classes`; the head's layer norm (`head.norm`) keeps its
  pretrained weights and is not part of `head_module`, so it trains with the body. Grad-CAM's
  target layer is the last stage (`stages[-1]`, `trailcam/evaluate/gradcam.py`).

All four load their ImageNet-1k weights by default, which downloads them once into the torch or
Hugging Face cache; `pretrained=False` builds the same architecture with random weights (the
tests, which have no network). Every pretrained weight set expects the ImageNet mean and standard
deviation that `trailcam/data/dataset.py` normalises with (R21); `tests/test_models.py` checks it.

ImageNet contains no CCT images, so the pretraining is not leakage.
"""

from __future__ import annotations

from torch import nn
from torchvision.models import ResNet, ResNet18_Weights, ResNet50_Weights, resnet18, resnet50

RESNET18_WEIGHTS = ResNet18_Weights.IMAGENET1K_V1
RESNET50_WEIGHTS = ResNet50_Weights.IMAGENET1K_V1
EFFICIENTNET_B0_TIMM_NAME = "efficientnet_b0.ra_in1k"
CONVNEXT_TINY_TIMM_NAME = "convnext_tiny.fb_in1k"  # ImageNet-1k; the bare name is ImageNet-12k


def build_resnet18(num_classes: int, *, pretrained: bool = True) -> ResNet:
    """ResNet-18 (ImageNet-1k weights unless `pretrained=False`) with a `num_classes`-way `fc`."""
    model: ResNet = resnet18(weights=RESNET18_WEIGHTS if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_resnet50(num_classes: int, *, pretrained: bool = True) -> ResNet:
    """ResNet-50 (ImageNet-1k V1 weights unless `pretrained=False`), a `num_classes`-way `fc`."""
    model: ResNet = resnet50(weights=RESNET50_WEIGHTS if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_efficientnet_b0(num_classes: int, *, pretrained: bool = True) -> nn.Module:
    """timm EfficientNet-B0 (ImageNet-1k weights unless `pretrained=False`), fresh head."""
    import timm

    return timm.create_model(
        EFFICIENTNET_B0_TIMM_NAME, pretrained=pretrained, num_classes=num_classes
    )


def build_convnext_tiny(num_classes: int, *, pretrained: bool = True) -> nn.Module:
    """timm ConvNeXt-Tiny (ImageNet-1k weights unless `pretrained=False`), fresh `head.fc`."""
    import timm

    return timm.create_model(
        CONVNEXT_TINY_TIMM_NAME, pretrained=pretrained, num_classes=num_classes
    )

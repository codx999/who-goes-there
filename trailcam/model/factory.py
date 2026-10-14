"""`build_model(name)` and the body / head parameter groups (plan.md sections 4 and 5).

The head is always sized from the class map, `trailcam/data/classes.py` (R05, R30): no caller
passes a class count. `parameter_groups` splits a built model's parameters into the body and the
head, so AdamW can put the pretrained body on a ten-times smaller learning rate than the new head
(plan.md section 5) and the head-only first epoch can freeze the body. Every parameter lands in
exactly one group.
"""

from __future__ import annotations

from typing import Final, NamedTuple

from torch import nn
from torchvision.models import ResNet

from trailcam.data import classes
from trailcam.model.backbone import build_efficientnet_b0, build_resnet18
from trailcam.model.small_cnn import SmallCNN

MODEL_NAMES: Final = ("small_cnn", "resnet18", "efficientnet_b0")


def build_model(name: str, *, pretrained: bool = True) -> nn.Module:
    """The named model with a head of `NUM_CLASSES` outputs.

    `pretrained` loads the ImageNet-1k weights of the two backbones (the default for training);
    the small CNN has none and is always built from random weights.
    """
    num_classes = classes.NUM_CLASSES
    model: nn.Module  # torchvision is untyped, so ResNet is Any to mypy
    if name == "small_cnn":
        model = SmallCNN(num_classes)
    elif name == "resnet18":
        model = build_resnet18(num_classes, pretrained=pretrained)
    elif name == "efficientnet_b0":
        model = build_efficientnet_b0(num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"unknown model {name!r}; expected one of {', '.join(MODEL_NAMES)}")
    return model


class ParameterGroups(NamedTuple):
    body: list[nn.Parameter]
    head: list[nn.Parameter]


def head_module(model: nn.Module) -> nn.Module:
    """The final classification layer that `build_model` sized from the class map."""
    if isinstance(model, SmallCNN):
        return model.classifier
    if isinstance(model, ResNet):
        fc: nn.Module = model.fc
        return fc
    get_classifier = getattr(model, "get_classifier", None)  # every timm model has one
    if callable(get_classifier):
        head = get_classifier()
        if isinstance(head, nn.Module):
            return head
    raise TypeError(f"no known head for {type(model).__name__}")


def parameter_groups(model: nn.Module) -> ParameterGroups:
    """The head's parameters and every other parameter, each once, in `model.parameters()` order."""
    head_ids = {id(p) for p in head_module(model).parameters()}
    body: list[nn.Parameter] = []
    head: list[nn.Parameter] = []
    for p in model.parameters():
        (head if id(p) in head_ids else body).append(p)
    return ParameterGroups(body=body, head=head)

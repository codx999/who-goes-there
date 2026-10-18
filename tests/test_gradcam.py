"""trailcam/evaluate/gradcam.py: the Grad-CAM map attention-in-box scores (R26, R23).

The known-answer test uses a two-layer model whose Grad-CAM is worked out by hand: a 1x1
convolution with identity weights (the target layer, at the input's size, so no upsampling),
global average pooling and a linear head. For class c the gradient of the logit with respect to
channel k of the activation A is w[c, k] / (H * W) everywhere, so the map is
ReLU(sum_k w[c, k] * A_k) / (H * W). Random-weight models only; no download.
"""

import pytest
import torch
from torch import Tensor, nn

from trailcam.evaluate.gradcam import gradcam, target_layer
from trailcam.model.factory import build_model

SIDE = 16


class _Tiny(nn.Module):
    def __init__(self, head: Tensor) -> None:
        super().__init__()
        self.features = nn.Conv2d(3, 3, kernel_size=1, bias=False)
        with torch.no_grad():
            self.features.weight.copy_(torch.eye(3)[:, :, None, None])
        self.head = nn.Linear(3, head.shape[0], bias=False)
        with torch.no_grad():
            self.head.weight.copy_(head)

    def forward(self, x: Tensor) -> Tensor:
        logits: Tensor = self.head(self.features(x).mean(dim=(2, 3)))
        return logits


def test_target_layers_are_the_last_feature_maps() -> None:
    resnet = build_model("resnet18", pretrained=False)
    assert target_layer(resnet) is resnet.layer4
    resnet50 = build_model("resnet50", pretrained=False)  # phase 2's p2_r50 (item 13.02)
    assert target_layer(resnet50) is resnet50.layer4
    small = build_model("small_cnn")
    assert target_layer(small) is small.conv_block_3
    efficientnet = build_model("efficientnet_b0", pretrained=False)
    assert target_layer(efficientnet) is efficientnet.bn2
    with pytest.raises(TypeError):
        target_layer(nn.Linear(2, 2))


def test_the_map_by_hand() -> None:
    """Class 0 weighs channel 0 by 2 and channel 1 by -1, class 1 channel 1 by 0.1. The image
    has a 4 x 4 patch of 1 on channel 0 in its top-left corner and a 4 x 4 patch of 0.5 on
    channel 1 in the bottom-right one. Logits: class 0 (2 * 16 - 0.5 * 16) / 256 = 0.09375,
    class 1 0.1 * 8 / 256 = 0.003125, so the map is class 0's and lies in the top-left patch."""
    head = torch.tensor([[2.0, -1.0, 0.0], [0.0, 0.1, 0.0]])
    model = _Tiny(head).eval()
    images = torch.zeros(1, 3, SIDE, SIDE)
    images[0, 0, :4, :4] = 1.0
    images[0, 1, 12:, 12:] = 0.5
    maps, logits = gradcam(model, images, model.features)
    torch.testing.assert_close(logits[0], torch.tensor([0.09375, 0.003125]))
    expected = torch.relu(2.0 * images[0, 0] - images[0, 1]) / SIDE**2
    torch.testing.assert_close(maps[0], expected)
    assert float(maps[0, :4, :4].sum()) == pytest.approx(float(maps[0].sum()))


def test_maps_are_non_negative_at_the_input_size_with_the_same_logits() -> None:
    torch.manual_seed(0)
    model = build_model("resnet18", pretrained=False).eval()
    images = torch.randn(3, 3, 224, 224)
    maps, logits = gradcam(model, images)
    assert maps.shape == (3, 224, 224)
    assert bool((maps >= 0).all())
    with torch.no_grad():
        torch.testing.assert_close(logits, model(images))


def test_each_map_depends_on_its_own_image_only() -> None:
    torch.manual_seed(1)
    model = build_model("small_cnn").eval()
    images = torch.randn(4, 3, 224, 224)
    maps, _ = gradcam(model, images)
    alone, _ = gradcam(model, images[2:3])
    torch.testing.assert_close(maps[2:3], alone, rtol=1e-4, atol=1e-6)


def test_no_hook_or_gradient_is_left_behind() -> None:
    torch.manual_seed(2)
    model = build_model("small_cnn").eval()
    layer = target_layer(model)
    gradcam(model, torch.randn(2, 3, 224, 224))
    assert not layer._forward_hooks
    assert all(p.grad is None for p in model.parameters())


def test_frozen_parameters_still_give_a_map() -> None:
    torch.manual_seed(3)
    model = build_model("small_cnn").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    maps, _ = gradcam(model, torch.randn(1, 3, 224, 224))
    assert maps.shape == (1, 224, 224)


def test_a_batch_without_its_batch_dimension_is_refused() -> None:
    with pytest.raises(ValueError):
        gradcam(build_model("small_cnn").eval(), torch.randn(3, 224, 224))

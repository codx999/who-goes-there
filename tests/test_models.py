"""trailcam/model: the small CNN, the backbones and the factory (plan.md section 4; R05 the head
sized from the class map, R21 one normalisation for every model, R30 the two models, and phase
2's ResNet-50 of `p2_r50`, item 13.02, with its `/review-shapes` table checked here).

No test downloads weights: every model is built with `pretrained=False`, the default's ImageNet
weights are checked by spying on the builders, and a module-wide fixture refuses any network
connection so a forgotten `pretrained=False` fails instead of downloading. It is module-scoped and
the shared `models` fixture depends on it, so it is already active while those models are built
(review 5.01: a per-test guard was set up after them).
"""

import socket
from collections.abc import Iterator
from typing import Any

import pytest
import timm
import torch
from torch import nn
from torchvision.models import ResNet, ResNet18_Weights, ResNet50_Weights

from trailcam.data import classes
from trailcam.data.classes import CLASSES
from trailcam.data.dataset import IMAGENET_MEAN, IMAGENET_STD
from trailcam.model import MODEL_NAMES, backbone, build_model, factory, parameter_groups
from trailcam.model.factory import head_module
from trailcam.model.small_cnn import SmallCNN


def _refuse_connection(*args: object, **kwargs: object) -> None:
    raise AssertionError("a model test tried to open a network connection")


@pytest.fixture(scope="module", autouse=True)
def no_network() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(socket.socket, "connect", _refuse_connection)
        yield


@pytest.fixture(scope="module")
def models(no_network: None) -> dict[str, nn.Module]:
    assert socket.socket.connect is _refuse_connection  # the guard is up before any build
    return {name: build_model(name, pretrained=False).eval() for name in MODEL_NAMES}


def test_the_network_guard_refuses_the_connection_a_download_opens() -> None:
    with pytest.raises(AssertionError, match="network connection"):
        socket.create_connection(("192.0.2.1", 443), timeout=1)  # TEST-NET-1, never routed


def test_model_names() -> None:
    assert MODEL_NAMES == ("small_cnn", "resnet18", "efficientnet_b0", "resnet50")


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_a_batch_of_two_maps_to_two_rows_of_logits(models: dict[str, nn.Module], name: str) -> None:
    with torch.no_grad():
        logits = models[name](torch.randn(2, 3, 224, 224))
    assert logits.shape == (2, len(CLASSES)) == (2, 12)
    assert logits.dtype == torch.float32


@pytest.mark.parametrize(
    ("name", "in_features"),
    [("small_cnn", 32 * 28 * 28), ("resnet18", 512), ("efficientnet_b0", 1280), ("resnet50", 2048)],
)
def test_the_head_is_one_linear_layer_sized_from_the_class_map(
    models: dict[str, nn.Module], name: str, in_features: int
) -> None:
    linears = [m for m in head_module(models[name]).modules() if isinstance(m, nn.Linear)]
    assert len(linears) == 1
    assert linears[0].out_features == len(CLASSES)
    assert linears[0].in_features == in_features  # the pooled feature size of the body


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_the_head_size_follows_the_class_map_not_a_literal(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setattr(classes, "NUM_CLASSES", 5)
    with torch.no_grad():
        logits = build_model(name, pretrained=False).eval()(torch.randn(1, 3, 224, 224))
    assert logits.shape == (1, 5)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_the_parameter_groups_partition_every_parameter_once(
    models: dict[str, nn.Module], name: str
) -> None:
    model = models[name]
    groups = parameter_groups(model)
    body, head = [id(p) for p in groups.body], [id(p) for p in groups.head]
    everything = [id(p) for p in model.parameters()]
    assert body and head
    assert len(set(body)) == len(body) and len(set(head)) == len(head)
    assert not set(body) & set(head)
    assert sorted(body + head) == sorted(everything)
    assert len(everything) == len({id(p) for _, p in model.named_parameters()})
    # the head group is the final linear layer's weight and bias and nothing else
    assert set(head) == {id(p) for p in head_module(model).parameters()}


@pytest.mark.parametrize("name", ["resnet18", "resnet50"])
def test_the_resnet_head_group_is_fc(name: str) -> None:
    model = build_model(name, pretrained=False)
    assert isinstance(model, ResNet)
    head_ids = {id(p) for p in parameter_groups(model).head}
    names = {n for n, p in model.named_parameters() if id(p) in head_ids}
    assert names == {"fc.weight", "fc.bias"}


def test_unknown_names_are_refused() -> None:
    with pytest.raises(ValueError, match="small_cnn, resnet18, efficientnet_b0, resnet50"):
        build_model("resnet101")
    with pytest.raises(TypeError):
        parameter_groups(nn.Sequential(nn.Linear(2, 2)))


def test_resnet18_loads_the_imagenet_1k_weights_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    requested: list[object] = []
    real = backbone.resnet18

    def spy(*, weights: ResNet18_Weights | None) -> ResNet:
        requested.append(weights)
        return real(weights=None)

    monkeypatch.setattr(backbone, "resnet18", spy)
    build_model("resnet18")
    build_model("resnet18", pretrained=False)
    assert requested == [ResNet18_Weights.IMAGENET1K_V1, None]


def test_resnet50_loads_the_imagenet_1k_v1_weights_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """V1, torchvision's original recipe, as ResNet-18's only weights are: V2's newer recipe
    would be a second change in `p2_r50`."""
    requested: list[object] = []
    real = backbone.resnet50

    def spy(*, weights: ResNet50_Weights | None) -> ResNet:
        requested.append(weights)
        return real(weights=None)

    monkeypatch.setattr(backbone, "resnet50", spy)
    build_model("resnet50")
    build_model("resnet50", pretrained=False)
    assert requested == [ResNet50_Weights.IMAGENET1K_V1, None]


def test_efficientnet_b0_loads_the_imagenet_1k_weights_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested: list[tuple[str, bool]] = []
    real = timm.create_model

    def spy(name: str, *, pretrained: bool, num_classes: int) -> Any:
        requested.append((name, pretrained))
        return real(name, pretrained=False, num_classes=num_classes)

    monkeypatch.setattr(timm, "create_model", spy)
    build_model("efficientnet_b0")
    build_model("efficientnet_b0", pretrained=False)
    assert requested == [("efficientnet_b0.ra_in1k", True), ("efficientnet_b0.ra_in1k", False)]


def test_the_pretrained_weights_expect_the_dataset_normalisation(
    models: dict[str, nn.Module],
) -> None:
    """One pipeline (R21): both weight sets were trained on the ImageNet mean and std."""
    resnet = ResNet18_Weights.IMAGENET1K_V1.transforms()
    assert tuple(resnet.mean) == IMAGENET_MEAN and tuple(resnet.std) == IMAGENET_STD
    config = models["efficientnet_b0"].pretrained_cfg
    assert config["tag"] == "ra_in1k"
    assert tuple(config["mean"]) == IMAGENET_MEAN and tuple(config["std"]) == IMAGENET_STD
    assert backbone.RESNET18_WEIGHTS.meta["num_params"] == 11_689_512  # ImageNet-1k, 1000 classes
    resnet50 = ResNet50_Weights.IMAGENET1K_V1.transforms()
    assert tuple(resnet50.mean) == IMAGENET_MEAN and tuple(resnet50.std) == IMAGENET_STD
    assert backbone.RESNET50_WEIGHTS.meta["num_params"] == 25_557_032


# The `/review-shapes` table of the phase-2 models (item 13.02): layer, input shape, output shape
# and parameters for a batch of two, traced by forward hooks on the top-level layers.
RESNET50_224 = [
    ("conv1", (2, 3, 224, 224), (2, 64, 112, 112), 9_408),
    ("bn1", (2, 64, 112, 112), (2, 64, 112, 112), 128),
    ("relu", (2, 64, 112, 112), (2, 64, 112, 112), 0),
    ("maxpool", (2, 64, 112, 112), (2, 64, 56, 56), 0),
    ("layer1", (2, 64, 56, 56), (2, 256, 56, 56), 215_808),
    ("layer2", (2, 256, 56, 56), (2, 512, 28, 28), 1_219_584),
    ("layer3", (2, 512, 28, 28), (2, 1024, 14, 14), 7_098_368),
    ("layer4", (2, 1024, 14, 14), (2, 2048, 7, 7), 14_964_736),
    ("avgpool", (2, 2048, 7, 7), (2, 2048, 1, 1), 0),
    ("fc", (2, 2048), (2, 12), 2048 * 12 + 12),
]


def _shape_table(
    model: nn.Module, side: int
) -> list[tuple[str, tuple[int, ...], tuple[int, ...], int]]:
    rows: list[tuple[str, tuple[int, ...], tuple[int, ...], int]] = []

    def hook(name: str) -> Any:
        def record(
            module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
        ) -> None:
            params = sum(p.numel() for p in module.parameters())
            rows.append((name, tuple(inputs[0].shape), tuple(output.shape), params))

        return record

    handles = [child.register_forward_hook(hook(n)) for n, child in model.named_children()]
    try:
        with torch.no_grad():
            model(torch.randn(2, 3, side, side))
    finally:
        for handle in handles:
            handle.remove()
    return rows


def test_the_resnet50_shape_table(models: dict[str, nn.Module]) -> None:
    """`p2_r50`: the pooled 2048 features feed the 12-way head; 23,532,620 parameters in all,
    the published 25,557,032 less the 1000-way head plus ours."""
    model = models["resnet50"]
    assert _shape_table(model, 224) == RESNET50_224
    total = sum(p.numel() for p in model.parameters())
    assert total == 25_557_032 - (2048 * 1000 + 1000) + (2048 * 12 + 12) == 23_532_620


@pytest.mark.parametrize(("name", "features"), [("resnet18", 512), ("resnet50", 2048)])
def test_the_resnets_take_320_px(models: dict[str, nn.Module], name: str, features: int) -> None:
    """`p2_res320`: at 320 px the last feature map is 10 x 10 instead of 7 x 7, and the adaptive
    pooling still gives the head its features, so the logits keep their shape."""
    table = {row[0]: row for row in _shape_table(models[name], 320)}
    assert table["layer4"][2] == (2, features, 10, 10)
    assert table["avgpool"][2] == (2, features, 1, 1)
    assert table["fc"][1:3] == ((2, features), (2, 12))


def test_the_small_cnn_is_three_tinyvgg_blocks_and_a_linear_head() -> None:
    model = SmallCNN(len(CLASSES))
    blocks = [model.conv_block_1, model.conv_block_2, model.conv_block_3]
    for block in blocks:
        assert [type(m) for m in block] == [nn.Conv2d, nn.ReLU, nn.Conv2d, nn.ReLU, nn.MaxPool2d]
    assert [type(m) for m in model.classifier] == [nn.Flatten, nn.Linear]
    assert [n for n, _ in model.named_children()] == [
        "conv_block_1",
        "conv_block_2",
        "conv_block_3",
        "classifier",
    ]
    assert sum(isinstance(m, nn.Conv2d) for m in model.modules()) == 6
    assert not any(isinstance(m, nn.BatchNorm2d | nn.Dropout) for m in model.modules())


def test_the_small_cnn_halves_the_side_three_times() -> None:
    model = SmallCNN(len(CLASSES), hidden_units=8)
    x = torch.randn(1, 3, 224, 224)
    sides = []
    for block in (model.conv_block_1, model.conv_block_2, model.conv_block_3):
        x = block(x)
        sides.append(tuple(x.shape[1:]))
    assert sides == [(8, 112, 112), (8, 56, 56), (8, 28, 28)]
    with pytest.raises(ValueError, match="divisible by 8"):
        SmallCNN(len(CLASSES), image_size=220)


def test_build_model_takes_no_class_count() -> None:
    """The class map is the only source of the head size: no caller can pass another."""
    with pytest.raises(TypeError):
        factory.build_model("small_cnn", num_classes=13)

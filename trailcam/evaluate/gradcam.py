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

**Overlays (item 8.02; R04, R26, R35).** `overlay` and `write_overlay` draw the map `gradcam`
returns, the one attention-in-box scores, over the photo it was computed from. The photo must lie
under the `assets/` folder (`ASSETS_DIR`, or another folder named `assets` passed as
`assets_root`, e.g. a clone's on Kaggle): the trans-test demo photos the user picks (item 8.05)
are the only dataset images allowed in the repo, so `assets_image` refuses any other path,
including one that leaves the folder through `..` or a symbolic link. The photo goes through the
eval transform; the overlay is drawn on that transform's output with the normalisation undone,
so it covers exactly the 224 x 224 pixels the model saw. For display the map is divided by its
maximum (attention-in-box divides by its sum; the two differ by a constant factor, so the shape
shown is the shape scored) and blended in one colour whose opacity grows with the map, up to
`OVERLAY_ALPHA`; where the map is zero the photo is untouched. The PNG holds the blended image
only, never the unblended photo: generated figures go to `docs/figures/`, which must never hold a
raw dataset image (`.claude/rules/data.md`; the user's decision, review 8.02).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor, nn
from torchvision.models import ResNet

from trailcam.data.classes import CLASSES
from trailcam.data.dataset import IMAGENET_MEAN, IMAGENET_STD, EvalTransform
from trailcam.model.small_cnn import SmallCNN

ASSETS_DIR: Final = Path(__file__).resolve().parents[2] / "assets"
OVERLAY_COLOUR: Final = (0xEB, 0x68, 0x34)  # categorical slot 2, orange: apart from foliage
OVERLAY_ALPHA: Final = 0.6  # the opacity at the map's maximum


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


def assets_image(path: Path, assets_root: Path = ASSETS_DIR) -> Path:
    """`path` resolved, if it is a file under `assets_root`, a folder named `assets` (R04, R35);
    anything else is refused. The message names neither path, so a refused dataset path is not
    echoed (R03)."""
    root = Path(assets_root).resolve()
    if root.name != "assets":
        raise ValueError("overlays are made from the photos of an assets/ folder only (R04, R35)")
    resolved = Path(path).resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError("an overlay's photo must lie under the assets/ folder (R04, R35)")
    if not resolved.is_file():
        raise FileNotFoundError("no such photo under the assets/ folder")
    return resolved


def denormalise(image: Tensor) -> Image.Image:
    """An eval-transform output, (3, H, W), back to RGB pixels: the normalisation undone,
    clamped to 0 to 255 and rounded."""
    if image.dim() != 3 or image.shape[0] != 3:
        raise ValueError(f"expected one (3, H, W) image, got shape {tuple(image.shape)}")
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    pixels = (image.detach().cpu().float() * std + mean).clamp(0.0, 1.0) * 255.0
    return Image.fromarray(pixels.round().to(torch.uint8).permute(1, 2, 0).numpy(), "RGB")


def blend(frame: Image.Image, cam: Tensor) -> Image.Image:
    """`frame` with `cam` (H, W, non-negative, the frame's size) drawn over it in
    `OVERLAY_COLOUR`: each pixel's opacity is `OVERLAY_ALPHA` times the map divided by its
    maximum. A map that is zero everywhere leaves the frame as it is."""
    rgb = np.asarray(frame.convert("RGB"), dtype=np.float64)
    if tuple(cam.shape) != rgb.shape[:2]:
        raise ValueError(f"map {tuple(cam.shape)} and frame {rgb.shape[:2]} differ")
    values = cam.detach().cpu().double().numpy()
    if (values < 0).any():
        raise ValueError("a Grad-CAM map has negative values; it is taken after a ReLU")
    peak = float(values.max())
    alpha = (values / peak if peak > 0 else np.zeros_like(values))[..., None] * OVERLAY_ALPHA
    mixed = (1.0 - alpha) * rgb + alpha * np.asarray(OVERLAY_COLOUR, dtype=np.float64)
    return Image.fromarray(mixed.round().clip(0, 255).astype(np.uint8), "RGB")


@dataclass(frozen=True)
class Overlay:
    """One photo's overlay: the eval frame with the top-1 class's Grad-CAM map blended in, and
    that class with its softmax confidence. The unblended frame is not kept, so no caller can
    write the raw photo out by mistake."""

    image: Image.Image
    prediction: int
    confidence: float


def overlay(model: nn.Module, path: Path, assets_root: Path = ASSETS_DIR) -> Overlay:
    """The Grad-CAM overlay of the photo at `path`, which must lie under `assets_root`
    (`assets_image`). Puts `model` in eval mode, as `gradcam` needs."""
    source = assets_image(path, assets_root)
    with Image.open(source) as opened:
        photo = opened.convert("RGB")
    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    batch = EvalTransform()(photo).unsqueeze(0)
    model.eval()
    maps, logits = gradcam(model, batch.to(device))
    confidence, prediction = logits[0].float().softmax(dim=0).max(dim=0)
    image = blend(denormalise(batch[0]), maps[0])
    return Overlay(image, int(prediction), float(confidence))


def write_overlay(
    model: nn.Module,
    path: Path,
    out_path: Path,
    label: int | None = None,
    assets_root: Path = ASSETS_DIR,
) -> Overlay:
    """`overlay` as a PNG at `out_path`: one panel, the blended image only (no raw photo),
    titled with the predicted class and its confidence, and the true class when `label` is
    given."""
    result = overlay(model, path, assets_root)
    title = f"predicted {CLASSES[result.prediction]} ({result.confidence:.2f})"
    if label is not None:
        verdict = "correct" if label == result.prediction else "wrong"
        title += f", true {CLASSES[label]}: {verdict}"
    figure = Figure(figsize=(3.6, 3.9), layout="constrained", facecolor="#fcfcfb")
    ax = figure.subplots()
    ax.imshow(result.image)
    ax.set_title("Grad-CAM of the top-1 class, eval frame 224 x 224", fontsize=8, color="#52514e")
    ax.set_axis_off()
    figure.suptitle(title, fontsize=9, color="#0b0b0b")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=120, facecolor="#fcfcfb")
    return result

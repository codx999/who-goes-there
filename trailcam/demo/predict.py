"""One photo's prediction for the demo and the Space (plan.md sections 7 and 8; R33).

`load_model(checkpoint, config_name)` loads a `train.py` checkpoint with `load_checkpoint` of
`trailcam/evaluate/attention.py`, the one loader, which refuses a checkpoint of another model or
class map; the config only names the model. `predict(model, photo)` returns the top class and the
two runners-up, each a (label, softmax probability) pair in descending order, and the seconds of
the forward pass. The three probabilities are three of the twelve classes' softmax, so they sum to
at most 1.

Cost and timing: the photo goes through the eval transform (`EvalTransform`: shorter side 256,
centre crop 224, normalise), then `timed_forward` of `trailcam/evaluate/metrics.py` runs one
untimed warm-up and one timed forward pass and returns that pass's logits beside its seconds. The
prediction is decoded from those logits, so a request costs two forward passes, not three.
Decoding the upload, the transform, the move to the device and the softmax are all outside the
timed region: the seconds shown are the model's forward pass only.

The import surface is this module and what it imports from `trailcam`: torch, torchvision,
Pillow, NumPy, PyYAML and matplotlib (through the evaluate package), and timm, imported only when
a timm model (EfficientNet-B0, ConvNeXt-Tiny) is built. gradio is the app's alone.

**The Space's config (item 13.11).** The Space's weights belong to the config named in
`config_name.txt` beside its `app.py` (`SPACE_CONFIG_FILE`, `space/config_name.txt` in the repo),
which `read_config_name` reads; `scripts/deploy_space.py` checks the weights against the same
file before an upload. It names `DEFAULT_CONFIG`, ResNet-18 arm B, until phase 3 freezes; a kept
phase-3 finalist is deployed by naming its `p3_` config there in a commit
(`docs/phase3-plan.md` section 11). `DEFAULT_CONFIG` stays the local demo's and the sorter's
default.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from PIL import Image
from torch import nn

from trailcam.data.classes import CLASSES
from trailcam.data.dataset import EvalTransform
from trailcam.evaluate.attention import load_checkpoint
from trailcam.evaluate.metrics import timed_forward
from trailcam.train.config import CONFIG_PATH, load_config

TOP_K: Final = 3  # the top class and two runners-up
DEFAULT_CONFIG: Final = "resnet18_arm_b"  # both arms are a ResNet-18; open question 7 picks one
PHOTO_SUFFIXES: Final = (".jpg", ".jpeg", ".png")  # the app's examples and the live sort
SPACE_CONFIG_FILE: Final = "config_name.txt"  # beside the Space's app.py: its weights' config
_CONFIG_NAME: Final = re.compile(r"[A-Za-z0-9_]+")


@dataclass(frozen=True)
class Prediction:
    """The top `TOP_K` (label, probability) pairs, most probable first, and the forward pass's
    seconds."""

    top: tuple[tuple[str, float], ...]
    seconds: float


def read_config_name(path: Path) -> str:
    """The config name in the file at `path` (the Space's `SPACE_CONFIG_FILE`): its one line that
    is neither empty nor a `#` comment. Refuses a file with none, with more than one, or with
    anything but a config name's characters on it."""
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    names = [line for line in lines if line and not line.startswith("#")]
    if len(names) != 1 or not _CONFIG_NAME.fullmatch(names[0]):
        raise ValueError(f"{path.name} must hold one config name on a line of its own")
    return names[0]


def load_model(checkpoint: Path, config_name: str, config_path: Path = CONFIG_PATH) -> nn.Module:
    """The model of `checkpoint`, in eval mode on the CPU; refuses one whose model or input size
    is not the config's, or whose class map is not `CLASSES` (`load_checkpoint`)."""
    model = load_checkpoint(checkpoint, load_config(config_name, config_path))
    model.eval()
    return model


def predict(model: nn.Module, photo: Image.Image) -> Prediction:
    """The top class, the two runners-up and the forward pass's seconds for one photo."""
    image = EvalTransform()(photo.convert("RGB"))
    logits, seconds = timed_forward(model, image)
    probabilities = logits[0].float().softmax(dim=0)
    values, indices = probabilities.topk(TOP_K)
    top = tuple(
        (CLASSES[int(index)], float(value)) for value, index in zip(values, indices, strict=True)
    )
    return Prediction(top, seconds)

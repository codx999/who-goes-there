"""Attention-in-box and the animal-removed test (plan.md sections 6 and 7; R23, R26).

Both measures run over the box-annotated images of one split, through the eval transform, with
the model in eval mode. The boxes come from the metadata (`Record.boxes`, fractions of the
image) and are used for scoring and for blanking pixels only: the model receives image tensors
and nothing else (R23).

**Attention-in-box.** The Grad-CAM map of the top-1 class (`trailcam/evaluate/gradcam.py`) is
taken over the whole 224 x 224 input and normalised by its total there; the score is the share of
that total that falls inside the box. The map is never cropped or rescaled before the division,
so a map spread evenly over the frame scores the box's share of the frame, not 1 (R26).

- Frame: the box is mapped into the eval transform's frame (shorter side resized to 256, then
  the centre 224 x 224 crop, `trailcam/data/dataset.py`) from the size of the image as loaded,
  so it lands where the transform puts the box's pixels. A box partly cut off by the centre crop
  is clipped to the crop.
- Several boxes on one image: the union of their clipped parts. Each box covers every pixel of
  the 224 x 224 grid by the share of that pixel's area it overlaps; where boxes overlap, the
  larger share counts, so no pixel is counted twice.
- Another input size (phase 2, item 13.05): every 224 above is the config's `input_size` and
  every 256 `eval_resize(input_size)` (366 for 320), as in the eval transform the model was
  scored with; the map, the mask and the frame share that side. Phase-1 configs are all 224.
- Excluded and counted: images without a box (about 10% of animal annotations have none, plan.md
  section 2), and images whose boxes all fall outside the centre crop (the model never saw the
  animal, so neither measure means anything there). Both counts are in the table.
- A map that is zero everywhere (the ReLU removed every value) has no mass to share: the image
  keeps its place in the animal-removed test but has no attention score, and is counted.

**Animal-removed test.** Every box of the image is filled with the image mean, then the image is
predicted again through the same eval transform. The image mean is the per-channel mean of the
whole decoded RGB image (box pixels included), on the 0 to 255 pixel values, before any resize or
normalisation, rounded to whole values; the fill is never zero or black (R26). The blanked region
is the box rounded outwards to whole pixels of the loaded image, so no animal pixel survives at
its edge. Reported: accuracy and mean top-1 confidence before and after blanking, and the share of
images whose prediction is unchanged (a model that learned scenery keeps its prediction).

**Reporting.** Each split is one camera group (`CAMERA_GROUPS`: val and cis_test are cis cameras,
trans_val and trans_test trans cameras), and every row is given for all scored images and for
those predicted correctly and wrongly on the unblanked image. Splits are never pooled, so val
never mixes with a test split. Output is counts and scores only, never an image id or file name
(R03). `write_report` writes `attention.md` (the tables) and `attention_in_box.png` (the
histogram) into the directory it is given; generated figures for the repo belong in
`docs/figures/`, which the user fills (item 8.06).

**Test-set isolation (R17, R26).** The command line accepts `val` and `trans_val` only and reads
no other split's records; cis_test and trans_test are scored once, at the end, by the final
scoring (item 9.01), which calls `attention_pass` and `write_report` directly.

**Input size (item 13.05).** The command line scores a checkpoint at its config's `input_size`:
224 for every phase-1 config, 320 for `p2_res320`. `load_checkpoint` refuses a checkpoint trained
at another size than the config's, so a model is never scored at a size it was not trained at.

    python -m trailcam.evaluate.attention --config-name resnet18_arm_b \\
        --checkpoint <run>.pt --data-root <cache> --split val --split trans_val --out <dir>
    python -m trailcam.evaluate.attention --dry-run=True --out <dir>

`--dry-run=True` takes the `dryrun` config, writes the synthetic fixture into a temporary
directory, builds the model with random weights and scores val (or the `--split` given).
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image, ImageStat
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from trailcam.data.classes import CLASSES, NUM_CLASSES
from trailcam.data.dataset import INPUT_SIZE, CCT20Dataset, EvalTransform, eval_resize
from trailcam.data.metadata import DAY_NIGHT_SIDECAR, Box, Record, load_metadata
from trailcam.data.split import make_split
from trailcam.data.synthetic import write_synthetic_fixture
from trailcam.evaluate.gradcam import gradcam, target_layer
from trailcam.model.factory import build_model
from trailcam.train.config import CONFIG_PATH, DRY_RUN, Config, ConfigError, load_config
from trailcam.train.train import (
    CACHE_ANNOTATIONS,
    CACHE_IMAGES,
    file_sha256,
    parse_bool,
    seed_everything,
)

CAMERA_GROUPS: Final = {
    "val": "cis",
    "cis_test": "cis",
    "trans_val": "trans",
    "trans_test": "trans",
}
CLI_SPLITS: Final = ("val", "trans_val")  # cis_test and trans_test: the final scoring only (9.01)
SUBSETS: Final = ("all", "correct", "wrong")
TABLE_NAME: Final = "attention.md"
HISTOGRAM_NAME: Final = "attention_in_box.png"
HISTOGRAM_BINS: Final = 20

# A box in the eval frame: (left, top, right, bottom) in pixels of the 224 x 224 input (or of
# the config's input size, phase 2).
FrameBox = tuple[float, float, float, float]


def eval_geometry(width: int, height: int, size: int = INPUT_SIZE) -> tuple[int, int, int, int]:
    """(resized width, resized height, crop left, crop top) of `EvalTransform(size)` for an
    image of `width` x `height`: torchvision's `Resize(256)` (shorter side to 256, the longer one
    scaled and truncated) and `CenterCrop(224)` (offsets rounded), or `Resize(eval_resize(size))`
    and `CenterCrop(size)` for another size. `tests/test_attention.py` checks it against the
    transform itself."""
    if width <= 0 or height <= 0:
        raise ValueError(f"an image of {width} x {height} pixels")
    resize = eval_resize(size)
    if width <= height:
        new_width, new_height = resize, int(resize * height / width)
    else:
        new_width, new_height = int(resize * width / height), resize
    left = int(round((new_width - size) / 2.0))
    top = int(round((new_height - size) / 2.0))
    return new_width, new_height, left, top


def box_in_frame(box: Box, width: int, height: int, size: int = INPUT_SIZE) -> FrameBox | None:
    """`box` (fractions of an image of `width` x `height`) in the eval frame, clipped to the
    `size` x `size` crop (224 x 224); None when no part of it lies inside the crop."""
    new_width, new_height, left, top = eval_geometry(width, height, size)
    x, y, w, h = box
    x0 = min(max(x * new_width - left, 0.0), float(size))
    x1 = min(max((x + w) * new_width - left, 0.0), float(size))
    y0 = min(max(y * new_height - top, 0.0), float(size))
    y1 = min(max((y + h) * new_height - top, 0.0), float(size))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def box_mask(boxes: Sequence[Box], width: int, height: int, size: int = INPUT_SIZE) -> Tensor:
    """The union of `boxes` on the `size` x `size` eval frame (224 x 224): each pixel holds the
    share of its area inside the boxes (the largest share where boxes overlap). All zero when
    every box falls outside the crop."""
    mask = torch.zeros(size, size, dtype=torch.float64)
    edges = torch.arange(size, dtype=torch.float64)
    for box in boxes:
        framed = box_in_frame(box, width, height, size)
        if framed is None:
            continue
        x0, y0, x1, y1 = framed
        columns = (torch.clamp(edges + 1, max=x1) - torch.clamp(edges, min=x0)).clamp(min=0)
        rows = (torch.clamp(edges + 1, max=y1) - torch.clamp(edges, min=y0)).clamp(min=0)
        mask = torch.maximum(mask, rows[:, None] * columns[None, :])
    return mask


def fraction_inside(heatmap: Tensor, mask: Tensor) -> float:
    """The share of `heatmap`'s total mass, over the whole frame, that lies inside `mask`.

    The map is normalised by its own total before the box is applied, never after: a map wholly
    inside the box scores 1.0, wholly outside 0.0, and one spread evenly over the frame the box's
    share of the frame. NaN for a map that is zero everywhere. Refuses a negative map.
    """
    if heatmap.shape != mask.shape:
        raise ValueError(f"map {tuple(heatmap.shape)} and mask {tuple(mask.shape)} differ")
    values = heatmap.detach().to("cpu", torch.float64)
    if bool((values < 0).any()):
        raise ValueError("a Grad-CAM map has negative values; it is taken after a ReLU")
    total = float(values.sum())
    if total <= 0.0:
        return math.nan
    return float((values * mask.to("cpu", torch.float64)).sum()) / total


def image_mean(image: Image.Image) -> tuple[int, int, int]:
    """The per-channel mean of the whole RGB image, on its 0 to 255 values, rounded."""
    red, green, blue = ImageStat.Stat(image.convert("RGB")).mean
    return round(red), round(green), round(blue)


def blank_region(box: Box, width: int, height: int) -> tuple[int, int, int, int]:
    """`box` in whole pixels of a `width` x `height` image, rounded outwards: (left, top, right,
    bottom), right and bottom exclusive."""
    x, y, w, h = box
    left = max(0, math.floor(x * width))
    top = max(0, math.floor(y * height))
    right = min(width, math.ceil((x + w) * width))
    bottom = min(height, math.ceil((y + h) * height))
    return left, top, right, bottom


def blank_boxes(image: Image.Image, boxes: Sequence[Box]) -> Image.Image:
    """A copy of `image` with every box filled with `image_mean(image)`; pixels outside the boxes
    are untouched."""
    blanked = image.convert("RGB")  # always a copy
    fill = image_mean(image)
    for box in boxes:
        blanked.paste(fill, blank_region(box, *image.size))
    return blanked


class AttentionDataset(Dataset[tuple[Tensor, Tensor, Tensor, int]]):
    """The boxed records of one split. An item is (image, blanked image, box mask, label): both
    images through `EvalTransform(size)`, the mask on the same `size` x `size` frame (224 x 224).
    The model is given the two images only; the mask is for scoring (R23)."""

    def __init__(self, records: Sequence[Record], image_root: Path, size: int = INPUT_SIZE) -> None:
        unboxed = sum(1 for r in records if not r.boxes)
        if unboxed:
            raise ValueError(f"{unboxed} of {len(records)} records have no box")
        self.size = size
        self.transform = EvalTransform(size)
        self.images = CCT20Dataset(records, image_root, self.transform)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, int]:
        record = self.images.records[index]
        image = self.images.load_image(index)
        original = self.transform(image)
        blanked = self.transform(blank_boxes(image, record.boxes))
        mask = box_mask(record.boxes, *image.size, self.size).to(torch.float32)
        return original, blanked, mask, record.label


@dataclass(frozen=True)
class ImageResult:
    """One boxed image; `attention` is NaN when `box_visible` is false or the map is zero.
    `box_share` is the boxes' share of the eval frame (224 x 224): what a map spread evenly over
    the frame would score, the baseline `attention` is read against."""

    label: int
    prediction: int
    confidence: float
    box_visible: bool
    attention: float
    box_share: float
    blanked_prediction: int
    blanked_confidence: float

    @property
    def correct(self) -> bool:
        return self.prediction == self.label


@dataclass(frozen=True)
class SplitRun:
    """The results of one split: `images` records in all, `unboxed` of them without a box (not
    scored), and one result per boxed image, in record order, on an eval frame of side
    `input_size` (224, or a phase-2 config's)."""

    split: str
    images: int
    unboxed: int
    results: tuple[ImageResult, ...]
    input_size: int = INPUT_SIZE

    @property
    def group(self) -> str:
        return CAMERA_GROUPS[self.split]

    @property
    def outside_crop(self) -> int:
        return sum(1 for r in self.results if not r.box_visible)

    @property
    def scored(self) -> tuple[ImageResult, ...]:
        return tuple(r for r in self.results if r.box_visible)

    @property
    def zero_maps(self) -> int:
        return sum(1 for r in self.scored if math.isnan(r.attention))


def attention_pass(
    model: nn.Module,
    records: Sequence[Record],
    split: str,
    image_root: Path,
    device: torch.device,
    *,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    size: int = INPUT_SIZE,
) -> SplitRun:
    """Both measures over the boxed records of `split`, a key of `CAMERA_GROUPS`, through the
    eval transform of side `size` (224; a phase-2 config's `input_size` from the command line).
    Every record must belong to `split`. This function scores any split it is given; keeping
    cis_test and trans_test for the final scoring is the caller's duty (the command line refuses
    them)."""
    if split not in CAMERA_GROUPS:
        raise ValueError(f"split must be one of {tuple(CAMERA_GROUPS)}, got {split!r}")
    others = sum(1 for r in records if r.split != split)
    if others:
        raise ValueError(f"{others} of {len(records)} records are not from {split}")
    boxed = [r for r in records if r.boxes]
    model.eval()
    layer = target_layer(model)
    results: list[ImageResult] = []
    if boxed:
        loader: DataLoader[tuple[Tensor, Tensor, Tensor, int]] = DataLoader(
            AttentionDataset(boxed, image_root, size),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        for images, blanked, masks, labels in loader:
            maps, logits = gradcam(model, images.to(device), layer)
            with torch.inference_mode():
                blanked_logits = model(blanked.to(device))
            confidence, prediction = logits.float().softmax(dim=1).max(dim=1)
            blanked_confidence, blanked_prediction = (
                blanked_logits.float().softmax(dim=1).max(dim=1)
            )
            for i in range(len(labels)):
                visible = bool(masks[i].sum() > 0)
                results.append(
                    ImageResult(
                        label=int(labels[i]),
                        prediction=int(prediction[i]),
                        confidence=float(confidence[i]),
                        box_visible=visible,
                        attention=fraction_inside(maps[i], masks[i]) if visible else math.nan,
                        box_share=float(masks[i].sum()) / size**2,
                        blanked_prediction=int(blanked_prediction[i]),
                        blanked_confidence=float(blanked_confidence[i]),
                    )
                )
    return SplitRun(split, len(records), len(records) - len(boxed), tuple(results), size)


@dataclass(frozen=True)
class Row:
    """One split and subset. Attention statistics are over the images with a score
    (`attention_n`); the animal-removed columns over all `n`. None where a subset is empty."""

    split: str
    group: str
    subset: str
    n: int
    attention_n: int
    attention_mean: float | None
    attention_median: float | None
    box_share: float | None
    accuracy: float | None
    blanked_accuracy: float | None
    confidence: float | None
    blanked_confidence: float | None
    kept: float | None


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def summarise(run: SplitRun) -> tuple[Row, ...]:
    """One row per subset in `SUBSETS` order: all scored images, those predicted correctly and
    those predicted wrongly before blanking."""
    rows = []
    for subset in SUBSETS:
        chosen = [r for r in run.scored if subset == "all" or r.correct == (subset == "correct")]
        attention = [r.attention for r in chosen if not math.isnan(r.attention)]
        rows.append(
            Row(
                split=run.split,
                group=run.group,
                subset=subset,
                n=len(chosen),
                attention_n=len(attention),
                attention_mean=_mean(attention),
                attention_median=statistics.median(attention) if attention else None,
                box_share=_mean([r.box_share for r in chosen if not math.isnan(r.attention)]),
                accuracy=_mean([float(r.correct) for r in chosen]),
                blanked_accuracy=_mean([float(r.blanked_prediction == r.label) for r in chosen]),
                confidence=_mean([r.confidence for r in chosen]),
                blanked_confidence=_mean([r.blanked_confidence for r in chosen]),
                kept=_mean([float(r.blanked_prediction == r.prediction) for r in chosen]),
            )
        )
    return tuple(rows)


def _cell(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _table(header: Sequence[str], lines: Sequence[Sequence[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + " --- |" * len(header),
        *("| " + " | ".join(line) + " |" for line in lines),
    ]


def render_report(runs: Sequence[SplitRun], header: Sequence[str] = ()) -> str:
    """The markdown tables: images per split, attention-in-box and the animal-removed test.
    `header` lines (the model, checkpoint and split) go first."""
    if not runs:
        raise ValueError("no split to report")
    sizes = sorted({run.input_size for run in runs})
    if len(sizes) != 1:
        raise ValueError(f"the splits were scored at different input sizes, {sizes}")
    side = sizes[0]
    rows = [row for run in runs for row in summarise(run)]
    lines = [
        "# Attention-in-box and the animal-removed test",
        "",
        *header,
        *([""] if header else []),
        "Boxes are read from the metadata and used to score and to blank only. Attention-in-box:",
        f"the share of the top-1 class's Grad-CAM mass, normalised over the whole {side} x {side}"
        " input,",
        "that falls inside the union of the image's boxes; a map spread evenly over the frame",
        "would score the boxes' share of the frame, given for comparison. Animal-removed: every",
        "box filled with the image's per-channel mean, then predicted again through the same eval",
        "transform.",
        f"Correct and wrong refer to the unblanked prediction. Chance accuracy is 1/{NUM_CLASSES}"
        f" = {1 / NUM_CLASSES:.4f}.",
        "",
        "## Images",
        "",
        *_table(
            (
                "split",
                "camera group",
                "images",
                "no box (excluded)",
                "box outside the centre crop (excluded)",
                "scored",
                "zero Grad-CAM map (no attention score)",
            ),
            [
                (
                    run.split,
                    run.group,
                    str(run.images),
                    str(run.unboxed),
                    str(run.outside_crop),
                    str(len(run.scored)),
                    str(run.zero_maps),
                )
                for run in runs
            ],
        ),
        "",
        "## Attention-in-box",
        "",
        *_table(
            (
                "split",
                "camera group",
                "predictions",
                "images",
                "mean",
                "median",
                "mean box share of the frame (an even map's score)",
            ),
            [
                (
                    r.split,
                    r.group,
                    r.subset,
                    str(r.attention_n),
                    _cell(r.attention_mean),
                    _cell(r.attention_median),
                    _cell(r.box_share),
                )
                for r in rows
            ],
        ),
        "",
        "## Animal-removed test",
        "",
        *_table(
            (
                "split",
                "camera group",
                "predictions",
                "images",
                "accuracy",
                "accuracy, box blanked",
                "mean top-1 confidence",
                "mean top-1 confidence, box blanked",
                "prediction kept",
            ),
            [
                (
                    r.split,
                    r.group,
                    r.subset,
                    str(r.n),
                    _cell(r.accuracy),
                    _cell(r.blanked_accuracy),
                    _cell(r.confidence),
                    _cell(r.blanked_confidence),
                    _cell(r.kept),
                )
                for r in rows
            ],
        ),
    ]
    return "\n".join(lines) + "\n"


def write_histogram(runs: Sequence[SplitRun], path: Path) -> None:
    """One panel per split: the attention-in-box scores of correct and wrong predictions, each
    as the share of its own images per bin, so the two shapes compare at any count."""
    if not runs:
        raise ValueError("no split to plot")
    bins = np.linspace(0.0, 1.0, HISTOGRAM_BINS + 1)
    figure = Figure(figsize=(5.0 * len(runs), 3.6), layout="constrained")
    axes = figure.subplots(1, len(runs), squeeze=False)[0]
    for ax, run in zip(axes, runs, strict=True):
        for subset, colour in (("correct", "#2a7ab9"), ("wrong", "#d9822b")):
            values = [
                r.attention
                for r in run.scored
                if r.correct == (subset == "correct") and not math.isnan(r.attention)
            ]
            if values:
                ax.hist(
                    values,
                    bins=bins,
                    weights=np.full(len(values), 1.0 / len(values)),
                    histtype="step",
                    linewidth=1.8,
                    color=colour,
                    label=f"{subset} (n={len(values)})",
                )
        ax.set_xlim(0.0, 1.0)
        ax.set_title(f"{run.split} ({run.group} cameras)")
        ax.set_xlabel("Grad-CAM mass inside the box")
        ax.set_ylabel("share of images")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)


def write_report(
    runs: Sequence[SplitRun], out_dir: Path, header: Sequence[str] = ()
) -> tuple[Path, Path]:
    """`attention.md` and `attention_in_box.png` under `out_dir`; returns both paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    table = out_dir / TABLE_NAME
    table.write_text(render_report(runs, header), encoding="utf-8")
    histogram = out_dir / HISTOGRAM_NAME
    write_histogram(runs, histogram)
    return table, histogram


# The keys of a `train.py` checkpoint (item 13.02 added `input_size`); phase 1's lack it.
CHECKPOINT_KEYS: Final = frozenset({"model", "classes", "state_dict", "input_size"})
PHASE1_CHECKPOINT_KEYS: Final = CHECKPOINT_KEYS - {"input_size"}


def load_checkpoint(path: Path, config: Config, *, eval_size: int = INPUT_SIZE) -> nn.Module:
    """The model of a `train.py` checkpoint, checked against `config`: the one loader of the
    evaluation tools, the final scoring, the demo, the Space and its deploy script.

    First, before the file is read, it refuses a config whose input size is not `eval_size`, the
    side of the caller's eval transform. The three evaluation command lines (`evaluate.py`, this
    module's and `slices.py`, item 13.05) build their eval transform at the config's input size
    and pass that side, so a phase-2 config such as `p2_res320` is scored at 320 px on val and
    trans_val. The final scoring, the demo, the Space and its deploy script use `EvalTransform()`
    at 224 px and pass none, so such a config is refused there rather than scored or served at
    the wrong size (its test scoring is item 13.06's). Then it refuses a checkpoint of another
    model, class map or input size. Since item 13.02 `train.py` saves the input size
    (`CHECKPOINT_KEYS`); a checkpoint without one holds exactly `PHASE1_CHECKPOINT_KEYS`, as phase
    1's do, and was trained at 224 px.
    """
    if config.input_size != eval_size:
        raise ValueError(
            f"the config's input size is {config.input_size} px; this tool's eval transform is "
            f"{eval_size} px (the final scoring, the demo and the Space run at {INPUT_SIZE} px "
            "only; docs/phase2-plan.md section 4)"
        )
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or set(saved) not in (CHECKPOINT_KEYS, PHASE1_CHECKPOINT_KEYS):
        raise ValueError(f"{path.name} is not a checkpoint written by trailcam.train.train")
    size = saved.get("input_size", INPUT_SIZE)
    if type(size) is not int:
        raise ValueError(f"{path.name} records an input size of {size!r}, not whole pixels")
    if size != config.input_size:
        raise ValueError(
            f"{path.name} was trained at {size} px, but the config's input size is "
            f"{config.input_size} px"
        )
    if saved["model"] != config.model:
        raise ValueError(f"{path.name} holds a {saved['model']}, the config a {config.model}")
    if list(saved["classes"]) != list(CLASSES):
        raise ValueError(f"{path.name} was trained on another class map")
    model = build_model(config.model, pretrained=False)
    model.load_state_dict(saved["state_dict"])
    return model


def size_note(config: Config) -> str:
    """`, input size <n> px` for a config of another input size than 224 (phase 2), for the
    tables' header lines; nothing for 224, so a phase-1 header reads as before."""
    return "" if config.input_size == INPUT_SIZE else f", input size {config.input_size} px"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m trailcam.evaluate.attention",
        description="Attention-in-box and the animal-removed test on val or trans_val.",
        allow_abbrev=False,
    )
    parser.add_argument("--config-name", help="the config the checkpoint was trained with")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="the config.yaml path")
    parser.add_argument("--checkpoint", type=Path, help="a checkpoint written by train.py")
    parser.add_argument("--data-root", type=Path, help="the 512 px cache")
    parser.add_argument(
        "--split",
        action="append",
        choices=CLI_SPLITS,
        help="val (the default) or trans_val; repeat for both. Test splits: item 9.01 only",
    )
    parser.add_argument("--out", type=Path, required=True, help="where the table and PNG go")
    parser.add_argument("--dry-run", type=parse_bool, default=False, metavar="{True,False}")
    return parser


def resolve(args: argparse.Namespace, parser: argparse.ArgumentParser) -> tuple[str, Config]:
    if args.dry_run:
        name = args.config_name or DRY_RUN
        if name != DRY_RUN:
            parser.error(f"--dry-run=True runs the {DRY_RUN} config, not {name}")
        if args.checkpoint is not None or args.data_root is not None:
            parser.error("the dry run uses random weights and its own fixture")
    else:
        name = args.config_name
        if name is None or args.checkpoint is None or args.data_root is None:
            parser.error("give --config-name, --checkpoint and --data-root, or --dry-run=True")
    try:
        config = load_config(name, args.config)
    except ConfigError as error:
        parser.error(str(error))
    if config.fixture != args.dry_run:
        parser.error(f"config {name} has fixture: {config.fixture}")
    return name, config


def run_splits(
    model: nn.Module,
    records: dict[str, tuple[Record, ...]],
    image_root: Path,
    config: Config,
) -> list[SplitRun]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return [
        attention_pass(
            model,
            split_records,
            name,
            image_root,
            device,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            size=config.input_size,
        )
        for name, split_records in records.items()
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    name, config = resolve(args, parser)
    splits = list(dict.fromkeys(args.split or ["val"]))
    if not set(splits) <= set(CLI_SPLITS):  # argparse's choices already refuse them
        parser.error(f"--split takes {CLI_SPLITS} only")

    if config.fixture:
        with tempfile.TemporaryDirectory(prefix="trailcam-attention-") as tmp:
            root = Path(tmp) / "fixture"
            fixture = write_synthetic_fixture(root)
            meta = load_metadata(root, fixture.paths["sidecar"])
            split = make_split(meta, config.split_seed)
            seed_everything(config.run_seed)
            model = build_model(config.model, pretrained=False)
            records = {s: split.records[s] for s in splits}
            runs = run_splits(model, records, fixture.image_root, config)
        header = [f"Model {config.model}, random weights (dry run on the synthetic fixture)."]
    else:
        # The eval transform is built at the config's input size (224 for phase 1), and the
        # checkpoint must have been trained at it (item 13.05).
        model = load_checkpoint(args.checkpoint, config, eval_size=config.input_size)
        root = args.data_root
        meta = load_metadata(root / CACHE_ANNOTATIONS, root / DAY_NIGHT_SIDECAR)
        split = make_split(meta, config.split_seed)
        records = {s: split.records[s] for s in splits}
        header = [
            f"Model {config.model}, config {name}{size_note(config)}, checkpoint sha256 "
            f"{file_sha256(args.checkpoint)}, split seed {config.split_seed}, split hash "
            f"{split.hash}.",
        ]
        del split, meta  # nothing past this line can reach cis_test or trans_test
        runs = run_splits(model, records, root / CACHE_IMAGES, config)

    table, histogram = write_report(runs, args.out, header)
    print(table.read_text(encoding="utf-8"))
    print(f"wrote {table.name} and {histogram.name} under {args.out}")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

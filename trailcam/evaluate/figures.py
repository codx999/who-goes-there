"""The evaluation figures: confusion matrix, per-class bars and per-camera bars (plan.md section 7;
R03, R04, R32).

Each writer takes scores already computed by `trailcam/evaluate/metrics.py` and writes one PNG to
the path it is given; `trailcam/evaluate/evaluate.py` puts them under its `--out` directory, and
the user copies the ones the repo keeps into `docs/figures/` (item 8.06), which has a 40-file,
20 MB budget in `scripts/guard_data.py`. Nothing here reads an image: the figures hold counts and
scores only, never a photo, an image id or a file name (R03, R04).

- **Confusion matrix**: rows are the true class, columns the predicted class, both in the class
  order of `trailcam/data/classes.py`. Each cell is labelled with its count and shaded by its
  share of the row (the row's recall spread over the predictions), so a rare class is as legible
  as a common one; a class absent from the split has an empty row, labelled "absent".
- **Per-class bars**: precision, recall and F1 for each class present in the split (absent
  classes are left out, as macro-F1 leaves them out), with the support under the class name.
- **Per-camera bars**: accuracy beside macro-F1 (R32) for each camera location of the split,
  worst accuracy first, with the image count under the location; a dashed line marks the split's
  overall accuracy.

Colours follow the reference data-viz palette: one blue ramp, light to dark, for the matrix, and
the first three categorical slots (blue, orange, aqua) for the bars, in that fixed order. Text
stays in the ink colours; a legend names every series.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from trailcam.data.classes import CLASSES, NUM_CLASSES
from trailcam.evaluate.metrics import CameraScores, ClassScores

SURFACE: Final = "#fcfcfb"
INK: Final = "#0b0b0b"
INK_SECONDARY: Final = "#52514e"
GRID: Final = "#e4e3df"
SERIES: Final = ("#2a78d6", "#eb6834", "#1baf7a")  # categorical slots 1 to 3, in fixed order
BLUE_RAMP: Final = ("#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b")
DARK_CELL: Final = 0.55  # a row share above this gets white text
DPI: Final = 120

MATRIX_COLOURS: Final = LinearSegmentedColormap.from_list("trailcam_blue", BLUE_RAMP)


def _style(ax: Axes) -> None:
    """Recessive axes: no top or right spine, a light horizontal grid behind the marks."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY, labelcolor=INK)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(figure: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=DPI, facecolor=SURFACE)
    return path


def row_shares(matrix: Sequence[Sequence[int]]) -> np.ndarray:
    """Each cell of a `NUM_CLASSES` x `NUM_CLASSES` count matrix as a share of its row: what the
    confusion matrix is shaded by. An all-zero row (an absent class) stays zero."""
    counts = np.asarray(matrix, dtype=np.int64)
    if counts.shape != (NUM_CLASSES, NUM_CLASSES):
        raise ValueError(f"expected a {NUM_CLASSES} x {NUM_CLASSES} matrix, got {counts.shape}")
    if (counts < 0).any():
        raise ValueError("a confusion matrix holds counts, not negative values")
    totals = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, totals, out=np.zeros(counts.shape), where=totals > 0)


def camera_order(scores: Mapping[int, CameraScores]) -> list[int]:
    """The camera locations of `scores` as the per-camera bars show them: worst accuracy first,
    ties by location."""
    return sorted(scores, key=lambda location: (scores[location].accuracy, location))


def write_confusion_matrix(matrix: Sequence[Sequence[int]], path: Path, title: str) -> Path:
    """The `NUM_CLASSES` x `NUM_CLASSES` matrix of `metrics.confusion_matrix` as a PNG at
    `path`: counts in the cells, shade by share of the row (`row_shares`)."""
    shares = row_shares(matrix)
    counts = np.asarray(matrix, dtype=np.int64)
    totals = counts.sum(axis=1, keepdims=True)

    figure = Figure(figsize=(8.6, 7.6), layout="constrained", facecolor=SURFACE)
    ax = figure.subplots()
    image = ax.imshow(shares, cmap=MATRIX_COLOURS, vmin=0.0, vmax=1.0)
    ticks = list(range(NUM_CLASSES))
    ax.set_xticks(ticks, labels=list(CLASSES), rotation=45, ha="right", color=INK)
    ax.set_yticks(
        ticks,
        labels=[
            f"{name} ({int(total)})" if total else f"{name} (absent)"
            for name, total in zip(CLASSES, totals[:, 0], strict=True)
        ],
        color=INK,
    )
    ax.set_xlabel("predicted class", color=INK)
    ax.set_ylabel("true class (images)", color=INK)
    for row in range(NUM_CLASSES):
        for column in range(NUM_CLASSES):
            if counts[row, column]:
                ax.text(
                    column,
                    row,
                    str(counts[row, column]),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#ffffff" if shares[row, column] > DARK_CELL else INK,
                )
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colourbar = figure.colorbar(image, ax=ax, shrink=0.8)
    colourbar.set_label("share of the true class's images", color=INK)
    colourbar.outline.set_visible(False)
    ax.set_title(title, color=INK, loc="left")
    return _save(figure, path)


def _grouped_bars(
    ax: Axes, values: Sequence[Sequence[float]], names: Sequence[str], labels: Sequence[str]
) -> None:
    """One group of bars per entry of `labels`, one bar per series in `names`, coloured by
    `SERIES` in order, with a small gap between neighbouring bars."""
    positions = np.arange(len(labels), dtype=np.float64)
    width = 0.8 / len(names)
    for k, (name, series) in enumerate(zip(names, values, strict=True)):
        offset = (k - (len(names) - 1) / 2) * width
        ax.bar(
            positions + offset,
            series,
            width=width * 0.9,
            color=SERIES[k],
            label=name,
            linewidth=0,
        )
    ax.set_xticks(list(positions), labels=list(labels), color=INK)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(-0.6, len(labels) - 0.4)


def write_per_class(scores: Mapping[int, ClassScores], path: Path, title: str) -> Path:
    """Precision, recall and F1 of each class in `scores` (`metrics.per_class_scores`, the
    classes present in the split) as grouped bars, a PNG at `path`."""
    if not scores:
        raise ValueError("no class to plot")
    classes = sorted(scores)
    figure = Figure(figsize=(max(6.0, 0.9 * len(classes) + 2.0), 4.4), layout="constrained")
    figure.set_facecolor(SURFACE)
    ax = figure.subplots()
    _style(ax)
    _grouped_bars(
        ax,
        [
            [scores[c].precision for c in classes],
            [scores[c].recall for c in classes],
            [scores[c].f1 for c in classes],
        ],
        ("precision", "recall", "F1"),
        [f"{CLASSES[c]}\n(n={scores[c].support})" for c in classes],
    )
    ax.tick_params(axis="x", labelsize=8)
    ax.set_ylabel("score", color=INK)
    ax.legend(frameon=False, ncols=3, loc="lower left", bbox_to_anchor=(0.0, 1.0))
    ax.set_title(title, color=INK, loc="left", pad=24)
    return _save(figure, path)


def write_per_camera(
    scores: Mapping[int, CameraScores], overall_accuracy: float, path: Path, title: str
) -> Path:
    """Accuracy and macro-F1 of each camera location in `scores` (`metrics.per_camera_scores`)
    as grouped bars in `camera_order`, with the split's overall accuracy
    as a dashed line; a PNG at `path`."""
    if not scores:
        raise ValueError("no camera to plot")
    if not 0.0 <= overall_accuracy <= 1.0:
        raise ValueError(f"an accuracy of {overall_accuracy}")
    order = camera_order(scores)
    figure = Figure(figsize=(max(6.0, 0.7 * len(order) + 2.0), 4.4), layout="constrained")
    figure.set_facecolor(SURFACE)
    ax = figure.subplots()
    _style(ax)
    _grouped_bars(
        ax,
        [[scores[c].accuracy for c in order], [scores[c].macro_f1 for c in order]],
        ("accuracy", "macro-F1"),
        [f"{c}\n(n={scores[c].images})" for c in order],
    )
    ax.axhline(overall_accuracy, color=INK_SECONDARY, linestyle="--", linewidth=1.2)
    ax.text(  # at the left, above the worst cameras, whose bars sit below the line
        -0.55,
        overall_accuracy,
        f"split accuracy {overall_accuracy:.3f}",
        ha="left",
        va="bottom",
        fontsize=8,
        color=INK_SECONDARY,
    )
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlabel("camera location (images)", color=INK)
    ax.set_ylabel("score", color=INK)
    ax.legend(frameon=False, ncols=2, loc="lower left", bbox_to_anchor=(0.0, 1.0))
    ax.set_title(title, color=INK, loc="left", pad=24)
    return _save(figure, path)

"""Data charts of CCT20 from the metadata alone (item 8.07): the data as it is before any model.

Five PNGs, written into the directory given as `--out` (`docs/figures/` for the committed set,
within the guard's budget, R04). They are drawn from the annotation files, the split of
`trailcam/data/split.py` and, when one is passed, the day/night sidecar; no image is opened, so no
photo can reach a chart. Every chart shows counts and shares only: no image id, sequence id or file
name, and camera locations as anonymous ranks (C1 to C10 for cis cameras, T1 to T10 for trans
cameras, by image count), never their ids (R03). Every chart carries the dataset citation (R06).

1. `class_counts.png`: each class's share of each split, with its count: the imbalance (opossum
   leads train) and the cis vs trans difference (raccoon in trans_test, classes trans_val lacks).
2. `camera_locations.png`: images per camera, stacked by split, cis cameras beside trans cameras.
3. `day_night.png`: the night share per split and images per capture hour by day and night, with
   hour 12 marked: it holds twice its neighbours' images and most of them are night frames, so the
   capture clock cannot stand in for the light. With `--sidecar` the shares are over kept images
   and the split of `--seed`; without one they are `CACHE_BUILD_NIGHT_BY_FILE` and
   `CACHE_BUILD_NIGHT_BY_HOUR`, the numbers the item 6.03 cache build printed over all 57,864
   annotated images before the drop rules, per official file. The chart names its source.
4. `box_fill.png`: the share of the image each box covers (width times height as fractions of
   the image), on a log scale, cis vs trans, with arm B's fill range (`ARM_B_FILL`, the box's area
   over the crop's, plan.md section 6) shaded: a box below 0.2 of the image needs arm B to crop in
   to reach the range; one above it is already there in the whole frame.
5. `burst_length.png`: sequences and images by burst length (`seq_num_frames`): frames of one
   burst are near-duplicates, which is why the split goes by sequence, not by image (plan.md
   section 3), with the count of sequences the official train and cis_val files share.

`python -m trailcam.data.datavis --root <dir with the 5 files> --seed 42 --out <dir>
[--sidecar day_night.json]` writes the five PNGs and prints counts only.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from trailcam.data.classes import CLASSES
from trailcam.data.metadata import Metadata, load_metadata
from trailcam.data.split import CIS_SPLITS, SPLIT_NAMES, TRANS_SPLITS, Split, make_split

CHART_NAMES: Final = (
    "class_counts.png",
    "camera_locations.png",
    "day_night.png",
    "box_fill.png",
    "burst_length.png",
)
CITATION: Final = (
    "Data: Caltech Camera Traps (CCT20), Beery, Van Horn & Perona, "
    '"Recognition in Terra Incognita", ECCV 2018. Licence CDLA-Permissive 1.0.'
)
# The box's area over the crop's area that arm B draws from (plan.md section 6). The sampler's own
# constant is `trailcam.data.boxcrop.FILL`, which only the training transform may import (R24);
# tests/test_datavis.py checks the two agree.
ARM_B_FILL: Final = (0.2, 1.0)
CLOCK_HOUR: Final = 12  # the capture hour that holds about twice its neighbours' images

# Printed by the item 6.03 cache build (scripts/kaggle_prepare_cct20.py on Kaggle) over all 57,864
# annotated images, before the drop rules, per official file: (night, images).
CACHE_BUILD_NIGHT_BY_FILE: Final[Mapping[str, tuple[int, int]]] = {
    "train": (8110, 13553),
    "cis_val": (2048, 3484),
    "trans_val": (1050, 1725),
    "cis_test": (9562, 15827),
    "trans_test": (16006, 23275),
}
# The same build, per capture hour 0 to 23: (night, images).
CACHE_BUILD_NIGHT_BY_HOUR: Final[tuple[tuple[int, int], ...]] = (
    (37, 203),
    (3087, 3247),
    (3360, 3485),
    (3370, 3497),
    (3333, 3420),
    (2850, 2958),
    (1450, 2162),
    (959, 2615),
    (341, 2810),
    (124, 2590),
    (58, 2168),
    (105, 2038),
    (3120, 5077),
    (58, 1492),
    (87, 1399),
    (78, 1103),
    (69, 1088),
    (263, 1086),
    (618, 1227),
    (1758, 2030),
    (2762, 2913),
    (3144, 3272),
    (2671, 2800),
    (3074, 3184),
)
CACHE_BUILD_SOURCE: Final = (
    "item 6.03 cache build: all 57,864 annotated images before the drop rules, per official file"
)

# Colours: the reference categorical palette of the dataviz method, checked with its validator
# (adjacent pairs, light surface). One colour per split in every chart; night and day their own.
SPLIT_COLOURS: Final[Mapping[str, str]] = {
    "train": "#2a78d6",
    "val": "#1baf7a",
    "cis_val": "#1baf7a",
    "cis_test": "#4a3aa7",
    "trans_val": "#eda100",
    "trans_test": "#eb6834",
}
NIGHT_COLOUR: Final = "#4a3aa7"
DAY_COLOUR: Final = "#eda100"
CIS_COLOUR: Final = "#2a78d6"
TRANS_COLOUR: Final = "#eb6834"
INK: Final = "#0b0b0b"
MUTED: Final = "#52514e"
GRID: Final = "#e4e3df"
BAND: Final = "#ecebe7"
DPI: Final = 120


@dataclass(frozen=True)
class NightTable:
    """Night counts: `rows` are (split or file, night, images); `hours` (night, images) per hour."""

    source: str
    rows: tuple[tuple[str, int, int], ...]
    hours: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class Camera:
    """One camera location under its anonymous rank label, with images per split."""

    label: str
    images: Mapping[str, int]


def class_counts(split: Split) -> dict[str, list[int]]:
    """Images per class index, per split, in `SPLIT_NAMES` order."""
    counts = {name: Counter(r.label for r in split.records[name]) for name in SPLIT_NAMES}
    return {name: [counts[name][i] for i in range(len(CLASSES))] for name in SPLIT_NAMES}


def cameras(split: Split) -> tuple[list[Camera], list[Camera]]:
    """Cis and trans cameras, each ranked by images (ties by location, which is not shown) and
    labelled C1, C2, ... and T1, T2, ...; images per split of the group."""
    groups: list[list[Camera]] = []
    for prefix, names in (("C", CIS_SPLITS), ("T", TRANS_SPLITS)):
        per_location: dict[int, Counter[str]] = {}
        for name in names:
            for r in split.records[name]:
                per_location.setdefault(r.location, Counter())[name] += 1
        ranked = sorted(per_location.items(), key=lambda kv: (-kv[1].total(), kv[0]))
        groups.append(
            [
                Camera(f"{prefix}{rank}", {name: c[name] for name in names})
                for rank, (_, c) in enumerate(ranked, start=1)
            ]
        )
    return groups[0], groups[1]


def night_table(split: Split) -> NightTable:
    """Night counts of kept images per split and per hour from the sidecar's flags, or the item
    6.03 cache-build numbers when the records carry no flag."""
    every = [r for name in SPLIT_NAMES for r in split.records[name]]
    if not every or any(r.night is None for r in every):
        return NightTable(
            CACHE_BUILD_SOURCE,
            tuple((name, n, total) for name, (n, total) in CACHE_BUILD_NIGHT_BY_FILE.items()),
            CACHE_BUILD_NIGHT_BY_HOUR,
        )
    rows = tuple(
        (name, sum(bool(r.night) for r in split.records[name]), len(split.records[name]))
        for name in SPLIT_NAMES
    )
    hours = tuple(
        (
            sum(1 for r in every if r.hour == hour and r.night),
            sum(1 for r in every if r.hour == hour),
        )
        for hour in range(24)
    )
    return NightTable(f"day/night sidecar: kept images, split seed {split.seed}", rows, hours)


def box_fills(split: Split) -> dict[str, list[float]]:
    """The share of the image each box covers (width x height, both fractions), per split."""
    return {
        name: [w * h for r in split.records[name] for (_, _, w, h) in r.boxes]
        for name in SPLIT_NAMES
    }


@dataclass(frozen=True)
class Bursts:
    """Sequences and kept images by burst length (`seq_num_frames`), images that share their
    burst with another kept image, and sequences with frames in both official train and cis_val."""

    sequences: Mapping[int, int]
    images: Mapping[int, int]
    images_with_a_sibling: int
    shared_train_cis_val: int


def bursts(meta: Metadata, split: Split) -> Bursts:
    every = [r for name in SPLIT_NAMES for r in split.records[name]]
    length: dict[str, int] = {}
    kept = Counter[str]()
    for r in every:
        length[r.seq_id] = max(length.get(r.seq_id, 0), r.seq_num_frames)
        kept[r.seq_id] += 1
    shared = {r.seq_id for r in meta.records["train"]} & {r.seq_id for r in meta.records["cis_val"]}
    return Bursts(
        sequences=dict(sorted(Counter(length.values()).items())),
        images=dict(sorted(Counter(length[r.seq_id] for r in every).items())),
        images_with_a_sibling=sum(n for n in kept.values() if n > 1),
        shared_train_cis_val=len(shared),
    )


def _figure(width: float, height: float, title: str, subtitle: str = "") -> Figure:
    """A constrained-layout figure with a title, an optional subtitle and the citation footer
    in a reserved strip at the bottom."""
    figure = Figure(figsize=(width, height), layout="constrained")
    footer = 0.32 / height  # the strip the citation sits in, in figure fractions
    engine = figure.get_layout_engine()
    assert engine is not None
    engine.set(rect=(0.0, footer, 1.0, 1.0 - footer))  # type: ignore[call-arg]
    figure.text(0.01, 0.012, CITATION, fontsize=7, color=MUTED, ha="left", va="bottom")
    figure.suptitle(title + (f"\n{subtitle}" if subtitle else ""), fontsize=12, color=INK)
    return figure


def _style(ax: Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
    ax.set_axisbelow(True)


def _percent(value: float, _: object = None) -> str:
    return f"{value:.0%}"


def plot_class_counts(split: Split) -> Figure:
    counts = class_counts(split)
    figure = _figure(
        15.0,
        5.2,
        "Images per class in each split (kept images)",
        f"bar: share of the split; number: images. Split seed {split.seed}; "
        "train and val are cis cameras re-split by sequence.",
    )
    axes = figure.subplots(1, len(SPLIT_NAMES), sharey=True)
    positions = np.arange(len(CLASSES))
    for ax, name in zip(axes, SPLIT_NAMES, strict=True):
        _style(ax)
        total = sum(counts[name])
        shares = [n / total if total else 0.0 for n in counts[name]]
        ax.barh(positions, shares, height=0.72, color=SPLIT_COLOURS[name])
        for y, (share, n) in enumerate(zip(shares, counts[name], strict=True)):
            ax.text(share + 0.008, y, f"{n:,}", va="center", fontsize=7, color=MUTED)
        group = "cis" if name in CIS_SPLITS else "trans"
        ax.set_title(f"{name} ({group} cameras)\n{total:,} images", fontsize=10, color=INK)
        ax.set_xlim(0.0, max(0.45, max(shares) + 0.12))
        ax.xaxis.set_major_formatter(FuncFormatter(_percent))
        ax.grid(axis="x", color=GRID, linewidth=0.8)
    axes[0].set_yticks(positions, CLASSES)
    axes[0].invert_yaxis()
    return figure


def plot_cameras(split: Split) -> Figure:
    cis, trans = cameras(split)
    figure = _figure(
        13.0,
        4.8,
        "Images per camera location: the cis cameras vs the trans cameras",
        "Cameras are ranked by images in their group (C = cis, T = trans); location ids are "
        "not shown. No camera is in both groups.",
    )
    axes = figure.subplots(1, 2, sharey=True, width_ratios=[len(cis) or 1, len(trans) or 1])
    for ax, group, names, heading in (
        (axes[0], cis, CIS_SPLITS, "cis cameras: train, val and cis_test share them"),
        (axes[1], trans, TRANS_SPLITS, "trans cameras: never seen in training"),
    ):
        _style(ax)
        x = np.arange(len(group))
        bottom = np.zeros(len(group))
        for name in names:
            heights = np.array([c.images[name] for c in group], dtype=float)
            ax.bar(
                x,
                heights,
                bottom=bottom,
                width=0.72,
                color=SPLIT_COLOURS[name],
                edgecolor="white",
                linewidth=1.0,
                label=name,
            )
            bottom += heights
        for xi, total in zip(x, bottom, strict=True):
            ax.text(xi, total, f"{int(total):,}", ha="center", va="bottom", fontsize=7, color=MUTED)
        ax.set_xticks(x, [c.label for c in group])
        ax.set_title(heading, fontsize=10, color=INK)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    axes[0].set_ylabel("kept images", color=INK)
    return figure


def plot_day_night(table: NightTable) -> Figure:
    figure = _figure(
        14.0,
        5.0,
        "Night (infrared) and day images: per split and per capture hour",
        f"Source: {table.source}.",
    )
    left, right = figure.subplots(1, 2, width_ratios=[1, 2.4])
    _style(left)
    names = [name for name, _, _ in table.rows]
    y = np.arange(len(names))
    shares = [n / total if total else 0.0 for _, n, total in table.rows]
    left.barh(y, shares, height=0.66, color=NIGHT_COLOUR, label="night")
    left.barh(y, [1 - s for s in shares], left=shares, height=0.66, color=DAY_COLOUR, label="day")
    for yi, (share, (_, n, total)) in enumerate(zip(shares, table.rows, strict=True)):
        left.text(
            0.02, yi, f"{share:.3f}  ({n:,} of {total:,})", va="center", fontsize=8, color="white"
        )
    left.set_yticks(y, names)
    left.invert_yaxis()
    left.set_xlim(0.0, 1.0)
    left.xaxis.set_major_formatter(FuncFormatter(_percent))
    left.set_title("night share", fontsize=10, color=INK)
    left.legend(frameon=False, fontsize=8, loc="lower right", ncols=2, bbox_to_anchor=(1.0, -0.2))

    _style(right)
    hours = np.arange(24)
    night = np.array([n for n, _ in table.hours], dtype=float)
    images = np.array([total for _, total in table.hours], dtype=float)
    right.bar(hours, night, width=0.78, color=NIGHT_COLOUR, label="night")
    right.bar(hours, images - night, bottom=night, width=0.78, color=DAY_COLOUR, label="day")
    right.set_xticks(hours, [f"{h:02d}" for h in hours])
    right.set_xlabel("capture hour (the camera's clock)", color=INK)
    right.set_ylabel("images", color=INK)
    right.grid(axis="y", color=GRID, linewidth=0.8)
    right.set_title("images per capture hour", fontsize=10, color=INK)
    right.set_ylim(0.0, max(images.max(initial=0.0), 1.0) * 1.12)  # room for the hour-12 note
    n12, total12 = table.hours[CLOCK_HOUR]
    if total12:
        right.annotate(
            f"hour {CLOCK_HOUR}: {total12:,} images, {n12:,} of them night frames.\n"
            "The clock is not the light: the flag comes from the pixels.",
            xy=(CLOCK_HOUR, total12),
            xytext=(CLOCK_HOUR + 1.0, total12 * 0.98),
            fontsize=8,
            color=INK,
            va="top",
            arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8},
        )
    right.legend(frameon=False, fontsize=8, loc="upper left")
    return figure


def plot_box_fill(split: Split) -> Figure:
    fills = box_fills(split)
    groups = {
        "cis (train, val, cis_test)": [f for n in CIS_SPLITS for f in fills[n]],
        "trans (trans_val, trans_test)": [f for n in TRANS_SPLITS for f in fills[n]],
    }
    train = fills["train"]
    below = sum(f < ARM_B_FILL[0] for f in train) / len(train) if train else 0.0
    figure = _figure(
        10.0,
        5.0,
        "How much of the image an animal's box covers",
        "box width x height as fractions of the image; each curve is the share of its boxes "
        "per bin (log scale)",
    )
    ax = figure.subplots()
    _style(ax)
    every = [f for values in groups.values() for f in values]
    low = math.floor(math.log10(min(every))) if every else -4
    bins = np.logspace(low, 0, 10 * -low + 1).tolist()
    ax.axvspan(*ARM_B_FILL, color=BAND, zorder=0)
    ax.text(
        math.sqrt(ARM_B_FILL[0] * ARM_B_FILL[1]),
        0.97,
        "arm B fill range:\nthe box is 20% to 100%\nof its training crop",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=8,
        color=INK,
    )
    for (label, values), colour in zip(groups.items(), (CIS_COLOUR, TRANS_COLOUR), strict=True):
        if values:
            ax.hist(
                values,
                bins=bins,
                weights=np.full(len(values), 1.0 / len(values)),
                histtype="step",
                linewidth=2.0,
                color=colour,
                label=f"{label}: {len(values):,} boxes",
            )
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.4g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(_percent))
    ax.set_xlabel("share of the image the box covers", color=INK)
    ax.set_ylabel("share of boxes", color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.text(
        0.01,
        0.80,
        f"train: {below:.0%} of {len(train):,} boxes cover less than 20% of the image;\n"
        "arm B's crop zooms in on those, the whole frame already holds the rest.",
        transform=ax.transAxes,
        fontsize=8,
        color=INK,
        va="top",
    )
    return figure


def plot_bursts(stats: Bursts) -> Figure:
    total_images = sum(stats.images.values())
    sibling = stats.images_with_a_sibling / total_images if total_images else 0.0
    figure = _figure(
        11.0,
        4.8,
        "Burst length: one trigger fires a sequence of near-identical frames",
        f"{sibling:.0%} of kept images share their burst with another kept image; "
        f"{stats.shared_train_cis_val:,} bursts have frames in both official train and "
        "cis_val,\nso the two are pooled and re-split by sequence (no burst in two splits).",
    )
    axes = figure.subplots(1, 2, sharex=True)
    for ax, values, ylabel in (
        (axes[0], stats.sequences, "sequences"),
        (axes[1], stats.images, "kept images"),
    ):
        _style(ax)
        lengths = list(values)
        ax.bar(lengths, [values[k] for k in lengths], width=0.7, color=CIS_COLOUR)
        for k in lengths:
            label = f"{values[k]:,}"
            ax.text(k, values[k], label, ha="center", va="bottom", fontsize=7, color=MUTED)
        ax.set_xticks(lengths)
        ax.set_xlabel("frames in the burst (seq_num_frames)", color=INK)
        ax.set_ylabel(ylabel, color=INK)
        ax.set_title(f"{ylabel} by burst length", fontsize=10, color=INK)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    return figure


def figures(meta: Metadata, split: Split) -> dict[str, Figure]:
    """The five charts by file name, in `CHART_NAMES` order."""
    drawn = (
        plot_class_counts(split),
        plot_cameras(split),
        plot_day_night(night_table(split)),
        plot_box_fill(split),
        plot_bursts(bursts(meta, split)),
    )
    return dict(zip(CHART_NAMES, drawn, strict=True))


def write_charts(meta: Metadata, split: Split, out_dir: Path) -> list[Path]:
    """The five PNGs of `CHART_NAMES` under `out_dir`; returns their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, figure in figures(meta, split).items():
        path = out_dir / name
        figure.savefig(path, dpi=DPI, facecolor="white")
        paths.append(path)
    return paths


def summary(meta: Metadata, split: Split) -> str:
    """Counts only: what the charts were drawn from."""
    cis, trans = cameras(split)
    table = night_table(split)
    stats = bursts(meta, split)
    n_boxes = sum(len(v) for v in box_fills(split).values())
    images = sum(len(split.records[name]) for name in SPLIT_NAMES)
    return "\n".join(
        [
            f"kept images {images}, split seed {split.seed}",
            f"cameras: {len(cis)} cis, {len(trans)} trans",
            f"boxes {n_boxes}",
            f"sequences {sum(stats.sequences.values())}, "
            f"shared by official train and cis_val {stats.shared_train_cis_val}",
            f"images sharing their burst with another kept image {stats.images_with_a_sibling}",
            f"night source: {table.source}",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m trailcam.data.datavis",
        description="Write the five data charts as PNG from the metadata; print counts only.",
        allow_abbrev=False,
    )
    parser.add_argument("--root", type=Path, required=True, help="directory holding the 5 files")
    parser.add_argument("--seed", type=int, required=True, help="split seed (42: the data card)")
    parser.add_argument("--out", type=Path, required=True, help="where the PNGs go")
    parser.add_argument("--sidecar", type=Path, default=None, help="day/night sidecar JSON")
    args = parser.parse_args(argv)
    meta = load_metadata(args.root, args.sidecar)
    split = make_split(meta, args.seed)
    paths = write_charts(meta, split, args.out)
    print(summary(meta, split))
    print(f"wrote {', '.join(p.name for p in paths)} under {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

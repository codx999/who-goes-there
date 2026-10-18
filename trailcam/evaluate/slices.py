"""Slices of one model's predictions and the feature map (plan.md section 7; R17, R32).

No extra training: one pass of a checkpoint over a split gives the predictions and the backbone
features, and everything here is cut from them.

**Day and night.** Each image's night flag comes from the cache's `day_night.json` sidecar
(`Record.night`, loaded with an explicit sidecar). Night share differs between splits (from the
6.03 cache build: train 0.598, cis_val 0.588, trans_val 0.609, cis_test 0.604, trans_test 0.688),
so part of a cis-to-trans gap can come from the mix of day and night images rather than
from the unseen cameras. The report gives accuracy and macro-F1 per split (one camera group each,
`CAMERA_GROUPS`) for all, day and night images, and, for each cis and trans pair given (val with
trans_val, cis_test with trans_test), the trans accuracy and macro-F1 re-weighted to the cis
night share. Each trans image is weighted by the night-share ratio, `p_cis / p_trans` for a night
image and `(1 - p_cis) / (1 - p_trans)` for a day image (the weights average 1), and both scores
are computed from the weighted counts: accuracy becomes `(1 - p_cis) * day accuracy + p_cis *
night accuracy`, and macro-F1 is the mean over the classes present of `2TP / (2TP + FP + FN)` on
the weighted confusion counts. Macro-F1 is not a weighted mean of its day and night values, so it
is re-weighted inside the counts rather than across the slices. The gap left at that common mix
is the part of the gap the day/night mix does not explain.

**Common and rare species.** The rule, fixed on the train counts before any test scoring (R17)
and never tuned on results: a class is rare when it holds fewer than `RARE_SHARE` (4%) of the
train images of the seed-42 split (`docs/data-card.md`, 13,587 images). That makes bird (509,
3.7%), rodent (440, 3.2%) and skunk (232, 1.7%) rare, the three the research report already calls
rare; the next class, car (603, 4.4%), is common. `RARE_CLASSES` holds the result as a constant,
and `tests/test_slices.py` checks it against the data card with `rare_classes`.

Every slice is scored with `trailcam/evaluate/metrics.py`: accuracy beside macro-F1 (R32), and
macro-F1 over the classes present in that slice, which a rare slice or a trans split lacks some
of. The two slices of a cut always add up to the split (day + night, common + rare).

**Feature map.** The backbone features are the input of the model's head (`head_module`),
flattened: 512 for ResNet-18, 1280 for EfficientNet-B0, `hidden_units * 28 * 28` for the small
CNN (25,088 per image; a caller that wants only the slice tables passes `keep_features=False`).
They are projected to two dimensions by PCA (centred, `torch.linalg.svd` in float64, each
component's sign fixed so its largest loading is positive; no new dependency), one projection per
checkpoint over the images of every split given, so cis and trans cameras share one map. Two PNGs,
each with one block of small panels per checkpoint, side by side (arm A beside arm B): one panel
per species, and one per camera location, each highlighting its images over the rest in grey.
Small panels rather than one panel in 12 or 20 colours, which no palette keeps apart.

**The pair (item 13.05).** Each `--checkpoint` names a config and a file, so the user names the
models compared. A phase-1 checkpoint is labelled by its model and arm, as before (`resnet18 arm
A` beside `resnet18 arm B`, one seed's pair); a phase-2 one by its config name, since several
phase-2 configs share a model and an arm (`p2_baseline` beside `p2_head3` or `p2_res320`, the
pairing of `docs/phase2-plan.md` section 3). Phase-1 and phase-2 checkpoints are never mixed in
one run: phase 2 compares each variant with `p2_baseline`, never with a phase-1 run. Each
checkpoint is scored at its own config's `input_size` (224 in phase 1, 320 for `p2_res320`), and
`load_checkpoint` refuses one trained at another size than its config's. Which run seeds are
paired is the caller's choice (the evaluation notebook pairs the same seed).

**Output.** `write_report` writes `slices.md` (the tables) and `write_feature_maps` the two PNGs
into the directory they are given; repo figures belong in `docs/figures/`, filled by the user
(item 8.06). Counts and scores only, never an image id or file name (R03).

**Test-set isolation (R17).** The command line accepts `val` and `trans_val` only and reads no
other split's records; cis_test and trans_test are reached only through the final scoring (item
9.01), which calls `predict_split`, `write_report`, `project` and `write_feature_maps` directly.

    python -m trailcam.evaluate.slices --checkpoint resnet18_arm_a <a>.pt \\
        --checkpoint resnet18_arm_b <b>.pt --data-root <cache> --out <dir>
    python -m trailcam.evaluate.slices --checkpoint p2_baseline <base>.pt \\
        --checkpoint p2_res320 <variant>.pt --data-root <cache> --out <dir>
    python -m trailcam.evaluate.slices --dry-run=True --out <dir>

`--split` defaults to both val and trans_val. `--dry-run=True` takes the `dryrun` config, writes
the synthetic fixture into a temporary directory and uses two models with random weights.
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from torch import Tensor, nn
from torch.utils.data import DataLoader

from trailcam.data.classes import CLASSES, NUM_CLASSES, index_of_name
from trailcam.data.dataset import INPUT_SIZE, CCT20Dataset, EvalTransform
from trailcam.data.metadata import DAY_NIGHT_SIDECAR, Record, load_metadata
from trailcam.data.split import make_split
from trailcam.data.synthetic import write_synthetic_fixture
from trailcam.evaluate.attention import CAMERA_GROUPS, CLI_SPLITS, load_checkpoint, size_note
from trailcam.evaluate.metrics import accuracy, macro_f1
from trailcam.model.factory import build_model, head_module
from trailcam.train.config import (
    CONFIG_PATH,
    DRY_RUN,
    Config,
    ConfigError,
    is_phase2,
    load_config,
)
from trailcam.train.train import (
    CACHE_ANNOTATIONS,
    CACHE_IMAGES,
    file_sha256,
    parse_bool,
    seed_everything,
)

RARE_SHARE: Final = 0.04
RARE_CLASSES: Final = frozenset(index_of_name(name) for name in ("bird", "rodent", "skunk"))
CUTS: Final = {"day/night": ("day", "night"), "species": ("common", "rare")}
PAIRS: Final = (("val", "trans_val"), ("cis_test", "trans_test"))
TABLE_NAME: Final = "slices.md"
SPECIES_FIGURE: Final = "features_by_species.png"
CAMERA_FIGURE: Final = "features_by_camera.png"
PANEL_COLUMNS: Final = 4
GROUP_COLOURS: Final = {"cis": "#2a78d6", "trans": "#eb6834"}  # categorical slots 1 and 2
SPECIES_COLOUR: Final = "#2a78d6"
BACKGROUND_COLOUR: Final = "#c9c8c1"


def rare_classes(train_counts: Mapping[int, int], share: float = RARE_SHARE) -> frozenset[int]:
    """The classes holding fewer than `share` of the train images in `train_counts` (class index
    to image count): the rule `RARE_CLASSES` was fixed with."""
    total = sum(train_counts.values())
    if total <= 0:
        raise ValueError("no train images to count")
    return frozenset(c for c, n in train_counts.items() if n < share * total)


@dataclass(frozen=True, eq=False)
class SplitPredictions:
    """One checkpoint's pass over one split, in record order: label, prediction, night flag and
    camera location per image, and the backbone features, (images, features) float32 on the CPU,
    when they were kept."""

    split: str
    labels: tuple[int, ...]
    predictions: tuple[int, ...]
    night: tuple[bool, ...]
    locations: tuple[int, ...]
    features: Tensor | None = None

    def __post_init__(self) -> None:
        if self.split not in CAMERA_GROUPS:
            raise ValueError(f"split must be one of {tuple(CAMERA_GROUPS)}, got {self.split!r}")
        n = len(self.labels)
        if not n:
            raise ValueError(f"no images in {self.split}")
        if not len(self.predictions) == len(self.night) == len(self.locations) == n:
            raise ValueError(f"{self.split}: labels, predictions, flags and locations differ")
        if self.features is not None and (self.features.dim() != 2 or len(self.features) != n):
            raise ValueError(f"{self.split}: features of shape {tuple(self.features.shape)}")

    @property
    def group(self) -> str:
        return CAMERA_GROUPS[self.split]

    @property
    def night_share(self) -> float:
        return sum(self.night) / len(self.night)

    def in_slice(self, name: str) -> tuple[int, ...]:
        """Positions of the images in slice `name`: all, day, night, common or rare."""
        if name == "all":
            return tuple(range(len(self.labels)))
        if name in ("day", "night"):
            return tuple(i for i, night in enumerate(self.night) if night == (name == "night"))
        if name in ("common", "rare"):
            rare = name == "rare"
            return tuple(i for i, y in enumerate(self.labels) if (y in RARE_CLASSES) == rare)
        raise ValueError(f"unknown slice {name!r}")


def predict_split(
    model: nn.Module,
    records: Sequence[Record],
    split: str,
    image_root: Path,
    device: torch.device,
    *,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    keep_features: bool = True,
    size: int = INPUT_SIZE,
) -> SplitPredictions:
    """Predictions (and, with `keep_features`, the head's input) for the records of `split`, a
    key of `CAMERA_GROUPS`, through the eval transform of side `size` (224; a phase-2 config's
    `input_size` from the command line) in eval mode. Every record must belong to
    `split` and carry a night flag. This function scores any split it is given; keeping cis_test
    and trans_test for the final scoring is the caller's duty (the command line refuses them)."""
    if split not in CAMERA_GROUPS:
        raise ValueError(f"split must be one of {tuple(CAMERA_GROUPS)}, got {split!r}")
    others = sum(1 for r in records if r.split != split)
    if others:
        raise ValueError(f"{others} of {len(records)} records are not from {split}")
    unflagged = sum(1 for r in records if r.night is None)
    if unflagged:
        raise ValueError(f"{unflagged} of {len(records)} records have no day/night flag")
    loader: DataLoader[tuple[Tensor, int]] = DataLoader(
        CCT20Dataset(records, image_root, EvalTransform(size)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    captured: list[Tensor] = []

    def keep(_module: nn.Module, inputs: tuple[Tensor, ...], _output: Tensor) -> None:
        captured.append(inputs[0].detach().flatten(1).float().cpu())

    hook = head_module(model).register_forward_hook(keep) if keep_features else None
    predictions: list[int] = []
    labels: list[int] = []
    model.eval()
    try:
        with torch.inference_mode():
            for images, targets in loader:
                logits = model(images.to(device))
                predictions += logits.argmax(dim=1).tolist()
                labels += targets.tolist()
    finally:
        if hook is not None:
            hook.remove()
    return SplitPredictions(
        split=split,
        labels=tuple(labels),
        predictions=tuple(predictions),
        night=tuple(bool(r.night) for r in records),
        locations=tuple(r.location for r in records),
        features=torch.cat(captured) if captured else None,
    )


@dataclass(frozen=True)
class SliceRow:
    """One slice of one split; scores are None for an empty slice. `classes` counts the classes
    present, the ones macro-F1 averages over."""

    split: str
    group: str
    cut: str
    slice: str
    images: int
    classes: int
    accuracy: float | None
    macro_f1: float | None


def slice_rows(run: SplitPredictions) -> tuple[SliceRow, ...]:
    """Per cut in `CUTS`: the whole split, then its two slices."""
    rows = []
    for cut, names in CUTS.items():
        for name in ("all", *names):
            chosen = run.in_slice(name)
            labels = [run.labels[i] for i in chosen]
            predictions = [run.predictions[i] for i in chosen]
            rows.append(
                SliceRow(
                    split=run.split,
                    group=run.group,
                    cut=cut,
                    slice=name,
                    images=len(chosen),
                    classes=len(set(labels)),
                    accuracy=accuracy(predictions, labels) if labels else None,
                    macro_f1=macro_f1(predictions, labels) if labels else None,
                )
            )
    return tuple(rows)


def weighted_scores(
    predictions: Sequence[int], labels: Sequence[int], weights: Sequence[float]
) -> tuple[float, float]:
    """Accuracy and macro-F1 with each image counted `weights[i]` times, under the conventions
    of `trailcam/evaluate/metrics.py`: macro-F1 is the mean over the classes with a positive
    weighted support of `2TP / (2TP + FP + FN)` on the weighted counts. With every weight 1 both
    equal `accuracy` and `macro_f1`."""
    if not len(predictions) == len(labels) == len(weights):
        raise ValueError(
            f"{len(predictions)} predictions, {len(labels)} labels, {len(weights)} weights"
        )
    if any(w < 0 for w in weights):
        raise ValueError("a weight is negative")
    total = sum(weights)
    if total <= 0:
        raise ValueError("no weight to score")
    accuracy(predictions, labels)  # the same checks on the class indices
    tp = [0.0] * NUM_CLASSES
    fp = [0.0] * NUM_CLASSES
    support = [0.0] * NUM_CLASSES
    for p, y, w in zip(predictions, labels, weights, strict=True):
        support[y] += w
        if p == y:
            tp[y] += w
        else:
            fp[p] += w
    present = [c for c in range(NUM_CLASSES) if support[c] > 0]
    f1 = [2 * tp[c] / (2 * tp[c] + fp[c] + support[c] - tp[c]) for c in present]
    return sum(tp) / total, sum(f1) / len(f1)


def mix_weights(night: Sequence[bool], night_share: float) -> tuple[float, ...] | None:
    """Per-image weights that move a split with flags `night` to `night_share` at night:
    `night_share / q` for a night image and `(1 - night_share) / (1 - q)` for a day image, q the
    split's own night share. None when the split has no day or no night image."""
    q = sum(night) / len(night)
    if q in (0.0, 1.0):
        return None
    day_weight, night_weight = (1 - night_share) / (1 - q), night_share / q
    return tuple(night_weight if flag else day_weight for flag in night)


@dataclass(frozen=True)
class MixRow:
    """A cis split and its trans counterpart, for one metric (accuracy or macro-F1);
    `trans_at_cis_mix` is the trans score with day and night re-weighted to the cis night share
    (None if the trans split lacks one of them)."""

    cis_split: str
    trans_split: str
    metric: str
    cis: float
    trans: float
    cis_night_share: float
    trans_night_share: float
    trans_at_cis_mix: float | None

    @property
    def gap(self) -> float:
        return self.cis - self.trans

    @property
    def gap_at_cis_mix(self) -> float | None:
        if self.trans_at_cis_mix is None:
            return None
        return self.cis - self.trans_at_cis_mix


def mix_rows(runs: Sequence[SplitPredictions]) -> tuple[MixRow, ...]:
    """Two rows, accuracy then macro-F1, per pair in `PAIRS` whose two splits are both in
    `runs`."""
    by_split = {run.split: run for run in runs}
    rows = []
    for cis_name, trans_name in PAIRS:
        if cis_name not in by_split or trans_name not in by_split:
            continue
        cis, trans = by_split[cis_name], by_split[trans_name]
        p = cis.night_share
        weights = mix_weights(trans.night, p)
        at_mix = (
            (None, None)
            if weights is None
            else weighted_scores(trans.predictions, trans.labels, weights)
        )
        for metric, score, mixed in (
            ("accuracy", accuracy, at_mix[0]),
            ("macro-F1", macro_f1, at_mix[1]),
        ):
            rows.append(
                MixRow(
                    cis_split=cis_name,
                    trans_split=trans_name,
                    metric=metric,
                    cis=score(cis.predictions, cis.labels),
                    trans=score(trans.predictions, trans.labels),
                    cis_night_share=p,
                    trans_night_share=trans.night_share,
                    trans_at_cis_mix=mixed,
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


def _cut_table(runs: Sequence[SplitPredictions], cut: str) -> list[str]:
    first, second = CUTS[cut]
    lines = []
    for run in runs:
        rows = {r.slice: r for r in slice_rows(run) if r.cut == cut}
        whole, a, b = rows["all"], rows[first], rows[second]
        lines.append(
            (
                run.split,
                run.group,
                str(whole.images),
                str(a.images),
                str(b.images),
                f"{b.images / whole.images:.3f}",
                _cell(whole.accuracy),
                _cell(a.accuracy),
                _cell(b.accuracy),
                _cell(whole.macro_f1),
                _cell(a.macro_f1),
                _cell(b.macro_f1),
                f"{whole.classes} / {a.classes} / {b.classes}",
            )
        )
    return _table(
        (
            "split",
            "camera group",
            "images",
            first,
            second,
            f"{second} share",
            "accuracy",
            f"accuracy, {first}",
            f"accuracy, {second}",
            "macro-F1",
            f"macro-F1, {first}",
            f"macro-F1, {second}",
            f"classes present (all / {first} / {second})",
        ),
        lines,
    )


def render_report(
    results: Mapping[str, Sequence[SplitPredictions]], header: Sequence[str] = ()
) -> str:
    """The markdown tables, one section per checkpoint label in `results`. `header` lines (the
    checkpoints and the split) go first."""
    if not results or not all(results.values()):
        raise ValueError("no split to report")
    rare = ", ".join(CLASSES[c] for c in sorted(RARE_CLASSES))
    lines = [
        "# Slices: day and night, common and rare species",
        "",
        *header,
        *([""] if header else []),
        "The same predictions cut two ways; each split is one camera group (cis cameras were",
        "seen in training, trans cameras never). Day and night come from the cache's day/night",
        f"flag. Rare species: fewer than {RARE_SHARE:.0%} of the train images (seed-42 split,",
        f"fixed before any test scoring): {rare}; the other classes are common. Macro-F1",
        "averages over the classes present in the slice. The two slices of a cut add up to the",
        "split.",
    ]
    for label, runs in results.items():
        lines += [
            "",
            f"## {label}",
            "",
            "### Day and night",
            "",
            *_cut_table(runs, "day/night"),
            "",
            "### Common and rare species",
            "",
            *_cut_table(runs, "species"),
        ]
        mixes = mix_rows(runs)
        if mixes:
            lines += [
                "",
                "### Cis and trans at the same day/night mix",
                "",
                "Trans scores re-weighted to the cis night share p: each trans image counts p / q",
                "at night and (1 - p) / (1 - q) by day, q the trans night share, and accuracy and",
                "macro-F1 are computed from the weighted counts. The gap left at the same mix is",
                "the part the day/night mix does not explain.",
                "",
                *_table(
                    (
                        "cis split",
                        "trans split",
                        "metric",
                        "cis",
                        "trans",
                        "gap",
                        "cis night share",
                        "trans night share",
                        "trans at the cis mix",
                        "gap at the cis mix",
                    ),
                    [
                        (
                            m.cis_split,
                            m.trans_split,
                            m.metric,
                            _cell(m.cis),
                            _cell(m.trans),
                            _cell(m.gap),
                            f"{m.cis_night_share:.3f}",
                            f"{m.trans_night_share:.3f}",
                            _cell(m.trans_at_cis_mix),
                            _cell(m.gap_at_cis_mix),
                        )
                        for m in mixes
                    ],
                ),
            ]
    return "\n".join(lines) + "\n"


def write_report(
    results: Mapping[str, Sequence[SplitPredictions]], out_dir: Path, header: Sequence[str] = ()
) -> Path:
    """`slices.md` under `out_dir`; returns its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / TABLE_NAME
    path.write_text(render_report(results, header), encoding="utf-8")
    return path


def pca_2d(features: Tensor) -> tuple[Tensor, tuple[float, float]]:
    """The rows of `features` (images, features) on their first two principal components, as
    (images, 2) float64, and the share of the total variance each component explains. Centred;
    each component's sign is fixed so its largest loading is positive, so the map is the same on
    every run."""
    if features.dim() != 2 or min(features.shape) < 2:
        raise ValueError(f"need at least 2 images and 2 features, got {tuple(features.shape)}")
    centred = features.detach().to("cpu", torch.float64)
    centred = centred - centred.mean(dim=0)
    _, singular, vh = torch.linalg.svd(centred, full_matrices=False)
    variance = singular**2
    total = float(variance.sum())
    if total <= 0.0:
        raise ValueError("the features do not vary")
    components = vh[:2].clone()
    for row in components:
        if row[row.abs().argmax()] < 0:
            row.neg_()
    explained = (float(variance[0]) / total, float(variance[1]) / total)
    return centred @ components.T, explained


@dataclass(frozen=True, eq=False)
class Projection:
    """One checkpoint's images of every split given, in split then record order, on two
    principal components, with each image's label, camera location and camera group."""

    coords: Tensor
    explained: tuple[float, float]
    labels: tuple[int, ...]
    locations: tuple[int, ...]
    groups: tuple[str, ...]

    @property
    def cameras(self) -> tuple[tuple[str, int], ...]:
        """(camera group, location) per image."""
        return tuple(zip(self.groups, self.locations, strict=True))


def project(runs: Sequence[SplitPredictions]) -> Projection:
    """PCA over the features of all `runs` together, so cis and trans images share one map."""
    if not runs:
        raise ValueError("no split to project")
    missing = [run.split for run in runs if run.features is None]
    if missing:
        raise ValueError(f"no features kept for {', '.join(missing)}")
    coords, explained = pca_2d(
        torch.cat([run.features for run in runs if run.features is not None])
    )
    return Projection(
        coords=coords,
        explained=explained,
        labels=tuple(y for run in runs for y in run.labels),
        locations=tuple(loc for run in runs for loc in run.locations),
        groups=tuple(run.group for run in runs for _ in run.labels),
    )


Panel = tuple[str, str, Sequence[bool]]  # title, highlight colour, which images to highlight


def _panels_figure(
    blocks: Mapping[str, tuple[Projection, Sequence[Panel]]], title: str, path: Path
) -> None:
    """One block of small panels per checkpoint, side by side: every image in grey, the panel's
    images on top in its colour; axes shared within a block."""
    most = max(len(panels) for _, panels in blocks.values())
    columns = min(PANEL_COLUMNS, most)
    rows = math.ceil(most / columns)
    figure = Figure(figsize=(2.25 * columns * len(blocks), 2.2 * rows + 0.9), layout="constrained")
    figure.suptitle(title)
    subfigures: Sequence[SubFigure] = figure.subfigures(1, len(blocks), squeeze=False)[0]
    for subfigure, (label, (projection, panels)) in zip(subfigures, blocks.items(), strict=True):
        first, second = projection.explained
        subfigure.suptitle(f"{label}: PC1 {first:.0%}, PC2 {second:.0%} of the variance")
        grid = subfigure.subplots(rows, columns, sharex=True, sharey=True, squeeze=False)
        x = projection.coords[:, 0].numpy()
        y = projection.coords[:, 1].numpy()
        axes: list[Axes] = [ax for line in grid for ax in line]
        for ax, (panel_title, colour, chosen) in zip(axes, panels, strict=False):
            ax.plot(x, y, linestyle="none", marker=".", markersize=1.5, color=BACKGROUND_COLOUR)
            mask = torch.tensor(list(chosen), dtype=torch.bool).numpy()
            ax.plot(x[mask], y[mask], linestyle="none", marker=".", markersize=2.5, color=colour)
            ax.set_title(f"{panel_title} (n={int(mask.sum())})", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        for ax in axes[len(panels) :]:
            ax.set_axis_off()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)


def write_feature_maps(projections: Mapping[str, Projection], out_dir: Path) -> tuple[Path, Path]:
    """`features_by_species.png` (a panel per class) and `features_by_camera.png` (a panel per
    camera location, cis cameras first, blue for cis and orange for trans) under `out_dir`, one
    block per checkpoint label; returns both paths."""
    if not projections:
        raise ValueError("no projection to plot")
    every = {camera for p in projections.values() for camera in p.cameras}
    cameras = sorted(
        every,
        key=lambda camera: (camera[0] != "cis", camera[1]),
    )
    by_species = {
        label: (
            p,
            [
                (name, SPECIES_COLOUR, [y == index for y in p.labels])
                for index, name in enumerate(CLASSES)
            ],
        )
        for label, p in projections.items()
    }
    by_camera = {
        label: (
            p,
            [
                (
                    f"location {loc}, {group}",
                    GROUP_COLOURS[group],
                    [camera == (group, loc) for camera in p.cameras],
                )
                for group, loc in cameras
            ],
        )
        for label, p in projections.items()
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    species_path, camera_path = out_dir / SPECIES_FIGURE, out_dir / CAMERA_FIGURE
    _panels_figure(by_species, "Backbone features, PCA to two dimensions, by species", species_path)
    _panels_figure(by_camera, "Backbone features, PCA to two dimensions, by camera", camera_path)
    return species_path, camera_path


def checkpoint_label(config: Config) -> str:
    """`small_cnn`, or the model and its arm, e.g. `resnet18 arm A`."""
    return config.model if config.model == "small_cnn" else f"{config.model} arm {config.arm}"


def model_label(name: str, config: Config) -> str:
    """The command line's label of config `name`: `checkpoint_label` for a phase-1 config, the
    config name for a phase-2 one, whose model and arm several phase-2 configs share."""
    return name if is_phase2(name) else checkpoint_label(config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m trailcam.evaluate.slices",
        description="Day/night and common/rare slices and the feature map on val or trans_val.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--checkpoint",
        nargs=2,
        action="append",
        metavar=("CONFIG_NAME", "PATH"),
        help=(
            "a config name and a checkpoint written by train.py; repeat for each model compared "
            "(arm A and arm B; in phase 2, p2_baseline and a variant)"
        ),
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="the config.yaml path")
    parser.add_argument("--data-root", type=Path, help="the 512 px cache")
    parser.add_argument(
        "--split",
        action="append",
        choices=CLI_SPLITS,
        help="val or trans_val (both by default); test splits: item 9.01 only",
    )
    parser.add_argument("--out", type=Path, required=True, help="where the table and PNGs go")
    parser.add_argument("--dry-run", type=parse_bool, default=False, metavar="{True,False}")
    return parser


def _load(name: str, path: Path, parser: argparse.ArgumentParser) -> Config:
    try:
        return load_config(name, path)
    except ConfigError as error:
        parser.error(str(error))


def resolve(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> list[tuple[str, Config, Path | None]]:
    """(label, config, checkpoint) per model: two random-weight models for the dry run."""
    if args.dry_run:
        if args.checkpoint or args.data_root is not None:
            parser.error("the dry run uses random weights and its own fixture")
        config = _load(DRY_RUN, args.config, parser)
        return [(f"random weights {n}", config, None) for n in (1, 2)]
    if not args.checkpoint or args.data_root is None:
        parser.error("give --checkpoint (once per model) and --data-root, or --dry-run=True")
    models: list[tuple[str, Config, Path | None]] = []
    for name, path in args.checkpoint:
        config = _load(name, args.config, parser)
        if config.fixture:
            parser.error(f"config {name} has fixture: true")
        models.append((model_label(name, config), config, Path(path)))
    phases = {is_phase2(name) for name, _ in args.checkpoint}
    if len(phases) != 1:
        parser.error(
            "phase-1 and phase-2 checkpoints in one run: phase 2 compares each variant with "
            "p2_baseline (docs/phase2-plan.md section 3)"
        )
    labels = [label for label, _, _ in models]
    if len(set(labels)) != len(labels):
        parser.error(f"one checkpoint per model and arm (in phase 2, per config), got {labels}")
    if len({config.split_seed for _, config, _ in models}) != 1:
        parser.error("the checkpoints' configs have different split seeds")
    return models


def run_splits(
    model: nn.Module,
    records: Mapping[str, tuple[Record, ...]],
    image_root: Path,
    config: Config,
) -> list[SplitPredictions]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return [
        predict_split(
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
    models = resolve(args, parser)
    splits = list(dict.fromkeys(args.split or CLI_SPLITS))
    if not set(splits) <= set(CLI_SPLITS):  # argparse's choices already refuse them
        parser.error(f"--split takes {CLI_SPLITS} only")

    results: dict[str, list[SplitPredictions]] = {}
    if args.dry_run:
        with tempfile.TemporaryDirectory(prefix="trailcam-slices-") as tmp:
            root = Path(tmp) / "fixture"
            fixture = write_synthetic_fixture(root)
            meta = load_metadata(root, fixture.paths["sidecar"])
            split = make_split(meta, models[0][1].split_seed)
            records = {s: split.records[s] for s in splits}
            for offset, (label, config, _) in enumerate(models):
                seed_everything(config.run_seed + offset)
                model = build_model(config.model, pretrained=False)
                results[label] = run_splits(model, records, fixture.image_root, config)
        header = [f"Model {models[0][1].model}, random weights (dry run on the synthetic fixture)."]
    else:
        root = args.data_root
        meta = load_metadata(root / CACHE_ANNOTATIONS, root / DAY_NIGHT_SIDECAR)
        split = make_split(meta, models[0][1].split_seed)
        records = {s: split.records[s] for s in splits}
        header = [f"Split seed {models[0][1].split_seed}, split hash {split.hash}.", ""]
        del split, meta  # nothing past this line can reach cis_test or trans_test
        for label, config, checkpoint in models:
            assert checkpoint is not None
            note = size_note(config)
            header.append(f"- {label}{note}: checkpoint sha256 {file_sha256(checkpoint)}")
            # Each checkpoint at its own config's input size (item 13.05).
            model = load_checkpoint(checkpoint, config, eval_size=config.input_size)
            results[label] = run_splits(model, records, root / CACHE_IMAGES, config)

    table = write_report(results, args.out, header)
    projections = {label: project(runs) for label, runs in results.items()}
    species, camera = write_feature_maps(projections, args.out)
    print(table.read_text(encoding="utf-8"))
    print(f"wrote {table.name}, {species.name} and {camera.name} under {args.out}")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

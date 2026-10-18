"""The evaluation entry point: one checkpoint over val or trans_val (plan.md section 7; R03, R04,
R17, R32).

One pass of the model over each split given, through the eval transform in eval mode, gives the
predictions; everything else is computed from them with `trailcam/evaluate/metrics.py`:

- `metrics.md`: per split, accuracy beside macro-F1 (R32, macro-F1 over the classes present),
  the per-class precision, recall and F1, the per-camera accuracy and macro-F1 (one row per
  camera location), and the confusion matrix (rows true, columns predicted, in the class order
  of `trailcam/data/classes.py`).
- Three PNGs per split, from `trailcam/evaluate/figures.py`: `<split>_confusion_matrix.png`,
  `<split>_per_class.png` and `<split>_per_camera.png`.
- With `--overlay`, one Grad-CAM overlay per photo (`trailcam/evaluate/gradcam.py`): photos
  under the `assets/` folder only (R04, R35), written as `gradcam_overlay_<n>.png` in the order
  given, so no file name, which may be an image id, reaches the output (R03).

Everything goes under `--out`. Repo figures belong in `docs/figures/` (40 files, 20 MB), which
the user fills from a real run (item 8.06); tests write to a temporary directory. Output is
counts, scores and camera location numbers only, never an image id or file name (R03).

**Test-set isolation (R17).** The command line accepts `val` and `trans_val` only and reads no
other split's records; cis_test and trans_test are scored once, at the end, by the final scoring
(`trailcam/evaluate/score_test.py`, item 9.01), which calls `write_report` and `write_overlays`
directly, on the one pass of `slices.predict_split` rather than a second one through `predict`.

**Other evaluation outputs.** Attention-in-box and the animal-removed test have their own entry
point (`python -m trailcam.evaluate.attention`, item 8.03), and so do the day/night and
common/rare slices with the feature map (`python -m trailcam.evaluate.slices`, item 8.04), which
takes arm A and arm B together. This module does not run them: each writes its own table and
figures under its own `--out`, and item 8.06 runs all three. A caller holding 8.04's
`SplitPredictions` can score it here with `SplitResult(run.split, run.labels, run.predictions,
run.locations)`, without a second pass.

    python -m trailcam.evaluate.evaluate --config resnet18_arm_b --checkpoint <run>.pt \\
        --data-root <cache> --split val --split trans_val --out <dir> \\
        --overlay assets/<photo>.jpg=coyote --overlay assets/<photo>.jpg
    python -m trailcam.evaluate.evaluate --config dryrun --split val --out <dir>

`--config` takes a config name of `config.yaml` (`--config-file` is that file's path, for a run from
a wheel on Kaggle). The checkpoint is scored, and its overlays drawn, at the config's `input_size`
(item 13.05): 224 for every phase-1 config, 320 for `p2_res320`; `load_checkpoint` refuses a
checkpoint trained at another size than the config's. A config with `fixture: true` (the `dryrun`
config) writes the synthetic fixture into a temporary directory and uses random weights; any other
needs `--checkpoint` and `--data-root`. An `--overlay` value is a photo path, optionally followed by
`=<class>` to title the overlay correct or wrong; `--assets` is the assets folder, the repo's
`assets/` by default.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from trailcam.data.classes import CLASS_TO_INDEX, CLASSES, NUM_CLASSES
from trailcam.data.dataset import INPUT_SIZE, CCT20Dataset, EvalTransform
from trailcam.data.metadata import DAY_NIGHT_SIDECAR, Record, load_metadata
from trailcam.data.split import make_split
from trailcam.data.synthetic import write_synthetic_fixture
from trailcam.evaluate.attention import CAMERA_GROUPS, CLI_SPLITS, load_checkpoint, size_note
from trailcam.evaluate.figures import write_confusion_matrix, write_per_camera, write_per_class
from trailcam.evaluate.gradcam import ASSETS_DIR, assets_image, write_overlay
from trailcam.evaluate.metrics import (
    accuracy,
    confusion_matrix,
    macro_f1,
    per_camera_scores,
    per_class_scores,
)
from trailcam.model.factory import build_model
from trailcam.train.config import CONFIG_PATH, Config, ConfigError, load_config
from trailcam.train.train import CACHE_ANNOTATIONS, CACHE_IMAGES, file_sha256, seed_everything

METRICS_NAME: Final = "metrics.md"
FIGURES: Final = ("confusion_matrix", "per_class", "per_camera")
OVERLAY_NAME: Final = "gradcam_overlay_{:02d}.png"


@dataclass(frozen=True)
class SplitResult:
    """One split's predictions, in record order: true class, predicted class and camera
    location per image."""

    split: str
    labels: tuple[int, ...]
    predictions: tuple[int, ...]
    locations: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.split not in CAMERA_GROUPS:
            raise ValueError(f"split must be one of {tuple(CAMERA_GROUPS)}, got {self.split!r}")
        if not self.labels:
            raise ValueError(f"no images in {self.split}")
        if not len(self.predictions) == len(self.locations) == len(self.labels):
            raise ValueError(f"{self.split}: labels, predictions and locations differ in length")

    @property
    def group(self) -> str:
        return CAMERA_GROUPS[self.split]


def predict(
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
) -> SplitResult:
    """The top-1 predictions of `model` for the records of `split`, a key of `CAMERA_GROUPS`,
    through the eval transform of side `size` (224; a phase-2 config's `input_size` from the
    command line) in eval mode. Every record must belong to `split`. This function scores any
    split it is given; keeping cis_test and trans_test for the final scoring is the caller's duty
    (the command line refuses them)."""
    if split not in CAMERA_GROUPS:
        raise ValueError(f"split must be one of {tuple(CAMERA_GROUPS)}, got {split!r}")
    others = sum(1 for r in records if r.split != split)
    if others:
        raise ValueError(f"{others} of {len(records)} records are not from {split}")
    loader: DataLoader[tuple[Tensor, int]] = DataLoader(
        CCT20Dataset(records, image_root, EvalTransform(size)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    predictions: list[int] = []
    labels: list[int] = []
    model.eval()
    with torch.inference_mode():
        for images, targets in loader:
            predictions += model(images.to(device)).argmax(dim=1).tolist()
            labels += targets.tolist()
    return SplitResult(split, tuple(labels), tuple(predictions), tuple(r.location for r in records))


def _table(header: Sequence[str], lines: Sequence[Sequence[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + " --- |" * len(header),
        *("| " + " | ".join(line) + " |" for line in lines),
    ]


def _split_section(result: SplitResult) -> list[str]:
    p, y = result.predictions, result.labels
    classes = per_class_scores(p, y)
    cameras = per_camera_scores(p, y, result.locations)
    matrix = confusion_matrix(p, y)
    absent = [CLASSES[c] for c in range(NUM_CLASSES) if c not in classes]
    return [
        f"## {result.split} ({result.group} cameras)",
        "",
        f"Classes absent from {result.split}, left out of macro-F1: "
        + (", ".join(absent) if absent else "none")
        + ".",
        "",
        "### Per class",
        "",
        *_table(
            ("class", "images", "predicted", "precision", "recall", "F1"),
            [
                (
                    CLASSES[c],
                    str(s.support),
                    str(s.predicted),
                    f"{s.precision:.4f}",
                    f"{s.recall:.4f}",
                    f"{s.f1:.4f}",
                )
                for c, s in classes.items()
            ],
        ),
        "",
        "### Per camera",
        "",
        *_table(
            ("camera location", "images", "accuracy", "macro-F1"),
            [
                (str(location), str(s.images), f"{s.accuracy:.4f}", f"{s.macro_f1:.4f}")
                for location, s in cameras.items()
            ],
        ),
        "",
        "### Confusion matrix",
        "",
        "Rows are the true class, columns the predicted class.",
        "",
        *_table(
            ("true \\ predicted", *CLASSES),
            [(CLASSES[c], *(str(n) for n in matrix[c])) for c in range(NUM_CLASSES)],
        ),
        "",
    ]


def render_metrics(results: Sequence[SplitResult], header: Sequence[str] = ()) -> str:
    """The markdown metrics table: accuracy beside macro-F1 per split, then each split's
    per-class, per-camera and confusion-matrix tables. `header` lines (the model, checkpoint
    and split) go first."""
    if not results:
        raise ValueError("no split to report")
    lines = [
        "# Metrics",
        "",
        *header,
        *([""] if header else []),
        "Macro-F1 averages the per-class F1 over the classes present in the split. Chance "
        f"accuracy is 1/{NUM_CLASSES} = {1 / NUM_CLASSES:.4f}.",
        "",
        *_table(
            ("split", "camera group", "images", "classes present", "accuracy", "macro-F1"),
            [
                (
                    r.split,
                    r.group,
                    str(len(r.labels)),
                    str(len(set(r.labels))),
                    f"{accuracy(r.predictions, r.labels):.4f}",
                    f"{macro_f1(r.predictions, r.labels):.4f}",
                )
                for r in results
            ],
        ),
        "",
    ]
    for result in results:
        lines += _split_section(result)
    return "\n".join(lines)


def figure_paths(out_dir: Path, split: str) -> tuple[Path, Path, Path]:
    """Where the confusion matrix, per-class and per-camera PNGs of `split` go."""
    first, second, third = (out_dir / f"{split}_{name}.png" for name in FIGURES)
    return first, second, third


def write_figures(result: SplitResult, out_dir: Path) -> tuple[Path, Path, Path]:
    """The three PNGs of one split under `out_dir`."""
    matrix_path, class_path, camera_path = figure_paths(out_dir, result.split)
    p, y = result.predictions, result.labels
    name = f"{result.split} ({result.group} cameras)"
    return (
        write_confusion_matrix(confusion_matrix(p, y), matrix_path, f"Confusion matrix, {name}"),
        write_per_class(per_class_scores(p, y), class_path, f"Per class, {name}"),
        write_per_camera(
            per_camera_scores(p, y, result.locations),
            accuracy(p, y),
            camera_path,
            f"Per camera, {name}",
        ),
    )


def write_report(
    results: Sequence[SplitResult], out_dir: Path, header: Sequence[str] = ()
) -> tuple[Path, list[Path]]:
    """`metrics.md` and three PNGs per split under `out_dir`; returns the table's path and the
    PNGs' paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    table = out_dir / METRICS_NAME
    table.write_text(render_metrics(results, header), encoding="utf-8")
    figures = [path for result in results for path in write_figures(result, out_dir)]
    return table, figures


def parse_overlay(text: str) -> tuple[Path, int | None]:
    """`<photo path>` or `<photo path>=<class name>`; whatever follows the last `=` must be a
    class name."""
    path, separator, name = text.rpartition("=")
    if not separator:
        return Path(text), None
    if name not in CLASS_TO_INDEX:
        raise argparse.ArgumentTypeError(f"unknown class {name!r}; one of {', '.join(CLASSES)}")
    return Path(path), CLASS_TO_INDEX[name]


def write_overlays(
    model: nn.Module,
    photos: Sequence[tuple[Path, int | None]],
    out_dir: Path,
    assets_root: Path = ASSETS_DIR,
    *,
    size: int = INPUT_SIZE,
) -> list[Path]:
    """One Grad-CAM overlay per photo, numbered in the order given, through the eval transform
    of side `size`; every photo is checked to lie under `assets_root` before the first is
    drawn."""
    for path, _ in photos:
        assets_image(path, assets_root)
    written = []
    for n, (path, label) in enumerate(photos, start=1):
        out_path = out_dir / OVERLAY_NAME.format(n)
        write_overlay(model, path, out_path, label, assets_root, size=size)
        written.append(out_path)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m trailcam.evaluate.evaluate",
        description="Metrics table and figures of one checkpoint on val or trans_val.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--config",
        "--config-name",
        dest="config_name",
        required=True,
        help="a config of config.yaml: the checkpoint's, or dryrun for the fixture",
    )
    parser.add_argument(
        "--config-file", type=Path, default=CONFIG_PATH, help="the config.yaml path"
    )
    parser.add_argument("--checkpoint", type=Path, help="a checkpoint written by train.py")
    parser.add_argument("--data-root", type=Path, help="the 512 px cache")
    parser.add_argument(
        "--split",
        action="append",
        choices=CLI_SPLITS,
        help="val (the default) or trans_val; repeat for both. Test splits: item 9.01 only",
    )
    parser.add_argument("--out", type=Path, required=True, help="where the table and PNGs go")
    parser.add_argument(
        "--overlay",
        action="append",
        type=parse_overlay,
        default=[],
        metavar="PHOTO[=CLASS]",
        help="a photo under the assets folder for a Grad-CAM overlay; repeat for more",
    )
    parser.add_argument(
        "--assets", type=Path, default=ASSETS_DIR, help="the assets folder (the repo's)"
    )
    return parser


def resolve(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Config:
    try:
        config = load_config(args.config_name, args.config_file)
    except ConfigError as error:
        parser.error(str(error))
    if config.fixture:
        if args.checkpoint is not None or args.data_root is not None:
            parser.error(f"config {args.config_name} uses random weights and its own fixture")
    elif args.checkpoint is None or args.data_root is None:
        parser.error(f"config {args.config_name} needs --checkpoint and --data-root")
    try:
        for path, _ in args.overlay:
            assets_image(path, args.assets)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    return config


def run_splits(
    model: nn.Module,
    records: dict[str, tuple[Record, ...]],
    image_root: Path,
    config: Config,
) -> list[SplitResult]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return [
        predict(
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
    config = resolve(args, parser)
    name = args.config_name
    splits = list(dict.fromkeys(args.split or ["val"]))
    if not set(splits) <= set(CLI_SPLITS):  # argparse's choices already refuse them
        parser.error(f"--split takes {CLI_SPLITS} only")

    if config.fixture:
        with tempfile.TemporaryDirectory(prefix="trailcam-evaluate-") as tmp:
            root = Path(tmp) / "fixture"
            fixture = write_synthetic_fixture(root)
            meta = load_metadata(root, fixture.paths["sidecar"])
            split = make_split(meta, config.split_seed)
            seed_everything(config.run_seed)
            model = build_model(config.model, pretrained=False)
            records = {s: split.records[s] for s in splits}
            results = run_splits(model, records, fixture.image_root, config)
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
        results = run_splits(model, records, root / CACHE_IMAGES, config)

    table, figures = write_report(results, args.out, header)
    overlays = write_overlays(model, args.overlay, args.out, args.assets, size=config.input_size)
    print(table.read_text(encoding="utf-8"))
    written = [table.name, *(p.name for p in figures), *(p.name for p in overlays)]
    print(f"wrote {', '.join(written)} under {args.out}")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""The live sort: copy a folder of photos into one folder per class (plan.md section 8; item 10.02).

For the live demo of about 200 trans-test photos. Every photo under `--src` (`.jpg`, `.jpeg` or
`.png`, at any depth) is classified and copied, never moved, into `--dst/<class>/`, one folder
per class of `trailcam/data/classes.py`. The model is loaded once with `load_model` of
`trailcam/demo/predict.py` (the one loader, `load_checkpoint`) and warmed up once with one
untimed forward pass; then each photo costs one forward pass. Each photo goes through the eval
transform (`EvalTransform`: shorter side 256, centre crop 224, normalise), and the top class is
the argmax of the logits, the same class `predict` ranks first.

Output: counts and seconds only. A line per photo gives its number, the running seconds, the
predicted class and that class's count so far; the end gives the count per class, the total and
the seconds. File names and paths of the photos are never printed (R03), nor is any accuracy: the
script reads no labels and no metadata, only the photos, so it cannot score anything. The test
sets are scored once, by items 9.01 and 9.02 (R17). A photo Pillow cannot read is counted as
unreadable and skipped.

`--src` is refused when it is inside a git repository, this one and its `data/` included: the
sort reads a folder of demo photos gathered outside the repo, never the dataset itself (R35).
`--dst` is refused when it is inside a git repository, so sorted photos can never be committed
(R01); when it is `--src` or inside it, so the sort cannot feed itself; and when it exists and is
not empty, so a rehearsal cannot mix with the run before.

    python -m trailcam.demo.sort --src <photos> --dst <outside the repo> --checkpoint <run>.pt

Without `--checkpoint` the config's model is built with random weights and the script says so:
a check of the pipeline, not a demo.
"""

from __future__ import annotations

import argparse
import shutil
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import torch
from PIL import Image, UnidentifiedImageError
from torch import nn

from trailcam.data.classes import CLASSES
from trailcam.data.dataset import INPUT_SIZE, EvalTransform
from trailcam.demo.predict import DEFAULT_CONFIG, PHOTO_SUFFIXES, load_model
from trailcam.model.factory import build_model
from trailcam.train.config import CONFIG_PATH, load_config

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
UNREADABLE: Final = "unreadable"


class SourceError(ValueError):
    """`--src` is inside a git repository."""


class DestinationError(ValueError):
    """`--dst` is inside a git repository, is `--src` or inside it, or is not empty."""


def _in_git_repository(path: Path) -> bool:
    """True when `path`, already resolved, is this repo or any git repository, or inside one."""
    return any(
        folder == REPO_ROOT or (folder / ".git").exists() for folder in (path, *path.parents)
    )


def check_source(src: Path) -> None:
    """Raise `SourceError` when `src` is inside a git repository (R35)."""
    if _in_git_repository(src.resolve()):
        raise SourceError("--src is inside a git repository; gather the photos outside it")


def check_destination(src: Path, dst: Path) -> None:
    """Raise `DestinationError` unless `dst` is a safe place for the sorted copies."""
    src, dst = src.resolve(), dst.resolve()
    if _in_git_repository(dst):
        raise DestinationError("--dst is inside a git repository; choose a folder outside it")
    if dst == src or src in dst.parents:
        raise DestinationError("--dst is --src or inside it")
    if dst.exists() and (not dst.is_dir() or any(dst.iterdir())):
        raise DestinationError("--dst exists and is not an empty folder")


def find_photos(src: Path) -> list[Path]:
    """The photos under `src`, at any depth, in a fixed order; hidden files are skipped."""
    return sorted(
        path
        for path in src.rglob("*")
        if path.is_file()
        and path.suffix.lower() in PHOTO_SUFFIXES
        and not any(part.startswith(".") for part in path.relative_to(src).parts)
    )


def warm_up(model: nn.Module) -> None:
    """One untimed forward pass on a blank input, so the first photo is not the slow one."""
    with torch.no_grad():
        model(torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE))


def top_class(model: nn.Module, photo: Image.Image) -> str:
    """The class of the largest logit for one photo: one forward pass."""
    image = EvalTransform()(photo.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = model(image)
    return CLASSES[int(logits[0].argmax())]


def _free_name(folder: Path, name: str) -> Path:
    """`folder / name`, or `name` with `_1`, `_2`, ... before its suffix if that is taken."""
    target = folder / name
    stem, suffix = target.stem, target.suffix
    number = 1
    while target.exists():
        target = folder / f"{stem}_{number}{suffix}"
        number += 1
    return target


def sort_photos(model: nn.Module, src: Path, dst: Path) -> Counter[str]:
    """Copy each photo under `src` into `dst/<class>/`, printing a line per photo; return the
    count per class (and of unreadable photos)."""
    check_source(src)
    check_destination(src, dst)
    photos = find_photos(src)
    for name in CLASSES:
        (dst / name).mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    width = len(str(len(photos)))
    start = time.perf_counter()
    for number, path in enumerate(photos, start=1):
        try:
            with Image.open(path) as photo:
                label = top_class(model, photo)
        except (UnidentifiedImageError, OSError):
            label = UNREADABLE
        else:
            shutil.copy2(path, _free_name(dst / label, path.name))
        counts[label] += 1
        seconds = time.perf_counter() - start
        print(f"{number:>{width}}/{len(photos)}  {seconds:7.2f} s  {label}: {counts[label]}")
    seconds = time.perf_counter() - start
    print(summary(counts, seconds))
    return counts


def summary(counts: Counter[str], seconds: float) -> str:
    """The count per class, the unreadable photos if any, the total and the seconds."""
    width = max(len(name) for name in (*CLASSES, UNREADABLE))
    lines = [f"{name:<{width}}  {counts[name]:>5}" for name in CLASSES]
    if counts[UNREADABLE]:
        lines.append(f"{UNREADABLE:<{width}}  {counts[UNREADABLE]:>5}")
    total = sum(counts.values())
    rate = total / seconds if seconds > 0 else 0.0
    lines.append(f"{total} photos in {seconds:.2f} s ({rate:.1f} photos per second)")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m trailcam.demo.sort", description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="the folder of photos")
    parser.add_argument("--dst", type=Path, required=True, help="a new folder outside the repo")
    parser.add_argument("--checkpoint", type=Path, help="a train.py checkpoint")
    parser.add_argument("--config-name", default=DEFAULT_CONFIG, help="the checkpoint's config")
    parser.add_argument("--config-path", type=Path, default=CONFIG_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.src.is_dir():
        parser.error("--src is not a folder")
    try:
        check_source(args.src)
        check_destination(args.src, args.dst)
    except (SourceError, DestinationError) as error:
        parser.error(str(error))
    start = time.perf_counter()
    if args.checkpoint is None:
        print("no --checkpoint: random weights, a pipeline check, not a demo")
        model = build_model(load_config(args.config_name, args.config_path).model, pretrained=False)
        model.eval()
    else:
        model = load_model(args.checkpoint, args.config_name, args.config_path)
    warm_up(model)
    print(f"model loaded and warmed up in {time.perf_counter() - start:.2f} s")
    sort_photos(model, args.src, args.dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

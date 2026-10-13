"""trailcam/data/synthetic.py, the images part (item 4.03; R03 counts only, R37 the fixture is
written only into the directory it is given).

The fixture is written to tmp_path and never committed. Expected values are read back from the
written annotation files and sidecar, so the images are checked against the metadata on disk, not
against the writer's own state. The metadata part is tested in tests/test_metadata.py.
"""

import json
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image

from trailcam.data import synthetic
from trailcam.data.dataset import INPUT_SIZE, CCT20Dataset, EvalTransform
from trailcam.data.metadata import DAY_NIGHT_SIDECAR, SPLITS, annotation_file_name, load_metadata
from trailcam.data.split import SPLIT_NAMES, make_split

REPO = Path(__file__).resolve().parents[1]
SEED = 42  # the split seed of the data card


@pytest.fixture(scope="module")
def written(tmp_path_factory: pytest.TempPathFactory) -> synthetic.Fixture:
    return synthetic.write_synthetic_fixture(tmp_path_factory.mktemp("fixture"))


def images_with_anns(fixture: synthetic.Fixture) -> Iterator[tuple[str, dict[str, Any], list[Any]]]:
    """(split, image entry, its annotations) for every image of the five written files."""
    for split in SPLITS:
        data = json.loads(fixture.paths[split].read_text())
        anns: dict[str, list[Any]] = {image["id"]: [] for image in data["images"]}
        for ann in data["annotations"]:
            anns[ann["image_id"]].append(ann)
        for image in data["images"]:
            yield split, image, anns[image["id"]]


def night_flags(fixture: synthetic.Fixture) -> dict[str, bool]:
    raw = json.loads(fixture.paths["sidecar"].read_text())
    return {image_id: entry["night"] for image_id, entry in raw.items()}


def pixel_boxes(anns: list[Any]) -> list[tuple[int, int, int, int]]:
    """Inclusive pixel corners (x0, y0, x1, y1) of the boxed annotations."""
    corners = []
    for ann in anns:
        if "bbox" in ann:
            x, y, w, h = (int(v) for v in ann["bbox"])
            corners.append((x, y, x + w - 1, y + h - 1))
    return corners


def ring(x0: int, y0: int, x1: int, y1: int) -> set[tuple[int, int]]:
    """The pixels on the edge of the inclusive rectangle."""
    return {(x, y) for x in range(x0, x1 + 1) for y in (y0, y1)} | {
        (x, y) for y in range(y0, y1 + 1) for x in (x0, x1)
    }


def inside(point: tuple[int, int], box: tuple[int, int, int, int]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


# What is written, and where (R37)


def test_writes_only_into_the_given_directory(tmp_path: Path) -> None:
    out = tmp_path / "new" / "fixture"
    fixture = synthetic.write_synthetic_fixture(out)
    assert [p.name for p in tmp_path.iterdir()] == ["new"]
    assert [p.name for p in (tmp_path / "new").iterdir()] == ["fixture"]
    expected = {annotation_file_name(s) for s in SPLITS} | {DAY_NIGHT_SIDECAR, synthetic.IMAGE_DIR}
    assert {p.name for p in out.iterdir()} == expected
    assert fixture.image_root == out / synthetic.IMAGE_DIR
    assert set(fixture.paths) == {*SPLITS, "sidecar"}
    assert all(p.parent == out for p in fixture.paths.values())
    images = list(fixture.image_root.iterdir())
    assert all(p.is_file() and p.suffix == ".jpg" for p in images)
    names = [image["file_name"] for _, image, _ in images_with_anns(fixture)]
    assert len(set(names)) == len(names)
    assert sorted(p.name for p in images) == sorted(names)


def test_refuses_a_non_empty_directory(tmp_path: Path) -> None:
    (tmp_path / "keep.txt").write_text("not the fixture's")
    with pytest.raises(FileExistsError, match="new or empty directory"):
        synthetic.write_synthetic_fixture(tmp_path)
    assert [p.name for p in tmp_path.iterdir()] == ["keep.txt"]


def test_an_empty_existing_directory_is_accepted(tmp_path: Path) -> None:
    fixture = synthetic.write_synthetic_fixture(tmp_path)
    assert fixture.counts.images == len(list(fixture.image_root.iterdir()))


# The images match the metadata


def test_one_jpeg_per_image_at_its_size_grayscale_at_night(written: synthetic.Fixture) -> None:
    night = night_flags(written)
    per_split = dict.fromkeys(SPLITS, 0)
    n_night = 0
    for split, image, _ in images_with_anns(written):
        with Image.open(written.image_root / image["file_name"]) as jpeg:
            assert jpeg.format == "JPEG"
            assert jpeg.size == (image["width"], image["height"])
            assert jpeg.mode == ("L" if night[image["id"]] else "RGB")
        per_split[split] += 1
        n_night += night[image["id"]]
    assert 0 < n_night < sum(per_split.values())
    assert dict(written.counts.per_split) == per_split
    assert written.counts.night == n_night


def test_a_rectangle_is_drawn_exactly_where_each_box_is(written: synthetic.Fixture) -> None:
    """Every box's edge is the bright outline and the pixels just outside it are background: a
    box drawn one pixel off, scaled, transposed or on another image fails. An image without a box
    has no bright pixels."""
    boxed = rectangles = 0
    for _, image, anns in images_with_anns(written):
        with Image.open(written.image_root / image["file_name"]) as jpeg:
            gray = jpeg.convert("L")
        boxes = pixel_boxes(anns)
        boxed += bool(boxes)
        rectangles += len(boxes)
        if not boxes:
            assert gray.getextrema()[1] < 160
        for box in boxes:
            edge = [gray.getpixel(p) for p in ring(*box)]
            assert sum(edge) / len(edge) >= 220
            outside = [
                gray.getpixel(p)
                for p in ring(box[0] - 1, box[1] - 1, box[2] + 1, box[3] + 1)
                if 0 <= p[0] < gray.width and 0 <= p[1] < gray.height
                if not any(inside(p, other) for other in boxes)
            ]
            if outside:
                assert sum(outside) / len(outside) <= 140
    assert written.counts.boxed == boxed
    assert written.counts.rectangles == rectangles
    assert 0 < boxed < written.counts.images


def test_rectangle_colour_follows_the_class(written: synthetic.Fixture) -> None:
    """The fill at the centre of a box on a day image is its category's colour (within JPEG
    error), so the fixture's boxes carry their class."""
    night = night_flags(written)
    checked = 0
    for _, image, anns in images_with_anns(written):
        boxes = pixel_boxes(anns)
        if night[image["id"]] or len(boxes) != 1:
            continue
        x0, y0, x1, y1 = boxes[0]
        with Image.open(written.image_root / image["file_name"]) as jpeg:
            pixel = jpeg.getpixel(((x0 + x1) // 2, (y0 + y1) // 2))
        colour = synthetic.box_colour(next(a["category_id"] for a in anns if "bbox" in a))
        assert isinstance(pixel, tuple)
        assert all(abs(p - c) <= 12 for p, c in zip(pixel, colour, strict=True))
        checked += 1
    assert checked > 100


def test_same_seed_same_bytes(tmp_path: Path) -> None:
    a = synthetic.write_synthetic_fixture(tmp_path / "a", seed=3)
    b = synthetic.write_synthetic_fixture(tmp_path / "b", seed=3)
    c = synthetic.write_synthetic_fixture(tmp_path / "c", seed=4)
    names = sorted(p.name for p in a.image_root.iterdir())
    assert names == sorted(p.name for p in b.image_root.iterdir())
    for name in names:
        assert (a.image_root / name).read_bytes() == (b.image_root / name).read_bytes()
    for key in a.paths:
        assert a.paths[key].read_bytes() == b.paths[key].read_bytes()
    assert any(
        (a.image_root / name).read_bytes() != (c.image_root / name).read_bytes() for name in names
    )


# The fixture loads through 3.02, splits through 3.03 and reads through 4.01


def test_fixture_loads_splits_and_reads(written: synthetic.Fixture, tmp_path: Path) -> None:
    root = written.paths["train"].parent
    meta = load_metadata(root, written.paths["sidecar"])
    for split in SPLITS:
        for record in meta.records[split]:
            with Image.open(written.image_root / record.file_name) as jpeg:
                assert jpeg.mode == ("L" if record.night else "RGB")
    split = make_split(meta, SEED)  # runs the leakage and coverage checks
    # The images part leaves the metadata as the metadata part alone writes it.
    synthetic.write_synthetic_metadata(tmp_path / "meta")
    assert split.hash == make_split(load_metadata(tmp_path / "meta"), SEED).hash
    for name in SPLIT_NAMES:
        dataset = CCT20Dataset(split.records[name], written.image_root, EvalTransform())
        assert len(dataset) > 0
        for index in range(len(dataset)):
            tensor, label = dataset[index]
            assert tuple(tensor.shape) == (3, INPUT_SIZE, INPUT_SIZE)
            assert tensor.dtype == torch.float32
            assert label == split.records[name][index].label


# The CLI prints counts only (R03)


def check_counts_only(out: str, fixture_dir: Path) -> None:
    assert "syn-" not in out and ".jpg" not in out and str(fixture_dir) not in out
    assert out.startswith("Synthetic CCT20 fixture, seed 0\n")
    images = sorted((fixture_dir / synthetic.IMAGE_DIR).iterdir())
    total = re.search(r"^  total\s+(\d+)$", out, re.MULTILINE)
    assert total is not None and int(total.group(1)) == len(images)
    for split in SPLITS:
        assert re.search(rf"^  {split}\s+\d+$", out, re.MULTILINE), split


@pytest.mark.parametrize("form", ["positional", "--root"])
def test_cli_prints_counts_only(
    form: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "fixture"
    argv = [str(out_dir)] if form == "positional" else ["--root", str(out_dir)]
    assert synthetic.main(argv) == 0
    check_counts_only(capsys.readouterr().out, out_dir)


def test_cli_needs_the_directory_once(tmp_path: Path) -> None:
    for argv in ([], [str(tmp_path / "a"), "--root", str(tmp_path / "b")]):
        with pytest.raises(SystemExit) as caught:
            synthetic.main(argv)
        assert caught.value.code == 2
    assert list(tmp_path.iterdir()) == []


def test_cli_refuses_a_non_empty_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "keep.txt").write_text("not the fixture's")
    with pytest.raises(SystemExit) as caught:
        synthetic.main([str(tmp_path)])
    assert caught.value.code == 2
    assert "new or empty directory" in capsys.readouterr().err
    assert [p.name for p in tmp_path.iterdir()] == ["keep.txt"]


def test_module_runs_as_a_script(tmp_path: Path) -> None:
    out_dir = tmp_path / "fixture"
    result = subprocess.run(
        [sys.executable, "-m", "trailcam.data.synthetic", str(out_dir)],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    check_counts_only(result.stdout, out_dir)

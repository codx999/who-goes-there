"""trailcam/data/metadata.py on the synthetic metadata from trailcam/data/synthetic.py (R03, R05,
R15).

The fixture is written to tmp_path and never committed. Expected values are read back from the
written JSON, so the loader is checked against the files, not against itself. One test runs on
the real metadata when `TRAILCAM_DATA_ROOT` is set (check.sh sets it locally; CI has no data).
"""

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from trailcam.data import metadata, synthetic
from trailcam.data.classes import (
    CLASSES,
    DROPPED_CCT_IDS,
    INDEX_TO_CCT_ID,
    NUM_CLASSES,
    index_of_cct_id,
)
from trailcam.data.metadata import DROP_RULES, SPLITS, Metadata, load_metadata


@pytest.fixture
def fixture_dir(tmp_path: Path) -> Path:
    synthetic.write_synthetic_metadata(tmp_path / "meta")
    return tmp_path / "meta"


@pytest.fixture
def meta(fixture_dir: Path) -> Metadata:
    return load_metadata(fixture_dir)


def read_split(fixture_dir: Path, split: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((fixture_dir / f"{split}_annotations.json").read_text())
    return data


def categories_by_image(data: dict[str, Any]) -> dict[str, list[int]]:
    cats: dict[str, list[int]] = {image["id"]: [] for image in data["images"]}
    for ann in data["annotations"]:
        cats[ann["image_id"]].append(ann["category_id"])
    return cats


# The synthetic writer


def test_synthetic_writes_only_the_five_files_and_the_sidecar(tmp_path: Path) -> None:
    paths = synthetic.write_synthetic_metadata(tmp_path)
    written = sorted(p.name for p in tmp_path.iterdir())
    assert written == sorted([f"{s}_annotations.json" for s in SPLITS] + ["day_night.json"])
    assert set(paths) == {*SPLITS, "sidecar"}


def test_synthetic_is_deterministic(tmp_path: Path) -> None:
    a = synthetic.write_synthetic_metadata(tmp_path / "a", seed=3)
    b = synthetic.write_synthetic_metadata(tmp_path / "b", seed=3)
    c = synthetic.write_synthetic_metadata(tmp_path / "c", seed=4)
    for key in a:
        assert a[key].read_bytes() == b[key].read_bytes()
    assert any(a[s].read_bytes() != c[s].read_bytes() for s in SPLITS)


def test_synthetic_shape(fixture_dir: Path) -> None:
    images: list[dict[str, Any]] = []
    anns: list[dict[str, Any]] = []
    locations: dict[str, set[int]] = {}
    for split in SPLITS:
        data = read_split(fixture_dir, split)
        images += data["images"]
        anns += data["annotations"]
        locations[split] = {image["location"] for image in data["images"]}
    cis = locations["train"] | locations["cis_val"] | locations["cis_test"]
    trans = locations["trans_val"] | locations["trans_test"]
    assert len(cis) == 2 and len(trans) == 1 and not cis & trans
    assert {image["frame_num"] for image in images} == {1, 2, 3}
    assert set(Counter(image["seq_id"] for image in images).values()) == {3}
    animal = [a for a in anns if a["category_id"] != 30]
    boxed = sum(1 for a in animal if "bbox" in a) / len(animal)
    assert 0.85 <= boxed <= 0.95
    # Enough sequences per class and cis location in the pool to stratify a re-split.
    pool = read_split(fixture_dir, "train")["images"] + read_split(fixture_dir, "cis_val")["images"]
    pool_cats = categories_by_image(read_split(fixture_dir, "train")) | categories_by_image(
        read_split(fixture_dir, "cis_val")
    )
    seqs: dict[tuple[int, int], set[str]] = {}
    for image in pool:
        (cct_id, *_) = pool_cats[image["id"]]
        seqs.setdefault((cct_id, image["location"]), set()).add(image["seq_id"])
    for cct_id in INDEX_TO_CCT_ID:
        for location in synthetic.CIS_LOCATIONS:
            assert len(seqs[(cct_id, location)]) >= synthetic.TRAIN_SEQUENCES + 1


# Loading: fields survive


def test_every_class_in_every_split(meta: Metadata) -> None:
    for split in ("train", "cis_val", "cis_test", "trans_val", "trans_test"):
        assert {r.label for r in meta.records[split]} == set(range(NUM_CLASSES)), split


def test_fields_boxes_and_hours_survive_loading(fixture_dir: Path, meta: Metadata) -> None:
    for split in SPLITS:
        data = read_split(fixture_dir, split)
        images = {image["id"]: image for image in data["images"]}
        cats = categories_by_image(data)
        boxes: dict[str, list[list[float]]] = {}
        for ann in data["annotations"]:
            if "bbox" in ann:
                boxes.setdefault(ann["image_id"], []).append(ann["bbox"])
        for r in meta.records[split]:
            image = images[r.image_id]
            assert r.split == split
            assert r.file_name == image["file_name"]
            assert r.location == image["location"]
            assert r.seq_id == image["seq_id"]
            assert r.frame_num == image["frame_num"]
            assert r.seq_num_frames == image["seq_num_frames"]
            assert (r.width, r.height) == (image["width"], image["height"])
            assert r.hour == int(image["date_captured"][11:13])
            assert [list(b) for b in r.boxes] == boxes.get(r.image_id, [])
            assert INDEX_TO_CCT_ID[r.label] == cats[r.image_id][0]


def test_hours_cover_the_day(meta: Metadata) -> None:
    hours = {r.hour for split in SPLITS for r in meta.records[split]}
    assert hours == set(range(24))


def test_capture_hour_parses_and_rejects() -> None:
    assert metadata.capture_hour("2012-05-09 00:59:59") == 0
    assert metadata.capture_hour("2012-05-09 23:00:00") == 23
    with pytest.raises(ValueError):
        metadata.capture_hour("2012-05-09T23:00:00")


def test_unboxed_animal_images_are_kept_in_every_split(fixture_dir: Path, meta: Metadata) -> None:
    """R15: the empty filter goes by category, never by a missing box."""
    for split in SPLITS:
        data = read_split(fixture_dir, split)
        cats = categories_by_image(data)
        unboxed = {r.image_id for r in meta.records[split] if not r.boxes}
        expected = {
            image_id
            for image_id, c in cats.items()
            if len(set(c)) == 1 and c[0] not in DROPPED_CCT_IDS
        } - {a["image_id"] for a in data["annotations"] if "bbox" in a}
        assert unboxed == expected
        assert expected, f"{split} has no unboxed animal image to test with"


def test_night_flag_from_sidecar_and_absent_without_it(fixture_dir: Path, meta: Metadata) -> None:
    flags = json.loads((fixture_dir / "day_night.json").read_text())
    for split in SPLITS:
        for r in meta.records[split]:
            assert r.night is flags[r.image_id]["night"]
    assert {r.night for r in meta.records["train"]} == {True, False}
    (fixture_dir / "day_night.json").unlink()
    bare = load_metadata(fixture_dir)
    assert all(r.night is None for split in SPLITS for r in bare.records[split])


def test_explicit_sidecar_path(tmp_path: Path) -> None:
    paths = synthetic.write_synthetic_metadata(tmp_path / "meta")
    moved = tmp_path / "cache" / "day_night.json"
    moved.parent.mkdir()
    paths["sidecar"].rename(moved)
    assert all(r.night is None for r in load_metadata(tmp_path / "meta").records["train"])
    with_flags = load_metadata(tmp_path / "meta", sidecar=moved)
    assert all(r.night is not None for r in with_flags.records["train"])


# Loading: the filters


def test_filters_remove_exactly_the_right_images(fixture_dir: Path, meta: Metadata) -> None:
    """R05: drop two-species, empty and rare images; keep same-species multi-annotation images."""
    for split in SPLITS:
        cats = categories_by_image(read_split(fixture_dir, split))
        expected_kept = {}
        expected_dropped = dict.fromkeys(DROP_RULES, 0)
        for image_id, c in cats.items():
            if len(set(c)) > 1:
                expected_dropped["two_species"] += 1
            elif not c:
                expected_dropped["unannotated"] += 1
            elif c[0] == 30:
                expected_dropped["empty"] += 1
            elif c[0] in DROPPED_CCT_IDS:
                expected_dropped["rare"] += 1
            else:
                expected_kept[image_id] = index_of_cct_id(c[0])
        assert {r.image_id: r.label for r in meta.records[split]} == expected_kept, split
        assert dict(meta.dropped[split]) == expected_dropped, split


def test_planted_cases_are_counted(meta: Metadata) -> None:
    """Pinned against the synthetic layout, so a filter that silently stops firing fails here."""
    dropped = {rule: [meta.dropped[s][rule] for s in SPLITS] for rule in DROP_RULES}
    # SPLITS order: train, cis_val, trans_val, cis_test, trans_test
    assert dropped["two_species"] == [1, 0, 0, 0, 1]
    assert dropped["empty"] == [0, 3, 3, 3, 3]
    assert dropped["rare"] == [3, 0, 0, 3, 3]
    assert dropped["unannotated"] == [0, 0, 0, 0, 0]
    for split in ("train", "cis_test"):
        multi = [r for r in meta.records[split] if len(r.boxes) == 2]
        same_species = [r for r in multi if CLASSES[r.label] == "opossum"]
        assert same_species, f"{split}: the two-opossum frame was not kept with both boxes"


def test_two_species_image_keeps_neither_label(fixture_dir: Path, meta: Metadata) -> None:
    cats = categories_by_image(read_split(fixture_dir, "train"))
    (mixed,) = [i for i, c in cats.items() if len(set(c)) > 1]
    assert mixed not in {r.image_id for r in meta.records["train"]}
    # The rest of its sequence is kept.
    seq = {
        image["seq_id"]
        for image in read_split(fixture_dir, "train")["images"]
        if image["id"] == mixed
    }
    kept_frames = [r for r in meta.records["train"] if r.seq_id in seq]
    assert sorted(r.frame_num for r in kept_frames) == [1, 3]


def test_unannotated_image_is_dropped(fixture_dir: Path) -> None:
    path = fixture_dir / "cis_test_annotations.json"
    data = json.loads(path.read_text())
    target = data["images"][0]["id"]
    data["annotations"] = [a for a in data["annotations"] if a["image_id"] != target]
    path.write_text(json.dumps(data))
    meta = load_metadata(fixture_dir)
    assert meta.dropped["cis_test"]["unannotated"] == 1
    assert target not in {r.image_id for r in meta.records["cis_test"]}


# Loading: malformed input fails loudly


def mutate(fixture_dir: Path, split: str, change: Any) -> None:
    path = fixture_dir / f"{split}_annotations.json"
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


def test_unknown_category_raises(fixture_dir: Path) -> None:
    def change(data: dict[str, Any]) -> None:
        data["categories"].append({"id": 77, "name": "moose"})

    mutate(fixture_dir, "train", change)
    with pytest.raises(KeyError):
        load_metadata(fixture_dir)


def test_renamed_category_raises(fixture_dir: Path) -> None:
    def change(data: dict[str, Any]) -> None:
        data["categories"] = [
            {**c, "name": "possum"} if c["id"] == 1 else c for c in data["categories"]
        ]

    mutate(fixture_dir, "trans_test", change)
    with pytest.raises(ValueError, match="category 1"):
        load_metadata(fixture_dir)


def test_orphan_annotation_and_repeated_image_raise(fixture_dir: Path) -> None:
    def orphan(data: dict[str, Any]) -> None:
        data["annotations"].append({"id": "x", "image_id": "nowhere", "category_id": 1})

    mutate(fixture_dir, "cis_val", orphan)
    with pytest.raises(ValueError, match="annotations for 1 image"):
        load_metadata(fixture_dir)


def test_repeated_image_raises(fixture_dir: Path) -> None:
    def repeat(data: dict[str, Any]) -> None:
        data["images"].append(data["images"][0])

    mutate(fixture_dir, "trans_val", repeat)
    with pytest.raises(ValueError, match="repeated image ids"):
        load_metadata(fixture_dir)


def test_malformed_box_raises(fixture_dir: Path) -> None:
    def bad_box(data: dict[str, Any]) -> None:
        next(a for a in data["annotations"] if "bbox" in a)["bbox"] = [1.0, 2.0, 0.0, 5.0]

    mutate(fixture_dir, "train", bad_box)
    with pytest.raises(ValueError, match="bbox"):
        load_metadata(fixture_dir)


def test_missing_or_duplicate_file_raises(fixture_dir: Path) -> None:
    copy = fixture_dir / "nested" / "train_annotations.json"
    copy.parent.mkdir()
    copy.write_bytes((fixture_dir / "train_annotations.json").read_bytes())
    with pytest.raises(FileNotFoundError, match="found 2"):
        load_metadata(fixture_dir)
    copy.unlink()
    (fixture_dir / "trans_test_annotations.json").unlink()
    with pytest.raises(FileNotFoundError, match="found 0"):
        load_metadata(fixture_dir)


def test_files_are_found_in_a_nested_folder(tmp_path: Path) -> None:
    synthetic.write_synthetic_metadata(tmp_path / "a" / "b")
    assert sum(len(r) for r in load_metadata(tmp_path).records.values()) > 0


# The CLI prints counts only (R03)


def test_cli_prints_counts_and_no_ids_or_file_names(
    fixture_dir: Path, meta: Metadata, capsys: pytest.CaptureFixture[str]
) -> None:
    assert metadata.main(["--root", str(fixture_dir)]) == 0
    out = capsys.readouterr().out
    assert "syn-" not in out and ".jpg" not in out
    for name in CLASSES:
        assert re.search(rf"^{name}\s", out, re.MULTILINE), name
    for rule in DROP_RULES:
        assert re.search(rf"^{rule}\s", out, re.MULTILINE), rule
    assert "Boxed fraction" in out and "Capture hour histogram" in out
    assert re.search(r"^  23\s+\d+$", out, re.MULTILINE)
    total = sum(len(meta.records[s]) for s in SPLITS)
    assert re.search(rf"^total\s.*\s{total}$", out, re.MULTILINE)


# The real metadata, when present (local runs; CI has no data)

REAL_ROOT = os.environ.get("TRAILCAM_DATA_ROOT", "")


@pytest.mark.skipif(not REAL_ROOT, reason="TRAILCAM_DATA_ROOT is not set (CI has no data)")
def test_real_metadata_drops_match_data_inspection() -> None:
    """Counts from docs/data-inspection.md: images per file, two-species and empty images."""
    meta = load_metadata(Path(REAL_ROOT))
    images = {"train": 13553, "cis_val": 3484, "trans_val": 1725, "cis_test": 15827,
              "trans_test": 23275}  # fmt: skip
    two_species = {"train": 7, "cis_val": 0, "trans_val": 61, "cis_test": 1, "trans_test": 9}
    empty = {"train": 0, "cis_val": 995, "trans_val": 61, "cis_test": 1180, "trans_test": 1778}
    for split in SPLITS:
        dropped = meta.dropped[split]
        assert len(meta.records[split]) + sum(dropped.values()) == images[split], split
        assert dropped["two_species"] == two_species[split], split
        assert dropped["empty"] == empty[split], split
        assert dropped["unannotated"] == 0, split
    assert sum(meta.dropped[s]["two_species"] for s in SPLITS) == 78

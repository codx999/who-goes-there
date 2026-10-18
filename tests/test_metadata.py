"""trailcam/data/metadata.py on the synthetic metadata from trailcam/data/synthetic.py (R03, R05,
R15).

The fixture is written to tmp_path and never committed. Expected values are read back from the
written JSON, so the loader is checked against the files, not against itself. Two tests run on
the real metadata in `TRAILCAM_DATA_ROOT` (check.sh sets it locally; CI has no data and skips
them; anywhere else a missing root fails).
"""

import dataclasses
import json
import os
import re
import subprocess
import sys
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
from trailcam.data.metadata import DROP_RULES, SPLITS, EmptyFrame, Metadata, load_metadata


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
            # The fixture's boxes lie inside the frame, so each is its pixel box over the size.
            expected = [
                (x / r.width, y / r.height, w / r.width, h / r.height)
                for x, y, w, h in boxes.get(r.image_id, [])
            ]
            assert r.boxes == pytest.approx(expected)
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
    sidecar = fixture_dir / "day_night.json"
    flags = json.loads(sidecar.read_text())
    flagged = load_metadata(fixture_dir, sidecar=sidecar)
    for split in SPLITS:
        for r in flagged.records[split]:
            assert r.night is flags[r.image_id]["night"]
    assert {r.night for r in flagged.records["train"]} == {True, False}
    assert all(r.night is None for split in SPLITS for r in meta.records[split])


def test_explicit_sidecar_path(tmp_path: Path) -> None:
    paths = synthetic.write_synthetic_metadata(tmp_path / "meta")
    moved = tmp_path / "cache" / "day_night.json"
    moved.parent.mkdir()
    paths["sidecar"].rename(moved)
    assert all(r.night is None for r in load_metadata(tmp_path / "meta").records["train"])
    with_flags = load_metadata(tmp_path / "meta", sidecar=moved)
    assert all(r.night is not None for r in with_flags.records["train"])


def test_sidecar_next_to_the_annotations_is_ignored_without_the_flag(fixture_dir: Path) -> None:
    """Plan section 2: training passes the sidecar path; without it the flag is absent, even when a
    day_night.json sits next to the annotation files (review 3.02, both reviewers)."""
    assert (fixture_dir / "day_night.json").is_file()
    meta = load_metadata(fixture_dir)
    assert all(r.night is None for split in SPLITS for r in meta.records[split])


def test_sidecar_keyed_by_file_name_raises(fixture_dir: Path) -> None:
    """A sidecar that matches no image id fails loudly instead of leaving every flag unset."""
    sidecar = fixture_dir / "day_night.json"
    flags = json.loads(sidecar.read_text())
    sidecar.write_text(json.dumps({f"{k}.jpg": v for k, v in flags.items()}))
    with pytest.raises(ValueError, match=r"missing from the day/night sidecar"):
        load_metadata(fixture_dir, sidecar=sidecar)


def test_sidecar_missing_one_kept_image_raises(fixture_dir: Path) -> None:
    sidecar = fixture_dir / "day_night.json"
    flags = json.loads(sidecar.read_text())
    kept = load_metadata(fixture_dir).records["trans_val"][0].image_id
    del flags[kept]
    sidecar.write_text(json.dumps(flags))
    with pytest.raises(ValueError, match=r"trans_val: 1 kept images missing"):
        load_metadata(fixture_dir, sidecar=sidecar)


def test_cli_tells_a_loaded_sidecar_from_none(
    fixture_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    metadata.main(["--root", str(fixture_dir)])
    assert "Night flag: no day/night sidecar" in capsys.readouterr().out
    metadata.main(["--root", str(fixture_dir), "--sidecar", str(fixture_dir / "day_night.json")])
    out = capsys.readouterr().out
    total = sum(len(r) for r in load_metadata(fixture_dir).records.values())
    assert re.search(rf"^Night flag: {total} of {total} images flagged, \d+ night", out, re.M)


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


def test_the_empty_images_are_kept_aside_as_empty_frames(fixture_dir: Path) -> None:
    """Item 13.04: every image of the `empty` rule, and no other, becomes an `EmptyFrame` of its
    file, in file order, with the file's location, sequence id and file name and the sidecar's
    night flag (None without a sidecar). None of them is a record."""
    sidecar = fixture_dir / "day_night.json"
    flags = json.loads(sidecar.read_text())
    flagged = load_metadata(fixture_dir, sidecar=sidecar)
    unflagged = load_metadata(fixture_dir)
    assert flagged.dropped == unflagged.dropped
    for split in SPLITS:
        ids = [r.image_id for r in unflagged.records[split]]
        assert [r.image_id for r in flagged.records[split]] == ids, split
    for split in SPLITS:
        data = read_split(fixture_dir, split)
        cats = categories_by_image(data)
        expected = [
            EmptyFrame(
                image_id=image["id"],
                file_name=image["file_name"],
                split=split,
                location=image["location"],
                seq_id=image["seq_id"],
                night=flags[image["id"]]["night"],
            )
            for image in data["images"]
            if cats[image["id"]] and set(cats[image["id"]]) == {30}
        ]
        assert list(flagged.empties[split]) == expected, split
        assert len(expected) == flagged.dropped[split]["empty"], split
        assert all(f.night is None for f in unflagged.empties[split]), split
        assert not {f.image_id for f in expected} & {r.image_id for r in flagged.records[split]}
    assert sum(len(flagged.empties[s]) for s in SPLITS) == 12
    assert {f.night for s in SPLITS for f in flagged.empties[s]} <= {True, False}


def test_an_empty_frame_needs_an_integer_location(fixture_dir: Path) -> None:
    """Arm C reads an empty frame's location, so a malformed one fails loudly, as a kept
    image's does, instead of being dropped unread."""
    path = fixture_dir / "cis_val_annotations.json"
    data = read_split(fixture_dir, "cis_val")
    cats = categories_by_image(data)
    empty = next(image for image in data["images"] if cats[image["id"]] == [30])
    empty["location"] = "101"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=r"^cis_val: a location is not an integer"):
        load_metadata(fixture_dir)


def test_metadata_built_by_hand_has_no_empties(meta: Metadata) -> None:
    """`empties` defaults to an empty mapping, so a hand-built Metadata gives no background."""
    built = Metadata(records=meta.records, dropped=meta.dropped, clipped_boxes=meta.clipped_boxes)
    assert dict(built.empties) == {}
    assert set(meta.empties) == set(SPLITS)


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


def test_boxes_are_fractions_inside_the_frame(meta: Metadata) -> None:
    boxes = [b for split in SPLITS for r in meta.records[split] for b in r.boxes]
    assert boxes
    for x, y, w, h in boxes:
        assert 0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 and 0 < h <= 1
        assert x + w <= 1 + 1e-9 and y + h <= 1 + 1e-9
    assert all(n == 0 for n in meta.clipped_boxes.values())


def test_normalise_box() -> None:
    inside, clipped = metadata.normalise_box([10, 20, 30, 40], 100, 200)
    assert inside == pytest.approx((0.1, 0.1, 0.3, 0.2)) and not clipped
    whole, clipped = metadata.normalise_box([0, 0, 100, 200], 100, 200)
    assert whole == pytest.approx((0.0, 0.0, 1.0, 1.0)) and not clipped
    over, clipped = metadata.normalise_box([-5, 190, 110, 20], 100, 200)
    assert over == pytest.approx((0.0, 0.95, 1.0, 0.05)) and clipped
    with pytest.raises(ValueError, match="outside"):
        metadata.normalise_box([100, 0, 10, 10], 100, 200)
    with pytest.raises(ValueError, match="image"):
        metadata.normalise_box([0, 0, 10, 10], 0, 200)


def test_overrunning_boxes_are_clipped_and_counted(fixture_dir: Path) -> None:
    path = fixture_dir / "trans_test_annotations.json"
    data = json.loads(path.read_text())
    images = {image["id"]: image for image in data["images"]}
    cats = categories_by_image(data)
    kept_boxed = [
        a
        for a in data["annotations"]
        if "bbox" in a
        and len(set(cats[a["image_id"]])) == 1
        and a["category_id"] not in DROPPED_CCT_IDS
    ]
    right, bottom = kept_boxed[0], kept_boxed[1]
    width, height = images[right["image_id"]]["width"], images[right["image_id"]]["height"]
    right["bbox"] = [width - 10.0, 0.0, 30.0, 12.0]  # 20 px past the right edge
    bottom["bbox"] = [-4.0, height - 6.0, 16.0, 9.0]  # past the left and bottom edges
    path.write_text(json.dumps(data))
    meta = load_metadata(fixture_dir)
    assert meta.clipped_boxes == {s: 2 if s == "trans_test" else 0 for s in SPLITS}
    by_id = {r.image_id: r for r in meta.records["trans_test"]}
    assert by_id[right["image_id"]].boxes[0] == pytest.approx(
        ((width - 10) / width, 0.0, 10 / width, 12 / height)
    )
    assert by_id[bottom["image_id"]].boxes[0] == pytest.approx(
        (0.0, (height - 6) / height, 12 / width, 6 / height)
    )


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
    assert "Boxes clipped to the frame" in out
    assert re.search(r"^  23\s+\d+$", out, re.MULTILINE)
    total = sum(len(meta.records[s]) for s in SPLITS)
    assert re.search(rf"^total\s.*\s{total}$", out, re.MULTILINE)


def test_reprs_show_no_image_id_seq_id_or_file_name(meta: Metadata) -> None:
    """R03: printing a Record or the Metadata (a log line, a failing test, a notebook cell) shows
    no id; equality and hashing still use every field."""
    records = [r for split in SPLITS for r in meta.records[split]]
    leaks = 0
    for r in records:
        text = repr(r)
        leaks += any(s in text for s in (r.image_id, r.seq_id, r.file_name, "syn-", ".jpg"))
    assert (leaks, len(records) > 0) == (0, True)

    first = records[0]
    name, frame = CLASSES[first.label], f"{first.frame_num}/{first.seq_num_frames}"
    assert repr(first) == (
        f"Record(class={name}, split={first.split}, location={first.location}, "
        f"frame={frame}, boxes={len(first.boxes)}, night={first.night})"
    )
    assert str(first) == repr(first)
    other = dataclasses.replace(first, image_id="other", file_name="other.jpg", seq_id="other")
    assert repr(other) == repr(first)
    assert other != first and dataclasses.replace(first) == first
    assert hash(dataclasses.replace(first)) == hash(first)

    frames = [f for s in SPLITS for f in meta.empties[s]]
    assert frames
    for f in frames:
        text = repr(f)
        assert not any(v in text for v in (f.image_id, f.seq_id, f.file_name, "syn-", ".jpg"))
        assert text == f"EmptyFrame(split={f.split}, location={f.location}, night={f.night})"

    text = repr(meta)
    assert "Record(" not in text and "syn-" not in text and ".jpg" not in text
    assert "EmptyFrame(" not in text
    sizes = ", ".join(f"{s}={len(meta.records[s])}" for s in SPLITS)
    drops = ", ".join(f"{rule}={sum(meta.dropped[s][rule] for s in SPLITS)}" for rule in DROP_RULES)
    clipped = sum(meta.clipped_boxes.values())
    assert text == f"Metadata(records: {sizes}; dropped: {drops}; clipped_boxes={clipped})"


# Two records that differ only in their ids, kept in a module of their own: pytest prints the
# failing test's source, so the ids must not be written in the test itself.
RECORDS_MODULE = """
from trailcam.data.metadata import EmptyFrame, Record

left = Record(image_id="leakmark-img-a", file_name="leakmark-img-a.jpg", split="train", label=0,
              location=1, seq_id="leakmark-seq-a", frame_num=0, seq_num_frames=1, width=8,
              height=8, hour=0, boxes=(), night=None)
right = Record(image_id="leakmark-img-b", file_name="leakmark-img-b.jpg", split="train", label=0,
               location=1, seq_id="leakmark-seq-b", frame_num=0, seq_num_frames=1, width=8,
               height=8, hour=0, boxes=(), night=None)
empty_left = EmptyFrame(image_id="leakmark-img-c", file_name="leakmark-img-c.jpg", split="cis_val",
                        location=1, seq_id="leakmark-seq-c", night=True)
empty_right = EmptyFrame(image_id="leakmark-img-d", file_name="leakmark-img-d.jpg", split="cis_val",
                         location=1, seq_id="leakmark-seq-d", night=True)
"""
FAILING_TEST = """
from leak_records import empty_left, empty_right, left, right


def test_records_equal() -> None:
    assert left == right


def test_empty_frames_equal() -> None:
    assert empty_left == empty_right
"""


def test_failed_record_equality_shows_no_id(tmp_path: Path) -> None:
    """R03: pytest explains a failed `==` between two dataclasses field by field, printing every
    differing field, ids included, without calling the repr (review 3.05, both reviewers). Runs a
    failing comparison of two records and of two empty frames (item 13.04) in a subprocess at
    default and -vv verbosity: the output holds the id-free reprs and no id."""
    (tmp_path / "leak_records.py").write_text(RECORDS_MODULE)
    (tmp_path / "test_leak.py").write_text(FAILING_TEST)
    root = str(Path(__file__).resolve().parents[1])
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(filter(None, [root, os.environ.get("PYTHONPATH")])),
    }
    outcomes = []
    for verbosity in ([], ["-vv"]):
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *verbosity, "test_leak.py"],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
        )
        out = result.stdout + result.stderr
        shown = "Record(class=" in out and "EmptyFrame(split=cis_val" in out
        outcomes.append((result.returncode, "2 failed" in out, shown, out.count("leakmark")))
    assert outcomes == [(1, True, True, 0), (1, True, True, 0)]


# The real metadata (local runs; CI has no data and skips them)

REAL_ROOT = os.environ.get("TRAILCAM_DATA_ROOT", "")


def real_root() -> str:
    """The real metadata root. CI has no data, so the real-data tests skip there; anywhere else a
    missing root fails, since item 3.03 allows the skip only in CI (review 3.03)."""
    if REAL_ROOT:
        return REAL_ROOT
    if os.environ.get("CI"):
        pytest.skip("TRAILCAM_DATA_ROOT is not set (CI has no data)")
    pytest.fail("TRAILCAM_DATA_ROOT is not set; bash scripts/check.sh sets it", pytrace=False)


def test_real_metadata_drops_match_data_inspection() -> None:
    """Counts from docs/data-inspection.md: images per file, two-species and empty images."""
    meta = load_metadata(Path(real_root()))
    images = {"train": 13553, "cis_val": 3484, "trans_val": 1725, "cis_test": 15827,
              "trans_test": 23275}  # fmt: skip
    two_species = {"train": 7, "cis_val": 0, "trans_val": 61, "cis_test": 1, "trans_test": 9}
    empty = {"train": 0, "cis_val": 995, "trans_val": 61, "cis_test": 1180, "trans_test": 1778}
    for split in SPLITS:
        dropped = meta.dropped[split]
        # Counts are taken before the assert, so a failure prints numbers, never a record (R03).
        kept = len(meta.records[split])
        assert kept + sum(dropped.values()) == images[split], split
        assert dropped["two_species"] == two_species[split], split
        assert dropped["empty"] == empty[split], split
        assert len(meta.empties[split]) == empty[split], split  # item 13.04's empty frames
        assert dropped["unannotated"] == 0, split
    assert sum(meta.dropped[s]["two_species"] for s in SPLITS) == 78


def test_real_metadata_boxes_and_rare_drops() -> None:
    """The box facts in plan section 2 and the rare drops of the item 3.02 CLI run (review 3.02).

    Boxes must be in the pixel space of each image's original width and height: 1024 px boxes
    divided by the 2048 to 2560 px originals would all end before 0.5 of the frame.
    """
    meta = load_metadata(Path(real_root()))
    rare = {"train": 50, "cis_val": 4, "trans_val": 0, "cis_test": 168, "trans_test": 15}
    assert {s: meta.dropped[s]["rare"] for s in SPLITS} == rare
    boxes = [b for s in SPLITS for r in meta.records[s] for b in r.boxes]
    n_boxes = len(boxes)  # counted before the assert, so a failure prints no box values (R03)
    assert n_boxes == 47729
    assert sum(meta.clipped_boxes.values()) == 100
    assert max(x + w for x, _, w, _ in boxes) > 0.9
    assert max(y + h for _, y, _, h in boxes) > 0.9
    assert sum(1 for x, _, w, _ in boxes if x + w > 0.6) / n_boxes > 0.2

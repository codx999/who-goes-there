"""trailcam/data/datavis.py: the five data charts from the metadata (item 8.07).

On the synthetic fixture (written into a temporary directory) and on the real metadata in
`TRAILCAM_DATA_ROOT` (check.sh sets it locally; CI has no data and skips the real runs, anywhere
else a missing root fails). A failure on the real metadata prints numbers, never an id or a record
(R03): asserts compare counts or booleans computed beforehand. Also checks the look-at-the-data
notebook's shape (outputs cleared, train images only, the citation).
"""

import ast
import json
import os
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.text import Text

from trailcam.data import boxcrop, datavis
from trailcam.data.classes import CLASSES
from trailcam.data.datavis import (
    ARM_B_FILL,
    CACHE_BUILD_NIGHT_BY_FILE,
    CACHE_BUILD_NIGHT_BY_HOUR,
    CACHE_BUILD_SOURCE,
    CHART_NAMES,
    CITATION,
    CLOCK_HOUR,
)
from trailcam.data.metadata import (
    SPLITS,
    Metadata,
    capture_hour,
    find_annotation_files,
    load_metadata,
    load_night_flags,
)
from trailcam.data.split import CIS_SPLITS, SPLIT_NAMES, TRANS_SPLITS, Split, make_split
from trailcam.data.synthetic import write_synthetic_metadata

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs" / "figures"
LOOK_NOTEBOOK = ROOT / "notebooks" / "kaggle_look_at_data.ipynb"
REAL_ROOT = os.environ.get("TRAILCAM_DATA_ROOT", "")
SEED = 42
LOCATION_SHIFT = 770000
# Released images per official file (docs/data-card.md), the cache build's denominators.
RELEASED = {
    "train": 13553,
    "cis_val": 3484,
    "trans_val": 1725,
    "cis_test": 15827,
    "trans_test": 23275,
}


def real_root() -> str:
    """The real metadata root; skipped in CI only, as in tests/test_split.py (review 3.03)."""
    if REAL_ROOT:
        return REAL_ROOT
    if os.environ.get("CI"):
        pytest.skip("TRAILCAM_DATA_ROOT is not set (CI has no data)")
    pytest.fail("TRAILCAM_DATA_ROOT is not set; bash scripts/check.sh sets it", pytrace=False)


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("datavis")
    write_synthetic_metadata(root)
    return root


@pytest.fixture(scope="module")
def flagged(fixture_dir: Path) -> tuple[Metadata, Split]:
    """The fixture with the sidecar's night flags."""
    meta = load_metadata(fixture_dir, fixture_dir / "day_night.json")
    return meta, make_split(meta, SEED)


@pytest.fixture(scope="module")
def unflagged(fixture_dir: Path) -> tuple[Metadata, Split]:
    """The fixture without a sidecar: every record has night=None."""
    meta = load_metadata(fixture_dir)
    return meta, make_split(meta, SEED)


def texts(figure: Figure) -> list[str]:
    """Every string drawn on the figure, tick labels included."""
    FigureCanvasAgg(figure).draw()
    return [t.get_text() for t in figure.findobj(Text) if t.get_text()]


def leaks(meta: Metadata, drawn: list[str], locations: bool = False) -> bool:
    """True if a drawn string holds an image id, sequence id or file name, or with `locations` a
    location id as a whole number. CCT20's location ids are small numbers that collide with counts
    and ticks, so locations are checked on `shifted`, whose ids no count can match."""
    records = [r for split in SPLITS for r in meta.records[split]]
    ids = {r.image_id for r in records} | {r.seq_id for r in records}
    ids |= {r.file_name for r in records} | {Path(r.file_name).stem for r in records}
    location_ids = {str(r.location) for r in records}
    for text in drawn:
        if any(i in text for i in ids):
            return True
        if locations and set(re.findall(r"\d+", text.replace(",", ""))) & location_ids:
            return True
    return False


@pytest.fixture(scope="module")
def shifted(fixture_dir: Path) -> tuple[Metadata, Split]:
    """The flagged fixture with every location id moved to a distinctive number (770101 ...)."""
    meta = load_metadata(fixture_dir, fixture_dir / "day_night.json")
    moved = Metadata(
        records={
            s: tuple(replace(r, location=r.location + LOCATION_SHIFT) for r in rs)
            for s, rs in meta.records.items()
        },
        dropped=meta.dropped,
        clipped_boxes=meta.clipped_boxes,
    )
    return moved, make_split(moved, SEED)


# The command line


@pytest.mark.parametrize("sidecar", [False, True])
def test_the_command_line_writes_the_five_pngs_under_out_and_nothing_in_docs_figures(
    fixture_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], sidecar: bool
) -> None:
    # R04: the charts come from the metadata alone; the fixture holds no image to draw from.
    assert not any(p.suffix.lower() in {".jpg", ".jpeg", ".png"} for p in fixture_dir.rglob("*"))
    before = sorted(FIGURES.rglob("*")) if FIGURES.exists() else []
    out = tmp_path / "out"
    argv = ["--root", str(fixture_dir), "--seed", str(SEED), "--out", str(out)]
    if sidecar:
        argv += ["--sidecar", str(fixture_dir / "day_night.json")]
    assert datavis.main(argv) == 0
    assert sorted(p.name for p in out.iterdir()) == sorted(CHART_NAMES)
    for name in CHART_NAMES:
        assert (out / name).read_bytes().startswith(b"\x89PNG")
    after = sorted(FIGURES.rglob("*")) if FIGURES.exists() else []
    assert after == before

    printed = capsys.readouterr().out
    meta = load_metadata(fixture_dir)
    assert not leaks(meta, printed.replace(str(out), "").splitlines())
    source = "day/night sidecar" if sidecar else "item 6.03 cache build"
    assert f"night source: {source}" in printed


@pytest.mark.parametrize("which", ["shifted", "unflagged"])
def test_every_chart_cites_the_data_and_shows_no_id_or_location(
    which: str, request: pytest.FixtureRequest
) -> None:
    meta, split = request.getfixturevalue(which)
    drawn = datavis.figures(meta, split)
    assert list(drawn) == list(CHART_NAMES)
    for figure in drawn.values():
        strings = texts(figure)
        assert CITATION in strings
        assert "Beery" in CITATION and "ECCV 2018" in CITATION
        assert not leaks(meta, strings, locations=which == "shifted")


def test_the_camera_chart_labels_cameras_by_rank_only(shifted: tuple[Metadata, Split]) -> None:
    meta, split = shifted
    figure = datavis.figures(meta, split)["camera_locations.png"]
    FigureCanvasAgg(figure).draw()
    ticks = [t.get_text() for ax in figure.axes for t in ax.get_xticklabels()]
    assert ticks == ["C1", "C2", "T1"]  # 2 cis and 1 trans fixture location


# What each chart is drawn from


def test_class_counts_sum_to_each_split(flagged: tuple[Metadata, Split]) -> None:
    _, split = flagged
    counts = datavis.class_counts(split)
    assert list(counts) == list(SPLIT_NAMES)
    for name in SPLIT_NAMES:
        assert len(counts[name]) == len(CLASSES)
        expected = Counter(r.label for r in split.records[name])
        assert counts[name] == [expected[i] for i in range(len(CLASSES))]


def test_cameras_are_anonymous_ranks_that_sum_to_their_group(
    flagged: tuple[Metadata, Split],
) -> None:
    _, split = flagged
    cis, trans = datavis.cameras(split)
    for group, prefix, names in ((cis, "C", CIS_SPLITS), (trans, "T", TRANS_SPLITS)):
        assert [c.label for c in group] == [f"{prefix}{i}" for i in range(1, len(group) + 1)]
        totals = [sum(c.images.values()) for c in group]
        assert totals == sorted(totals, reverse=True)
        assert sum(totals) == sum(len(split.records[n]) for n in names)
        n_locations = len({r.location for n in names for r in split.records[n]})
        assert len(group) == n_locations
        assert all(set(c.images) == set(names) for c in group)


def test_the_night_table_counts_kept_images_from_the_sidecar(
    flagged: tuple[Metadata, Split],
) -> None:
    _, split = flagged
    table = datavis.night_table(split)
    assert table.source.startswith("day/night sidecar")
    every = [r for name in SPLIT_NAMES for r in split.records[name]]
    assert [name for name, _, _ in table.rows] == list(SPLIT_NAMES)
    for name, night, images in table.rows:
        assert images == len(split.records[name])
        assert night == sum(1 for r in split.records[name] if r.night)
    assert len(table.hours) == 24
    assert sum(images for _, images in table.hours) == len(every)
    assert sum(night for night, _ in table.hours) == sum(1 for r in every if r.night)
    for hour, (night, images) in enumerate(table.hours):
        assert images == sum(1 for r in every if r.hour == hour)
        assert night == sum(1 for r in every if r.hour == hour and r.night)


def test_without_a_sidecar_the_night_table_is_the_cache_build_s(
    unflagged: tuple[Metadata, Split],
) -> None:
    _, split = unflagged
    table = datavis.night_table(split)
    assert table.source == CACHE_BUILD_SOURCE
    assert table.rows == tuple((f, n, t) for f, (n, t) in CACHE_BUILD_NIGHT_BY_FILE.items())
    assert table.hours == CACHE_BUILD_NIGHT_BY_HOUR


def test_the_cache_build_numbers_add_up() -> None:
    """The numbers item 6.03 printed: all 57,864 annotated images, 36,776 night, hour 12."""
    assert {f: t for f, (_, t) in CACHE_BUILD_NIGHT_BY_FILE.items()} == RELEASED
    assert sum(t for _, t in CACHE_BUILD_NIGHT_BY_FILE.values()) == 57864
    assert sum(n for n, _ in CACHE_BUILD_NIGHT_BY_FILE.values()) == 36776
    assert len(CACHE_BUILD_NIGHT_BY_HOUR) == 24
    assert sum(t for _, t in CACHE_BUILD_NIGHT_BY_HOUR) == 57864
    assert sum(n for n, _ in CACHE_BUILD_NIGHT_BY_HOUR) == 36776
    assert all(0 <= n <= t for n, t in CACHE_BUILD_NIGHT_BY_HOUR)
    assert CACHE_BUILD_NIGHT_BY_HOUR[CLOCK_HOUR] == (3120, 5077)
    images = [t for _, t in CACHE_BUILD_NIGHT_BY_HOUR]
    assert images[CLOCK_HOUR] > 2 * max(images[CLOCK_HOUR - 1], images[CLOCK_HOUR + 1])


def test_box_fill_is_width_times_height_and_arm_b_s_range_is_the_sampler_s(
    flagged: tuple[Metadata, Split],
) -> None:
    _, split = flagged
    fills = datavis.box_fills(split)
    for name in SPLIT_NAMES:
        boxes = [b for r in split.records[name] for b in r.boxes]
        assert fills[name] == [w * h for (_, _, w, h) in boxes]
        assert all(0.0 < f <= 1.0 for f in fills[name])
    assert ARM_B_FILL == boxcrop.FILL


def test_bursts_count_sequences_once_and_the_shared_train_cis_val_burst(
    flagged: tuple[Metadata, Split],
) -> None:
    meta, split = flagged
    stats = datavis.bursts(meta, split)
    every = [r for name in SPLIT_NAMES for r in split.records[name]]
    assert sum(stats.sequences.values()) == len({r.seq_id for r in every})
    assert sum(stats.images.values()) == len(every)
    assert set(stats.sequences) == {3}  # every fixture sequence has 3 frames
    per_sequence = Counter(r.seq_id for r in every)
    assert stats.images_with_a_sibling == sum(n for n in per_sequence.values() if n > 1)
    assert stats.shared_train_cis_val == 1  # the fixture plants one, as CCT20 has 224


# The real metadata


def test_real_metadata_charts_match_the_data_card(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = Path(real_root())
    out = tmp_path / "out"
    assert datavis.main(["--root", str(root), "--seed", str(SEED), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert sorted(p.name for p in out.iterdir()) == sorted(CHART_NAMES)
    assert "kept images 53535, split seed 42" in printed
    assert "cameras: 10 cis, 10 trans" in printed
    assert "boxes 47729" in printed
    assert "sequences 20952, shared by official train and cis_val 222" in printed
    assert "images sharing their burst with another kept image 48701" in printed

    meta = load_metadata(root)
    split = make_split(meta, SEED)
    leaked = any(leaks(meta, texts(f)) for f in datavis.figures(meta, split).values())
    leaked_print = leaks(meta, printed.replace(str(out), "").splitlines())
    assert (leaked, leaked_print) == (False, False)


# Kept images per split with the sidecar (split seed 42): (night, images); docs/data-card.md.
KEPT_NIGHT = (
    ("train", 8246, 13587),
    ("val", 1436, 2394),
    ("trans_val", 1014, 1603),
    ("cis_test", 9066, 14478),
    ("trans_test", 15384, 21473),
)


def test_real_sidecar_gives_the_data_card_s_kept_image_night_shares() -> None:
    """The 6.03 sidecar, copied into the main checkout's data/ (gitignored, never committed): the
    night counts of the kept images per split and at hour 12, as the data card gives them."""
    root = Path(real_root())
    sidecar = root / "day_night.json"
    if not sidecar.is_file():
        pytest.fail("no day_night.json in the data root (the 6.03 sidecar)", pytrace=False)
    table = datavis.night_table(make_split(load_metadata(root, sidecar), SEED))
    assert table.source.startswith("day/night sidecar")
    assert table.rows == KEPT_NIGHT
    assert table.hours[CLOCK_HOUR] == (2990, 4525)
    assert sum(n for n, _ in table.hours) == 35146


def test_real_annotation_files_hold_the_cache_build_s_images_per_hour() -> None:
    """The cache build's image counts per file and per hour are the annotation files' own, before
    the drop rules (the night counts are checked against the sidecar by the next test)."""
    files = find_annotation_files(Path(real_root()))
    per_file = {}
    per_hour = Counter[int]()
    for split, path in files.items():
        images = json.loads(path.read_text())["images"]
        per_file[split] = len(images)
        per_hour.update(capture_hour(image["date_captured"]) for image in images)
    assert per_file == RELEASED
    assert [per_hour[h] for h in range(24)] == [t for _, t in CACHE_BUILD_NIGHT_BY_HOUR]


def test_real_sidecar_holds_the_cache_build_s_night_counts_per_file_and_hour() -> None:
    """The 6.03 numbers recomputed from the sidecar and the raw annotation files, before the drop
    rules: night per official file and per capture hour (review 8.07). Every annotated image must
    have a flag, so no image can drop out of the count unseen."""
    root = Path(real_root())
    sidecar = root / "day_night.json"
    if not sidecar.is_file():
        pytest.fail("no day_night.json in the data root (the 6.03 sidecar)", pytrace=False)
    flags = load_night_flags(sidecar)
    per_file: dict[str, tuple[int, int]] = {}
    per_hour = {hour: [0, 0] for hour in range(24)}
    missing = 0
    for split, path in find_annotation_files(root).items():
        night = images = 0
        for image in json.loads(path.read_text())["images"]:
            flag = flags.get(image["id"])
            missing += flag is None
            counts = per_hour[capture_hour(image["date_captured"])]
            counts[0] += bool(flag)
            counts[1] += 1
            night += bool(flag)
            images += 1
        per_file[split] = (night, images)
    assert missing == 0
    assert per_file == dict(CACHE_BUILD_NIGHT_BY_FILE)
    assert tuple((n, t) for n, t in per_hour.values()) == CACHE_BUILD_NIGHT_BY_HOUR


# The look-at-the-data notebook (runs on Kaggle only; its shape is checked here)


def _notebook() -> tuple[dict[str, object], list[str], str]:
    nb = json.loads(LOOK_NOTEBOOK.read_text(encoding="utf-8"))
    cells = nb["cells"]
    code = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
    markdown = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "markdown")
    return nb, code, markdown


def test_the_look_notebook_has_no_outputs_and_says_never_to_commit_them() -> None:
    nb, code, markdown = _notebook()
    cells = nb["cells"]
    assert isinstance(cells, list)
    for cell in cells:
        assert "attachments" not in cell
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
    assert "never commit" in markdown.lower()
    assert "Clear All Outputs" in markdown
    assert "docs/figures/" in markdown
    assert "Beery" in markdown and "CDLA-Permissive 1.0" in markdown
    for source in code:
        ast.parse(source)


def test_the_look_notebook_shows_train_photos_only() -> None:
    """Test-set isolation (R17): the photos come from the train split; no other split's records
    are read for display."""
    _, code, _ = _notebook()
    every = "\n".join(code)
    for name in ("val", "trans_val", "cis_test", "trans_test", "cis_val"):
        assert f'records["{name}"]' not in every
    assert 'split.records["train"]' in every
    assert every.count(".records[") == 1


def test_the_look_notebook_names_no_id_and_reads_file_names_only_to_open_photos() -> None:
    """R03's exception for the photo grid (review 8.07, fix round): no code cell touches an image
    id, sequence id or location in any form (attribute, name or string key), and `file_name` is
    read only inside the one `Image.open(...)` call, so no title, label, text or print can show a
    file name, however it is built."""
    _, code, _ = _notebook()
    trees = [ast.parse(source) for source in code]
    nodes = [node for tree in trees for node in ast.walk(tree)]
    forbidden = {"image_id", "seq_id", "location"}
    for node in nodes:
        assert not (isinstance(node, ast.Attribute) and node.attr in forbidden)
        assert not (isinstance(node, ast.Name) and node.id in forbidden)
        assert not (isinstance(node, ast.Constant) and node.value in forbidden | {"file_name"})
    opens = [
        node
        for node in nodes
        if isinstance(node, ast.Call) and ast.unparse(node.func) in ("Image.open", "PIL.Image.open")
    ]
    assert len(opens) == 1
    inside_open = {id(n) for arg in opens[0].args for n in ast.walk(arg)}
    file_names = [n for n in nodes if isinstance(n, ast.Attribute) and n.attr == "file_name"]
    assert len(file_names) == 1
    assert all(id(n) in inside_open for n in file_names)
    assert not any(isinstance(n, ast.Name) and n.id == "file_name" for n in nodes)


def test_the_look_notebook_s_trailcam_names_exist() -> None:
    import importlib

    _, code, _ = _notebook()
    for tree in (ast.parse(source) for source in code):
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("trail"):
                module = importlib.import_module(node.module)
                for alias in node.names:
                    assert hasattr(module, alias.name), f"{node.module}.{alias.name}"

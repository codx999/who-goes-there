"""trailcam/data/classes.py: the 12-class map (R05). The expected order and ids are read from the
class table in docs/plan.md section 2 and the category table in docs/data-inspection.md, so the
code and the documents cannot drift apart unnoticed."""

import re
from pathlib import Path

import pytest

from trailcam.data import classes

ROOT = Path(__file__).resolve().parents[1]
ROW = re.compile(r"^\|(.+)\|\s*$")


def table_rows(path: Path, heading: str) -> list[list[str]]:
    """Data rows (cells stripped) of the first markdown table after the line `heading`."""
    lines = path.read_text().splitlines()
    start = lines.index(heading)
    rows: list[list[str]] = []
    for line in lines[start + 1 :]:
        match = ROW.match(line)
        if match is None:
            if rows:
                break
            continue
        cells = [cell.strip() for cell in match.group(1).split("|")]
        if not all(set(cell) <= set("-: ") for cell in cells):
            rows.append(cells)
    return rows[1:]  # drop the header row


PLAN_ROWS = table_rows(ROOT / "docs" / "plan.md", "## 2. Dataset and data decisions")
INSPECTION_IDS = {
    name: int(cct_id)
    for cct_id, name in table_rows(ROOT / "docs" / "data-inspection.md", "## Categories")
}


def test_twelve_classes_in_plan_order() -> None:
    assert classes.NUM_CLASSES == 12
    assert len(set(classes.CLASSES)) == 12
    assert [(int(r[0]), r[1]) for r in PLAN_ROWS] == list(enumerate(classes.CLASSES))


def test_name_to_index_covers_0_to_11() -> None:
    assert sorted(classes.CLASS_TO_INDEX.values()) == list(range(12))
    for i, name in enumerate(classes.CLASSES):
        assert classes.CLASS_TO_INDEX[name] == i
        assert classes.index_of_name(name) == i


def test_cct_ids_match_plan_and_data_inspection() -> None:
    plan_ids = {r[1]: int(r[2]) for r in PLAN_ROWS}
    for i, name in enumerate(classes.CLASSES):
        cct_id = classes.INDEX_TO_CCT_ID[i]
        assert cct_id == plan_ids[name] == INSPECTION_IDS[name]
        assert classes.CCT_ID_TO_INDEX[cct_id] == i
        assert classes.index_of_cct_id(cct_id) == i
    assert len(classes.CCT_ID_TO_INDEX) == 12


def test_kept_and_dropped_cover_the_16_categories_once() -> None:
    assert dict(classes.DROPPED_CCT_IDS) == {30: "empty", 21: "badger", 51: "fox", 34: "deer"}
    for cct_id, name in classes.DROPPED_CCT_IDS.items():
        assert INSPECTION_IDS[name] == cct_id
    kept = set(classes.CCT_ID_TO_INDEX)
    dropped = set(classes.DROPPED_CCT_IDS)
    assert kept.isdisjoint(dropped)
    assert kept | dropped == set(INSPECTION_IDS.values())
    assert len(INSPECTION_IDS) == 16


@pytest.mark.parametrize("cct_id", [30, 21, 51, 34])
def test_dropped_ids_are_rejected(cct_id: int) -> None:
    assert classes.is_dropped_cct_id(cct_id)
    with pytest.raises(classes.DroppedClassError):
        classes.index_of_cct_id(cct_id)
    with pytest.raises(classes.DroppedClassError):
        classes.index_of_name(classes.DROPPED_CCT_IDS[cct_id])


@pytest.mark.parametrize("cct_id", [0, 2, 4, 12, 100, -1])
def test_unknown_ids_are_rejected(cct_id: int) -> None:
    assert not classes.is_dropped_cct_id(cct_id)
    with pytest.raises(KeyError) as info:
        classes.index_of_cct_id(cct_id)
    assert not isinstance(info.value, classes.DroppedClassError)


@pytest.mark.parametrize("name", ["Opossum", "fox ", "other", ""])
def test_unknown_names_are_rejected(name: str) -> None:
    with pytest.raises(KeyError):
        classes.index_of_name(name)


def test_maps_are_read_only() -> None:
    with pytest.raises(TypeError):
        classes.CLASS_TO_INDEX["empty"] = 12  # type: ignore[index]
    with pytest.raises(TypeError):
        classes.CCT_ID_TO_INDEX[30] = 12  # type: ignore[index]
    with pytest.raises(TypeError):
        classes.DROPPED_CCT_IDS[1] = "opossum"  # type: ignore[index]

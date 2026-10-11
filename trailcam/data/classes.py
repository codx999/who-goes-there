"""The class-to-index map, written once (plan.md section 2) and imported everywhere else.

12 classes, indexed 0 to 11 in the plan's training-count order, each with its CCT category id.
CCT ids are not contiguous, so the map from CCT id to index is explicit. The four dropped
categories (empty, badger, fox, deer) have no index and every lookup rejects them.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final

# (class name, CCT category id) in index order: plan.md section 2, ids from docs/data-inspection.md.
_KEPT: Final = (
    ("opossum", 1),
    ("rabbit", 10),
    ("coyote", 9),
    ("cat", 16),
    ("squirrel", 5),
    ("raccoon", 3),
    ("dog", 8),
    ("bobcat", 6),
    ("car", 33),
    ("bird", 11),
    ("rodent", 99),
    ("skunk", 7),
)

CLASSES: Final[tuple[str, ...]] = tuple(name for name, _ in _KEPT)
NUM_CLASSES: Final = len(CLASSES)
INDEX_TO_CCT_ID: Final[tuple[int, ...]] = tuple(cct_id for _, cct_id in _KEPT)
CLASS_TO_INDEX: Final = MappingProxyType({name: i for i, name in enumerate(CLASSES)})
CCT_ID_TO_INDEX: Final = MappingProxyType({cct_id: i for i, cct_id in enumerate(INDEX_TO_CCT_ID)})

# Dropped CCT categories (plan.md section 2): empty has no training images, the rest are too rare.
DROPPED_CCT_IDS: Final = MappingProxyType({30: "empty", 21: "badger", 51: "fox", 34: "deer"})


class DroppedClassError(KeyError):
    """A lookup named a category the task drops (empty, badger, fox, deer)."""


def is_dropped_cct_id(cct_id: int) -> bool:
    """True for the four dropped categories; the metadata loader filters their images out."""
    return cct_id in DROPPED_CCT_IDS


def index_of_cct_id(cct_id: int) -> int:
    """Index 0 to 11 for a kept CCT category id.

    Raises DroppedClassError for a dropped id and KeyError for an id CCT20 does not have.
    """
    if cct_id in DROPPED_CCT_IDS:
        raise DroppedClassError(f"CCT id {cct_id} ({DROPPED_CCT_IDS[cct_id]}) is dropped")
    if cct_id not in CCT_ID_TO_INDEX:
        raise KeyError(f"unknown CCT category id {cct_id}")
    return CCT_ID_TO_INDEX[cct_id]


def index_of_name(name: str) -> int:
    """Index 0 to 11 for a kept class name.

    Raises DroppedClassError for a dropped name and KeyError for an unknown one.
    """
    if name in DROPPED_CCT_IDS.values():
        raise DroppedClassError(f"class {name!r} is dropped")
    if name not in CLASS_TO_INDEX:
        raise KeyError(f"unknown class name {name!r}")
    return CLASS_TO_INDEX[name]

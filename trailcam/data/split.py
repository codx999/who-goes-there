"""The split (plan.md section 3): pool official train and cis_val, re-split them by sequence, and
pass cis_test, trans_val and trans_test through as loaded.

The official train and cis_val files share 224 sequences (bursts), so the two are pooled and
re-split by `seq_id` into train and val, about 85 to 15, stratified by the sequence's majority
class and by location. trans_val, cis_test and trans_test are the tuples `load_metadata` returns
(class filters of plan section 2 applied), never re-split, re-balanced or subsampled. For a seed:

1. The pool is grouped by `seq_id`. A sequence's class is the majority label over its kept frames
   (ties go to the lower class index); every image keeps its own label. A sequence spanning two
   locations is refused.
2. The strata are (class, location). A class with n sequences gets round(15% of n) val sequences,
   at least 1 and at most n - 1 when n >= 2 (none when n = 1), spread over its locations by largest
   remainder: each stratum gets the floor or the ceiling of 15% of its sequences.
3. In each stratum the val sequences are the ones with the smallest SHA-256 of (seed, seq_id).

Every draw is a SHA-256 of the seed and an id; no random generator is used. The split therefore
depends only on the metadata and the seed: not on record order, dict or set iteration, the global
random state or the Python version, so the hash computed locally matches the one a Kaggle run
records. The split hash is a SHA-256 over (split, image id, label) for all five splits, sorted by
image id. The leakage and coverage checks run before a split is returned; a failure raises
`SplitError` with counts only. Nothing here writes a file: a seed and a hash reproduce the split.
`python -m trailcam.data.split --root <dir> --seed <n>` prints counts and the hash only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Final

from trailcam.data.classes import CLASSES
from trailcam.data.metadata import Metadata, Record, load_metadata

# In the order of the plan section 3 table.
SPLIT_NAMES: Final = ("train", "val", "trans_val", "cis_test", "trans_test")
POOLED_FILES: Final = ("train", "cis_val")
PASSED_THROUGH: Final = ("trans_val", "cis_test", "trans_test")
CIS_SPLITS: Final = ("train", "val", "cis_test")
TRANS_SPLITS: Final = ("trans_val", "trans_test")
VAL_FRACTION: Final = Fraction(3, 20)

Stratum = tuple[int, int]  # (majority class index, location)


class SplitError(ValueError):
    """The split breaks a leakage or coverage rule of plan.md section 3."""


@dataclass(frozen=True, repr=False)
class Split:
    """Records per split in `SPLIT_NAMES` order, each with `split` set to the split it is in.

    train and val hold the pool in file order (official train, then cis_val); trans_val, cis_test
    and trans_test are the loaded tuples themselves. The mapping is read-only, so the hash cannot
    go stale. `mixed_label_sequences` counts pool sequences whose kept frames carry more than one
    label, `majority_ties` those among them with no single majority. The repr shows the seed, the
    hash and the sizes, never a record (R03).
    """

    seed: int
    records: Mapping[str, tuple[Record, ...]]
    hash: str
    mixed_label_sequences: int
    majority_ties: int

    def __repr__(self) -> str:
        sizes = ", ".join(f"{name}={len(rs)}" for name, rs in self.records.items())
        return f"Split(seed={self.seed}, hash={self.hash}, {sizes})"


def _draw(seed: int, *parts: object) -> str:
    """A seeded SHA-256 of `parts`, the split's only source of randomness."""
    text = "\x1f".join(str(part) for part in ("trailcam.split", seed, *parts))
    return hashlib.sha256(text.encode()).hexdigest()


def majority_label(frames: Iterable[Record]) -> int:
    """The most common label over a sequence's kept frames; ties go to the lower class index."""
    counts = Counter(r.label for r in frames)
    if not counts:
        raise ValueError("a sequence has no frames")
    return max(counts, key=lambda label: (counts[label], -label))


def val_quotas(
    sizes: Mapping[Stratum, int], seed: int, fraction: Fraction = VAL_FRACTION
) -> dict[Stratum, int]:
    """Val sequences per (class, location) stratum, given each stratum's size in sequences.

    A class with n sequences gets round(fraction * n), half up, clamped to 1 .. n - 1 when n >= 2
    and 0 when n = 1. Every stratum gets the floor of fraction * its size; the class's remaining
    val sequences go one each to the strata with the largest remainders, ties by a seeded draw.
    """
    if not 0 < fraction < 1:
        raise ValueError("the val fraction must lie strictly between 0 and 1")
    if any(n < 1 for n in sizes.values()):
        raise ValueError("a stratum has no sequences")
    by_class: dict[int, list[Stratum]] = {}
    for stratum in sorted(sizes):
        by_class.setdefault(stratum[0], []).append(stratum)
    quotas: dict[Stratum, int] = {}
    for strata in by_class.values():
        total = sum(sizes[s] for s in strata)
        target = min(max(math.floor(fraction * total + Fraction(1, 2)), 1), total - 1)
        exact = {s: fraction * sizes[s] for s in strata}
        floors = {s: math.floor(exact[s]) for s in strata}
        extra = target - sum(floors.values())
        by_remainder = sorted(
            strata, key=lambda s: (floors[s] - exact[s], _draw(seed, "stratum", *s))
        )
        bumped = set(by_remainder[:extra])
        for s in strata:
            quotas[s] = floors[s] + (1 if s in bumped else 0)
    return quotas


def split_hash(records: Mapping[str, Sequence[Record]]) -> str:
    """SHA-256 over (split, image id, label) for every split, sorted by image id."""
    body = [[name, sorted([r.image_id, r.label] for r in records[name])] for name in SPLIT_NAMES]
    return hashlib.sha256(json.dumps(body, separators=(",", ":")).encode()).hexdigest()


def check_split(records: Mapping[str, Sequence[Record]]) -> None:
    """Raise SplitError on a leakage or coverage failure (plan.md section 3, tests 1 to 4).

    Checked: an image id twice anywhere, a `seq_id` in two splits, a location in both a cis split
    (train, val, cis_test) and a trans split, a cis location missing from train or from val.
    The per-class minimum of val images is a test on the real metadata, not a check here.
    """
    problems: list[str] = []
    images = Counter(r.image_id for name in SPLIT_NAMES for r in records[name])
    repeated = sum(1 for n in images.values() if n > 1)
    if repeated:
        problems.append(f"{repeated} image ids appear more than once")
    owners: dict[str, set[str]] = {}
    for name in SPLIT_NAMES:
        for r in records[name]:
            owners.setdefault(r.seq_id, set()).add(name)
    shared = sum(1 for names in owners.values() if len(names) > 1)
    if shared:
        problems.append(f"{shared} sequences in more than one split")
    locations = {name: {r.location for r in records[name]} for name in SPLIT_NAMES}
    cis = set().union(*(locations[name] for name in CIS_SPLITS))
    trans = set().union(*(locations[name] for name in TRANS_SPLITS))
    if cis & trans:
        problems.append(f"{len(cis & trans)} locations in both a cis and a trans split")
    for name in ("train", "val"):
        missing = len(cis - locations[name])
        if missing:
            problems.append(f"{missing} cis locations missing from {name}")
    if problems:
        raise SplitError("; ".join(problems))


def make_split(meta: Metadata, seed: int) -> Split:
    """Re-split the pooled official train and cis_val by sequence and pass the rest through."""
    pool = [r for name in POOLED_FILES for r in meta.records[name]]
    sequences: dict[str, list[Record]] = {}
    for r in pool:
        sequences.setdefault(r.seq_id, []).append(r)

    spanning = sum(1 for frames in sequences.values() if len({r.location for r in frames}) > 1)
    if spanning:
        raise SplitError(f"{spanning} pool sequences span more than one location")
    strata: dict[Stratum, list[str]] = {}
    mixed = ties = 0
    for seq_id, frames in sequences.items():
        counts = Counter(r.label for r in frames)
        if len(counts) > 1:
            mixed += 1
            top = sorted(counts.values(), reverse=True)
            ties += top[0] == top[1]
        strata.setdefault((majority_label(frames), frames[0].location), []).append(seq_id)

    quotas = val_quotas({s: len(ids) for s, ids in strata.items()}, seed)
    val_sequences: set[str] = set()
    for stratum, seq_ids in strata.items():
        ranked = sorted(seq_ids, key=lambda seq_id: (_draw(seed, "sequence", seq_id), seq_id))
        val_sequences.update(ranked[: quotas[stratum]])

    assigned: dict[str, tuple[Record, ...]] = {
        "train": tuple(replace(r, split="train") for r in pool if r.seq_id not in val_sequences),
        "val": tuple(replace(r, split="val") for r in pool if r.seq_id in val_sequences),
        **{name: meta.records[name] for name in PASSED_THROUGH},
    }
    records = {name: assigned[name] for name in SPLIT_NAMES}
    check_split(records)
    return Split(
        seed=seed,
        records=MappingProxyType(records),
        hash=split_hash(records),
        mixed_label_sequences=mixed,
        majority_ties=ties,
    )


def summary(split: Split) -> str:
    """Counts only: images per class per split, sequences and locations per split, the val share
    of the pool, the mixed-label sequences and the hash."""
    records = split.records
    width = max(len(name) for name in (*CLASSES, "locations")) + 2
    head = "".join(f"{name:>12}" for name in SPLIT_NAMES) + f"{'all':>12}"
    lines = [
        f"Split seed {split.seed}",
        f"Split hash {split.hash}",
        "",
        "Images per class per split",
        f"{'class':<{width}}{head}",
    ]
    per_split = {name: Counter(r.label for r in records[name]) for name in SPLIT_NAMES}
    for index, name in enumerate(CLASSES):
        row = [per_split[split_name][index] for split_name in SPLIT_NAMES]
        lines.append(f"{name:<{width}}" + "".join(f"{n:>12}" for n in row) + f"{sum(row):>12}")
    columns = [records[name] for name in SPLIT_NAMES]
    columns.append(tuple(r for column in columns for r in column))
    rows = {
        "total": [len(rs) for rs in columns],
        "sequences": [len({r.seq_id for r in rs}) for rs in columns],
        "locations": [len({r.location for r in rs}) for rs in columns],
    }
    for label, row in rows.items():
        lines.append(f"{label:<{width}}" + "".join(f"{n:>12}" for n in row))

    pool = records["train"] + records["val"]
    pool_sequences = len({r.seq_id for r in pool})
    val_sequences = len({r.seq_id for r in records["val"]})
    shares = []
    for location in sorted({r.location for r in pool}):
        at = [r for r in pool if r.location == location]
        in_val = len({r.seq_id for r in at if r.split == "val"})
        shares.append(in_val / len({r.seq_id for r in at}))
    lines += [
        "",
        "Pool (official train + cis_val) re-split by sequence",
        f"  val images     {len(records['val']):>6} of {len(pool):>6}  "
        f"{len(records['val']) / len(pool):.3f}",
        f"  val sequences  {val_sequences:>6} of {pool_sequences:>6}  "
        f"{val_sequences / pool_sequences:.3f}",
        f"  val share of sequences per pool location: {min(shares):.3f} to {max(shares):.3f}"
        f" over {len(shares)} locations",
        f"  mixed-label sequences {split.mixed_label_sequences}"
        f" (majority ties {split.majority_ties})",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Re-split train and cis_val; print counts only.")
    parser.add_argument("--root", type=Path, required=True, help="directory holding the 5 files")
    parser.add_argument("--seed", type=int, required=True, help="split seed")
    args = parser.parse_args(argv)
    print(summary(make_split(load_metadata(args.root), args.seed)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

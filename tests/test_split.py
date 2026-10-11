"""trailcam/data/split.py: the re-split of the pooled train and cis_val, the pass-through test files
and the leakage and coverage tests of plan.md section 3 (R10 to R15; R27 for the split seed).

Every rule test runs twice: on the synthetic metadata from trailcam/data/synthetic.py (written to
a temporary directory, never committed) and on the real metadata in `TRAILCAM_DATA_ROOT` (check.sh
sets it locally; CI has no data and skips the real runs). Expectations are computed here from the
records, not by the module's own checks.

A failure on the real metadata prints numbers, never an id or a record (R03): every assert
compares values taken beforehand, never an expression over records; the tests take a `Case`,
whose repr names its source only, because pytest prints a failing test's arguments; and
`quietly` runs `make_split` and the CLI on the real metadata, re-raising a failure as its type
and message, without the frames whose arguments hold records.
"""

import itertools
import math
import os
import random
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

import pytest

from trailcam.data import split as split_module
from trailcam.data import synthetic
from trailcam.data.classes import CLASSES, NUM_CLASSES, index_of_name
from trailcam.data.metadata import Metadata, Record, load_metadata
from trailcam.data.split import (
    PASSED_THROUGH,
    POOLED_FILES,
    SPLIT_NAMES,
    VAL_FRACTION,
    Split,
    SplitError,
    majority_label,
    make_split,
    split_hash,
    val_quotas,
)

ROOT = Path(__file__).resolve().parents[1]
REAL_ROOT = os.environ.get("TRAILCAM_DATA_ROOT", "")
NO_DATA = "TRAILCAM_DATA_ROOT is not set (CI has no data)"
SEED = 42  # the split seed of item 3.03's CLI run and of the data card (item 3.04)

CIS = ("train", "val", "cis_test")
TRANS = ("trans_val", "trans_test")

# Pinned for SEED (plan section 3, test 5). Images per class in CLASSES order, per split.
SYNTHETIC_COUNTS = {
    "train": [39, 39, 36, 36, 36, 36, 36, 38, 36, 36, 36, 36],
    "val": [6] * NUM_CLASSES,
    "trans_val": [3] * NUM_CLASSES,
    "cis_test": [9] + [6] * 11,
    "trans_test": [3] * 7 + [5] + [3] * 4,
}
SYNTHETIC_SEQUENCES = {"train": 147, "val": 24, "trans_val": 12, "cis_test": 25, "trans_test": 13}
SYNTHETIC_HASH = "7a13478e3a541b4ddbc4493d4b802450a9422aa24f0e454ff7b676ee9f99df0c"

REAL_COUNTS = {
    "train": [2733, 2494, 1437, 1257, 1250, 1016, 817, 799, 603, 509, 440, 232],
    "val": [483, 442, 251, 222, 221, 180, 142, 140, 106, 89, 78, 40],
    "trans_val": [444, 9, 49, 72, 0, 129, 96, 734, 0, 7, 0, 63],
    "cis_test": [4524, 1757, 1326, 1632, 659, 1047, 845, 897, 791, 573, 233, 194],
    "trans_test": [5504, 843, 2249, 1418, 1050, 5469, 888, 2329, 1113, 224, 58, 328],
}
REAL_SEQUENCES = {"train": 5334, "val": 940, "trans_val": 538, "cis_test": 5882, "trans_test": 8258}
REAL_LOCATIONS = {"train": 10, "val": 10, "trans_val": 1, "cis_test": 10, "trans_test": 9}
REAL_HASH = "c78a9c1048f63aa52f34578ca17a6391315afdc2e478b4dbfd8a3aa954d1b009"
REAL_POOL_IMAGES = 15981  # plan section 3: 17,037 before the class filters
REAL_CROSS_FILE_SEQUENCES = 222  # of the 224 shared; two keep frames in one file after filtering

UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE)


@dataclass(frozen=True, repr=False)
class Case:
    """One metadata source and its split for SEED. The repr names the source only."""

    source: str
    meta: Metadata
    split: Split

    def __repr__(self) -> str:
        return f"Case({self.source})"


def quietly[T](call: Callable[..., T], *args: object) -> T:
    """`call(*args)`, with a failure re-raised as its type and message only (see the docstring)."""
    __tracebackhide__ = True
    try:
        return call(*args)
    except Exception as error:
        name = getattr(call, "__name__", "call")
        raise AssertionError(f"{name} failed: {type(error).__name__}: {error}") from None


def resplit(meta: Metadata, seed: int) -> Split:
    __tracebackhide__ = True
    return quietly(make_split, meta, seed)


# Fixtures: the synthetic and the real metadata, loaded and split once per module


@pytest.fixture(scope="module")
def synthetic_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("meta")
    synthetic.write_synthetic_metadata(out)
    return out


@pytest.fixture(scope="module")
def synthetic_meta(synthetic_dir: Path) -> Metadata:
    return load_metadata(synthetic_dir)


@pytest.fixture(scope="module")
def synthetic_case(synthetic_meta: Metadata) -> Case:
    return Case("synthetic", synthetic_meta, resplit(synthetic_meta, SEED))


@pytest.fixture(scope="module")
def real_case() -> Case:
    if not REAL_ROOT:
        pytest.skip(NO_DATA)
    meta = load_metadata(Path(REAL_ROOT))
    return Case("real", meta, resplit(meta, SEED))


@pytest.fixture(scope="module", params=["synthetic", "real"])
def case(request: pytest.FixtureRequest) -> Case:
    found: Case = request.getfixturevalue(f"{request.param}_case")
    return found


# Helpers, written independently of trailcam/data/split.py


def pool_of(meta: Metadata) -> list[Record]:
    return [r for name in ("train", "cis_val") for r in meta.records[name]]


def class_counts(records: Sequence[Record]) -> list[int]:
    counts = Counter(r.label for r in records)
    return [counts[i] for i in range(NUM_CLASSES)]


def sequences_of(records: Sequence[Record]) -> dict[str, list[Record]]:
    frames: dict[str, list[Record]] = {}
    for r in records:
        frames.setdefault(r.seq_id, []).append(r)
    return frames


def strata_of(records: Sequence[Record]) -> dict[tuple[int, int], set[str]]:
    """(majority class, location) to seq_ids; ties go to the lower class index."""
    strata: dict[tuple[int, int], set[str]] = {}
    for seq_id, frames in sequences_of(records).items():
        counts = Counter(r.label for r in frames)
        top = max(counts.values())
        label = min(label for label, n in counts.items() if n == top)
        # make_split refuses a sequence spanning two locations (tested below), so min is its one.
        strata.setdefault((label, min(r.location for r in frames)), set()).add(seq_id)
    return strata


def shared_counts(records: Mapping[str, Sequence[Record]], field: str) -> dict[str, int]:
    """For every pair of splits, how many distinct values of `field` they share."""
    values = {name: {getattr(r, field) for r in records[name]} for name in SPLIT_NAMES}
    return {f"{a}/{b}": len(values[a] & values[b]) for a, b in itertools.combinations(values, 2)}


def stratification_errors(split: Split) -> dict[str, int]:
    """Counts of strata and classes whose val share breaks the rule in plan section 3."""
    pool = [*split.records["train"], *split.records["val"]]
    in_val = {r.seq_id for r in split.records["val"]}
    errors = {"stratum off floor or ceiling": 0, "class off its rounded share": 0}
    per_class: dict[int, list[int]] = {}
    for (label, _), seq_ids in strata_of(pool).items():
        n, v = len(seq_ids), len(seq_ids & in_val)
        errors["stratum off floor or ceiling"] += not (
            math.floor(Fraction(3, 20) * n) <= v <= math.ceil(Fraction(3, 20) * n)
        )
        total, val = per_class.setdefault(label, [0, 0])
        per_class[label] = [total + n, val + v]
    for total, val in per_class.values():
        target = math.floor(Fraction(3, 20) * total + Fraction(1, 2))
        expected = min(max(target, 1), total - 1) if total >= 2 else 0
        errors["class off its rounded share"] += val != expected
    return errors


def with_records(meta: Metadata, **records: tuple[Record, ...]) -> Metadata:
    return Metadata(
        records={**meta.records, **records}, dropped=meta.dropped, clipped_boxes=meta.clipped_boxes
    )


# Leakage and coverage (plan section 3 tests 1 to 4), on the synthetic and the real metadata


def test_split_names_and_fraction() -> None:
    assert SPLIT_NAMES == ("train", "val", "trans_val", "cis_test", "trans_test")
    assert POOLED_FILES == ("train", "cis_val")
    assert PASSED_THROUGH == ("trans_val", "cis_test", "trans_test")
    assert VAL_FRACTION == Fraction(3, 20)


def test_no_location_in_a_cis_and_a_trans_split(case: Case) -> None:
    """R10, test 1: train and val (and cis_test) share no location with trans_val or trans_test."""
    locations = {name: {r.location for r in case.split.records[name]} for name in SPLIT_NAMES}
    shared = {f"{a}/{b}": len(locations[a] & locations[b]) for a in CIS for b in TRANS}
    assert shared == dict.fromkeys(shared, 0)
    n_trans = len(locations["trans_val"] | locations["trans_test"])
    assert n_trans >= 1  # the check is not vacuous


def test_no_seq_id_in_two_splits(case: Case) -> None:
    """R11, test 2."""
    shared = shared_counts(case.split.records, "seq_id")
    assert shared == dict.fromkeys(shared, 0)


def test_every_sequence_split_across_the_files_lies_in_one_split(case: Case) -> None:
    """R11: the bursts the official files split between train and cis_val are kept whole."""
    by_file = {name: {r.seq_id for r in case.meta.records[name]} for name in POOLED_FILES}
    crossing = by_file["train"] & by_file["cis_val"]
    homes = {n: {r.seq_id for r in case.split.records[n]} & crossing for n in ("train", "val")}
    n_crossing, n_train, n_val = len(crossing), len(homes["train"]), len(homes["val"])
    n_both = len(homes["train"] & homes["val"])
    assert n_crossing >= 1
    assert (n_train + n_val, n_both) == (n_crossing, 0)


def test_no_image_id_in_two_splits(case: Case) -> None:
    """R12, test 3, and no image id twice within one split."""
    shared = shared_counts(case.split.records, "image_id")
    assert shared == dict.fromkeys(shared, 0)
    repeats = {n: len(rs) - len({r.image_id for r in rs}) for n, rs in case.split.records.items()}
    assert repeats == dict.fromkeys(SPLIT_NAMES, 0)


def test_train_and_val_partition_the_pool_unchanged(case: Case) -> None:
    """R12: every pool image lands in train or val once, with every field but `split` as loaded."""
    pool = {r.image_id: r for r in pool_of(case.meta)}
    out = [*case.split.records["train"], *case.split.records["val"]]
    n_out, n_ids, n_pool = len(out), len({r.image_id for r in out}), len(pool)
    assert n_out == n_ids == n_pool
    foreign = sum(1 for r in out if r.image_id not in pool)
    assert foreign == 0
    changed = sum(1 for r in out if replace(r, split=pool[r.image_id].split) != pool[r.image_id])
    assert changed == 0


def test_every_record_names_its_split(case: Case) -> None:
    names = tuple(case.split.records)
    assert names == SPLIT_NAMES
    wrong = {n: sum(1 for r in rs if r.split != n) for n, rs in case.split.records.items()}
    assert wrong == dict.fromkeys(SPLIT_NAMES, 0)


def test_every_cis_location_in_train_and_val(case: Case) -> None:
    """R13, test 4: every cis location (train, val, cis_test) is in both train and val."""
    locations = {name: {r.location for r in case.split.records[name]} for name in SPLIT_NAMES}
    cis = locations["train"] | locations["val"] | locations["cis_test"]
    missing = {name: len(cis - locations[name]) for name in ("train", "val")}
    assert missing == {"train": 0, "val": 0}
    n_cis = len(cis)
    assert n_cis >= 2


def test_every_class_in_train_and_val(case: Case) -> None:
    for name in ("train", "val"):
        absent = class_counts(case.split.records[name]).count(0)
        assert absent == 0, name


def test_stratified_by_majority_class_and_location(case: Case) -> None:
    """R13: per (class, location) stratum, the floor or ceiling of 15% of its sequences go to val;
    per class, 15% rounded half up (at least one, at most all but one)."""
    errors = stratification_errors(case.split)
    assert errors == dict.fromkeys(errors, 0)


def test_val_is_about_15_percent(case: Case) -> None:
    records = case.split.records
    n_val, n_train = len(records["val"]), len(records["train"])
    seq_val = len({r.seq_id for r in records["val"]})
    seq_train = len({r.seq_id for r in records["train"]})
    images, sequences = n_val / (n_val + n_train), seq_val / (seq_val + seq_train)
    assert 0.13 <= images <= 0.17
    assert 0.13 <= sequences <= 0.17


def test_trans_and_cis_test_pass_through_as_loaded(case: Case) -> None:
    """R15: trans_val, cis_test and trans_test are the loaded tuples, never re-split, re-balanced
    or subsampled."""
    same = {n: case.split.records[n] is case.meta.records[n] for n in PASSED_THROUGH}
    assert same == dict.fromkeys(PASSED_THROUGH, True)


# Determinism and the hash (plan section 3 test 5, R14, R27)


def test_same_seed_same_split_and_hash(case: Case) -> None:
    again = resplit(case.meta, SEED)
    same = {n: again.records[n] == case.split.records[n] for n in SPLIT_NAMES}
    assert same == dict.fromkeys(SPLIT_NAMES, True)
    first, second = again.hash, case.split.hash
    assert first == second


def test_different_seeds_give_different_splits(case: Case) -> None:
    hashes = {resplit(case.meta, seed).hash for seed in (0, 1, 2, 43)} | {case.split.hash}
    n_hashes = len(hashes)
    assert n_hashes == 5
    other = {r.image_id for r in resplit(case.meta, SEED + 1).records["val"]}
    moved = len(other ^ {r.image_id for r in case.split.records["val"]})
    assert moved > 0


def test_split_ignores_record_order(case: Case) -> None:
    """R14: shuffling the records of every file changes neither the assignment nor the hash."""
    rng = random.Random(7)
    shuffled = {name: tuple(rng.sample(rs, len(rs))) for name, rs in case.meta.records.items()}
    again = resplit(with_records(case.meta, **shuffled), SEED)
    moved = {
        n: len({r.image_id for r in again.records[n]} ^ {r.image_id for r in case.split.records[n]})
        for n in SPLIT_NAMES
    }
    assert moved == dict.fromkeys(SPLIT_NAMES, 0)
    first, second = again.hash, case.split.hash
    assert first == second


def test_split_ignores_the_global_random_state(case: Case) -> None:
    """R27: the split draws from its seed only, whatever seed the caller set for `random`."""
    random.seed(1)
    first = resplit(case.meta, SEED).hash
    random.seed(2)
    random.random()
    second = resplit(case.meta, SEED).hash
    pinned = case.split.hash
    assert first == second == pinned


def test_split_ignores_the_string_hash_seed(synthetic_dir: Path, synthetic_case: Case) -> None:
    """R14: set and dict iteration order over strings changes with PYTHONHASHSEED; the hash does
    not."""
    code = (
        "import sys; from pathlib import Path; "
        "from trailcam.data.metadata import load_metadata; "
        "from trailcam.data.split import make_split; "
        "print(make_split(load_metadata(Path(sys.argv[1])), int(sys.argv[2])).hash)"
    )
    hashes = set()
    for hash_seed in ("0", "1", "4242"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        result = subprocess.run(
            [sys.executable, "-c", code, str(synthetic_dir), str(SEED)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        hashes.add(result.stdout.strip())
    assert hashes == {synthetic_case.split.hash}


def test_hash_covers_assignment_and_labels(case: Case) -> None:
    """Moving one sequence from val to train, or relabelling one image, changes the hash."""
    records = dict(case.split.records)
    moved_seq = records["val"][0].seq_id
    moved = {
        **records,
        "train": records["train"] + tuple(r for r in records["val"] if r.seq_id == moved_seq),
        "val": tuple(r for r in records["val"] if r.seq_id != moved_seq),
    }
    first = records["cis_test"][0]
    relabelled = {
        **records,
        "cis_test": (replace(first, label=(first.label + 1) % NUM_CLASSES),)
        + records["cis_test"][1:],
    }
    same, after_move, after_relabel = split_hash(records), split_hash(moved), split_hash(relabelled)
    pinned = case.split.hash
    assert same == pinned
    assert len({same, after_move, after_relabel}) == 3
    assert re.fullmatch(r"[0-9a-f]{64}", pinned)


def test_hash_ignores_order_within_a_split(case: Case) -> None:
    reversed_hash = split_hash({n: tuple(reversed(rs)) for n, rs in case.split.records.items()})
    pinned = case.split.hash
    assert reversed_hash == pinned


def test_repr_shows_seed_hash_and_sizes_only(case: Case) -> None:
    """R03: printing a Split (a log line, a notebook cell) shows no record."""
    text = repr(case.split)
    n_ids, n_records, n_files = len(UUID.findall(text)), text.count("Record("), text.count(".jpg")
    assert (n_ids, n_records, n_files, "syn-" in text) == (0, 0, 0, False)
    sizes = ", ".join(f"{n}={len(rs)}" for n, rs in case.split.records.items())
    assert text == f"Split(seed={SEED}, hash={case.split.hash}, {sizes})"


def test_records_are_read_only(synthetic_case: Case) -> None:
    """The hash is computed once, so the mapping it covers cannot be changed afterwards."""
    with pytest.raises(TypeError):
        synthetic_case.split.records["val"] = ()  # type: ignore[index]


def test_synthetic_split_is_pinned(synthetic_case: Case) -> None:
    """Plan section 3 test 5 on the fixture, so CI catches a change to the algorithm."""
    pinned = synthetic_case.split
    assert {n: class_counts(rs) for n, rs in pinned.records.items()} == SYNTHETIC_COUNTS
    sequences = {n: len({r.seq_id for r in rs}) for n, rs in pinned.records.items()}
    assert sequences == SYNTHETIC_SEQUENCES
    assert (pinned.mixed_label_sequences, pinned.majority_ties) == (0, 0)
    assert pinned.hash == SYNTHETIC_HASH


# The real metadata only: the 30-image floor and the pinned split for the seed


@pytest.mark.skipif(not REAL_ROOT, reason=NO_DATA)
def test_real_every_class_has_30_val_images(real_case: Case) -> None:
    """R13, test 4 (smallest on the real metadata for seed 42: skunk, 40)."""
    val = class_counts(real_case.split.records["val"])
    below = sum(1 for n in val if n < 30)
    assert below == 0, val


@pytest.mark.skipif(not REAL_ROOT, reason=NO_DATA)
def test_real_split_is_pinned(real_case: Case) -> None:
    """Plan section 3 test 5: sizes, per-class counts, sequences and the hash for seed 42."""
    records, meta = real_case.split.records, real_case.meta
    counts = {n: class_counts(rs) for n, rs in records.items()}
    assert counts == REAL_COUNTS
    sequences = {n: len({r.seq_id for r in rs}) for n, rs in records.items()}
    assert sequences == REAL_SEQUENCES
    locations = {n: len({r.location for r in rs}) for n, rs in records.items()}
    assert locations == REAL_LOCATIONS
    pool = len(meta.records["train"]) + len(meta.records["cis_val"])
    assert pool == REAL_POOL_IMAGES
    by_file = {name: {r.seq_id for r in meta.records[name]} for name in POOLED_FILES}
    crossing = len(by_file["train"] & by_file["cis_val"])
    assert crossing == REAL_CROSS_FILE_SEQUENCES
    mixed = (real_case.split.mixed_label_sequences, real_case.split.majority_ties)
    assert mixed == (3, 0)
    pinned = real_case.split.hash
    assert pinned == REAL_HASH


@pytest.mark.skipif(not REAL_ROOT, reason=NO_DATA)
def test_real_cli_prints_counts_and_the_hash_only(capsys: pytest.CaptureFixture[str]) -> None:
    """R03: the item's CLI run prints counts and the hash, no image or sequence id."""
    status = quietly(split_module.main, ["--root", REAL_ROOT, "--seed", str(SEED)])
    assert status == 0
    out = capsys.readouterr().out
    n_ids, n_files = len(UUID.findall(out)), out.count(".jpg")
    assert (n_ids, n_files) == (0, 0)
    assert f"Split hash {REAL_HASH}" in out
    assert re.search(r"^skunk\s+232\s+40\s+63\s+194\s+328\s+857$", out, re.M)
    assert re.search(r"mixed-label sequences 3 \(majority ties 0\)$", out, re.M)


# The majority class and the val quotas


def make_record(label: int, frame: int = 1, seq_id: str = "s", location: int = 1) -> Record:
    return Record(
        image_id=f"{seq_id}-{frame}",
        file_name=f"{seq_id}-{frame}.jpg",
        split="train",
        label=label,
        location=location,
        seq_id=seq_id,
        frame_num=frame,
        seq_num_frames=3,
        width=96,
        height=72,
        hour=12,
        boxes=(),
        night=None,
    )


def test_majority_label() -> None:
    assert majority_label([make_record(5), make_record(2, 2), make_record(5, 3)]) == 5
    assert majority_label([make_record(7), make_record(1, 2)]) == 1  # a tie: the lower index
    assert majority_label([make_record(1, 2), make_record(7)]) == 1  # whatever the frame order
    assert majority_label([make_record(9), make_record(3, 2), make_record(4, 3)]) == 3
    with pytest.raises(ValueError, match="no frames"):
        majority_label([])


def test_sequences_are_stratified_by_their_majority_class(
    synthetic_meta: Metadata, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R13: a sequence counts in the stratum of its majority class (a tie: the lower index), not
    in its first frame's; every frame keeps its own label."""
    coyote, bird, bobcat, rabbit = (
        index_of_name(n) for n in ("coyote", "bird", "bobcat", "rabbit")
    )
    sequences = sequences_of(synthetic_meta.records["train"])
    three = next(fs for fs in sequences.values() if len(fs) == 3 and fs[0].label == coyote)
    # The planted bobcat sequence whose frame 2 is dropped as two-species: frames 1 and 3 remain.
    two = next(fs for fs in sequences.values() if len(fs) == 2 and fs[0].label == bobcat)
    # coyote, bird, bird: majority bird. bobcat, rabbit: a tie, so rabbit, the lower index.
    relabel = {three[1].image_id: bird, three[2].image_id: bird, two[1].image_id: rabbit}
    train = tuple(
        replace(r, label=relabel.get(r.image_id, r.label)) for r in synthetic_meta.records["train"]
    )
    seen: list[dict[tuple[int, int], int]] = []
    original = split_module.val_quotas

    def spy(
        sizes: Mapping[tuple[int, int], int], seed: int, fraction: Fraction = VAL_FRACTION
    ) -> dict[tuple[int, int], int]:
        seen.append(dict(sizes))
        return original(sizes, seed, fraction)

    monkeypatch.setattr(split_module, "val_quotas", spy)
    mixed = make_split(with_records(synthetic_meta, train=train), SEED)
    assert (mixed.mixed_label_sequences, mixed.majority_ties) == (2, 1)
    pool = [*mixed.records["train"], *mixed.records["val"]]
    labels = {r.image_id: r.label for r in pool}
    assert {i: labels[i] for i in relabel} == relabel
    expected = {stratum: len(ids) for stratum, ids in strata_of(pool).items()}
    assert seen == [expected]
    here, there = three[0].location, two[0].location
    assert (expected[(coyote, here)], expected[(bird, here)]) == (6, 8)
    assert (expected[(bobcat, there)], expected[(rabbit, there)]) == (7, 8)
    errors = stratification_errors(mixed)
    assert errors == dict.fromkeys(errors, 0)


def test_val_quotas_small_classes() -> None:
    sizes = {(0, 1): 1, (1, 1): 2, (2, 1): 3, (3, 1): 7, (3, 2): 7, (4, 1): 10, (5, 1): 30}
    # 15% of 1, 2, 3, 14, 10 and 30 is 0.15, 0.3, 0.45, 2.1, 1.5 and 4.5: rounded half up (not to
    # even), then a class with two or more sequences gets at least one, one with a single none.
    assert val_quotas(sizes, SEED) == {
        (0, 1): 0,
        (1, 1): 1,
        (2, 1): 1,
        (3, 1): 1,
        (3, 2): 1,
        (4, 1): 2,
        (5, 1): 5,
    }


def test_val_quotas_follow_the_rule_on_random_sizes() -> None:
    rng = random.Random(0)
    for trial in range(300):
        fraction = rng.choice([Fraction(3, 20), Fraction(1, 10), Fraction(1, 2), Fraction(9, 10)])
        sizes = {
            (label, location): rng.randint(1, 40)
            for label in range(rng.randint(1, 4))
            for location in range(rng.randint(1, 5))
        }
        quotas = val_quotas(sizes, trial, fraction)
        assert quotas.keys() == sizes.keys()
        for stratum, n in sizes.items():
            assert math.floor(fraction * n) <= quotas[stratum] <= math.ceil(fraction * n)
        for label in {label for label, _ in sizes}:
            total = sum(n for (c, _), n in sizes.items() if c == label)
            val = sum(q for (c, _), q in quotas.items() if c == label)
            target = math.floor(fraction * total + Fraction(1, 2))
            assert val == (min(max(target, 1), total - 1) if total >= 2 else 0)


def test_val_quotas_ties_depend_on_the_seed() -> None:
    """Equal remainders are broken by a seeded draw, not always towards the same location."""
    # 15% of 70 is 10.5, rounded up to 11; each stratum's floor is 1 (15% of 7 is 1.05), so one
    # of the ten, all with the same remainder, gets a second.
    sizes = {(0, location): 7 for location in range(10)}
    bumped = set()
    for seed in range(20):
        quotas = val_quotas(sizes, seed)
        (extra,) = [s for s, q in quotas.items() if q == 2]
        bumped.add(extra)
    n_bumped = len(bumped)
    assert n_bumped > 1


def test_val_quotas_reject_bad_input() -> None:
    with pytest.raises(ValueError, match="fraction"):
        val_quotas({(0, 1): 5}, SEED, Fraction(1))
    with pytest.raises(ValueError, match="fraction"):
        val_quotas({(0, 1): 5}, SEED, Fraction(0))
    with pytest.raises(ValueError, match="no sequences"):
        val_quotas({(0, 1): 0}, SEED)


# Leakage in the input is refused, with counts only


def test_pool_sequence_spanning_two_locations_raises(synthetic_meta: Metadata) -> None:
    train = synthetic_meta.records["train"]
    moved = (replace(train[0], location=999),) + train[1:]
    with pytest.raises(SplitError, match=r"^1 pool sequences span more than one location$"):
        make_split(with_records(synthetic_meta, train=moved), SEED)


def test_pool_sequence_also_in_a_test_file_raises(synthetic_meta: Metadata) -> None:
    cis_test = synthetic_meta.records["cis_test"]
    pool_seq = synthetic_meta.records["cis_val"][0].seq_id
    leaked = (replace(cis_test[0], seq_id=pool_seq),) + cis_test[1:]
    with pytest.raises(SplitError, match=r"1 sequences in more than one split"):
        make_split(with_records(synthetic_meta, cis_test=leaked), SEED)


def test_image_id_in_the_pool_and_a_test_file_raises(synthetic_meta: Metadata) -> None:
    trans_test = synthetic_meta.records["trans_test"]
    pool_image = synthetic_meta.records["train"][0].image_id
    leaked = (replace(trans_test[0], image_id=pool_image),) + trans_test[1:]
    with pytest.raises(SplitError, match=r"1 image ids appear more than once"):
        make_split(with_records(synthetic_meta, trans_test=leaked), SEED)


def test_pool_location_in_a_trans_file_raises(synthetic_meta: Metadata) -> None:
    trans_val = synthetic_meta.records["trans_val"]
    cis_location = synthetic_meta.records["train"][0].location
    leaked = tuple(replace(r, location=cis_location) for r in trans_val)
    with pytest.raises(SplitError, match=r"1 locations in both a cis and a trans split"):
        make_split(with_records(synthetic_meta, trans_val=leaked), SEED)


def test_cis_location_missing_from_train_or_val_raises(synthetic_meta: Metadata) -> None:
    """One sequence alone at a cis location cannot be in both train and val."""
    train = synthetic_meta.records["train"]
    alone = train[0].seq_id
    moved = tuple(replace(r, location=103) if r.seq_id == alone else r for r in train)
    with pytest.raises(SplitError, match=r"1 cis locations missing from (train|val)"):
        make_split(with_records(synthetic_meta, train=moved), SEED)


def test_cis_test_location_missing_from_the_pool_raises(synthetic_meta: Metadata) -> None:
    cis_test = synthetic_meta.records["cis_test"]
    moved = tuple(replace(r, location=104) for r in cis_test[:3]) + cis_test[3:]
    with pytest.raises(SplitError, match=r"1 cis locations missing from train; 1 .* from val"):
        make_split(with_records(synthetic_meta, cis_test=moved), SEED)


# The CLI prints counts and the hash only, and writes no file (R03, R14)


def test_cli_prints_counts_and_the_hash(
    synthetic_dir: Path, synthetic_meta: Metadata, capsys: pytest.CaptureFixture[str]
) -> None:
    assert split_module.main(["--root", str(synthetic_dir), "--seed", str(SEED)]) == 0
    out = capsys.readouterr().out
    assert "syn-" not in out and ".jpg" not in out
    assert re.search(rf"^Split seed {SEED}$", out, re.M)
    assert re.search(rf"^Split hash {SYNTHETIC_HASH}$", out, re.M)
    assert re.search(r"^class\s+train\s+val\s+trans_val\s+cis_test\s+trans_test\s+all$", out, re.M)
    for index, name in enumerate(CLASSES):
        row = [SYNTHETIC_COUNTS[n][index] for n in SPLIT_NAMES]
        cells = r"\s+".join(str(n) for n in [*row, sum(row)])
        assert re.search(rf"^{name}\s+{cells}$", out, re.M), name
    assert re.search(r"^sequences\s+147\s+24\s+12\s+25\s+13\s+221$", out, re.M)
    assert re.search(r"mixed-label sequences 0 \(majority ties 0\)$", out, re.M)
    total = sum(len(rs) for rs in synthetic_meta.records.values())
    assert re.search(rf"^total\s.*\s{total}$", out, re.M)


def test_cli_writes_no_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R14: the split is never written to a file, neither in the working directory nor the root."""
    synthetic.write_synthetic_metadata(tmp_path / "meta")
    monkeypatch.chdir(tmp_path)

    def snapshot() -> dict[str, bytes]:
        return {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}

    before = snapshot()
    assert split_module.main(["--root", str(tmp_path / "meta"), "--seed", str(SEED)]) == 0
    assert snapshot() == before


def test_split_module_has_no_file_writes() -> None:
    """R14: no file is written anywhere, not only in the directories the CLI test watches."""
    source = (ROOT / "trailcam" / "data" / "split.py").read_text()
    for call in ("open(", "write_text", "write_bytes", "json.dump(", "to_csv", "savetxt"):
        assert call not in source, call

"""trailcam/train/config.py and the root config.yaml (item 5.02; R25 arm A and arm B differ in
`arm` alone, R31 the training recipe of plan.md section 5, for the phase-1 configs).

Phase 2 (item 13.02, docs/phase2-plan.md): every `p2_` config is `resnet18_arm_a` with the one
change `P2_CHANGES` names for it, so a phase-2 comparison changes one thing; arm C's one change
(item 13.04) is the arm with its own setting, `paste_p`. `input_size` and `paste_p`, the fields
phase 2 adds, leave every phase-1 config hash as `docs/ledger.md` records it.

Phase 3 (item 13.11, docs/phase3-plan.md section 5): `p3_baseline` is `resnet18_arm_a` exactly,
`p3_convnext` changes the model alone, and `p3_convnext_cp` is `p3_convnext` with arm C at
`paste_p` 0.5 (`P3_CHANGES`, each against its own base). Phase 3 adds no field: every phase-1 and
phase-2 hash is pinned unchanged, and the phase-3 hashes are pinned before any phase-3 run.

The first tests read the committed config.yaml; the rest feed `parse_configs` small texts built
from it, so each refusal is checked against a file that is otherwise valid.
"""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, fields, replace

import pytest
import yaml
from PIL import Image

from trailcam.data.dataset import ARMS, INPUT_SIZE, TrainTransform
from trailcam.data.metadata import Box
from trailcam.model import factory
from trailcam.train import config as config_module
from trailcam.train import train as train_module
from trailcam.train.config import (
    CONFIG_PATH,
    DRY_RUN,
    FIELDS,
    HASH_EXCLUDED,
    HASH_NEUTRAL,
    MODEL_NAMES,
    OPTIONAL_FIELDS,
    PHASE2_PREFIX,
    PHASE3_PREFIX,
    PHASE_PREFIXES,
    Config,
    ConfigError,
    config_hash,
    is_phase2,
    is_phase3,
    load_config,
    load_configs,
    parse_configs,
    phase_of,
    with_overrides,
)
from trailcam.train.ledger import check_ledger
from trailcam.train.train import LEDGER_PATH

PHASE1 = ("small_cnn", "resnet18_arm_a", "resnet18_arm_b")  # the training configs R31 holds for
P2_BASE = "resnet18_arm_a"
P2_CHANGES: dict[str, dict[str, object]] = {
    "p2_baseline": {},  # the control re-run at the phase-2 commit
    "p2_arm_c": {"arm": "C", "paste_p": 0.5},  # copy-paste (item 13.04): the arm and its setting
    "p2_arm_c_p25": {"arm": "C", "paste_p": 0.25},  # arm C's pre-declared follow-up (item 13.08)
    "p2_head3": {"head_only_epochs": 3},  # the body trains 7 of the 10 epochs
    "p2_res320": {"input_size": 320},  # eval resize round(320 * 256 / 224) = 366
    "p2_r50": {"model": "resnet50"},
}
# Phase 3 (docs/phase3-plan.md sections 3 and 5): each config's base and its one change from it.
P3_CHANGES: dict[str, tuple[str, dict[str, object]]] = {
    "p3_baseline": ("resnet18_arm_a", {}),  # the control re-run at the phase-3 commit
    "p3_convnext": ("resnet18_arm_a", {"model": "convnext_tiny"}),  # candidate (a)
    "p3_convnext_cp": ("p3_convnext", {"arm": "C", "paste_p": 0.5}),  # candidate (b)
}
# The config hashes of docs/ledger.md's rows (items 6.04, 6.05, 7.02, 7.03); score_test.py
# refuses a config whose hash no longer matches its row.
PHASE1_HASHES = {
    "small_cnn": "f40ed6e3960104f10a78e3d33a8f530cde82e23a555def76f694e19ab09a3e95",
    "resnet18_arm_a": "483aa055ab221fe445add05623a061a980fdf728f63c819c78caccd5a94c6203",
    "resnet18_arm_b": "483aa055ab221fe445add05623a061a980fdf728f63c819c78caccd5a94c6203",
}


@pytest.fixture(scope="module")
def configs() -> dict[str, Config]:
    return dict(load_configs())


def test_all_configs_load(configs: dict[str, Config]) -> None:
    assert list(configs) == [DRY_RUN, *PHASE1, *P2_CHANGES, *P3_CHANGES]
    assert all(isinstance(c, Config) for c in configs.values())
    assert load_config("resnet18_arm_b") == configs["resnet18_arm_b"]


def test_the_dry_run_config(configs: dict[str, Config]) -> None:
    dry = configs[DRY_RUN]
    assert dry.max_epochs == 1
    assert dry.batch_size == 2
    assert dry.fixture is True
    assert dry.num_workers == 0
    assert dry.pin_memory is False
    assert dry.pretrained is False  # random weights: the dry run needs no download


def test_every_config_has_both_seeds(configs: dict[str, Config]) -> None:
    for config in configs.values():
        assert type(config.split_seed) is int and type(config.run_seed) is int
        assert config.split_seed == 42  # the seed of the pinned split tests and the data card


def test_the_phase_of_a_config_is_its_name(configs: dict[str, Config]) -> None:
    """Phases 2 and 3 are named by the prefix alone: the unprefixed training configs are phase
    1's, the `p2_` ones are `P2_CHANGES`, the `p3_` ones `P3_CHANGES`, and the dry run is none
    of them (phase 1 by name, a fixture by its field)."""
    assert (PHASE2_PREFIX, PHASE3_PREFIX) == ("p2_", "p3_")
    assert dict(PHASE_PREFIXES) == {2: "p2_", 3: "p3_"}
    training = {name for name, c in configs.items() if not c.fixture}
    assert {name for name in training if phase_of(name) == 1} == set(PHASE1)
    assert {name for name in configs if is_phase2(name)} == set(P2_CHANGES)
    assert {name for name in configs if is_phase3(name)} == set(P3_CHANGES)
    assert configs[DRY_RUN].fixture and phase_of(DRY_RUN) == 1
    assert is_phase2("p2_anything") and not is_phase2("resnet18_p2_")
    assert is_phase3("p3_anything") and not is_phase3("p2_p3_") and not is_phase2("p3_p2_")
    assert [phase_of(n) for n in ("small_cnn", "p2_x", "p3_x", "P3_x", "xp3_")] == [1, 2, 3, 1, 1]


def test_every_config_has_loader_settings(configs: dict[str, Config]) -> None:
    for name in (*PHASE1, *P2_CHANGES, *P3_CHANGES):
        assert (configs[name].num_workers, configs[name].pin_memory) == (4, True), name
        assert configs[name].fixture is False, name


def test_arm_a_and_arm_b_differ_in_arm_alone(configs: dict[str, Config]) -> None:
    """R25: same backbone, seeds, epochs, optimizer, schedule, batch size; only the sampler."""
    a, b = asdict(configs["resnet18_arm_a"]), asdict(configs["resnet18_arm_b"])
    assert [key for key in FIELDS if a[key] != b[key]] == ["arm"]
    assert (a["arm"], b["arm"]) == ("A", "B")


def test_the_full_configs_carry_the_recipe(configs: dict[str, Config]) -> None:
    """R31, for the phase-1 configs: AdamW 1e-4 body / 1e-3 head, weight decay 1e-2, cosine with
    a one-epoch warmup, patience 3, batch 64 at 224 px, mixed precision; up to 10 backbone epochs
    in all, the first of them head-only (it counts inside max_epochs, item 5.03), up to 15 for the
    small CNN, which has no pretrained body and trains it at the head's rate. The phase-2 configs
    are held to arm A's recipe with one change instead (`test_every_p2_config_...`)."""
    for name in PHASE1:
        c = configs[name]
        assert (c.weight_decay, c.scheduler, c.warmup_epochs, c.patience) == (
            1e-2,
            "cosine",
            1,
            3,
        ), name
        assert (c.batch_size, c.mixed_precision, c.lr_head) == (64, True, 1e-3), name
        assert c.input_size == INPUT_SIZE == 224, name
    for name in ("resnet18_arm_a", "resnet18_arm_b"):
        c = configs[name]
        assert (c.model, c.pretrained, c.lr_body, c.max_epochs, c.head_only_epochs) == (
            "resnet18",
            True,
            1e-4,
            10,
            1,
        ), name
    c = configs["small_cnn"]
    assert (c.model, c.arm, c.lr_body, c.max_epochs, c.head_only_epochs) == (
        "small_cnn",
        "A",
        1e-3,
        15,
        0,
    )


@pytest.mark.parametrize("name", list(P2_CHANGES))
def test_every_p2_config_is_arm_a_with_its_one_change(
    configs: dict[str, Config], name: str
) -> None:
    """Phase 2's one change per comparison: the config differs from `resnet18_arm_a` in exactly
    the fields `P2_CHANGES` names, to exactly those values, and in nothing else. An arm's own
    setting (`OPTIONAL_FIELDS`: arm C's `paste_p`) belongs to the arm's change, not a second one."""
    base, variant = asdict(configs[P2_BASE]), asdict(configs[name])
    changed = {key: variant[key] for key in FIELDS if variant[key] != base[key]}
    assert changed == P2_CHANGES[name]
    assert len(set(changed) - set(OPTIONAL_FIELDS)) <= 1


@pytest.mark.parametrize("name", list(P3_CHANGES))
def test_every_p3_config_is_its_base_with_its_one_change(
    configs: dict[str, Config], name: str
) -> None:
    """Phase 3 (docs/phase3-plan.md section 5): `p3_baseline` differs from `resnet18_arm_a` in
    nothing, `p3_convnext` in `model` alone, and `p3_convnext_cp` from `p3_convnext` in `arm` and
    arm C's own `paste_p` alone; against the baseline, (b) is judged as one recipe."""
    base_name, change = P3_CHANGES[name]
    base, variant = asdict(configs[base_name]), asdict(configs[name])
    changed = {key: variant[key] for key in FIELDS if variant[key] != base[key]}
    assert changed == change
    assert len(set(changed) - set(OPTIONAL_FIELDS)) <= 1
    arm_a, cp = asdict(configs[P2_BASE]), asdict(configs["p3_convnext_cp"])
    assert [key for key in FIELDS if arm_a[key] != cp[key]] == ["model", "arm", "paste_p"]
    assert asdict(configs["p2_arm_c"])["paste_p"] == cp["paste_p"] == 0.5  # phase 2's setting


def test_the_p2_config_hashes(configs: dict[str, Config]) -> None:
    """The control re-run shares arm A's config hash (it is arm A); every variant has its own,
    so its rows can never pass as another recipe's."""
    arm_a = config_hash(configs[P2_BASE])
    assert config_hash(configs["p2_baseline"]) == arm_a
    variants = [config_hash(configs[name]) for name in P2_CHANGES if name != "p2_baseline"]
    assert len(set(variants)) == len(variants)
    assert arm_a not in variants and config_hash(configs["small_cnn"]) not in variants


def test_the_phase1_config_hashes_are_the_ledgers(configs: dict[str, Config]) -> None:
    """Adding `input_size` and `paste_p` left every phase-1 config hash as it was: each
    docs/ledger.md row's config, at the row's run seed, still hashes to the row's config hash."""
    for name, digest in PHASE1_HASHES.items():
        assert config_hash(configs[name]) == digest, name
    rows, problems = check_ledger(LEDGER_PATH)
    assert problems == [] and rows
    for row in rows:
        config = with_overrides(
            row["config"], configs[row["config"]], {"run_seed": int(row["run seed"])}
        )
        assert config_hash(config) == row["config hash"] == PHASE1_HASHES[row["config"]]


# The config hashes of docs/ledger-phase2.md's configs (item 13.08, review 13.08): the configs with
# rows and the pre-declared ones, so a config edited after its run, or before it, fails here.
PHASE2_HASHES = {
    "p2_baseline": PHASE1_HASHES["resnet18_arm_a"],  # input_size 224 and paste_p None: neutral
    "p2_arm_c": "5c7c51c2faea18b8f814eb58cb54ca34ade489302e2689781092ab633c313b51",
    "p2_arm_c_p25": "42d47bfda80f99a821b0257a1f1c74543d93587cf2f5966a735899df54410fb9",
    "p2_head3": "a21b04f00a14b244a72e0cc425c575dfebafc7e263ab803e145fa78af8073b56",
    "p2_res320": "5a20cf0771a6e3f81b2fbf4ee1a23bbb19738fcca761bf87ef07e7cf825febfd",
    "p2_r50": "05a685a9c19ef0b43693be03373de08dce8af240c1c02c1f99ff1fa3427a78ed",
}


def test_the_phase2_config_hashes_are_the_ledgers(configs: dict[str, Config]) -> None:
    """Review 13.08, R25's phase-2 line: the phase-2 checker compares rows with each other only,
    so this ties each docs/ledger-phase2.md row to today's config.yaml, as the phase-1 test does:
    the row's config at the row's run seed hashes to the row's config hash. `p2_baseline` rows
    carry arm A's phase-1 hash (`input_size` 224 and `paste_p` None are hash-neutral); arm C's
    rows do not (a set `paste_p` is hashed). Every phase-2 config's hash is pinned too."""
    assert set(PHASE2_HASHES) == set(P2_CHANGES)
    for name, digest in PHASE2_HASHES.items():
        assert config_hash(configs[name]) == digest, name
    rows, problems = check_ledger(train_module.PHASE2_LEDGER_PATH)
    assert problems == [] and rows
    for row in rows:
        config = with_overrides(
            row["config"], configs[row["config"]], {"run_seed": int(row["run seed"])}
        )
        assert config_hash(config) == row["config hash"] == PHASE2_HASHES[row["config"]], row[
            "run id"
        ]
    by_config = {row["config"]: row["config hash"] for row in rows}
    assert by_config["p2_baseline"] == PHASE1_HASHES["resnet18_arm_a"]
    assert by_config["p2_arm_c"] != PHASE1_HASHES["resnet18_arm_a"]


# The config hashes of the phase-3 configs, pinned before any phase-3 run (item 13.11): a config
# edited before its run, or after it, fails here. `p3_baseline` is arm A's phase-1 hash.
PHASE3_HASHES = {
    "p3_baseline": PHASE1_HASHES["resnet18_arm_a"],
    "p3_convnext": "7240842c4d79b3d9401065374eb0aa35c4cdfc253f95917be54646fccc919e59",
    "p3_convnext_cp": "a8ed0edc19dfb978a3bc34b78f94c6ed5a4dd063ca0e3dad116989701c8a0a27",
}


def test_the_phase3_config_hashes_are_pinned_and_the_others_unchanged(
    configs: dict[str, Config],
) -> None:
    """Phase 3 adds no field, so `HASH_NEUTRAL` is phase 2's and every phase-1 and phase-2 hash
    is the pinned one (the two tests above); `p3_baseline` carries arm A's hash, so its control
    row is comparable with `resnet18_arm_a`'s, and the two candidates have hashes of their own.
    Every row of docs/ledger-phase3.md, however many it holds, must hash to its config's."""
    assert set(PHASE3_HASHES) == set(P3_CHANGES)
    for name, digest in PHASE3_HASHES.items():
        assert config_hash(configs[name]) == digest, name
    assert dict(HASH_NEUTRAL) == {"input_size": INPUT_SIZE, "paste_p": None}
    others = {*PHASE1_HASHES.values(), *PHASE2_HASHES.values()}
    assert PHASE3_HASHES["p3_baseline"] in others
    assert {PHASE3_HASHES["p3_convnext"], PHASE3_HASHES["p3_convnext_cp"]}.isdisjoint(others)
    assert PHASE3_HASHES["p3_convnext"] != PHASE3_HASHES["p3_convnext_cp"]
    rows, problems = check_ledger(train_module.PHASE3_LEDGER_PATH)
    assert problems == []
    for row in rows:
        config = with_overrides(
            row["config"], configs[row["config"]], {"run_seed": int(row["run seed"])}
        )
        assert config_hash(config) == row["config hash"] == PHASE3_HASHES[row["config"]]


def test_input_size_is_left_out_of_the_hash_at_224_only(configs: dict[str, Config]) -> None:
    """`HASH_NEUTRAL`: at 224 the hash is the one of the fields without `input_size`, as before
    the field existed; any other value changes it."""
    assert dict(HASH_NEUTRAL) == {"input_size": INPUT_SIZE, "paste_p": None}
    assert INPUT_SIZE == 224
    a = configs[P2_BASE]
    old = {k: v for k, v in asdict(a).items() if k not in (*HASH_EXCLUDED, *HASH_NEUTRAL)}
    text = json.dumps(old, sort_keys=True, separators=(",", ":"))
    assert config_hash(a) == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert config_hash(replace(a, input_size=320)) == config_hash(configs["p2_res320"])
    assert config_hash(replace(a, input_size=320)) != config_hash(a)


def _no_paste(
    image: Image.Image, boxes: Sequence[Box], night: bool | None, location: int | None
) -> tuple[Image.Image, tuple[Box, ...]]:
    return image, tuple(boxes)


def test_every_arm_is_a_training_transform_arm(configs: dict[str, Config]) -> None:
    """Arm C's transform needs the run's paste step, which train.py builds from the background
    pool; a stand-in that pastes nothing is enough here."""
    for config in configs.values():
        assert config.arm in ARMS
        paste = _no_paste if config.arm == "C" else None
        assert TrainTransform(config.arm, paste=paste).arm == config.arm
        assert config.model in MODEL_NAMES
    assert ARMS == ("A", "B", "C")


def test_arm_c_is_arm_a_with_arm_c_and_its_paste_probability(configs: dict[str, Config]) -> None:
    """Item 13.04: the phase-2 arm C config is resnet18_arm_a's recipe, field for field, but
    `arm` and `paste_p` (R25's rule for arm C against arm A); only arm C sets `paste_p`."""
    a, c = asdict(configs["resnet18_arm_a"]), asdict(configs["p2_arm_c"])
    assert [key for key in FIELDS if a[key] != c[key]] == ["arm", "paste_p"]
    assert (c["arm"], c["paste_p"]) == ("C", 0.5)
    assert is_phase2("p2_arm_c") and "p2_arm_c" in P2_CHANGES
    for name, config in configs.items():
        assert (config.paste_p is not None) == (config.arm == "C"), name


def test_arm_c_p25_is_p2_arm_c_at_a_quarter_paste_probability(configs: dict[str, Config]) -> None:
    """Item 13.08, arm C's pre-declared follow-up: it loads as a phase-2 config, differs from
    `p2_baseline` in `arm` and `paste_p` alone (one change, as `p2_arm_c` does) and from
    `p2_arm_c` in `paste_p` alone, and its hashed `paste_p` gives it a config hash of its own, so
    its rows never pass as `p2_arm_c`'s. Its refusal against docs/ledger.md is in
    tests/test_train.py (`PHASE_MIXUPS`) and tests/test_notebook.py."""
    assert load_config("p2_arm_c_p25") == configs["p2_arm_c_p25"]
    assert is_phase2("p2_arm_c_p25")
    base, c, p25 = (asdict(configs[n]) for n in ("p2_baseline", "p2_arm_c", "p2_arm_c_p25"))
    assert [key for key in FIELDS if base[key] != p25[key]] == ["arm", "paste_p"]
    assert [key for key in FIELDS if c[key] != p25[key]] == ["paste_p"]
    assert (p25["arm"], p25["paste_p"], p25["input_size"]) == ("C", 0.25, 224)
    assert "input_size" not in _raw()["configs"]["p2_arm_c_p25"]  # from common
    digests = [config_hash(configs[n]) for n in ("p2_baseline", "p2_arm_c", "p2_arm_c_p25")]
    assert len(set(digests)) == 3
    assert config_hash(replace(configs["p2_arm_c_p25"], paste_p=0.5)) == digests[1]


def test_the_model_names_are_the_factorys() -> None:
    """The names are imported from trailcam/model/factory.py, not copied (review 5.02)."""
    assert config_module.MODEL_NAMES is factory.MODEL_NAMES


# --- the config hash of the ledger (review 5.02, user's decision) and the overrides ---


def test_arm_a_and_arm_b_share_the_config_hash(configs: dict[str, Config]) -> None:
    """R25: the hash leaves out `arm` and `run_seed` alone, so the two arms of one seed, and the
    seeds of one arm, share it; item 7.01's ledger checker compares it."""
    a, b = configs["resnet18_arm_a"], configs["resnet18_arm_b"]
    assert HASH_EXCLUDED == ("arm", "run_seed")
    assert config_hash(a) == config_hash(b) == config_hash(replace(b, run_seed=2))
    assert len(config_hash(a)) == 64 and int(config_hash(a), 16) >= 0
    assert config_hash(a) != config_hash(configs["small_cnn"])


@pytest.mark.parametrize("field", [f for f in FIELDS if f not in ("arm", "run_seed")])
def test_every_other_field_changes_the_config_hash(configs: dict[str, Config], field: str) -> None:
    """A phase-2 field is checked on the config that sets it."""
    c = configs["p2_arm_c" if field in OPTIONAL_FIELDS else "resnet18_arm_b"]
    value = getattr(c, field)
    if isinstance(value, bool):
        other: object = not value
    elif isinstance(value, int):
        other = value + 1
    elif isinstance(value, float):
        other = value * 2
    else:
        other = value + "x"
    assert config_hash(replace(c, **{field: other})) != config_hash(c)


def test_paste_p_is_hashed_only_when_set(configs: dict[str, Config]) -> None:
    """`HASH_NEUTRAL` holds `paste_p` at None, its value when a config leaves it out: a phase-1
    config hashes as before the field existed. Set, it changes the hash, so arm C does not share
    arm A's hash (the user's decision on 13.04; its rows go to the phase-2 ledger)."""
    a = configs["resnet18_arm_a"]
    assert a.paste_p is None and HASH_NEUTRAL["paste_p"] is None
    before = {k: v for k, v in asdict(a).items() if k not in (*HASH_EXCLUDED, *HASH_NEUTRAL)}
    text = json.dumps(before, sort_keys=True, separators=(",", ":"))
    assert config_hash(a) == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert OPTIONAL_FIELDS == ("paste_p",) and HASH_EXCLUDED == ("arm", "run_seed")
    c = configs["p2_arm_c"]
    assert config_hash(c) != config_hash(a)
    assert config_hash(replace(c, arm="A", paste_p=None)) == config_hash(a)
    assert config_hash(replace(c, paste_p=0.25)) != config_hash(c)


def test_with_overrides_applies_and_checks(configs: dict[str, Config]) -> None:
    dry = configs[DRY_RUN]
    assert with_overrides(DRY_RUN, dry, {"max_epochs": 3, "run_seed": 1}) == replace(
        dry, max_epochs=3, run_seed=1
    )
    assert with_overrides(DRY_RUN, dry, {}) == dry
    for changes in ({"max_epochs": 0}, {"batch_size": True}, {"lr_head": "1e-3"}, {"nope": 1}):
        with pytest.raises(ConfigError):
            with_overrides(DRY_RUN, dry, changes)
    arm = configs["resnet18_arm_a"]
    with pytest.raises(ConfigError, match="head_only_epochs"):
        with_overrides("resnet18_arm_a", arm, {"max_epochs": 1})
    with pytest.raises(ConfigError, match="only arm C pastes"):
        with_overrides("resnet18_arm_a", arm, {"paste_p": 0.5})
    with pytest.raises(ConfigError, match="paste_p: expected float"):
        with_overrides("p2_arm_c", configs["p2_arm_c"], {"paste_p": None})


# --- refusals, each on a copy of the committed file with one change ---


def _raw() -> dict[str, dict[str, object]]:
    raw: dict[str, dict[str, object]] = yaml.safe_load(CONFIG_PATH.read_text())
    return raw


def _parse(raw: object) -> None:
    parse_configs(yaml.safe_dump(raw, sort_keys=False))


def test_the_dumped_copy_parses_the_same() -> None:
    assert parse_configs(yaml.safe_dump(_raw(), sort_keys=False)) == load_configs()


@pytest.mark.parametrize(
    ("block", "key", "value", "message"),
    [
        ("resnet18_arm_b", "model", "resnet101", "resnet18_arm_b.model"),
        ("resnet18_arm_b", "arm", "C", "resnet18_arm_b.arm"),
        ("resnet18_arm_b", "arm", "b", "resnet18_arm_b.arm"),
        ("common", "scheduler", "step", "scheduler"),
        ("resnet18_arm_a", "lr_body", "1e-4", "write floats with a dot"),
        ("resnet18_arm_a", "lr_head", True, "lr_head: expected float"),
        ("resnet18_arm_a", "lr_head", 0.0, "must be above 0"),
        ("common", "batch_size", True, "batch_size: expected int"),
        ("common", "batch_size", 64.0, "batch_size: expected int"),
        ("common", "batch_size", 0, "at least 1"),
        ("common", "input_size", 16, "at least 32"),
        ("common", "input_size", 224.0, "input_size: expected int"),
        ("small_cnn", "input_size", 320, "must be 224 for small_cnn"),
        ("common", "split_seed", -1, "0 or more"),
        ("common", "split_seed", "42", "split_seed: expected int"),
        ("common", "pin_memory", "true", "pin_memory: expected bool"),
        ("common", "pin_memory", 1, "pin_memory: expected bool"),
        ("common", "num_workers", -1, "0 or more"),
        ("common", "patience", 0, "at least 1"),
        ("common", "weight_decay", -0.01, "0 or more"),
        ("dryrun", "max_epochs", 0, "at least 1"),
        ("resnet18_arm_a", "head_only_epochs", 10, "head_only_epochs"),
        ("dryrun", "warmup_epochs", 2, "warmup_epochs"),
        # Arm C's paste probability (item 13.04): a float from 0 to 1, for arm C alone.
        ("p2_arm_c", "paste_p", 1.5, r"p2_arm_c\.paste_p: must be between 0 and 1"),
        ("p2_arm_c", "paste_p", -0.1, r"p2_arm_c\.paste_p: must be between 0 and 1"),
        ("p2_arm_c", "paste_p", float("nan"), r"p2_arm_c\.paste_p: must be between 0 and 1"),
        ("p2_arm_c", "paste_p", None, r"p2_arm_c\.paste_p: expected float"),
        ("p2_arm_c", "paste_p", "0.5", r"p2_arm_c\.paste_p: expected float"),
        ("p2_arm_c", "paste_p", True, r"p2_arm_c\.paste_p: expected float"),
        ("p2_arm_c", "arm", "A", r"p2_arm_c\.paste_p: only arm C pastes; arm A"),
        ("resnet18_arm_b", "paste_p", 0.5, r"resnet18_arm_b\.paste_p: only arm C pastes"),
        ("common", "paste_p", 0.5, r"dryrun\.paste_p: only arm C pastes"),
    ],
)
def test_a_bad_value_is_refused(block: str, key: str, value: object, message: str) -> None:
    raw = _raw()
    target = raw["common"] if block == "common" else raw["configs"][block]
    assert isinstance(target, dict)
    target[key] = value
    with pytest.raises(ConfigError, match=message):
        _parse(raw)


def test_an_unknown_field_is_refused() -> None:
    raw = _raw()
    raw["configs"]["resnet18_arm_b"]["label_smoothing"] = 0.1  # type: ignore[index]
    with pytest.raises(ConfigError, match=r"resnet18_arm_b: unknown fields \['label_smoothing'\]"):
        _parse(raw)


@pytest.mark.parametrize("key", [f.name for f in fields(Config) if f.name not in OPTIONAL_FIELDS])
def test_a_missing_field_is_refused(key: str) -> None:
    raw = _raw()
    for block in (raw["common"], *raw["configs"].values()):
        assert isinstance(block, dict)
        block.pop(key, None)
    with pytest.raises(ConfigError, match=f"missing fields \\['{key}'\\]"):
        _parse(raw)


def test_a_phase_2_field_may_be_left_out_but_arm_c_needs_paste_p() -> None:
    """The phase-1 configs name no phase-2 field and load with it None; arm C without its paste
    probability is refused, as is the field written out as null."""
    for name in (*PHASE1, *P2_CHANGES):
        assert ("paste_p" in _raw()["configs"][name]) == ("paste_p" in P2_CHANGES.get(name, {})), (
            name
        )
    raw = _raw()
    del raw["configs"]["p2_arm_c"]["paste_p"]  # type: ignore[attr-defined]
    with pytest.raises(ConfigError, match=r"p2_arm_c\.arm: arm C needs paste_p"):
        _parse(raw)


def test_a_repeated_key_is_refused() -> None:
    """Plain YAML keeps the last of two keys: a second `arm:` in arm A's block would silently
    make it arm B."""
    text = CONFIG_PATH.read_text()
    block = text.index("  resnet18_arm_a:\n")
    line = text.index("    arm: A\n", block)
    doubled = text[:line] + "    arm: A\n    arm: B\n" + text[line + len("    arm: A\n") :]
    with pytest.raises(ConfigError, match="repeated key 'arm'"):
        parse_configs(doubled)


@pytest.mark.parametrize(
    "text",
    [
        "- a list\n",
        "common: {}\n",
        "common: {}\nconfigs: {}\n",
        "common: {}\nconfigs: {}\nextra: 1\n",
        "common: []\nconfigs: {x: {}}\n",
        "common: {}\nconfigs: {x: [1]}\n",
        "common: {1: 2}\nconfigs: {x: {}}\n",
    ],
)
def test_a_malformed_file_is_refused(text: str) -> None:
    with pytest.raises(ConfigError):
        parse_configs(text)


def test_an_unknown_name_is_refused() -> None:
    with pytest.raises(ConfigError, match="no config 'resnet18'"):
        load_config("resnet18")


def test_a_config_is_frozen(configs: dict[str, Config]) -> None:
    with pytest.raises(AttributeError):
        configs["resnet18_arm_b"].arm = "A"  # type: ignore[misc]

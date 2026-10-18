"""trailcam/train/config.py and the root config.yaml (item 5.02; R25 arm A and arm B differ in
`arm` alone, R31 the training recipe of plan.md section 5; item 13.04: arm C's `paste_p`, the
first phase-2 setting, and the phase-1 config hashes it must leave unchanged).

The first tests read the committed config.yaml; the rest feed `parse_configs` small texts built
from it, so each refusal is checked against a file that is otherwise valid.
"""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, fields, replace
from pathlib import Path

import pytest
import yaml
from PIL import Image

from trailcam.data.dataset import ARMS, TrainTransform
from trailcam.data.metadata import Box
from trailcam.model import factory
from trailcam.train import config as config_module
from trailcam.train.config import (
    CONFIG_PATH,
    DRY_RUN,
    FIELDS,
    HASH_EXCLUDED,
    MODEL_NAMES,
    PHASE2_FIELDS,
    Config,
    ConfigError,
    config_hash,
    load_config,
    load_configs,
    parse_configs,
    with_overrides,
)
from trailcam.train.ledger import check_ledger

FULL = ("small_cnn", "resnet18_arm_a", "resnet18_arm_b")
PHASE2 = ("p2_arm_c",)
LEDGER = Path(__file__).resolve().parents[1] / "docs" / "ledger.md"


@pytest.fixture(scope="module")
def configs() -> dict[str, Config]:
    return dict(load_configs())


def test_all_configs_load(configs: dict[str, Config]) -> None:
    assert list(configs) == [DRY_RUN, *FULL, *PHASE2]
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


def test_every_config_has_loader_settings(configs: dict[str, Config]) -> None:
    for name in (*FULL, *PHASE2):
        assert (configs[name].num_workers, configs[name].pin_memory) == (4, True), name
        assert configs[name].fixture is False, name


def test_arm_a_and_arm_b_differ_in_arm_alone(configs: dict[str, Config]) -> None:
    """R25: same backbone, seeds, epochs, optimizer, schedule, batch size; only the sampler."""
    a, b = asdict(configs["resnet18_arm_a"]), asdict(configs["resnet18_arm_b"])
    assert [key for key in FIELDS if a[key] != b[key]] == ["arm"]
    assert (a["arm"], b["arm"]) == ("A", "B")


def test_the_full_configs_carry_the_recipe(configs: dict[str, Config]) -> None:
    """R31: AdamW 1e-4 body / 1e-3 head, weight decay 1e-2, cosine with a one-epoch warmup,
    patience 3, batch 64, mixed precision; up to 10 backbone epochs in all, the first of them
    head-only (it counts inside max_epochs, item 5.03), up to 15 for the small CNN, which has no
    pretrained body and trains it at the head's rate."""
    for name in FULL:
        c = configs[name]
        assert (c.weight_decay, c.scheduler, c.warmup_epochs, c.patience) == (
            1e-2,
            "cosine",
            1,
            3,
        ), name
        assert (c.batch_size, c.mixed_precision, c.lr_head) == (64, True, 1e-3), name
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
    assert all(name.startswith("p2_") for name in PHASE2)
    for name, config in configs.items():
        assert (config.paste_p is not None) == (config.arm == "C"), name


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
    c = configs["p2_arm_c" if field in PHASE2_FIELDS else "resnet18_arm_b"]
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


def test_the_phase_1_config_hashes_are_the_ledgers(configs: dict[str, Config]) -> None:
    """docs/phase2-plan.md section 4: a phase-2 setting must leave the phase-1 hashes as they
    are, or `score_test.py` refuses the phase-1 checkpoints. Every row of `docs/ledger.md` names a
    phase-1 config whose hash today is the row's, and every phase-1 config but the dry run has
    rows (seven at 7.03: one small CNN, three seeds of each ResNet-18 arm)."""
    rows, problems = check_ledger(LEDGER)
    assert problems == []
    assert len(rows) >= 7
    assert {row["config"] for row in rows} == set(FULL)
    for row in rows:
        assert config_hash(configs[row["config"]]) == row["config hash"], row.line
    for name in FULL:
        assert all(getattr(configs[name], key) is None for key in PHASE2_FIELDS), name


def test_a_phase_2_field_is_hashed_only_when_set(configs: dict[str, Config]) -> None:
    """Left out (None), a phase-2 field leaves the hash as it was before the field existed: the
    hash of the other fields, `arm` and `run_seed` aside. Set, it changes the hash, so arm C does
    not share arm A's hash (its rows go to the phase-2 ledger, item 13.02)."""
    a = configs["resnet18_arm_a"]
    before = {k: v for k, v in asdict(a).items() if k not in (*HASH_EXCLUDED, *PHASE2_FIELDS)}
    text = json.dumps(before, sort_keys=True, separators=(",", ":"))
    assert config_hash(a) == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert PHASE2_FIELDS == ("paste_p",) and HASH_EXCLUDED == ("arm", "run_seed")
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
        ("resnet18_arm_b", "model", "resnet50", "resnet18_arm_b.model"),
        ("resnet18_arm_b", "arm", "C", "resnet18_arm_b.arm"),
        ("resnet18_arm_b", "arm", "b", "resnet18_arm_b.arm"),
        ("common", "scheduler", "step", "scheduler"),
        ("resnet18_arm_a", "lr_body", "1e-4", "write floats with a dot"),
        ("resnet18_arm_a", "lr_head", True, "lr_head: expected float"),
        ("resnet18_arm_a", "lr_head", 0.0, "must be above 0"),
        ("common", "batch_size", True, "batch_size: expected int"),
        ("common", "batch_size", 64.0, "batch_size: expected int"),
        ("common", "batch_size", 0, "at least 1"),
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


@pytest.mark.parametrize("key", [f.name for f in fields(Config) if f.name not in PHASE2_FIELDS])
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
    for name in FULL:
        assert "paste_p" not in _raw()["configs"][name]
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

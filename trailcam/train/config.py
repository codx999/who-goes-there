"""The training configs of the root `config.yaml` (plan.md sections 5 and 6).

`load_configs(path)` reads the file and returns one frozen `Config` per name under `configs`,
each the `common` block updated with its own. Nothing is defaulted in code: a config that misses
a field, names an unknown one, repeats a key, or gives a value of the wrong type or range is
refused with a `ConfigError` naming the config and the field. The arm values are `ARMS` of
`trailcam/data/dataset.py`, the switch of `TrainTransform`; the model names are `MODEL_NAMES` of
`trailcam/model/factory.py`, the names `build_model` accepts.

A config's name is its key, not a field, so the configs of arm A and arm B differ in `arm` alone
(R25); the ledger records the name in its config column. `DRY_RUN` is the config `/dryrun` uses:
one epoch, batch 2, the synthetic fixture (`fixture: true`) and no DataLoader workers.
`with_overrides` applies the few command-line changes `train.py` allows and checks the result as a
loaded config is checked. `config_hash` is the ledger's config hash: a SHA-256 of the resolved
config without `arm` and `run_seed` (`HASH_EXCLUDED`), so the arm A and arm B rows of one seed,
and the seeds of one arm, share it, and a run whose values differ from the committed file does not.

Phase 2 (`docs/phase2-plan.md`, item 13.02). A config whose name starts with `PHASE2_PREFIX`
(`p2_`) is a phase-2 config: its runs go to `docs/ledger-phase2.md`, never `docs/ledger.md`
(`trailcam/train/train.py` and `trailcam/train/ledger.py` refuse both mix-ups). A field added for
phase 2 is listed in `HASH_NEUTRAL` with the value phase 1 used: at that value it is left out of
the hash, so every phase-1 config hash stays the one in `docs/ledger.md` (which `score_test.py`
checks), and any other value changes the hash. `input_size` is the first: the side of the
training crop and of the eval centre crop, 224 in phase 1 (`trailcam/data/dataset.py`). The small
CNN's flatten head is sized for 224, so its configs keep 224. `paste_p`, arm C's paste
probability (item 13.04), is the second, and the one field a config may leave out
(`OPTIONAL_FIELDS`, the one exception to "nothing is defaulted"): left out it is None, its
phase-1 value. Arm C sets it and no other arm may, so `p2_arm_c` hashes it and has a config hash
of its own (the user's decision on 13.04).

Phase 3 (`docs/phase3-plan.md`, item 13.11) adds a second prefix the same way: a config whose
name starts with `PHASE3_PREFIX` (`p3_`) is a phase-3 config, whose runs go to
`docs/ledger-phase3.md` only. `phase_of` gives a name's phase from `PHASE_PREFIXES` (1 for any
name without a phase prefix), and `train.py` and `ledger.py` route by it. Phase 3 adds no field,
so `HASH_NEUTRAL` is unchanged and `p3_baseline`, `resnet18_arm_a` exactly, shares its hash.

The file sits at the repository root, outside the package. `CONFIG_PATH` finds it from a checkout
or an editable install; a run from an installed wheel passes its path.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Hashable, Mapping
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import yaml

from trailcam.data.dataset import ARMS, INPUT_SIZE
from trailcam.model.factory import MODEL_NAMES

CONFIG_PATH: Final = Path(__file__).resolve().parents[2] / "config.yaml"
DRY_RUN: Final = "dryrun"
PHASE2_PREFIX: Final = "p2_"  # phase-2 config names (docs/phase2-plan.md section 4)
PHASE3_PREFIX: Final = "p3_"  # phase-3 config names (docs/phase3-plan.md section 4)
PHASE_PREFIXES: Final = MappingProxyType({2: PHASE2_PREFIX, 3: PHASE3_PREFIX})  # 1: no prefix
SCHEDULERS: Final = ("cosine",)  # cosine annealing after a linear warmup (plan.md section 5)


class ConfigError(ValueError):
    """The config file or one of its configs is malformed."""


@dataclass(frozen=True)
class Config:
    """One training run's settings; `loaders` and `train.py` read them, nothing else sets them."""

    model: str
    arm: str
    pretrained: bool  # ImageNet-1k weights for a backbone; the small CNN has none
    split_seed: int
    run_seed: int
    max_epochs: int
    head_only_epochs: int  # the first epochs train the head alone (plan.md section 5)
    batch_size: int
    input_size: int  # train crop and eval centre crop side, px; 224 in phase 1 (HASH_NEUTRAL)
    lr_body: float
    lr_head: float
    weight_decay: float
    scheduler: str
    warmup_epochs: int
    patience: int  # epochs without a better val macro-F1 before early stopping
    mixed_precision: bool
    num_workers: int
    pin_memory: bool
    fixture: bool  # the dry run: train on the synthetic fixture in a temporary directory
    # Phase 2: None when the config leaves it out (OPTIONAL_FIELDS), then out of the hash.
    paste_p: float | None = None  # arm C: share of boxed train images pasted (copypaste.py)


FIELDS: Final = tuple(f.name for f in fields(Config))
HASH_EXCLUDED: Final = ("arm", "run_seed")  # the fields arm A and B, or two seeds, may differ in
# Phase-2 fields, left out of the hash at their phase-1 value so the phase-1 hashes stay the same.
# `paste_p` did not exist in phase 1: None, which is also its value when a config leaves it out.
HASH_NEUTRAL: Final = MappingProxyType({"input_size": INPUT_SIZE, "paste_p": None})
OPTIONAL_FIELDS: Final = ("paste_p",)  # may be left out (None); arm C sets it, no other arm does
_TYPES: Final = {f.name: str(f.type).removesuffix(" | None") for f in fields(Config)}


class _UniqueKeyLoader(yaml.SafeLoader):
    """A safe loader that refuses a repeated key, which plain YAML resolves to the last value."""


def _construct_unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode) -> dict[Any, Any]:
    seen: set[Hashable] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node)
        if key in seen:
            raise ConfigError(f"repeated key {key!r} (line {key_node.start_mark.line + 1})")
        seen.add(key)
    return loader.construct_mapping(node)


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _value(name: str, key: str, value: object) -> object:
    """`value` checked against the type of field `key`; an int is accepted for a float."""
    kind = _TYPES[key]
    if kind == "bool":
        ok = isinstance(value, bool)
    elif kind == "int":
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif kind == "float":
        ok = isinstance(value, int | float) and not isinstance(value, bool)
        if ok and isinstance(value, int | float):
            value = float(value)
    else:
        ok = isinstance(value, str)
    if not ok:
        hint = ""
        if kind == "float":
            hint = " (write floats with a dot, 1.0e-4, or YAML reads a string)"
        raise ConfigError(f"{name}.{key}: expected {kind}, got {value!r}{hint}")
    return value


def _check(name: str, config: Config) -> None:
    def refuse(key: str, why: str) -> None:
        raise ConfigError(f"{name}.{key}: {why}, got {getattr(config, key)!r}")

    if config.model not in MODEL_NAMES:
        refuse("model", f"expected one of {MODEL_NAMES}")
    if config.arm not in ARMS:
        refuse("arm", f"expected one of {ARMS}")
    if config.scheduler not in SCHEDULERS:
        refuse("scheduler", f"expected one of {SCHEDULERS}")
    for key in ("split_seed", "run_seed", "num_workers", "head_only_epochs", "warmup_epochs"):
        if getattr(config, key) < 0:
            refuse(key, "must be 0 or more")
    for key in ("max_epochs", "batch_size", "patience"):
        if getattr(config, key) < 1:
            refuse(key, "must be at least 1")
    if config.input_size < 32:
        refuse("input_size", "must be at least 32 (a ResNet or ConvNeXt halves it five times)")
    if config.model == "small_cnn" and config.input_size != INPUT_SIZE:
        refuse("input_size", f"must be {INPUT_SIZE} for small_cnn, whose head is sized for it")
    for key in ("lr_body", "lr_head"):
        if not getattr(config, key) > 0:
            refuse(key, "must be above 0")
    if not config.weight_decay >= 0:
        refuse("weight_decay", "must be 0 or more")
    if config.head_only_epochs >= config.max_epochs:
        refuse("head_only_epochs", f"must leave the body one of {config.max_epochs} epochs")
    if config.warmup_epochs > config.max_epochs:
        refuse("warmup_epochs", f"must fit in {config.max_epochs} epochs")
    if config.arm == "C" and config.paste_p is None:
        refuse("arm", "arm C needs paste_p, its paste probability")
    if config.arm != "C" and config.paste_p is not None:
        refuse("paste_p", f"only arm C pastes; arm {config.arm} takes no paste_p")
    if config.paste_p is not None and not 0.0 <= config.paste_p <= 1.0:
        refuse("paste_p", "must be between 0 and 1")


def _config(name: str, common: Mapping[str, object], own: Mapping[str, object]) -> Config:
    raw = {**common, **own}
    unknown = sorted(set(raw) - set(FIELDS), key=str)
    if unknown:
        raise ConfigError(f"{name}: unknown fields {unknown}")
    missing = [key for key in FIELDS if key not in raw and key not in OPTIONAL_FIELDS]
    if missing:
        raise ConfigError(f"{name}: missing fields {missing}")
    values = {key: _value(name, key, raw[key]) for key in FIELDS if key in raw}
    config = Config(**values)  # type: ignore[arg-type]
    _check(name, config)
    return config


def _mapping(where: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(k, str) for k in value):
        raise ConfigError(f"{where}: expected a mapping with string keys")
    return value


def parse_configs(text: str) -> Mapping[str, Config]:
    """Every config of a `config.yaml` text, by name, in file order."""
    top = _mapping("config file", yaml.load(text, Loader=_UniqueKeyLoader))
    if set(top) != {"common", "configs"}:
        raise ConfigError(f"config file: expected the keys common and configs, got {sorted(top)}")
    common = _mapping("common", top["common"])
    named = _mapping("configs", top["configs"])
    if not named:
        raise ConfigError("configs: no config")
    configs = {name: _config(name, common, _mapping(name, own)) for name, own in named.items()}
    return MappingProxyType(configs)


def load_configs(path: Path = CONFIG_PATH) -> Mapping[str, Config]:
    """Every config of the file at `path`, by name."""
    return parse_configs(path.read_text(encoding="utf-8"))


def load_config(name: str, path: Path = CONFIG_PATH) -> Config:
    """The config called `name`; refuses an unknown name with the names there are."""
    configs = load_configs(path)
    if name not in configs:
        raise ConfigError(f"no config {name!r} in {path.name}; expected one of {list(configs)}")
    return configs[name]


def with_overrides(name: str, config: Config, changes: Mapping[str, object]) -> Config:
    """`config` with `changes` applied, each value and the result checked as a loaded config is."""
    unknown = sorted(set(changes) - set(FIELDS))
    if unknown:
        raise ConfigError(f"{name}: unknown fields {unknown}")
    checked = {key: _value(name, key, value) for key, value in changes.items()}
    result = replace(config, **checked)  # type: ignore[arg-type]
    _check(name, result)
    return result


def phase_of(name: str) -> int:
    """The phase of config `name`: that of its prefix in `PHASE_PREFIXES`, else 1."""
    for phase, prefix in PHASE_PREFIXES.items():
        if name.startswith(prefix):
            return phase
    return 1


def is_phase2(name: str) -> bool:
    """Whether config `name` is a phase-2 config (`PHASE2_PREFIX`), whose runs go to
    `docs/ledger-phase2.md` only."""
    return phase_of(name) == 2


def is_phase3(name: str) -> bool:
    """Whether config `name` is a phase-3 config (`PHASE3_PREFIX`), whose runs go to
    `docs/ledger-phase3.md` only."""
    return phase_of(name) == 3


def config_hash(config: Config) -> str:
    """SHA-256 (hex) of the config's fields but `HASH_EXCLUDED`, as sorted-key JSON; a field of
    `HASH_NEUTRAL` is left out while it holds its phase-1 value."""
    values = {
        key: value
        for key, value in asdict(config).items()
        if key not in HASH_EXCLUDED and not (key in HASH_NEUTRAL and value == HASH_NEUTRAL[key])
    }
    text = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

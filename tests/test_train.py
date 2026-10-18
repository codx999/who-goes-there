"""trailcam/train/train.py: the training loop, its command line and the ledger row (plan.md
section 5; R16 leakage alarms, R17 selection on val only, R25 the arm and the overrides, R27 the
run seed, R31 the recipe, R37 the dry run, R40 the ledger; item 13.02, phase 2: a config's input
size reaches both transforms, and a `p2_` row goes to docs/ledger-phase2.md and nowhere else).

The loop tests train a random-weight ResNet-18 on 32 px random tensors, so they are quick; the
early-stopping tests replace the val scoring with a scripted sequence of macro-F1 values. The
end-to-end test runs `main` as `/dryrun` does, on the synthetic fixture the script writes itself,
with the small CNN. No test opens a network connection or downloads weights.
"""

import inspect
import random
import re
import socket
import tempfile
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from trailcam.data import synthetic
from trailcam.data.boxcrop import BoxCropSampler
from trailcam.data.classes import NUM_CLASSES
from trailcam.data.dataset import CCT20Dataset, ControlCropSampler, EvalTransform
from trailcam.data.metadata import load_metadata
from trailcam.data.split import make_split
from trailcam.evaluate.attention import CHECKPOINT_KEYS, load_checkpoint
from trailcam.model.factory import build_model, head_module, parameter_groups
from trailcam.train import train
from trailcam.train.config import (
    CONFIG_PATH,
    DRY_RUN,
    Config,
    config_hash,
    load_config,
    load_configs,
)
from trailcam.train.ledger import check_ledger
from trailcam.train.train import (
    LEDGER_COLUMNS,
    LEDGER_PATH,
    PHASE2_LEDGER_PATH,
    EarlyStopping,
    LeakageError,
    Scores,
    append_row,
    check_loaders,
    fit,
    ledger_header,
    ledger_row,
    lr_factor,
    score_alarms,
    seed_everything,
)

CPU = torch.device("cpu")


def _refuse_connection(*args: object, **kwargs: object) -> None:
    raise AssertionError("a training test tried to open a network connection")


@pytest.fixture(scope="module", autouse=True)
def no_network() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(socket.socket, "connect", _refuse_connection)
        yield


def config(**changes: Any) -> Config:
    """The dry-run config (random weights, no workers) with `changes`."""
    return replace(load_config(DRY_RUN), **changes)


def tensor_loaders(seed: int = 0, n: int = 8) -> dict[str, DataLoader[Any]]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, 32, 32, generator=g)
    y = torch.arange(n) % NUM_CLASSES
    shuffle = torch.Generator().manual_seed(seed)
    return {
        "train": DataLoader(TensorDataset(x, y), batch_size=4, shuffle=True, generator=shuffle),
        "val": DataLoader(TensorDataset(x, y), batch_size=4),
    }


def resnet(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return build_model("resnet18", pretrained=False)


def state(model: nn.Module) -> dict[str, Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def same(a: dict[str, Tensor], b: dict[str, Tensor]) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


class ScriptedVal:
    """Stands in for `train.evaluate`: returns the next scripted val macro-F1 and keeps a copy of
    the weights it was called with, by epoch."""

    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.states: dict[int, dict[str, Tensor]] = {}

    def __call__(self, model: nn.Module, loader: object, device: object, amp: bool) -> Scores:
        epoch = len(self.states) + 1
        self.states[epoch] = state(model)
        return Scores(loss=1.0, accuracy=0.5, macro_f1=self.scores[epoch - 1])


# Early stopping and the restored checkpoint (R17, R31).


def test_early_stopping_stops_after_patience_epochs_without_improvement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Best at epoch 2; epochs 3, 4 (a tie is no improvement) and 5 bring nothing higher, so
    patience 3 stops after epoch 5 of 10, and epoch 2's weights are the ones restored."""
    scripted = ScriptedVal([0.2, 0.5, 0.4, 0.5, 0.3, 0.9, 0.9, 0.9, 0.9, 0.9])
    monkeypatch.setattr(train, "evaluate", scripted)
    model = resnet()
    result = fit(
        model, tensor_loaders(), config(max_epochs=10, patience=3), CPU, log=lambda _: None
    )
    assert len(result.epochs) == 5
    assert result.best_epoch == 2
    assert same(state(model), scripted.states[2])
    assert not same(state(model), scripted.states[5])


def test_a_new_best_resets_the_patience(monkeypatch: pytest.MonkeyPatch) -> None:
    scripted = ScriptedVal([0.1, 0.2, 0.1, 0.1, 0.3, 0.1, 0.1, 0.1, 0.9, 0.9])
    monkeypatch.setattr(train, "evaluate", scripted)
    model = resnet()
    result = fit(
        model, tensor_loaders(), config(max_epochs=10, patience=3), CPU, log=lambda _: None
    )
    assert (len(result.epochs), result.best_epoch) == (8, 5)
    assert same(state(model), scripted.states[5])


def test_a_run_that_keeps_improving_runs_every_epoch(monkeypatch: pytest.MonkeyPatch) -> None:
    scripted = ScriptedVal([0.1, 0.2, 0.3, 0.4])
    monkeypatch.setattr(train, "evaluate", scripted)
    model = resnet()
    result = fit(model, tensor_loaders(), config(max_epochs=4, patience=1), CPU, log=lambda _: None)
    assert (len(result.epochs), result.best_epoch) == (4, 4)
    assert same(state(model), scripted.states[4])


def test_each_epoch_line_gives_its_seconds_and_images_per_second(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Item 6.08: a committed Kaggle run's Logs tab shows each epoch as it ends, with the epoch's
    seconds and training images per second; the scores in the line are unchanged."""
    monkeypatch.setattr(train, "evaluate", ScriptedVal([0.1, 0.2]))
    lines: list[str] = []
    fit(resnet(), tensor_loaders(), config(max_epochs=2, patience=3), CPU, log=lines.append)
    epochs = [line for line in lines if line.startswith("epoch ")]
    assert len(epochs) == 2
    for line in epochs:
        assert re.fullmatch(
            r"epoch \d/2( head-only)?: train loss \d+\.\d{4}, val loss \d+\.\d{4}, "
            r"val accuracy \d\.\d{4}, val macro-F1 \d\.\d{4} \(best epoch \d\); "
            r"\d+\.\d s, \d+\.\d train images/s",
            line,
        ), line


def test_log_lines_are_flushed_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Item 6.08: `fit` and `train_run` print through `log_line`, which flushes every line."""
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(train, "print", lambda *a, **k: calls.append((a, k)), raising=False)
    train.log_line("epoch 1/1")
    assert calls == [(("epoch 1/1",), {"flush": True})]
    for function in (train.fit, train.train_run):
        assert inspect.signature(function).parameters["log"].default is train.log_line


def test_early_stopping_keeps_the_first_of_equal_scores() -> None:
    stopper = EarlyStopping(patience=2)
    model = nn.Linear(2, 2)
    assert not stopper.update(1, 0.5, model)
    assert not stopper.update(2, 0.5, model)
    assert stopper.update(3, 0.4, model)
    assert stopper.best_epoch == 1


# The recipe (R31).


def test_the_schedule_warms_up_linearly_then_anneals_by_cosine() -> None:
    """4 steps an epoch, one warmup epoch of 3: 1/4, 2/4, 3/4, 4/4, then the cosine from 1 at
    step 4 to 0.5 at the midpoint (step 8 of the 8 cosine steps 4 to 11)."""
    c = config(max_epochs=3, warmup_epochs=1, head_only_epochs=0)
    factors = [lr_factor(step, 4, c) for step in range(12)]
    assert factors[:4] == pytest.approx([0.25, 0.5, 0.75, 1.0])
    assert factors[4] == pytest.approx(1.0)
    assert factors[8] == pytest.approx(0.5)
    assert factors[11] == pytest.approx(0.5 * (1 + np.cos(np.pi * 7 / 8)))
    assert all(a >= b for a, b in zip(factors[4:], factors[5:], strict=False))


def test_adamw_gets_the_body_and_head_groups_at_their_rates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    made: list[torch.optim.AdamW] = []
    real = torch.optim.AdamW

    def spy(*args: Any, **kwargs: Any) -> torch.optim.AdamW:
        made.append(real(*args, **kwargs))
        return made[-1]

    monkeypatch.setattr(train.torch.optim, "AdamW", spy)
    model = resnet()
    c = config(max_epochs=1, lr_body=1e-4, lr_head=1e-3, weight_decay=1e-2)
    fit(model, tensor_loaders(), c, CPU, log=lambda _: None)
    groups = parameter_groups(model)
    body, head = made[0].param_groups
    assert [id(p) for p in body["params"]] == [id(p) for p in groups.body]
    assert [id(p) for p in head["params"]] == [id(p) for p in groups.head]
    assert (body["initial_lr"], head["initial_lr"]) == (1e-4, 1e-3)
    assert body["weight_decay"] == head["weight_decay"] == 1e-2


def test_the_head_only_epoch_leaves_the_body_as_it_was(monkeypatch: pytest.MonkeyPatch) -> None:
    """Epoch 1 of 2 is head-only: body parameters and batch-norm statistics unchanged, the head
    changed; epoch 2 trains the body."""
    scripted = ScriptedVal([0.1, 0.2])
    monkeypatch.setattr(train, "evaluate", scripted)
    model = resnet()
    head_keys = {f"fc.{k}" for k in head_module(model).state_dict()}
    before = state(model)
    fit(model, tensor_loaders(), config(max_epochs=2, head_only_epochs=1), CPU, log=lambda _: None)

    def body(s: dict[str, Tensor]) -> dict[str, Tensor]:
        return {k: v for k, v in s.items() if k not in head_keys}

    def head(s: dict[str, Tensor]) -> dict[str, Tensor]:
        return {k: v for k, v in s.items() if k in head_keys}

    after_1, after_2 = scripted.states[1], scripted.states[2]
    assert any("running_mean" in k for k in body(before))
    assert same(body(after_1), body(before))
    assert not same(head(after_1), head(before))
    assert not same(body(after_2), body(after_1))


# Seeds (R27).


def test_seed_everything_seeds_random_numpy_and_torch() -> None:
    def draws() -> tuple[float, float, float, float]:
        g = seed_everything(7)
        return (
            random.random(),
            float(np.random.rand()),
            float(torch.rand(1)),
            float(torch.rand(1, generator=g)),
        )

    assert draws() == draws()


def test_the_same_run_seed_trains_the_same_weights() -> None:
    def run() -> dict[str, Tensor]:
        seed_everything(3)
        model = build_model("resnet18", pretrained=False)
        fit(model, tensor_loaders(), config(max_epochs=1), CPU, log=lambda _: None)
        return state(model)

    assert same(run(), run())


# Leakage alarms (R16).


@pytest.fixture(scope="module")
def fixture_split(tmp_path_factory: pytest.TempPathFactory) -> Any:
    root = tmp_path_factory.mktemp("train") / "fixture"
    fixture = synthetic.write_synthetic_fixture(root)
    meta = load_metadata(root, fixture.paths["sidecar"])
    return make_split(meta, 42), fixture.image_root, meta


def test_a_train_id_in_another_split_stops_the_run(fixture_split: Any) -> None:
    split, root, _ = fixture_split
    train_records = split.records["train"]
    leaked = (*split.records["val"], replace(train_records[0], split="val"))

    def loader(records: Any) -> DataLoader[Any]:
        return DataLoader(CCT20Dataset(records, root, EvalTransform()))

    loaders = {"train": loader(train_records), "val": loader(leaked)}
    loaders["trans_val"] = loader(split.records["trans_val"])
    with pytest.raises(LeakageError, match="val=1"):
        check_loaders(loaders)
    loaders["val"] = loader(split.records["val"])
    check_loaders(loaders)
    with pytest.raises(LeakageError, match="cis_test"):
        check_loaders({**loaders, "cis_test": loader(split.records["cis_test"])})


@pytest.mark.parametrize(
    ("val", "trans", "fired"),
    [
        (0.80, 0.60, 0),
        (0.80, 0.785, 1),  # within two points
        (0.80, 0.85, 1),  # trans above cis
        (0.96, 0.60, 1),  # val beyond the benchmark
        (0.97, 0.96, 2),
    ],
)
def test_the_score_alarms(val: float, trans: float, fired: int) -> None:
    assert len(score_alarms(val, trans)) == fired


# The arm reaches the training transform (R25; reviews 4.04 and 5.02).


@pytest.mark.parametrize(
    ("name", "sampler", "size"),
    [
        ("resnet18_arm_a", ControlCropSampler, 224),
        ("resnet18_arm_b", BoxCropSampler, 224),
        ("p2_res320", ControlCropSampler, 320),
    ],
)
def test_the_config_arm_builds_its_crop_sampler(
    monkeypatch: pytest.MonkeyPatch, fixture_split: Any, name: str, sampler: type, size: int
) -> None:
    """The arm picks the sampler; the config's input size (224 in phase 1, 320 for
    `p2_res320`) reaches the training and the eval transform alike."""
    split, root, meta = fixture_split
    seen: dict[str, Any] = {}

    class Stop(Exception):
        pass

    def spy(*args: Any, **kwargs: Any) -> None:
        seen.update(kwargs)
        raise Stop

    monkeypatch.setattr(train, "make_loaders", spy)
    monkeypatch.setattr(train, "build_model", lambda *a, **k: nn.Linear(1, 1))
    with pytest.raises(Stop):
        train.train_run(name, load_config(name), meta, root, Path("unused"), log=lambda _: None)
    assert seen["train_transform"].arm == load_config(name).arm
    assert isinstance(seen["train_transform"].sampler, sampler)
    assert seen["train_transform"].size == seen["eval_transform"].size == size
    assert isinstance(seen["eval_transform"], EvalTransform)
    assert (seen["num_workers"], seen["pin_memory"]) == (4, True)


@pytest.mark.parametrize(
    ("name", "size", "other"),
    [("resnet18_arm_a", 224, "p2_res320"), ("p2_res320", 320, "resnet18_arm_a")],
)
def test_the_checkpoint_carries_the_config_input_size(
    monkeypatch: pytest.MonkeyPatch,
    fixture_split: Any,
    tmp_path: Path,
    name: str,
    size: int,
    other: str,
) -> None:
    """Item 13.02, the review's low finding: `train_run` saves the config's input size beside
    the model, the class list and the weights, and `load_checkpoint` loads that file under its
    own config and refuses it under a config of the other size. Training and scoring are
    replaced (random weights, no download); the checkpoint is the real one `train_run` writes."""
    _, root, meta = fixture_split
    config = replace(load_config(name), pretrained=False, num_workers=0, pin_memory=False)
    fake_fit = train.FitResult(
        (train.Epoch(1, True, 1.0, Scores(1.0, 0.5, 0.5)),), 1, (2, NUM_CLASSES)
    )
    monkeypatch.setattr(train, "fit", lambda *a, **k: fake_fit)
    monkeypatch.setattr(train, "evaluate", lambda *a, **k: Scores(1.0, 0.2, 0.1))
    run = train.train_run(name, config, meta, root, tmp_path, log=lambda _: None)
    (checkpoint,) = tmp_path.glob("*.pt")
    assert train.file_sha256(checkpoint) == run.checkpoint_sha256
    saved = torch.load(checkpoint, weights_only=True)
    assert set(saved) == CHECKPOINT_KEYS
    assert (saved["model"], saved["input_size"]) == ("resnet18", size)
    load_checkpoint(checkpoint, config, eval_size=size)
    with pytest.raises(ValueError, match=f"trained at {size} px"):
        load_checkpoint(checkpoint, load_config(other), eval_size=load_config(other).input_size)


# The command line (R25, R37).


def parse(*argv: str) -> tuple[str, Config]:
    parser = train.build_parser()
    return train.resolve(parser.parse_args(list(argv)), parser)


def test_the_dryrun_flags_parse_as_dryrun_passes_them() -> None:
    name, c = parse("--max-epochs=1", "--batch-size=2", "--dry-run=True")
    assert (name, c.max_epochs, c.batch_size, c.fixture) == (DRY_RUN, 1, 2, True)
    assert parse("--dry-run=true", "--max-epochs=2")[1].max_epochs == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["--dry-run=yes"],
        ["--dry-run=1"],
        ["--dry=True"],  # no abbreviations
        ["--dry-run=True", "--arm=A"],
        ["--dry-run=True", "--config-name=resnet18_arm_a"],
        ["--dry-run=True", "--data-root=/tmp"],
        ["--dry-run=True", "--max-epochs=0"],
        ["--config-name=dryrun", "--data-root=/x", "--out-dir=/y"],  # fixture without the flag
        ["--config-name=resnet18_arm_b", "--data-root=/x", "--out-dir=/y", "--max-epochs=11"],
        ["--config-name=resnet18_arm_b", "--data-root=/x", "--out-dir=/y", "--batch-size=32"],
        ["--config-name=resnet18_arm_b", "--out-dir=/y"],
        ["--data-root=/x", "--out-dir=/y"],
    ],
)
def test_the_command_line_refuses(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        parse(*argv)


def test_a_full_run_overrides_the_run_seed_alone() -> None:
    """R25: the run seed is the one field a full run may change from the command line."""
    base = load_config("resnet18_arm_b")
    name, c = parse(
        "--config-name=resnet18_arm_b", "--data-root=/x", "--out-dir=/y", "--run-seed=2"
    )
    assert name == "resnet18_arm_b"
    assert c == replace(base, run_seed=2)


def test_the_options_are_the_allowed_ones() -> None:
    """R25: no option sets the arm, the model, a rate or anything else of a config."""
    options = {s for a in train.build_parser()._actions for s in a.option_strings}
    assert options == {
        "-h",
        "--help",
        "--config-name",
        "--config",
        "--data-root",
        "--out-dir",
        "--ledger",
        "--run-seed",
        "--max-epochs",
        "--batch-size",
        "--dry-run",
    }


# The ledger (R40).


def test_the_ledger_file_is_its_text_then_the_header_then_rows_that_pass_the_check() -> None:
    """What holds for the life of `docs/ledger.md` (item 6.09; 5.03's test, written before any
    run, wanted no rows): the explanatory text, the header of `LEDGER_COLUMNS`, then zero or more
    rows, every one passing `trailcam.train.ledger`'s checks. That the dry run never appends is
    `test_the_dry_run_trains_on_its_own_fixture_and_never_appends`."""
    lines = LEDGER_PATH.read_text().splitlines()
    header = ledger_header().splitlines()
    start = lines.index(header[0])
    assert lines[start : start + 2] == header
    text = lines[:start]
    assert any(line.strip() for line in text)
    assert not any(line.lstrip().startswith("|") for line in text)
    rows, problems = check_ledger(LEDGER_PATH)
    assert problems == []
    assert len(rows) == len(lines) - start - 2


def test_the_phase2_ledger_is_beside_the_phase1_one_with_the_same_header() -> None:
    """Item 13.02: docs/ledger-phase2.md holds its own text, the header of `LEDGER_COLUMNS` and
    zero or more rows that pass the phase-2 checks; the phase is the file name."""
    assert PHASE2_LEDGER_PATH == LEDGER_PATH.parent / "ledger-phase2.md"
    assert train.is_phase2_ledger(PHASE2_LEDGER_PATH) and not train.is_phase2_ledger(LEDGER_PATH)
    lines = PHASE2_LEDGER_PATH.read_text().splitlines()
    header = ledger_header().splitlines()
    start = lines.index(header[0])
    assert lines[start : start + 2] == header
    assert not any(line.lstrip().startswith("|") for line in lines[:start])
    rows, problems = check_ledger(PHASE2_LEDGER_PATH)
    assert problems == []
    assert len(rows) == len(lines) - start - 2


def test_the_header_follows_the_plan_with_the_config_hash_after_the_config() -> None:
    """The plan's columns (section 5) with the config hash after the config (review 5.02) and
    trans_val macro-F1 after trans_val accuracy (review 5.03)."""
    plan = [
        "run id",
        "model",
        "arm",
        "config",
        "split seed",
        "run seed",
        "split hash",
        "epochs run",
        "best epoch",
        "scheduler",
        "val accuracy",
        "val macro-F1",
        "trans_val accuracy",
        "trans_val macro-F1",
        "checkpoint sha256",
    ]
    assert [c for c in LEDGER_COLUMNS if c != "config hash"] == plan
    assert LEDGER_COLUMNS.index("config hash") == LEDGER_COLUMNS.index("config") + 1


def values(**changes: str) -> dict[str, str]:
    return {column: changes.get(column, "x") for column in LEDGER_COLUMNS}


def test_a_row_has_the_columns_in_order() -> None:
    row = ledger_row(values(arm="B"))
    assert row.count("|") == len(LEDGER_COLUMNS) + 1
    assert row.split(" | ")[LEDGER_COLUMNS.index("arm")] == "B"
    with pytest.raises(ValueError):
        ledger_row(dict(reversed(values().items())))
    with pytest.raises(ValueError):
        ledger_row(values(config="a|b"))


def test_append_row_adds_one_line_to_the_table(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.md"
    ledger.write_text(LEDGER_PATH.read_text())
    row = ledger_row(values())
    append_row(ledger, row)
    lines = ledger.read_text().splitlines()
    assert lines[-1] == row
    assert lines[:-1] == LEDGER_PATH.read_text().splitlines()


def test_append_row_refuses_a_missing_ledger_or_another_header(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        append_row(tmp_path / "missing.md", ledger_row(values()))
    other = tmp_path / "other.md"
    other.write_text(ledger_header().replace("| arm | config |", "| config | arm |") + "\n")
    with pytest.raises(ValueError):
        append_row(other, ledger_row(values()))


# A phase-2 config with a phase-1 ledger, or a phase-1 config with the phase-2 ledger, is
# refused before training and at the append (item 13.02).
PHASE_MIXUPS = [
    ("p2_head3", "ledger.md"),
    ("p2_baseline", "other.md"),
    ("resnet18_arm_a", "ledger-phase2.md"),
]


@pytest.mark.parametrize(("name", "ledger_name"), PHASE_MIXUPS)
def test_a_config_of_the_other_phase_than_the_ledger_stops_before_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    name: str,
    ledger_name: str,
) -> None:
    monkeypatch.setattr(train, "load_metadata", _no_training)
    monkeypatch.setattr(train, "train_run", _no_training)
    ledger = tmp_path / ledger_name
    ledger.write_text(LEDGER_PATH.read_text())
    argv = [f"--config-name={name}", f"--data-root={tmp_path}", f"--out-dir={tmp_path / 'out'}"]
    with pytest.raises(SystemExit) as exit_info:
        train.main([*argv, f"--ledger={ledger}"])
    assert exit_info.value.code == 2
    err = capsys.readouterr().err
    assert "before training" in err and "phase-2" in err
    assert ledger.read_text() == LEDGER_PATH.read_text()


@pytest.mark.parametrize(("name", "ledger_name"), PHASE_MIXUPS)
def test_append_row_refuses_a_row_of_the_other_phase(
    tmp_path: Path, name: str, ledger_name: str
) -> None:
    ledger = tmp_path / ledger_name
    ledger.write_text(LEDGER_PATH.read_text())
    with pytest.raises(ValueError, match="phase-2"):
        append_row(ledger, ledger_row(values(config=name)))
    assert ledger.read_text() == LEDGER_PATH.read_text()
    with pytest.raises(ValueError, match="cells"):
        append_row(ledger, "| p2_head3 |")


def test_a_phase2_run_appends_to_the_phase2_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = values(config="p2_head3")
    monkeypatch.setattr(train, "load_metadata", lambda *a, **k: None)
    monkeypatch.setattr(train, "train_run", lambda *a, **k: SimpleNamespace(alarms=()))
    monkeypatch.setattr(train, "ledger_values", lambda r: row)
    ledger = tmp_path / "ledger-phase2.md"
    ledger.write_text(PHASE2_LEDGER_PATH.read_text())
    argv = ["--config-name=p2_head3", f"--data-root={tmp_path}", f"--out-dir={tmp_path / 'out'}"]
    assert train.main([*argv, f"--ledger={ledger}"]) == 0
    assert ledger.read_text().splitlines() == [
        *PHASE2_LEDGER_PATH.read_text().splitlines(),
        ledger_row(row),
    ]


# A real run fails before training on a bad out dir or ledger, and its alarms always show
# (review 5.03).


def test_preflight_accepts_a_writable_out_dir_and_the_ledger(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.md"
    ledger.write_text(LEDGER_PATH.read_text())
    train.preflight(tmp_path / "new" / "out", ledger)
    assert list((tmp_path / "new" / "out").iterdir()) == []


def test_preflight_refuses_a_bad_ledger(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="--ledger"):
        train.preflight(tmp_path / "out", tmp_path / "missing.md")
    other = tmp_path / "other.md"
    other.write_text(ledger_header().replace("| arm | config |", "| config | arm |") + "\n")
    with pytest.raises(ValueError):
        train.preflight(tmp_path / "out", other)
    trailing = tmp_path / "trailing.md"
    trailing.write_text(LEDGER_PATH.read_text() + "\nA paragraph after the table.\n")
    with pytest.raises(ValueError, match="end with its table"):
        train.preflight(tmp_path / "out", trailing)
    read_only = tmp_path / "read_only.md"
    read_only.write_text(LEDGER_PATH.read_text())
    read_only.chmod(0o444)
    try:
        with pytest.raises(PermissionError):
            train.preflight(tmp_path / "out", read_only)
    finally:
        read_only.chmod(0o644)
    assert read_only.read_text() == LEDGER_PATH.read_text()


def test_preflight_refuses_an_out_dir_it_cannot_write(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.md"
    ledger.write_text(LEDGER_PATH.read_text())
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        with pytest.raises(OSError):
            train.preflight(locked, ledger)
        with pytest.raises(OSError):
            train.preflight(locked / "sub", ledger)
    finally:
        locked.chmod(0o700)


def _no_training(*args: Any, **kwargs: Any) -> None:
    raise AssertionError("the run started before its paths were checked")


@pytest.mark.parametrize("bad", ["ledger", "out_dir"])
def test_a_real_run_with_a_bad_path_stops_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], bad: str
) -> None:
    monkeypatch.setattr(train, "load_metadata", _no_training)
    monkeypatch.setattr(train, "train_run", _no_training)
    ledger = tmp_path / "ledger.md"
    ledger.write_text(LEDGER_PATH.read_text())
    out_dir = tmp_path / "out"
    if bad == "ledger":
        ledger = tmp_path / "missing.md"
    else:
        out_dir.write_text("a file, not a directory")
    argv = ["--config-name=resnet18_arm_a", f"--data-root={tmp_path}"]
    with pytest.raises(SystemExit) as exit_info:
        train.main([*argv, f"--out-dir={out_dir}", f"--ledger={ledger}"])
    assert exit_info.value.code == 2
    assert "before training" in capsys.readouterr().err


def fake_run() -> train.RunResult:
    """A finished arm A run with distinct scores in every cell and a trans_val alarm."""
    return train.RunResult(
        run_id="resnet18_arm_a-seed0-20261013T000000Z",
        config_name="resnet18_arm_a",
        config=load_config("resnet18_arm_a"),
        split=SimpleNamespace(hash="0" * 64),  # type: ignore[arg-type]
        fit=train.FitResult(
            epochs=(train.Epoch(1, True, 1.0, Scores(1.0, 0.8, 0.7)),),
            best_epoch=1,
            logits_shape=(64, NUM_CLASSES),
        ),
        trans_val=Scores(1.0, 0.79, 0.6),
        checkpoint_sha256="1" * 64,
        alarms=score_alarms(0.8, 0.79),
    )


def test_the_row_carries_the_best_epochs_val_scores_and_both_trans_val_scores() -> None:
    got = train.ledger_values(fake_run())
    assert list(got) == list(LEDGER_COLUMNS)
    assert (got["val accuracy"], got["val macro-F1"]) == ("0.8000", "0.7000")
    assert (got["trans_val accuracy"], got["trans_val macro-F1"]) == ("0.7900", "0.6000")
    assert (got["epochs run"], got["best epoch"], got["scheduler"]) == ("1", "1", "cosine warmup 1")
    assert got["config hash"] == config_hash(load_config("resnet18_arm_b"))


def test_the_alarms_print_before_the_ledger_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A real run whose append fails after training still shows its row and its alarms."""
    run = fake_run()
    monkeypatch.setattr(train, "load_metadata", lambda *a, **k: None)
    monkeypatch.setattr(train, "train_run", lambda *a, **k: run)

    def failing_append(ledger: Path, row: str) -> None:
        raise OSError("the ledger went away")

    monkeypatch.setattr(train, "append_row", failing_append)
    ledger = tmp_path / "ledger.md"
    ledger.write_text(LEDGER_PATH.read_text())
    argv = ["--config-name=resnet18_arm_a", f"--data-root={tmp_path}"]
    with pytest.raises(OSError):
        train.main([*argv, f"--out-dir={tmp_path / 'out'}", f"--ledger={ledger}"])
    captured = capsys.readouterr()
    assert run.run_id in captured.out
    assert "LEAKAGE ALARM: trans_val accuracy 0.7900" in captured.err


def test_a_real_run_with_an_alarm_appends_then_exits_3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run = SimpleNamespace(alarms=score_alarms(0.8, 0.79))
    monkeypatch.setattr(train, "load_metadata", lambda *a, **k: None)
    monkeypatch.setattr(train, "train_run", lambda *a, **k: run)
    monkeypatch.setattr(train, "ledger_values", lambda r: values())
    ledger = tmp_path / "ledger.md"
    ledger.write_text(LEDGER_PATH.read_text())
    argv = ["--config-name=resnet18_arm_a", f"--data-root={tmp_path}"]
    code = train.main([*argv, f"--out-dir={tmp_path / 'out'}", f"--ledger={ledger}"])
    assert code == train.ALARM_EXIT
    assert ledger.read_text().splitlines()[-1] == ledger_row(values())
    assert "LEAKAGE ALARM" in capsys.readouterr().err


# The dry run end to end (R37, R17).


def test_the_dry_run_trains_on_its_own_fixture_and_never_appends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`/dryrun`'s flags with the dry-run config switched to the small CNN for speed. The fixture
    goes into a temporary directory that is removed afterwards; the ledger is untouched; the
    test splits are never read, and trans_val only after the last train and val image."""
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    raw["configs"][DRY_RUN]["model"] = "small_cnn"
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(raw))
    assert load_configs(config_file)[DRY_RUN].model == "small_cnn"
    ledger = tmp_path / "ledger.md"
    ledger.write_text(LEDGER_PATH.read_text())
    temp = tmp_path / "temp"
    temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp))

    reads: list[str] = []
    real = CCT20Dataset.__getitem__

    def spy(self: CCT20Dataset, index: int) -> tuple[Tensor, int]:
        reads.append(self.records[index].split)
        return real(self, index)

    monkeypatch.setattr(CCT20Dataset, "__getitem__", spy)
    argv = ["--max-epochs=1", "--batch-size=2", "--dry-run=True"]
    code = train.main([*argv, f"--config={config_file}", f"--ledger={ledger}"])
    out = capsys.readouterr().out

    assert code == 0
    assert "logits shape (2, 12)" in out
    assert "epoch 1/1: train loss" in out
    header = ledger_header().splitlines()
    lines = out.splitlines()
    row = lines[lines.index(header[1]) + 1]
    cells = row.strip("|").split(" | ")
    assert len(cells) == len(LEDGER_COLUMNS)
    got = dict(zip(LEDGER_COLUMNS, (c.strip() for c in cells), strict=True))
    assert (got["model"], got["arm"], got["config"], got["epochs run"]) == (
        "small_cnn",
        "B",
        DRY_RUN,
        "1",
    )
    assert len(got["checkpoint sha256"]) == len(got["split hash"]) == 64
    assert 0 <= float(got["trans_val macro-F1"]) <= 1
    assert ledger.read_text() == LEDGER_PATH.read_text()
    assert list(temp.iterdir()) == []
    assert {"train", "val", "trans_val"} == set(reads)
    first_trans = reads.index("trans_val")
    assert set(reads[first_trans:]) == {"trans_val"}

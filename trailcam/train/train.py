"""The training loop, its command line and the ledger row (plan.md sections 5 and 6).

`python -m trailcam.train.train --config-name resnet18_arm_b --data-root <cache> --out-dir <dir>`
trains one config of `config.yaml` on the 512 px cache of item 6.01 (`annotations/`, `images/`
and the day/night sidecar under `--data-root`), prints one ledger row and appends it to
`docs/ledger.md`. `--dry-run=True` (what `/dryrun` runs, with `--max-epochs=1 --batch-size=2`)
takes the `dryrun` config instead, writes the synthetic fixture of `trailcam/data/synthetic.py`
into a temporary directory itself, trains on it, prints its row and never appends it (R37).

The recipe (R31) is the config's: unweighted cross-entropy; AdamW over the body and head
parameter groups of `trailcam/model/factory.py` at `lr_body` and `lr_head`; a per-step schedule,
linear warmup over the first `warmup_epochs` epochs and cosine annealing to zero at `max_epochs`;
mixed precision on a GPU only. The first `head_only_epochs` epochs count inside `max_epochs` and
train the head alone: the body's parameters are frozen and its batch-norm layers kept in eval
mode, so the body leaves those epochs exactly as it entered them. In the ResNet-18 configs the one
warmup epoch and the one head-only epoch coincide: the head warms up while the body is frozen,
and the body joins in epoch 2 at the top of the cosine. Ten epochs in all, not eleven.

Selection (R17): after every epoch the model is scored on val, accuracy and macro-F1 over the
classes present (`trailcam/evaluate/metrics.py`); the best val macro-F1 so far is kept, and
`patience` epochs in a row without a higher one stop the run. The best epoch's weights are then
restored, saved as the checkpoint and hashed. trans_val is scored once, after that restore, so it
cannot steer anything; cis_test and trans_test are never read here (their loaders are dropped as
soon as `make_loaders` returns).

Seeds (R27): the split seed draws the split (`make_split`); the run seed seeds `random`, NumPy and
torch (every device) before the model is built, and the train loader's shuffle generator.

Leakage alarms (R16): an image id of the train loader in another split stops the run before any
training (`make_loaders` checks it too; this re-checks the loaders the loop reads). After
training, trans_val accuracy within `TRANS_GAP_ALARM` of val accuracy (or above it), or val
accuracy above `VAL_ACCURACY_ALARM`, is reported as an alarm: a real run prints its row and the
alarms, then appends the row, so nothing is lost, and exits with `ALARM_EXIT`, so the notebook
cell fails and the user looks before the work moves on. A dry run (random weights on synthetic
images) prints them and exits 0.

Before a real run trains, `preflight` checks that `--out-dir` can be written and that `--ledger`
exists with the header of `LEDGER_COLUMNS` (review 5.03), so a bad path fails in seconds instead
of after the GPU hour.

Command-line overrides (R25): `--run-seed` for any config; `--max-epochs` and `--batch-size` with
`--dry-run=True` only. No other field can be changed from the command line, so the arm A and arm
B runs of one seed differ in `arm` alone. Booleans are written out: `--dry-run=True` or
`--dry-run=False`.

On Kaggle the package comes from a wheel, which ships neither `config.yaml` nor `docs/`, so the
defaults `CONFIG_PATH` and `LEDGER_PATH` (both relative to a checkout) do not exist there. Pass
both from a clone at the commit of the installed package: `--config <clone>/config.yaml` and
`--ledger <clone>/docs/ledger.md`, with `--out-dir` under `/kaggle/working`.

The ledger row (R40) has the columns of `LEDGER_COLUMNS` in that order: the plan's fields with the
config hash (`trailcam/train/config.py`) after the config name, by the user's decision of review
5.02, and trans_val macro-F1 after trans_val accuracy, by the user's decision of review 5.03.
`docs/ledger.md` holds the header; `append_row` refuses a ledger whose header differs.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np  # a dependency of torchvision
import torch
from torch import Tensor, nn
from torch.optim.lr_scheduler import LambdaLR

from trailcam.data.classes import CLASSES
from trailcam.data.dataset import CCT20Dataset, TrainTransform
from trailcam.data.loaders import Loader, make_loaders
from trailcam.data.metadata import DAY_NIGHT_SIDECAR, Metadata, load_metadata
from trailcam.data.split import Split, make_split
from trailcam.data.synthetic import write_synthetic_fixture
from trailcam.evaluate.metrics import accuracy, macro_f1
from trailcam.model.factory import build_model, parameter_groups
from trailcam.train.config import (
    CONFIG_PATH,
    DRY_RUN,
    Config,
    ConfigError,
    config_hash,
    load_config,
    with_overrides,
)

LEDGER_PATH: Final = Path(__file__).resolve().parents[2] / "docs" / "ledger.md"
LEDGER_COLUMNS: Final = (
    "run id",
    "model",
    "arm",
    "config",
    "config hash",
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
)
USED_SPLITS: Final = ("train", "val", "trans_val")  # cis_test and trans_test are never read here
CACHE_ANNOTATIONS: Final = "annotations"  # the 512 px cache of item 6.01
CACHE_IMAGES: Final = "images"

# Leakage alarms (R16, plan.md section 3). Beery et al. 2018 report 80.9% cis and 59.0% trans
# top-1 for a whole-image classifier, and 91.9% cis on ground-truth box crops; a whole-image
# val accuracy above 95% beats even the crops, and trans within two points of cis is no gap.
TRANS_GAP_ALARM: Final = 0.02
VAL_ACCURACY_ALARM: Final = 0.95
ALARM_EXIT: Final = 3


class LeakageError(RuntimeError):
    """An image id of the train loader is in another split (R16); counts only."""


@dataclass(frozen=True)
class Scores:
    loss: float
    accuracy: float
    macro_f1: float


@dataclass(frozen=True)
class Epoch:
    number: int  # from 1
    head_only: bool
    train_loss: float
    val: Scores


@dataclass(frozen=True)
class FitResult:
    epochs: tuple[Epoch, ...]
    best_epoch: int
    logits_shape: tuple[int, ...]  # of the first training batch

    @property
    def best(self) -> Epoch:
        return self.epochs[self.best_epoch - 1]


@dataclass(frozen=True)
class RunResult:
    run_id: str
    config_name: str
    config: Config
    split: Split
    fit: FitResult
    trans_val: Scores
    checkpoint_sha256: str
    alarms: tuple[str, ...]


def seed_everything(seed: int) -> torch.Generator:
    """Seed `random`, NumPy and torch (CPU and every GPU); return a generator for the shuffle."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def lr_factor(step: int, steps_per_epoch: int, config: Config) -> float:
    """The learning-rate multiplier for optimizer step `step` (from 0): linear warmup to 1 over
    `warmup_epochs`, then cosine annealing towards 0 at the end of `max_epochs`."""
    warmup = config.warmup_epochs * steps_per_epoch
    total = config.max_epochs * steps_per_epoch
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


class EarlyStopping:
    """Keeps the weights of the epoch with the highest score; `update` says when to stop."""

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best_score = -math.inf
        self.best_epoch = 0
        self.best_state: dict[str, Tensor] | None = None
        self.stale = 0

    def update(self, epoch: int, score: float, model: nn.Module) -> bool:
        """Record epoch `epoch`'s score; True once `patience` epochs in a row brought no higher
        one. A tie is not an improvement, so the earlier epoch stays the best."""
        if score > self.best_score:
            self.best_score, self.best_epoch, self.stale = score, epoch, 0
            self.best_state = {
                k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()
            }
        else:
            self.stale += 1
        return self.stale >= self.patience


def _autocast(device: torch.device, enabled: bool) -> torch.autocast:
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)


def _set_head_only(model: nn.Module, body: Sequence[nn.Parameter], head_only: bool) -> None:
    """Freeze or free the body; in a head-only epoch its batch-norm layers stay in eval mode."""
    model.train()
    for p in body:
        p.requires_grad_(not head_only)
    if head_only:
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()


def evaluate(model: nn.Module, loader: Loader, device: torch.device, amp: bool) -> Scores:
    """Mean cross-entropy, accuracy and macro-F1 of `model` over every batch of `loader`."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total = 0.0
    predictions: list[int] = []
    labels: list[int] = []
    with torch.inference_mode():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            with _autocast(device, amp):
                logits = model(images)
            total += float(loss_fn(logits.float(), targets))
            predictions += logits.argmax(dim=1).tolist()
            labels += targets.tolist()
    return Scores(total / len(labels), accuracy(predictions, labels), macro_f1(predictions, labels))


def fit(
    model: nn.Module,
    loaders: Mapping[str, Loader],
    config: Config,
    device: torch.device,
    log: Callable[[str], None] = print,
) -> FitResult:
    """Train on `loaders["train"]`, select on `loaders["val"]` macro-F1 with early stopping, and
    leave the best epoch's weights in `model`. No other loader is read."""
    groups = parameter_groups(model)
    optimizer = torch.optim.AdamW(
        [
            {"params": groups.body, "lr": config.lr_body},
            {"params": groups.head, "lr": config.lr_head},
        ],
        weight_decay=config.weight_decay,
    )
    steps = len(loaders["train"])
    scheduler = LambdaLR(optimizer, lambda step: lr_factor(step, steps, config))
    amp = config.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loss_fn = nn.CrossEntropyLoss()
    stopper = EarlyStopping(config.patience)
    epochs: list[Epoch] = []
    logits_shape: tuple[int, ...] = ()

    for number in range(1, config.max_epochs + 1):
        head_only = number <= config.head_only_epochs
        _set_head_only(model, groups.body, head_only)
        total, seen = 0.0, 0
        for images, targets in loaders["train"]:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, amp):
                logits = model(images)
                loss = loss_fn(logits.float(), targets)
            if not logits_shape:
                logits_shape = tuple(logits.shape)
                log(f"logits shape {logits_shape}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += float(loss.detach()) * len(targets)
            seen += len(targets)
        val = evaluate(model, loaders["val"], device, amp)
        epochs.append(Epoch(number, head_only, total / seen, val))
        stop = stopper.update(number, val.macro_f1, model)
        log(
            f"epoch {number}/{config.max_epochs}{' head-only' if head_only else ''}: "
            f"train loss {total / seen:.4f}, val loss {val.loss:.4f}, "
            f"val accuracy {val.accuracy:.4f}, val macro-F1 {val.macro_f1:.4f} "
            f"(best epoch {stopper.best_epoch})"
        )
        if stop:
            log(f"early stop: {config.patience} epochs without a higher val macro-F1")
            break

    if stopper.best_state is None:  # unreachable: macro-F1 is finite, so epoch 1 sets it
        raise RuntimeError("no epoch was scored")
    model.load_state_dict(stopper.best_state)
    log(f"restored the weights of epoch {stopper.best_epoch} (best val macro-F1)")
    return FitResult(tuple(epochs), stopper.best_epoch, logits_shape)


def check_loaders(loaders: Mapping[str, Loader]) -> None:
    """Refuse loaders whose train images reappear in another split (R16), or that include a
    test split, before any training."""
    extra = sorted(set(loaders) - set(USED_SPLITS))
    if extra:
        raise LeakageError(f"the training loop takes {USED_SPLITS} only, got {extra} too")
    ids: dict[str, set[str]] = {}
    for name, loader in loaders.items():
        if not isinstance(loader.dataset, CCT20Dataset):
            raise TypeError("the training loop reads CCT20Dataset loaders only")
        ids[name] = {r.image_id for r in loader.dataset.records}
    shared = {name: len(ids["train"] & ids[name]) for name in USED_SPLITS if name != "train"}
    if any(shared.values()):
        counts = ", ".join(f"{name}={n}" for name, n in shared.items() if n)
        raise LeakageError(f"train image ids also appear in: {counts}")


def score_alarms(val_accuracy: float, trans_val_accuracy: float) -> tuple[str, ...]:
    """The after-training leakage alarms of R16; empty when none fires."""
    alarms = []
    if trans_val_accuracy >= val_accuracy - TRANS_GAP_ALARM:
        alarms.append(
            f"trans_val accuracy {trans_val_accuracy:.4f} is within {TRANS_GAP_ALARM:.2f} of "
            f"val accuracy {val_accuracy:.4f} or above it: check the split for leakage"
        )
    if val_accuracy > VAL_ACCURACY_ALARM:
        alarms.append(
            f"val accuracy {val_accuracy:.4f} is above {VAL_ACCURACY_ALARM:.2f}, beyond the "
            f"published benchmark: check the split for leakage"
        )
    return tuple(alarms)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def train_run(
    config_name: str,
    config: Config,
    meta: Metadata,
    image_root: Path,
    out_dir: Path,
    log: Callable[[str], None] = print,
) -> RunResult:
    """Split, seed, build, train, restore, save and hash, then score trans_val once."""
    started = datetime.now(UTC)
    run_id = f"{config_name}-seed{config.run_seed}-{started:%Y%m%dT%H%M%SZ}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"run {run_id}: model {config.model}, arm {config.arm}, device {device.type}")
    log(f"seeds: split {config.split_seed}, run {config.run_seed} (random, numpy, torch, shuffle)")
    if config.warmup_epochs == config.head_only_epochs:
        overlap = "the same epochs as the head-only ones"
    else:
        overlap = f"head-only epochs {config.head_only_epochs}"
    log(
        f"schedule: at most {config.max_epochs} epochs; warmup {config.warmup_epochs} epoch(s), "
        f"{overlap}, then cosine; early stopping on val macro-F1, patience {config.patience}"
    )

    split = make_split(meta, config.split_seed)
    log(repr(split))
    generator = seed_everything(config.run_seed)
    model = build_model(config.model, pretrained=config.pretrained)
    every = make_loaders(
        split,
        image_root,
        config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        train_transform=TrainTransform(config.arm),
        generator=generator,
    )
    loaders = {name: every[name] for name in USED_SPLITS}
    del every
    check_loaders(loaders)

    model.to(device)
    result = fit(model, {"train": loaders["train"], "val": loaders["val"]}, config, device, log)

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / f"{run_id}.pt"
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"model": config.model, "classes": list(CLASSES), "state_dict": state}, checkpoint)
    digest = file_sha256(checkpoint)
    log(f"checkpoint {checkpoint.name}, sha256 {digest}")

    amp = config.mixed_precision and device.type == "cuda"
    trans_val = evaluate(model, loaders["trans_val"], device, amp)
    log(
        f"trans_val (reported, never selected on): accuracy {trans_val.accuracy:.4f}, "
        f"macro-F1 {trans_val.macro_f1:.4f}"
    )
    alarms = score_alarms(result.best.val.accuracy, trans_val.accuracy)
    return RunResult(run_id, config_name, config, split, result, trans_val, digest, alarms)


def ledger_header() -> str:
    """The two header lines of the ledger table: the column names and the separator."""
    return "\n".join(
        ["| " + " | ".join(LEDGER_COLUMNS) + " |", "|" + " --- |" * len(LEDGER_COLUMNS)]
    )


def ledger_values(run: RunResult) -> dict[str, str]:
    c, best = run.config, run.fit.best
    return {
        "run id": run.run_id,
        "model": c.model,
        "arm": c.arm,
        "config": run.config_name,
        "config hash": config_hash(c),
        "split seed": str(c.split_seed),
        "run seed": str(c.run_seed),
        "split hash": run.split.hash,
        "epochs run": str(len(run.fit.epochs)),
        "best epoch": str(run.fit.best_epoch),
        "scheduler": f"{c.scheduler} warmup {c.warmup_epochs}",
        "val accuracy": f"{best.val.accuracy:.4f}",
        "val macro-F1": f"{best.val.macro_f1:.4f}",
        "trans_val accuracy": f"{run.trans_val.accuracy:.4f}",
        "trans_val macro-F1": f"{run.trans_val.macro_f1:.4f}",
        "checkpoint sha256": run.checkpoint_sha256,
    }


def ledger_row(values: Mapping[str, str]) -> str:
    """One markdown table line, the values in `LEDGER_COLUMNS` order."""
    if list(values) != list(LEDGER_COLUMNS):
        raise ValueError(f"expected the columns {LEDGER_COLUMNS}, got {tuple(values)}")
    cells = list(values.values())
    if any("|" in cell or "\n" in cell for cell in cells):
        raise ValueError("a ledger cell holds a | or a line break")
    return "| " + " | ".join(cells) + " |"


def check_ledger(ledger: Path) -> str:
    """The text of `ledger`; refuses a missing ledger, another header, or one that does not end
    with its table."""
    if not ledger.is_file():
        raise FileNotFoundError(f"no ledger at {ledger}; pass --ledger <clone>/docs/ledger.md")
    text = ledger.read_text(encoding="utf-8")
    lines = text.splitlines()
    if lines.count(ledger_header().splitlines()[0]) != 1 or ledger_header() not in text:
        raise ValueError(f"{ledger} does not hold the header of LEDGER_COLUMNS exactly once")
    if not lines[-1].startswith("|"):
        raise ValueError(f"{ledger} does not end with its table")
    return text


def append_row(ledger: Path, row: str) -> None:
    """Append `row` to the table that ends `ledger` (checked by `check_ledger`)."""
    text = check_ledger(ledger)
    ledger.write_text(text + ("" if text.endswith("\n") else "\n") + row + "\n", encoding="utf-8")


def preflight(out_dir: Path, ledger: Path) -> None:
    """Before a real run trains: `out_dir` can be written and `ledger` takes a row."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=out_dir, prefix=".write-check-"):
        pass
    check_ledger(ledger)
    with ledger.open("a", encoding="utf-8"):  # appends nothing; refuses a read-only ledger
        pass


def parse_bool(text: str) -> bool:
    """`True` / `False` as `/dryrun` writes them (also lower case); anything else is refused."""
    value = {"true": True, "false": False}.get(text.lower())
    if value is None:
        raise argparse.ArgumentTypeError(f"expected True or False, got {text!r}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m trailcam.train.train",
        description="Train one config of config.yaml and print its ledger row.",
        allow_abbrev=False,
    )
    parser.add_argument("--config-name", help=f"a config of the file ({DRY_RUN} for a dry run)")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="the config.yaml path")
    parser.add_argument("--data-root", type=Path, help="the 512 px cache (real runs)")
    parser.add_argument("--out-dir", type=Path, help="where the checkpoint goes (real runs)")
    parser.add_argument("--ledger", type=Path, default=LEDGER_PATH, help="the ledger to append to")
    parser.add_argument("--run-seed", type=int, help="override the config's run seed")
    parser.add_argument("--max-epochs", type=int, help="dry run only")
    parser.add_argument("--batch-size", type=int, help="dry run only")
    parser.add_argument("--dry-run", type=parse_bool, default=False, metavar="{True,False}")
    return parser


def resolve(args: argparse.Namespace, parser: argparse.ArgumentParser) -> tuple[str, Config]:
    """The config name and the config with the allowed command-line overrides applied."""
    if args.dry_run:
        name = args.config_name or DRY_RUN
        if name != DRY_RUN:
            parser.error(f"--dry-run=True runs the {DRY_RUN} config, not {name}")
        if args.data_root is not None or args.out_dir is not None:
            parser.error("the dry run writes its own fixture: no --data-root or --out-dir")
    else:
        name = args.config_name
        if name is None:
            parser.error("give --config-name, or --dry-run=True")
        if args.max_epochs is not None or args.batch_size is not None:
            parser.error("--max-epochs and --batch-size are for the dry run only (R25)")
        if args.data_root is None or args.out_dir is None:
            parser.error("a real run needs --data-root and --out-dir")
    changes = {
        key: value
        for key, value in (
            ("run_seed", args.run_seed),
            ("max_epochs", args.max_epochs),
            ("batch_size", args.batch_size),
        )
        if value is not None
    }
    try:
        config = with_overrides(name, load_config(name, args.config), changes)
    except ConfigError as error:
        parser.error(str(error))
    if config.fixture != args.dry_run:
        parser.error(
            f"config {name} has fixture: {config.fixture}; it runs with --dry-run={config.fixture}"
        )
    return name, config


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    name, config = resolve(args, parser)
    if not config.fixture:
        try:
            preflight(args.out_dir, args.ledger)
        except (OSError, ValueError) as error:
            parser.error(f"before training: {error}")
    if config.fixture:
        with tempfile.TemporaryDirectory(prefix="trailcam-dryrun-") as tmp:
            root = Path(tmp) / "fixture"
            fixture = write_synthetic_fixture(root)
            meta = load_metadata(root, fixture.paths["sidecar"])
            run = train_run(name, config, meta, fixture.image_root, Path(tmp) / "out")
    else:
        root = args.data_root
        meta = load_metadata(root / CACHE_ANNOTATIONS, root / DAY_NIGHT_SIDECAR)
        run = train_run(name, config, meta, root / CACHE_IMAGES, args.out_dir)

    row = ledger_row(ledger_values(run))
    if config.fixture:
        print("ledger row (dry run, not appended):")
    else:
        print("ledger row:")
    print(ledger_header())
    print(row)
    sys.stdout.flush()
    for alarm in run.alarms:  # before the append, so a failing append cannot hide them
        prefix = "alarm (dry run, reported only)" if config.fixture else "LEAKAGE ALARM"
        print(f"{prefix}: {alarm}", file=sys.stderr)
    sys.stderr.flush()
    if not config.fixture:
        append_row(args.ledger, row)
        print(f"appended to {args.ledger}")
    return ALARM_EXIT if run.alarms and not config.fixture else 0


if __name__ == "__main__":
    raise SystemExit(main())

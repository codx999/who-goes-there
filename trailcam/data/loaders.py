"""The DataLoaders for the five splits (plan.md sections 3 and 5).

`make_loaders` takes a `Split` from `make_split`, never a list of records, and returns one loader
per name in `SPLIT_NAMES` order:

- train: `CCT20Dataset(split.records["train"], ...)`, the re-split train (R12), with the training
  transform; a `RandomSampler` (shuffle on), and the last incomplete batch dropped so batch norm
  never sees a batch of one. The reshuffle each epoch puts every image in some epoch's batches.
- val, trans_val, cis_test and trans_test: the eval transform, a `SequentialSampler` and every
  image, so the batches come in record order and predictions line up with the records (R22).

Before building anything, two checks refuse with counts only (R12, R16): the records must still
match `split.hash`, so a filter after the split that moves or relabels an image is caught, and no
train image id may appear in another split, which also covers a `Split` built by hand with a
recomputed hash. `num_workers` and `pin_memory` are arguments; item 5.02 reads them from the
config. `generator` seeds the train shuffle and the workers' base seed. With workers, that seeds
the augmentation too; with `num_workers=0` the training transform draws from torch's global
generator in the main process, so reproducing it also needs `torch.manual_seed`. The dataset
pickles, so workers started with spawn (macOS) receive it.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from trailcam.data.dataset import CCT20Dataset, EvalTransform, TrainTransform
from trailcam.data.split import SPLIT_NAMES, Split, split_hash

Loader = DataLoader[tuple[Tensor, int]]

EVAL_SPLITS: Final = tuple(name for name in SPLIT_NAMES if name != "train")


def make_loaders(
    split: Split,
    image_root: Path,
    batch_size: int,
    *,
    num_workers: int,
    pin_memory: bool,
    train_transform: TrainTransform | None = None,
    generator: torch.Generator | None = None,
) -> Mapping[str, Loader]:
    """One loader per split in `SPLIT_NAMES` order; shuffle on train only.

    `train_transform` defaults to the control arm; the training script passes
    `TrainTransform(arm)` for the run's arm.
    """
    if not isinstance(split, Split):
        raise TypeError("make_loaders takes the Split from make_split, not a list of records")
    if split_hash(split.records) != split.hash:
        sizes = ", ".join(f"{name}={len(split.records[name])}" for name in SPLIT_NAMES)
        raise ValueError(
            f"the split's records no longer match its hash; build the loaders from make_split's "
            f"output, not an edited copy ({sizes})"
        )
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    if num_workers < 0:
        raise ValueError(f"num_workers must be 0 or more, got {num_workers}")
    train = split.records["train"]
    if len(train) < batch_size:
        raise ValueError(
            f"the train split has {len(train)} images, fewer than one batch of {batch_size}"
        )
    train_ids = {r.image_id for r in train}
    shared = {
        name: sum(1 for r in split.records[name] if r.image_id in train_ids) for name in EVAL_SPLITS
    }
    if any(shared.values()):
        counts = ", ".join(f"{name}={n}" for name, n in shared.items() if n)
        raise ValueError(f"train image ids also appear in other splits: {counts}")

    def loader(dataset: CCT20Dataset, shuffle: bool) -> Loader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            generator=generator if shuffle else None,
        )

    if train_transform is None:
        train_transform = TrainTransform()
    loaders = {"train": loader(CCT20Dataset(train, image_root, train_transform), shuffle=True)}
    for name in EVAL_SPLITS:
        dataset = CCT20Dataset(split.records[name], image_root, EvalTransform())
        loaders[name] = loader(dataset, shuffle=False)
    return MappingProxyType(loaders)

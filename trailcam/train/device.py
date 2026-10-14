"""The Kaggle session's GPU and CPU checks, run by the notebook before training (item 6.06).

plan.md section 5 trains on one T4 of Kaggle's GPU T4 x2: the notebook sets
`CUDA_VISIBLE_DEVICES=0` before torch is imported, so torch sees one device, and
`check_gpu` refuses a session without a GPU, with more than one visible device (the variable was
set too late), or whose device's architecture is not in torch's build. The last is the P100
case: Kaggle's torch 2.11 is a CUDA 12.8 build for sm_75 and newer only (checked 2026-10-14), so
a P100 (sm_60) would fail at the first CUDA kernel, after the setup cells; `check_gpu` makes it
fail before them instead.

`arch_supported` is an exact match of the device's `sm_XY` against `torch.cuda.get_arch_list()`.
CUDA can also run a cubin built for a lower minor version of the same major (sm_86 code on an
sm_89 device) or JIT a `compute_XY` entry; the check refuses those cases rather than relying on
them, since every run here uses the same T4 and sm_75 is in the build.

`workers_warning` compares a config's DataLoader workers with the session's CPUs and only warns:
`num_workers` is in the config hash, so changing it is the user's decision.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Final

import torch

ACCELERATOR: Final = "GPU T4 x2"  # Kaggle's Session options > Accelerator (plan.md section 5)
VISIBLE_DEVICES: Final = "0"  # CUDA_VISIBLE_DEVICES: training uses the first GPU only


def sm_arch(capability: tuple[int, int]) -> str:
    """`(7, 5)` -> `"sm_75"`, the form of `torch.cuda.get_arch_list()`."""
    major, minor = capability
    return f"sm_{major}{minor}"


def arch_supported(capability: tuple[int, int], arch_list: Sequence[str]) -> bool:
    """Whether torch's build has kernels compiled for exactly this capability."""
    return sm_arch(capability) in arch_list


def check_gpu() -> str:
    """Refuse a session torch cannot train on; return a line naming the one visible device."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"no GPU: set Session options > Accelerator to {ACCELERATOR} (plan.md section 5), "
            "then run again"
        )
    visible = torch.cuda.device_count()
    if visible != 1:
        raise RuntimeError(
            f"{visible} GPUs visible, expected one: CUDA_VISIBLE_DEVICES={VISIBLE_DEVICES} must "
            "be set before torch is imported; restart the session (Run > Factory reset) and run "
            "again"
        )
    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    archs = torch.cuda.get_arch_list()
    build = f"torch {torch.__version__} (CUDA {torch.version.cuda})"
    if not arch_supported(capability, archs):
        raise RuntimeError(
            f"{name} is {sm_arch(capability)}, but {build} is built for {', '.join(archs)}: "
            f"training would fail at its first CUDA call; set Session options > Accelerator to "
            f"{ACCELERATOR} (plan.md section 5), then run again"
        )
    pinned = os.environ.get("CUDA_VISIBLE_DEVICES")
    return (
        f"GPU {name} ({sm_arch(capability)}), the one visible device "
        f"(CUDA_VISIBLE_DEVICES={pinned}): {ACCELERATOR} has two and training uses one; "
        f"{build} built for {', '.join(archs)}"
    )


def workers_warning(num_workers: int, cpus: int | None) -> str | None:
    """A warning naming both numbers if the config asks for more workers than there are CPUs."""
    if cpus is None or num_workers <= cpus:
        return None
    return (
        f"WARNING: the config's num_workers is {num_workers} but this session has {cpus} CPUs; "
        "num_workers is in the config hash, so any change is the user's decision"
    )

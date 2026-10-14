"""trailcam/train/device.py: the notebook's GPU and CPU checks (item 6.06).

No GPU here, so `check_gpu` runs against a patched `torch.cuda` that reports what Kaggle's
sessions reported on 2026-10-14.
"""

import pytest
import torch

from trailcam.train import device

# torch.cuda.get_arch_list() of Kaggle's torch 2.11.0+cu128 (GPU T4 x2 session, 2026-10-14).
KAGGLE_ARCHS = ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]


def test_sm_arch() -> None:
    assert device.sm_arch((7, 5)) == "sm_75"
    assert device.sm_arch((6, 0)) == "sm_60"
    assert device.sm_arch((12, 0)) == "sm_120"


@pytest.mark.parametrize(
    ("capability", "supported"),
    [
        ((7, 5), True),  # T4
        ((8, 0), True),  # A100
        ((12, 0), True),
        ((6, 0), False),  # P100: Pascal, absent from Kaggle's build
        ((8, 9), False),  # same major as sm_86, but not an exact match
    ],
)
def test_arch_supported_against_kaggles_build(capability: tuple[int, int], supported: bool) -> None:
    assert device.arch_supported(capability, KAGGLE_ARCHS) is supported


def test_arch_supported_against_a_given_list() -> None:
    assert device.arch_supported((6, 0), ["sm_50", "sm_60", "sm_70"])
    assert not device.arch_supported((7, 5), ["sm_50", "sm_60", "sm_70"])
    assert not device.arch_supported((7, 5), [])
    assert not device.arch_supported((7, 5), ["compute_75"])  # an exact cubin entry only


def _patch_cuda(
    monkeypatch: pytest.MonkeyPatch,
    *,
    available: bool = True,
    count: int = 1,
    name: str = "Tesla T4",
    capability: tuple[int, int] = (7, 5),
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index=None: name)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=None: capability)
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: list(KAGGLE_ARCHS))
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", device.VISIBLE_DEVICES)


def test_check_gpu_accepts_one_t4(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_cuda(monkeypatch)
    line = device.check_gpu()
    assert "Tesla T4 (sm_75)" in line and "CUDA_VISIBLE_DEVICES=0" in line
    assert "GPU T4 x2" in line and "uses one" in line
    assert "CUDA 12.8" in line and "sm_120" in line


def test_check_gpu_refuses_a_p100(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_cuda(monkeypatch, name="Tesla P100-PCIE-16GB", capability=(6, 0))
    with pytest.raises(RuntimeError) as refused:
        device.check_gpu()
    message = str(refused.value)
    assert "Tesla P100-PCIE-16GB is sm_60" in message
    assert ", ".join(KAGGLE_ARCHS) in message and "CUDA 12.8" in message
    assert "Accelerator to GPU T4 x2" in message


def test_check_gpu_refuses_no_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_cuda(monkeypatch, available=False)
    with pytest.raises(RuntimeError, match="no GPU: .*Accelerator to GPU T4 x2"):
        device.check_gpu()


@pytest.mark.parametrize("count", [0, 2])
def test_check_gpu_refuses_other_than_one_visible_device(
    monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    """Two visible T4s mean CUDA_VISIBLE_DEVICES came after torch had initialised CUDA."""
    _patch_cuda(monkeypatch, count=count)
    with pytest.raises(RuntimeError, match=f"{count} GPUs visible.*Factory reset"):
        device.check_gpu()


@pytest.mark.parametrize(
    ("num_workers", "cpus", "warns"),
    [(4, 4, False), (2, 4, False), (0, 1, False), (4, None, False), (4, 2, True), (1, 0, True)],
)
def test_workers_warning(num_workers: int, cpus: int | None, warns: bool) -> None:
    warning = device.workers_warning(num_workers, cpus)
    if warns:
        assert warning is not None
        assert f"num_workers is {num_workers}" in warning and f"has {cpus} CPUs" in warning
    else:
        assert warning is None

"""Runtime dependency minimums (invariant R38; review 4.01).

The Kaggle notebook installs `trailcam` with pip, which ignores `[tool.uv.sources]`. A minimum
above Kaggle's preinstalled torch would make pip replace it with PyPI's default CUDA build, so the
minimums are the oldest versions `tests/test_dataset.py` was run on (2026-10-12: torch 2.3.0,
torchvision 0.18.0, Pillow 10.1.0, Python 3.12, NumPy 2.5), not the locked ones. Raising one
needs a reason, such as an API the code starts to use.
"""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

VERIFIED_MINIMUMS = {"pillow": ">=10.1", "torch": ">=2.3", "torchvision": ">=0.18"}


def _locked_versions() -> dict[str, set[str]]:
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    versions: dict[str, set[str]] = {}
    for package in lock["package"]:
        versions.setdefault(package["name"], set()).add(package["version"].split("+")[0])
    return versions


def _specifiers() -> dict[str, str]:
    dependencies = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    specifiers = {}
    for dependency in dependencies:
        name, _, rest = dependency.partition(">=")
        specifiers[name.strip()] = f">={rest.strip()}"
    return specifiers


def test_runtime_minimums_are_the_verified_ones() -> None:
    specifiers = _specifiers()
    for name, minimum in VERIFIED_MINIMUMS.items():
        assert specifiers.get(name) == minimum, name


def test_minimums_sit_below_the_locked_versions() -> None:
    """A minimum equal to the version locked today is the defect of review 4.01."""
    locked = _locked_versions()
    specifiers = _specifiers()
    for name in VERIFIED_MINIMUMS:
        floor = tuple(int(part) for part in specifiers[name].removeprefix(">=").split("."))
        for version in locked[name]:
            assert floor < tuple(int(part) for part in version.split(".")), name

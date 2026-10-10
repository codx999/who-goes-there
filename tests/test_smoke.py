"""Smoke test: the package imports and carries a version."""

import trailcam


def test_package_imports() -> None:
    assert trailcam.__version__ == "0.1.0"

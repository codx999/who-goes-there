"""trailcam/data/daynight.py: the day/night flag from the pixels (item 6.01, plan.md section 2).

Images are synthetic and written to tmp_path: a grayscale JPEG (mode L, and grey saved as RGB, the
two ways an infrared frame can be stored) must be flagged night, a colour JPEG day. The synthetic
fixture of item 4.03, whose night frames are mode L JPEGs, is checked image by image against its
own sidecar.
"""

import json
import random
from pathlib import Path

import pytest
from PIL import Image, ImageFilter

from trailcam.data import synthetic
from trailcam.data.daynight import NIGHT_MAX_SPREAD, channel_spread, is_night
from trailcam.data.metadata import SPLITS

SIZE = (160, 120)


def textured_grey(seed: int = 0) -> Image.Image:
    """A blurred noise scene in one band, so the JPEG has detail to quantise."""
    rng = random.Random(seed)
    noise = Image.frombytes("L", SIZE, rng.randbytes(SIZE[0] * SIZE[1]))
    return noise.filter(ImageFilter.GaussianBlur(1))


def tinted(rgb: tuple[int, int, int], seed: int = 0) -> Image.Image:
    """The textured grey scene blended half and half with one colour."""
    return Image.blend(Image.new("RGB", SIZE, rgb), textured_grey(seed).convert("RGB"), 0.5)


def saved(image: Image.Image, path: Path, quality: int = 90) -> Image.Image:
    image.save(path, "JPEG", quality=quality)
    with Image.open(path) as reread:
        reread.load()
        return reread


@pytest.mark.parametrize("quality", [75, 90, 95])
def test_grayscale_jpeg_is_night(tmp_path: Path, quality: int) -> None:
    single_band = saved(textured_grey(), tmp_path / "l.jpg", quality)
    assert single_band.mode == "L"
    assert is_night(single_band)

    three_band = saved(textured_grey().convert("RGB"), tmp_path / "rgb.jpg", quality)
    assert three_band.mode == "RGB"
    assert channel_spread(three_band) <= NIGHT_MAX_SPREAD / 2
    assert is_night(three_band)


@pytest.mark.parametrize("rgb", [(153, 141, 122), (60, 110, 40), (90, 120, 170)])
def test_colour_jpeg_is_day(tmp_path: Path, rgb: tuple[int, int, int]) -> None:
    image = saved(tinted(rgb), tmp_path / "day.jpg")
    assert image.mode == "RGB"
    assert channel_spread(image) > 2 * NIGHT_MAX_SPREAD
    assert not is_night(image)


def test_washed_out_colour_is_still_day(tmp_path: Path) -> None:
    """A scene at HSV saturation 0.03 (153, 151, 148), blended half with grey: about 3 levels."""
    image = saved(tinted((153, 151, 148)), tmp_path / "pale.jpg")
    assert NIGHT_MAX_SPREAD < channel_spread(image) < 2 * NIGHT_MAX_SPREAD
    assert not is_night(image)


def test_spread_is_the_mean_largest_minus_smallest_channel() -> None:
    image = Image.new("RGB", (4, 1))
    image.putdata([(10, 10, 10), (0, 40, 20), (200, 100, 150), (5, 5, 9)])
    # Per pixel: 0, 40, 100, 4.
    assert channel_spread(image) == pytest.approx(36.0)


@pytest.mark.parametrize("mode", ["1", "L", "LA", "I", "F"])
def test_grey_modes_score_zero(mode: str) -> None:
    assert channel_spread(textured_grey().convert(mode)) == 0.0


def test_palette_and_alpha_modes_are_measured_in_colour() -> None:
    colour = tinted((60, 110, 40))
    assert channel_spread(colour.convert("RGBA")) == pytest.approx(channel_spread(colour))
    assert not is_night(colour.convert("P"))
    assert is_night(textured_grey().convert("RGB").convert("P"))


def test_synthetic_fixture_flags_match_its_sidecar(tmp_path: Path) -> None:
    """Every JPEG of the 4.03 fixture: night exactly where its sidecar says night."""
    fixture = synthetic.write_synthetic_fixture(tmp_path / "fixture")
    flags = {k: v["night"] for k, v in json.loads(fixture.paths["sidecar"].read_text()).items()}
    checked = nights = 0
    for split in SPLITS:
        for image in json.loads(fixture.paths[split].read_text())["images"]:
            with Image.open(fixture.image_root / image["file_name"]) as jpeg:
                assert is_night(jpeg) == flags[image["id"]]
            checked += 1
            nights += flags[image["id"]]
    assert checked == len(flags)
    assert 0 < nights < checked

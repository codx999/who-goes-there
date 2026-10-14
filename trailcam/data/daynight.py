"""Day/night flag of a camera-trap image, from its pixels (plan.md section 2).

Infrared night frames are grayscale: saved either as single-band (mode L) JPEGs or as RGB JPEGs
whose three channels are near-identical. Day frames are in colour. The clock alone is unreliable
(hour 12 holds about twice as many images as its neighbours), so the flag comes from the pixels.

`channel_spread` is the mean, over all pixels, of the largest channel minus the smallest, in grey
levels (0 to 255); a single-band image scores 0. An image is night when its spread is at most
`NIGHT_MAX_SPREAD`. A grey image saved as an RGB JPEG decodes with a spread of exactly 0 (its
chroma planes are flat); a washed-out colour scene (HSV saturation 0.03) scores about 3. The
threshold sits between the two; `scripts/kaggle_prepare_cct20.py` (item 6.01) prints the
histogram of spreads over CCT20 so the gap can be checked on the real images, and stores each
image's spread in the sidecar.
"""

from __future__ import annotations

from typing import Final

from PIL import Image, ImageChops, ImageStat

NIGHT_MAX_SPREAD: Final = 2.0
# Modes with a single grey band (plus alpha for LA): no colour to measure.
GREY_MODES: Final = frozenset(("1", "L", "LA", "I", "I;16", "F"))


def channel_spread(image: Image.Image) -> float:
    """Mean over pixels of max(R, G, B) - min(R, G, B), 0 to 255; 0.0 for a grey-mode image."""
    if image.mode in GREY_MODES:
        return 0.0
    red, green, blue = image.convert("RGB").split()
    spread = ImageChops.lighter(
        ImageChops.lighter(ImageChops.difference(red, green), ImageChops.difference(green, blue)),
        ImageChops.difference(red, blue),
    )
    return float(ImageStat.Stat(spread).mean[0])


def is_night(image: Image.Image) -> bool:
    """True for a grayscale (infrared night) frame: channel spread at most `NIGHT_MAX_SPREAD`."""
    return channel_spread(image) <= NIGHT_MAX_SPREAD

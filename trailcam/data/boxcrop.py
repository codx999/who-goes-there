"""The arm B crop sampler: box-guided training crops (plan.md section 6, R24).

`BoxCropSampler` is the crop sampler of arm B, which `TrainTransform("B")` in
`trailcam/data/dataset.py` builds; no other module of the package imports it, and loading the
eval transform never loads it. For an image with at least one box, with probability `BOX_P` (0.5)
the crop comes from the box path (`box_crop`), otherwise from arm A's control sampler. An image
without boxes always gets the control sampler and never reaches the box path. The box path:

1. Converts each box from fractions of the image, as `trailcam/data/metadata.py` stores them
   (plan section 2), to pixels of the image as loaded, once, in `to_pixels`: x and width times
   the image width, y and height times its height. A box outside 0 to 1 is refused, so a box
   already in pixels cannot pass.
2. With several boxes (several animals of the image's one species; two-species images are
   dropped), draws one uniformly. One animal per crop keeps the fill about that animal, where the
   union of two animals at opposite edges would be most of the frame. 425 of the 11,633 boxed
   train images (split seed 42) have more than one box.
3. Keeps the crop's shape in arm A's aspect-ratio range, 3/4 to 4/3, so the resize to 224 x 224
   never stretches the animal more than arm A stretches its crops. 6,676 of the 12,177 train
   boxes lie outside that range, most wider than 4:3; a crop in the range carries background
   beside such a box, so the fill's upper end drops (a 2:1 box tops out at 67%). For a box
   covering most of the image, the largest crop in the range (497 x 373 of a 512 x 373 frame)
   raises the lower end. Only when no crop in the range holds the box at a fill of 0.2 inside the
   image (a box spanning nearly the whole width, or more elongated than about 6.7:1) is the range
   left, for the closest shape: containment and fill come first.
4. Draws the fill, the box's area over the crop's area (not a ratio of sides), uniformly over
   what step 3 allows within 0.2 to 1.0, then the aspect ratio log-uniform, as arm A does, among
   the shapes of that area that hold the box and fit in the image.
5. Draws the position uniformly among the placements that keep the whole box inside the crop and
   the crop inside the image: the jitter. A box at the image edge stays whole.

The crop is in whole pixels and runs from the box's first pixel or before to its last pixel or
after, rounding the drawn shape down, never the box. A box under about two pixels a side cannot
fill 20% of any whole-pixel crop around it and is skipped; an image whose boxes are all skipped
gets the control sampler (no train box is under 8 px a side at 512 px).

All randomness comes from torch's generator, as in the control sampler. With `p=0` no coin is
drawn, so the sampler then gives arm A's crops exactly for the same seed; with `p=1` every image
with a usable box takes the box path.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import torch
from PIL import Image

from trailcam.data.dataset import CROP_RATIO, ControlCropSampler, Crop
from trailcam.data.metadata import Box

BOX_P: Final = 0.5  # share of boxed images whose crop comes from the box path
FILL: Final = (0.2, 1.0)  # the box's area over the crop's area
EDGE_TOLERANCE: Final = 1e-6  # float slack on x + width and y + height, as fractions

# A box in pixels of the image as loaded: (x0, y0, x1, y1), the right and bottom edges excluded.
PixelBox = tuple[float, float, float, float]


def to_pixels(box: Box, width: int, height: int) -> PixelBox:
    """`box`, (x, y, width, height) as fractions of the image, in pixels of a `width` x `height`
    image. The only place a box is scaled."""
    x, y, w, h = box
    inside = x >= 0 and y >= 0 and x + w <= 1 + EDGE_TOLERANCE and y + h <= 1 + EDGE_TOLERANCE
    if not (inside and w > 0 and h > 0):
        raise ValueError(
            "a box must be fractions of the image: x and y at least 0, width and height above 0, "
            "x + width and y + height at most 1"
        )
    return x * width, y * height, (x + w) * width, (y + h) * height


@dataclass(frozen=True)
class _Target:
    """A box on a `width` x `height` image: its whole pixels (`left` and `top` included, `right`
    and `bottom` excluded), its own area in pixels and the largest crop area that keeps the fill
    at 0.2 or more and fits in the image."""

    left: int
    top: int
    right: int
    bottom: int
    area: float
    max_area: float

    @classmethod
    def of(cls, box: Box, width: int, height: int) -> _Target:
        x0, y0, x1, y1 = to_pixels(box, width, height)
        area = (box[2] * width) * (box[3] * height)
        return cls(
            left=max(math.floor(x0), 0),
            top=max(math.floor(y0), 0),
            right=min(math.ceil(x1), width),
            bottom=min(math.ceil(y1), height),
            area=area,
            max_area=min(area / FILL[0], float(width * height)),
        )

    @property
    def usable(self) -> bool:
        """Some whole-pixel crop contains the box at a fill of 0.2 or more."""
        return (self.right - self.left) * (self.bottom - self.top) <= self.max_area


def _shaped_areas(
    box_w: int, box_h: int, width: int, height: int, max_area: float
) -> tuple[float, float] | None:
    """The smallest and largest areas of a crop in arm A's aspect-ratio range that holds a
    `box_w` x `box_h` box, fits a `width` x `height` image and keeps the fill at 0.2 or more
    (crop area at most `max_area`). None when there is no such crop."""
    ratio_low, ratio_high = CROP_RATIO
    # The smallest: the box, widened or heightened to the nearest ratio in the range.
    min_w, min_h = max(box_w, box_h * ratio_low), max(box_h, box_w / ratio_high)
    # The largest: the image, narrowed or shortened to the nearest ratio in the range.
    max_w, max_h = min(width, height * ratio_high), min(height, width / ratio_low)
    smallest, largest = min_w * min_h, min(max_w * max_h, max_area)
    if min_w > width or min_h > height or smallest > largest:
        return None
    return smallest, largest


def _uniform(low: float, high: float) -> float:
    return low + (high - low) * float(torch.rand(1))


def _randint(low: int, high: int) -> int:
    """Uniform over `low` to `high`, both included."""
    return int(torch.randint(low, high + 1, (1,)))


class BoxCropSampler:
    """Arm B: with probability `p` per image with a box, a crop around one of its boxes (the box
    path, `box_crop`); otherwise, and for every image without a box, arm A's control sampler,
    `self.control`. Training only; the boxes are fractions of the image."""

    def __init__(self, p: float = BOX_P) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be between 0 and 1, got {p}")
        self.p = p
        self.control = ControlCropSampler()

    def __call__(self, image: Image.Image, boxes: Sequence[Box]) -> Crop:
        if boxes and self._coin():
            width, height = image.size
            crop = self.box_crop(width, height, boxes)
            if crop is not None:
                return crop
        return self.control(image, boxes)

    def _coin(self) -> bool:
        """True with probability `p`. At 0 and 1 nothing is drawn."""
        if self.p == 0.0 or self.p == 1.0:
            return self.p == 1.0
        return float(torch.rand(1)) < self.p

    def box_crop(self, width: int, height: int, boxes: Sequence[Box]) -> Crop | None:
        """The box path: a crop of a `width` x `height` image holding one of `boxes` whole, drawn
        uniformly among the usable ones, with the fill, aspect ratio and jitter of the module
        docstring. None when no box is usable."""
        targets = [t for t in (_Target.of(box, width, height) for box in boxes) if t.usable]
        if not targets:
            return None
        target = targets[_randint(0, len(targets) - 1)] if len(targets) > 1 else targets[0]
        box_w, box_h = target.right - target.left, target.bottom - target.top

        # The crop areas allowed: those of arm A's shapes when one holds the box (step 3), else
        # any that holds the box at a fill of 0.2 or more inside the image.
        areas = _shaped_areas(box_w, box_h, width, height, target.max_area)
        smallest, largest = areas or (float(box_w * box_h), target.max_area)

        # The fill, uniform over those areas: from 0.2 or the box's share of the largest crop to
        # 1.0 or its share of the smallest; and the crop area it gives.
        fill = _uniform(target.area / largest, target.area / smallest)
        crop_area = min(max(target.area / fill, smallest), largest)

        # The crop width: wide enough for the box and for the height to fit the image, narrow
        # enough to fit the image and for the height to hold the box; log-uniform in the aspect
        # range where it overlaps (always, up to float rounding, when `areas` is not None), else
        # the closest end.
        low, high = max(box_w, crop_area / height), min(width, crop_area / box_h)
        ratio_low = math.sqrt(crop_area * CROP_RATIO[0])  # the width at aspect ratio 3/4
        ratio_high = math.sqrt(crop_area * CROP_RATIO[1])  # and at 4/3
        if max(low, ratio_low) <= min(high, ratio_high):
            log_w = _uniform(math.log(max(low, ratio_low)), math.log(min(high, ratio_high)))
            crop_w = min(max(math.exp(log_w), low), high)
        else:
            crop_w = low if ratio_high < low else high

        # Whole pixels: round the shape down, which keeps the fill at or above the draw; the
        # clamps only absorb float rounding at the bounds.
        w = min(max(math.floor(crop_w), box_w), width)
        h = min(max(math.floor(crop_area / crop_w), box_h), height)

        # The jitter: any placement holding the whole box and inside the image.
        left = _randint(max(0, target.right - w), min(target.left, width - w))
        top = _randint(max(0, target.bottom - h), min(target.top, height - h))
        return top, left, h, w

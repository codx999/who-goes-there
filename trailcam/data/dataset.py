"""The torch Dataset over one split and its two transforms (plan.md sections 5 and 6).

`CCT20Dataset` follows the custom-`Dataset` pattern of the course's chapter 04: it holds the
records of one split (for example `split.records["train"]`) and the folder of images they name,
loads an image with `load_image` and returns `(tensor, class index)` from `__getitem__`. The
boxes never leave it (R23): it hands a record's boxes to the training transform, whose crop
sampler may read them, and calls the eval transform with the image alone.

- `TrainTransform`: a crop drawn by the crop sampler and resized to 224 x 224, horizontal flip,
  mild colour jitter, random grayscale with probability 0.1 and ImageNet normalisation. Its `arm`
  switch picks the sampler and nothing else (R25): arm A, the control, draws torchvision's random
  resized crop at scale 0.65 to 1.0 and ignores the boxes; arm B draws box-guided crops
  (`trailcam/data/boxcrop.py`, item 4.04) and falls back to the control sampler.
- `EvalTransform`, for val, the tests, the demo and the Space: resize so the shorter side is 256,
  centre crop 224 and ImageNet normalisation. It takes an image and nothing else, so no box can
  reach it.

All randomness comes from torch's generator, so `torch.manual_seed` (and the DataLoader's
per-worker seeding) makes the training transform reproducible.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final, Protocol

import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from trailcam.data.classes import CLASS_TO_INDEX, CLASSES
from trailcam.data.metadata import Box, Record

INPUT_SIZE: Final = 224
EVAL_RESIZE: Final = 256  # shorter side before the centre crop
IMAGENET_MEAN: Final = (0.485, 0.456, 0.406)
IMAGENET_STD: Final = (0.229, 0.224, 0.225)

# The control arm (plan.md section 5). "Mild" colour jitter is set here: brightness, contrast and
# saturation by up to 20%, no hue shift.
CROP_SCALE: Final = (0.65, 1.0)
CROP_RATIO: Final = (3 / 4, 4 / 3)  # torchvision's default aspect-ratio range
ARMS: Final = ("A", "B")  # arm A the control, arm B box-guided crops (plan.md section 6)
FLIP_P: Final = 0.5
JITTER: Final = 0.2
GRAYSCALE_P: Final = 0.1

# A crop as torchvision's `resized_crop` takes it: (top, left, height, width) in pixels.
Crop = tuple[int, int, int, int]


class CropSampler(Protocol):
    """Draws the training crop for one image; `boxes` are the image's boxes, as fractions."""

    def __call__(self, image: Image.Image, boxes: Sequence[Box]) -> Crop: ...


class ControlCropSampler:
    """Arm A: torchvision's random resized crop at scale 0.65 to 1.0. The boxes are not read."""

    def __call__(self, image: Image.Image, boxes: Sequence[Box]) -> Crop:
        top, left, height, width = T.RandomResizedCrop.get_params(
            image, scale=list(CROP_SCALE), ratio=list(CROP_RATIO)
        )
        return int(top), int(left), int(height), int(width)


def make_sampler(arm: str) -> CropSampler:
    """The crop sampler of `arm`: "A" the control sampler, "B" the box-guided sampler."""
    if arm == "A":
        return ControlCropSampler()
    if arm == "B":
        # Imported on use: loading this module, and with it the eval transform, never loads the
        # box sampler (R24), and boxcrop, which imports this module, is not imported back.
        from trailcam.data.boxcrop import BoxCropSampler

        return BoxCropSampler()
    raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")


class TrainTransform:
    """The training transform: the sampler's crop, resized to 224 x 224, then flip, colour
    jitter, random grayscale, tensor and normalisation. Train records only.

    `arm` ("A" by default, or "B") picks the crop sampler, the one difference between the arms.
    `sampler` replaces it with another (the tests' spies); pass one or the other, not both.
    """

    def __init__(self, arm: str | None = None, *, sampler: CropSampler | None = None) -> None:
        if sampler is None:
            self.arm: str | None = "A" if arm is None else arm
            self.sampler: CropSampler = make_sampler(self.arm)
        elif arm is None:
            self.arm, self.sampler = None, sampler
        else:
            raise ValueError("pass an arm or a sampler, not both")
        self.after_crop: Callable[[Image.Image], Tensor] = T.Compose(
            [
                T.RandomHorizontalFlip(p=FLIP_P),
                T.ColorJitter(brightness=JITTER, contrast=JITTER, saturation=JITTER),
                T.RandomGrayscale(p=GRAYSCALE_P),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __call__(self, image: Image.Image, boxes: Sequence[Box]) -> Tensor:
        top, left, height, width = self.sampler(image, boxes)
        cropped: Image.Image = F.resized_crop(
            image, top, left, height, width, [INPUT_SIZE, INPUT_SIZE], antialias=True
        )
        return self.after_crop(cropped)


class EvalTransform:
    """Val, test, demo and Space: shorter side to 256, centre crop 224, normalise. Deterministic;
    it takes the image only."""

    def __init__(self) -> None:
        self.pipeline: Callable[[Image.Image], Tensor] = T.Compose(
            [
                T.Resize(EVAL_RESIZE, antialias=True),
                T.CenterCrop(INPUT_SIZE),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __call__(self, image: Image.Image) -> Tensor:
        return self.pipeline(image)


class ImageReadError(OSError):
    """An image could not be opened or decoded. The message names the item's position, class
    and split, never its file name, which is the image id (R03)."""


class CCT20Dataset(Dataset[tuple[Tensor, int]]):
    """The records of one split and the folder holding their images (`file_name` under
    `image_root`). Items are `(float32 tensor of shape (3, 224, 224), class index 0 to 11)`.

    The training transform is refused for records outside the train split (augmentation runs on
    the training transform only, R20); any other transform type is refused too.
    """

    def __init__(
        self,
        records: Sequence[Record],
        image_root: Path,
        transform: TrainTransform | EvalTransform,
    ) -> None:
        if isinstance(transform, TrainTransform):
            others = sum(1 for r in records if r.split != "train")
            if others:
                raise ValueError(
                    f"the training transform is for train records only; {others} of "
                    f"{len(records)} records are from another split"
                )
        elif not isinstance(transform, EvalTransform):
            raise TypeError("transform must be a TrainTransform or an EvalTransform")
        self.records = tuple(records)
        self.image_root = Path(image_root)
        self.transform = transform
        self.classes = CLASSES
        self.class_to_idx = dict(CLASS_TO_INDEX)  # a mappingproxy cannot be pickled for workers

    def load_image(self, index: int) -> Image.Image:
        """The image of item `index` as RGB (a grayscale JPEG gets three equal channels)."""
        record = self.records[index]
        try:
            with Image.open(self.image_root / record.file_name) as image:
                return image.convert("RGB")
        except OSError as error:
            raise ImageReadError(
                f"item {index} of {len(self)} ({CLASSES[record.label]}, {record.split}) "
                f"could not be read: {type(error).__name__}"
            ) from None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        record = self.records[index]
        image = self.load_image(index)
        if isinstance(self.transform, TrainTransform):
            tensor = self.transform(image, record.boxes)
        else:
            tensor = self.transform(image)
        return tensor, record.label

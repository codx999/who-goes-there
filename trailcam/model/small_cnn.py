"""The small CNN trained from scratch (plan.md section 4): the warm-up and the baseline.

The course's chapter 04 TinyVGG pattern at 224 px: three conv blocks of (conv 3x3, ReLU, conv
3x3, ReLU, max-pool 2), each keeping `hidden_units` channels and halving the spatial size, then a
linear head on the flattened features. No batch norm, no dropout and no fourth block (R30): the
point of the model is to show what pretraining buys over the plainest network.

Shapes for the default 224 px input: (N, 3, 224, 224) -> (N, h, 112, 112) -> (N, h, 56, 56) ->
(N, h, 28, 28) -> flatten (N, h * 28 * 28) -> (N, num_classes).
"""

from __future__ import annotations

from typing import Final

from torch import Tensor, nn

DEFAULT_HIDDEN_UNITS: Final = 32
NUM_BLOCKS: Final = 3  # each halves height and width


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


class SmallCNN(nn.Module):
    """TinyVGG with three conv blocks and a linear head; `classifier` is the head."""

    def __init__(
        self,
        num_classes: int,
        hidden_units: int = DEFAULT_HIDDEN_UNITS,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        if image_size % 2**NUM_BLOCKS:
            raise ValueError(f"image_size {image_size} is not divisible by {2**NUM_BLOCKS}")
        side = image_size // 2**NUM_BLOCKS
        self.conv_block_1 = _conv_block(3, hidden_units)
        self.conv_block_2 = _conv_block(hidden_units, hidden_units)
        self.conv_block_3 = _conv_block(hidden_units, hidden_units)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * side * side, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_block_3(self.conv_block_2(self.conv_block_1(x)))
        logits: Tensor = self.classifier(x)
        return logits

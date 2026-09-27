"""Tile CNN and the same preprocess used in the training notebook."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

IMAGE_SIZE = 64


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class TileCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 16),
            conv_block(16, 32),
            conv_block(32, 64),
        )
        self.classifier = nn.Linear(64 * 8 * 8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def image_to_tensor(path: Union[str, Path]) -> torch.Tensor:
    with Image.open(path) as img:
        image = img.convert("RGB")
    if image.size != (IMAGE_SIZE, IMAGE_SIZE):
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(pixels).permute(2, 0, 1)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    rgb = image.convert("RGB")
    if rgb.size != (IMAGE_SIZE, IMAGE_SIZE):
        rgb = rgb.resize((IMAGE_SIZE, IMAGE_SIZE))
    pixels = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(pixels).permute(2, 0, 1)

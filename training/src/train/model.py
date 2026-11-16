"""Residual U-Net: output = input colour + correction.
- bilinear upsampling + conv (no checkerboard artefacts of transposed convs);
- no normalisation layers (16-crop batch statistics are noisy; AA depends on absolute intensities);
- zero-initialised output conv: training starts exactly at the 'do nothing' baseline."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

def block(a: int, b: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.LeakyReLU(0.1, inplace=True),
                         nn.Conv2d(b, b, 3, padding=1), nn.LeakyReLU(0.1, inplace=True))

class Up(nn.Module):
    def __init__(self, low: int, skip: int):
        super().__init__()
        self.reduce = nn.Conv2d(low, skip, 3, padding=1)
        self.fuse = block(2 * skip, skip)

    def forward(self, h, skip):
        h = self.reduce(F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False))
        return self.fuse(torch.cat([h, skip], dim=1))

class UNet(nn.Module):
    def __init__(self, in_channels: int, channels=(32, 64, 128, 256)):
        super().__init__()
        pairs = list(zip(channels[:-1], channels[1:]))
        self.inc = block(in_channels, channels[0])
        self.down = nn.ModuleList(nn.Sequential(nn.MaxPool2d(2), block(a, b)) for a, b in pairs)
        self.up = nn.ModuleList(Up(b, a) for a, b in reversed(pairs))
        self.out = nn.Conv2d(channels[0], 3, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.multiple = 2 ** len(pairs)  # H and W must be divisible by this (1080 / 8 = 135: no padding)

    def forward(self, x):
        skips = [self.inc(x)]
        for down in self.down:
            skips.append(down(skips[-1]))
        h = skips.pop()
        for up in self.up:
            h = up(h, skips.pop())
        return x[:, :3] + self.out(h)





"""Is the GT the same picture as the aliased input, only anti-aliased? Plus clipping and brightness distribution.

The 4x4 box filter keeps the average of the image, so block-averaged GT and input must agree; only edges and
fine detail may differ. A GT/input brightness ratio far from 1 means the two are shaded differently.
A non-zero shift means the GT is misaligned: a network trained on it learns to shift and blur.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .base import Analysis, Histogram
from .ops import block_mean, gaussian_blur, luminance

CLIP = 1 - 0.5 / 255  # top code value of the 8-bit render targets

def shift_system(moving: torch.Tensor, reference: torch.Tensor, sigma: float = 2.0) -> np.ndarray:
    """Normal equations of a global Lucas-Kanade shift d with moving(x) ~ reference(x + d), (1, H, W) images.
    Returns [sxx, sxy, syy, sxr, syr]. Summing them over frames gives the shift of a whole sequence."""
    a, g = gaussian_blur(moving, sigma)[0], gaussian_blur(reference, sigma)[0]
    gx = ((g.roll(-1, 1) - g.roll(1, 1)) / 2)[4:-4, 4:-4]
    gy = ((g.roll(-1, 0) - g.roll(1, 0)) / 2)[4:-4, 4:-4]
    r = (a - g)[4:-4, 4:-4]
    return torch.stack([(gx * gx).sum(), (gx * gy).sum(), (gy * gy).sum(), (gx * r).sum(), (gy * r).sum()]) \
        .cpu().double().numpy()

def solve_shift(system: np.ndarray) -> list[float] | None:
    sxx, sxy, syy, sxr, syr = system
    det = sxx * syy - sxy * sxy
    if det <= 1e-9 * max(sxx * syy, 1e-30):
        return None  # no texture to lock onto
    return [(syy * sxr - sxy * syr) / det, (sxx * syr - sxy * sxr) / det]

class ConsistencyAnalysis(Analysis):
    name = "consistency"

    def begin_scene(self, scene: dict) -> None:
        super().begin_scene(scene)
        self.sums = np.zeros((2, 3))  # per-channel sum of input, GT
        self.block_errors = []        # per frame: mean |block mean(input) - block mean(GT)|
        self.shift = np.zeros(5)
        self.frame_shifts = []
        self.clipped = np.zeros(3)    # pixels, clipped input pixels, clipped GT pixels
        self.histograms = {"input": Histogram(np.linspace(0, 1, 33)), "gt": Histogram(np.linspace(0, 1, 33))}

    def update(self, frame, previous) -> None:
        a, g = frame.albedo, frame.gt
        block = self.params.get("blockSize", 8)
        row = torch.stack([
            *a.flatten(1).sum(1), *g.flatten(1).sum(1),
            (block_mean(a, block) - block_mean(g, block)).abs().mean(),
            (a >= CLIP).any(0).sum().float(), (g >= CLIP).any(0).sum().float(),
        ]).cpu().double().numpy()
        self.sums += row[:6].reshape(2, 3)
        self.block_errors.append(row[6])
        self.clipped += [a.shape[1] * a.shape[2], row[7], row[8]]

        system = shift_system(luminance(a), luminance(g))
        self.shift += system
        self.frame_shifts.append(solve_shift(system))
        self.histograms["input"].update(luminance(a))
        self.histograms["gt"].update(luminance(g))

    def end_scene(self) -> dict:
        if not self.block_errors:
            return {**self.scene, "framesChecked": 0, "ok": False, "problems": ["no readable frames"]}
        ratio = self.sums[1] / np.maximum(self.sums[0], 1e-12)
        block_error = float(np.mean(self.block_errors))
        shift = solve_shift(self.shift)
        frame_shift_max = max((math.hypot(*s) for s in self.frame_shifts if s), default=None)
        pixels, clipped_input, clipped_gt = self.clipped

        problems = []
        if np.any(np.abs(ratio - 1) > self.params.get("brightnessTolerance", 0.02)):
            problems.append(f"GT brightness differs from the input: GT / input = {np.round(ratio, 3).tolist()} (R, G, B)")
        if block_error > self.params.get("blockErrorThreshold", 0.02):
            problems.append(f"GT differs from the input beyond aliasing (mean block difference {block_error:.4f})")
        if shift and math.hypot(*shift) > self.params.get("shiftThresholdPx", 0.05):
            problems.append(f"GT is shifted against the input by ({shift[0]:.3f}, {shift[1]:.3f}) px")
        if clipped_input / pixels > self.params.get("clipThreshold", 0.01):
            problems.append(f"{100 * clipped_input / pixels:.1f}% of input pixels are clipped at 1.0")

        return {
            "ok": not problems,
            "problems": problems,
            **self.scene,
            "framesChecked": len(self.block_errors),
            "brightnessRatioGtToInput": ratio.tolist(),
            "meanBlockDifference": block_error,
            "blockDifferencePerFrame": [float(e) for e in self.block_errors],
            "shiftPx": shift,  # input(x) ~ GT(x + shift); expected (0, 0)
            "maxFrameShiftPx": frame_shift_max,
            "clippedFraction": {"input": clipped_input / pixels, "gt": clipped_gt / pixels},
            "luminanceHistogram": {"edges": self.histograms["input"].edges.tolist(),
                                   "input": self.histograms["input"].fractions(),
                                   "gt": self.histograms["gt"].fractions()},
        }

    def describe(self, result: dict) -> str:
        if not result["framesChecked"]:
            return "no frames"
        shift = result["shiftPx"]
        shift = f"({shift[0]:.3f}, {shift[1]:.3f}) px" if shift else "n/a"
        status = "OK" if result["ok"] else "PROBLEMS: " + "; ".join(result["problems"])
        return (f"GT/input brightness {np.round(result['brightnessRatioGtToInput'], 3).tolist()}, shift {shift}, "
                f"clipped {100 * result['clippedFraction']['input']:.2f}% | {status}")



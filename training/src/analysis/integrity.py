"""Base class of an analysis and a streaming statistics helper."""
from __future__ import annotations

import numpy as np
import torch

class Analysis:
    """Per scene the runner calls begin_scene(), then update() for every frame in index order (already on the device)
    or skip() for an unreadable one, then end_scene(). `previous` is frame index - 1 when it was loaded, else None.
    Don't keep frames around (a sequence is ~32 GB); accumulate statistics instead.
    """
    name = ""

    def __init__(self, params: dict, device: torch.device):
        self.params, self.device = params, device

    def begin_scene(self, scene: dict) -> None:
        """scene: {"scene", "split", "framesOnDisk", "missingFrames"}."""
        self.scene = scene

    def update(self, frame, previous) -> None:
        raise NotImplementedError

    def skip(self, index: int, error: str) -> None:
        pass

    def end_scene(self) -> dict:
        """JSON-serialisable result for the current scene."""
        raise NotImplementedError

    def describe(self, result: dict) -> str:
        """One console line for a scene result."""
        return ""

class ChannelStats:
    """Per-channel count / NaN / Inf / min / max / mean / std over many frames.
    Per-frame reductions run on the device in float32 (works on DirectML); totals add up on the CPU in float64."""

    def __init__(self, channels):
        self.channels = list(channels)
        self.sums = np.zeros((5, len(self.channels)))  # count, sum, sum of squares, NaN, Inf
        self.min = np.full(len(self.channels), np.inf)
        self.max = np.full(len(self.channels), -np.inf)

    def update(self, data: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        """data (C, H, W); mask (1, H, W) bool restricts the region."""
        x = data.flatten(1)
        region = torch.ones_like(x, dtype=torch.bool) if mask is None else mask.flatten(1).expand_as(x)
        valid = region & x.isfinite()
        values = torch.where(valid, x, 0.0)
        row = torch.stack([
            valid.sum(1).float(), values.sum(1), (values * values).sum(1),
            (region & x.isnan()).sum(1).float(), (region & x.isinf()).sum(1).float(),
            torch.where(valid, x, float("inf")).amin(1), torch.where(valid, x, float("-inf")).amax(1),
        ]).cpu().double().numpy()
        self.sums += row[:5]
        self.min = np.minimum(self.min, row[5])
        self.max = np.maximum(self.max, row[6])

    @property
    def non_finite(self) -> int:
        return int(self.sums[3:].sum())

    def as_dict(self) -> dict:
        count, total, total_sq, nan, inf = self.sums
        mean = total / np.maximum(count, 1)
        std = np.sqrt(np.maximum(total_sq / np.maximum(count, 1) - mean ** 2, 0))
        return {name: {"count": int(count[i]), "nan": int(nan[i]), "inf": int(inf[i]), "min": float(self.min[i]),
                       "max": float(self.max[i]), "mean": float(mean[i]), "std": float(std[i])}
                for i, name in enumerate(self.channels)}

class Histogram:
    """Value distribution over fixed bin edges, accumulated over many frames; gives approximate quantiles.
    Counting is done with comparisons (no histc / bucketize), so it runs on every backend."""

    def __init__(self, edges):
        self.edges = np.asarray(edges, dtype=np.float64)
        self.total = 0.0
        self.at_least = np.zeros(len(self.edges))  # number of values >= each edge

    def update(self, values: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        v = values.flatten()
        m = torch.ones_like(v, dtype=torch.bool) if mask is None else mask.flatten()
        counts = torch.stack([m.sum()] + [(m & (v >= float(edge))).sum() for edge in self.edges])
        counts = counts.float().cpu().double().numpy()
        self.total += counts[0]
        self.at_least += counts[1:]

    def quantile(self, q: float) -> float | None:
        """Linear interpolation inside the bin; None when there is no data or the quantile is past the last edge."""
        if not self.total:
            return None
        below = self.total - self.at_least
        target = q * self.total
        i = int(np.searchsorted(below, target))
        if i == 0:
            return float(self.edges[0])
        if i == len(self.edges):
            return None
        span = below[i] - below[i - 1]
        t = (target - below[i - 1]) / span if span else 0.0
        return float(self.edges[i - 1] + t * (self.edges[i] - self.edges[i - 1]))

    def fractions(self) -> list[float]:
        """Share of values below the first edge, in each [edge_i, edge_i+1) and at / above the last edge."""
        if not self.total:
            return []
        counts = np.concatenate([[self.total], self.at_least]) - np.concatenate([self.at_least, [0.0]])
        return (counts / self.total).tolist()
"""Endless stream of training sequences from the EXR captures (~100 GB: never loaded whole).

A sample is K = sequenceLength consecutive frames cropped at one position, so the network is unrolled over them as at
inference (step 0 without history, later steps with its own warped previous output). The previous frame comes from
the previous file. Each worker walks random runs of consecutive frames keeping the last K decoded frames; after each
new frame it emits `cropsPerFrame` sequences ending at that frame (decoding is the bottleneck), mixed in a shuffle
buffer. Positions: `errorSampling` share proportional to the last frame's input-vs-GT error, the rest uniform;
positions with less than `minGeometry` geometry are skipped (empty sky in sponza_d)."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, get_worker_info

from common.exr import load_frame

from .inputs import MOTION_X, build_frame, history_valid

BLOCK = 32
KEYS = ("x", "y", "motion", "valid", "geo")

class PatchStream(IterableDataset):
    def __init__(self, scenes: dict[str, dict[int, Path]], params: dict, depth_scale: float, seed: int):
        self.scenes, self.p, self.depth_scale, self.seed = scenes, params, depth_scale, seed
        self.names = sorted(scenes)

    def __iter__(self):
        info = get_worker_info()
        rng = np.random.default_rng([self.seed, info.id if info else 0])
        buffer, size = [], self.p["shuffleBuffer"]
        while True:
            for sample in self._segment(rng):
                if len(buffer) < size:
                    buffer.append(sample)
                    continue
                i = rng.integers(size)
                yield buffer[i]
                buffer[i] = sample

    def _segment(self, rng):

        name = self.names[rng.integers(len(self.names))]
        frames = self.scenes[name]
        indices = sorted(frames)
        k = self.p["sequenceLength"]
        start = rng.integers(max(1, len(indices) - k + 1))
        window, previous = deque(maxlen=k), None
        
        for index in indices[start:start + self.p["segmentLength"]]:
            try:
                frame = load_frame(frames[index], name).to("cpu")
            except Exception as error:  # a broken file must not kill a multi-hour run
                print(f"skipping {frames[index]}: {error}")
                window.clear()
                previous = None
                continue
            x, y, motion, geo = build_frame(frame, self.depth_scale)
            if previous is not None and previous.index == index - 1:
                valid = history_valid(frame, previous)
            else:  # gap: start a new sequence
                window.clear()
                valid = torch.zeros_like(geo)
            window.append({"x": x.half(), "y": y.half(), "motion": motion, "valid": valid, "geo": geo})
            previous = frame
            if len(window) == k:
                for top, left in self._positions(window[-1], rng):
                    yield self._crop(window, top, left, rng)

    def _positions(self, entry, rng):
        s = self.p["cropSize"]
        x, y = entry["x"].float(), entry["y"].float()
        h, w = x.shape[-2:]
        error = (x[:3] - y).abs().amax(0, keepdim=True)
        grid = F.avg_pool2d(torch.cat([error, entry["geo"].float()])[None], BLOCK)
        window = F.avg_pool2d(grid, s // BLOCK, stride=1)[0]  # (2, rows, cols): mean error / geometry per crop
        e = window[0].flatten().double().numpy()
        ok = window[1].flatten().numpy() >= self.p["minGeometry"]
        if not ok.any():
            return []
        a = self.p["errorSampling"]
        p = np.where(ok, a * e / max(e[ok].sum(), 1e-12) + (1 - a) / ok.sum(), 0.0)
        picks = rng.choice(len(p), size=self.p["cropsPerFrame"], p=p / p.sum())
        cols = window.shape[-1]
        return [(min(r * BLOCK + rng.integers(BLOCK), h - s), min(c * BLOCK + rng.integers(BLOCK), w - s))
                for r, c in zip(picks // cols, picks % cols)]

    def _crop(self, window, top, left, rng):
        s = self.p["cropSize"]
        item = {key: torch.stack([e[key][..., top:top + s, left:left + s] for e in window]) for key in KEYS}  # (K, C, s, s)
        if self.p["flip"] and rng.random() < 0.5:
            item = {key: v.flip(-1) for key, v in item.items()}
            item["x"][:, MOTION_X] = -item["x"][:, MOTION_X]
            item["motion"][:, 0] = -item["motion"][:, 0]
        return item




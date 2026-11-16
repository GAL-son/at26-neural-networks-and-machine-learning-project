"""Validation on full frames of the val scenes: PSNR / SSIM gain over the baseline, output flicker ratio, previews."""
from __future__ import annotations

import numpy as np
import torch

from analysis.ops import reproject, warp
from common.exr import load_frame

from .inputs import build_frame, history_valid, warp
from .losses import ssim

import torch.nn.functional as F

def psnr(a, b):
    return float(10 * torch.log10(1 / ((a - b) ** 2).mean().clamp_min(1e-12)))

def usable(frame, previous, tolerance=0.02):
    """(H, W) pixels with valid history: geometry, source on screen, not disoccluded (as in analysis/temporal.py)."""
    h = frame.header
    vp, prev_vp = (np.asarray(h[k], np.float64).T for k in ("viewProjection", "prevViewProjection"))
    _, expected, in_front = reproject(frame.depth, vp, prev_vp, h["nearPlane"], h["farPlane"])
    warped_depth, inside = warp(previous.depth, frame.motion, mode="nearest")
    moved = (warped_depth[0] - expected).abs() > tolerance * expected.abs()
    return frame.geometry_mask()[0] & inside & in_front & ~moved


class ValSet:
    """Full-frame clips of the val scenes, run recurrently exactly like inference (frame 0 of a clip: no history).
    PSNR / SSIM against the baseline on frames 1.. (history available); flicker on consecutive outputs over
    history-valid pixels. About 2 GB of RAM for 2 scenes x 2 clips x 8 frames."""

    def __init__(self, scenes: dict, params: dict, depth_scale: float):
        n = params["clipLength"]
        self.clips = {}
        for name, frames in scenes.items():
            indices = sorted(frames)
            starts = np.linspace(0, len(indices) - n, params["clipsPerScene"]).astype(int)
            self.clips[name] = [self._load([frames[indices[s + i]] for i in range(n)], name, depth_scale) for s in starts]

    @staticmethod
    def _load(paths, name, depth_scale):
        clip, previous = [], None
        for path in paths:
            frame = load_frame(path, name).to("cpu")
            x, y, motion, geo = build_frame(frame, depth_scale)
            valid = history_valid(frame, previous) if previous is not None else torch.zeros_like(geo)
            clip.append({"x": x.half(), "y": y.half(), "motion": motion, "valid": valid})
            previous = frame
        return clip

    @torch.no_grad()
    def evaluate(self, model, device, amp, previews=False):
        model.eval()
        results, tiles = {}, {}
        for name, clips in self.clips.items():
            rows, errors = [], np.zeros(3)  # error vs warped previous: output, input, GT
            for c, clip in enumerate(clips):
                prev = None
                for item in clip:
                    x, y = item["x"].to(device).float()[None], item["y"].to(device).float()[None]
                    motion, valid = item["motion"].to(device)[None], item["valid"].to(device)[None]
                    if prev is None:
                        history, ok = torch.zeros_like(y), torch.zeros_like(y[:, :1])
                    else:
                        history, inside = warp(prev[0], motion)
                        ok = (inside & valid).float()
                    with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                        out = model(torch.cat([x, history, ok], dim=1)).float().clamp(0, 1)
                    if prev is not None:
                        rows.append([psnr(out, y), psnr(x[:, :3], y), float(ssim(out, y)), float(ssim(x[:, :3], y))])
                        mask = ok[:, 0] > 0.5
                        for i, (a, b) in enumerate([(out, prev[0]), (x[:, :3], prev[1]), (y, prev[2])]):
                            errors[i] += float((a - warp(b, motion)[0]).abs().mean(1)[mask].mean())
                    prev = (out, x[:, :3], y)
                if previews and c == 0:
                    tiles[name] = self._tile(x[0, :3], out[0], y[0])
            p_out, p_base, s_out, s_base = np.mean(rows, axis=0)
            results[name] = {"psnr": p_out, "psnrBaseline": p_base, "psnrGain": p_out - p_base,
                             "ssim": s_out, "ssimBaseline": s_base, "ssimGain": s_out - s_base,
                             "flicker": errors[0] / errors[2], "flickerInput": errors[1] / errors[2]}
        model.train()
        return results, tiles

    @staticmethod
    def _tile(x, out, y, size=256):
        """[input | output | GT | |output - GT| x8] around the largest input error of the frame."""
        err = F.avg_pool2d((x - y).abs().mean(0, keepdim=True)[None], size, stride=size // 2)[0, 0]
        r, c = divmod(int(err.argmax()), err.shape[1])
        t, l = r * size // 2, c * size // 2
        crop = lambda img: img[:, t:t + size, l:l + size]
        return torch.cat([crop(x), crop(out), crop(y), (crop(out) - crop(y)).abs().mul(8).clamp(0, 1)], dim=2).cpu()
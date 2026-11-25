"""Explanation cases: the exact network input at frame t, history included, and the region whose output is explained."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from common.dataset import iter_frames
from train.inputs import build_frame, history_valid, warp

HARD_FRACTION = 0.10  # "hard-pixels" region: the 10% of pixels with the largest input error

def crop(t, window):
    """Last two dims of t cut to the window (top, left, height, width)."""
    top, left, h, w = window
    return t[..., top:top + h, left:left + w]

@torch.no_grad()
def run_to_frame(model, scene, frames, index, device, depth_scale, warmup=None, workers=4):
    """Runs the network recurrently up to frame `index`, exactly like inference (train/evaluate.py), from frame 0
    or from `warmup` frames before it. Returns that frame's input x (1, 13, H, W), GT y and output, all float32."""
    
    first = 0 if warmup is None else max(0, index - warmup)
    indices = [i for i in sorted(frames) if first <= i <= index]

    if not indices or indices[-1] != index:
        raise ValueError(f"{scene}: no frame {index}")
    
    x = y = out = previous = prev_out = None
    for item in iter_frames(scene, frames, indices, workers=workers):
        if isinstance(item, tuple):  # unreadable file: the history chain breaks
            previous = prev_out = None
            continue

        frame = item.to(device)
        if previous is not None and frame.index != previous.index + 1:
            previous = prev_out = None  # gap: no valid history

        current, y, motion, _ = build_frame(frame, depth_scale)
        current, y, motion = current[None], y[None], motion[None]

        if prev_out is None:
            history, ok = torch.zeros_like(y), torch.zeros_like(y[:, :1])
        else:
            history, inside = warp(prev_out, motion)
            ok = (inside & history_valid(frame, previous)[None]).float()

        x = torch.cat([current, history, ok], dim=1)
        out = model(x)
        previous, prev_out = frame, out.clamp(0, 1)

    if previous is None or previous.index != index:
        raise ValueError(f"{scene}: frame {index} is unreadable")
    
    return x, y, out

def select_region(spec, x, y, out, size=64):
    """Region R as a (1, 1, H, W) bool mask.
    "max-input-error"  / "max-output-error": the size x size tile where the input / output is furthest from the GT
    "hard-pixels": the HARD_FRACTION of pixels with the largest input error in the whole frame
    [top, left]: an explicit size x size tile"""

    h, w = x.shape[-2:]
    if spec == "hard-pixels":
        err = (x[:, :3] - y).abs().mean(1, keepdim=True)
        return err >= err.flatten().topk(max(1, int(err.numel() * HARD_FRACTION))).values[-1]
    
    if spec in ("max-input-error", "max-output-error"):
        source = x[:, :3] if spec == "max-input-error" else out.clamp(0, 1)
        err = F.avg_pool2d((source - y).abs().mean(1, keepdim=True), size, stride=size // 2)
        r, c = divmod(int(err.flatten().argmax()), err.shape[-1])
        top, left = r * (size // 2), c * (size // 2)
    elif isinstance(spec, (list, tuple)):
        top, left = spec
    else:
        raise ValueError(f"unknown region '{spec}'")
    
    mask = torch.zeros(1, 1, h, w, dtype=torch.bool, device=x.device)
    mask[..., top:top + size, left:left + size] = True
    return mask

def window_around(mask, size=384, multiple=8):
    """(top, left, height, width) of the displayed crop: size x size centred on R, aligned to the U-Net's coarsest
    grid (multiple of 8), or the whole frame if R does not fit."""

    h, w = mask.shape[-2:]
    m = mask[0, 0]
    rows, cols = m.any(1).nonzero().flatten(), m.any(0).nonzero().flatten()
    r0, r1, c0, c1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1

    if r1 - r0 > size or c1 - c0 > size or size > min(h, w):
        return 0, 0, h, w
    
    top = min(max((r0 + r1 - size) // 2, 0), h - size)
    left = min(max((c0 + c1 - size) // 2, 0), w - size)
    return top - top % multiple, left - left % multiple, size, size

def region_box(mask):
    """[top, left, bottom, right] of R."""
    
    m = mask[0, 0]
    rows, cols = m.any(1).nonzero().flatten(), m.any(0).nonzero().flatten()
    return [int(rows[0]), int(cols[0]), int(rows[-1]) + 1, int(cols[-1]) + 1]


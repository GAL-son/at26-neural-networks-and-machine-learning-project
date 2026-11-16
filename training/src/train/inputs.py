"""Network input: the only definition of channels and normalisation (training, validation, inference).

Frame t   (FRAME_CHANNELS = 9):   sRGB colour 3 | world normal 3 | normalised log depth 1 | motion px / 128  2
History   (HISTORY_CHANNELS = 4): previous network output warped to frame t 3 | history valid 1
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from analysis.ops import reproject, srgb
from analysis.ops import warp as warp_single

FRAME_CHANNELS, HISTORY_CHANNELS = 9, 4
IN_CHANNELS = FRAME_CHANNELS + HISTORY_CHANNELS
MOTION_X = 7
MOTION_SCALE = 128.0

def build_frame(frame, depth_scale: float = 0.35):
    """Frame (float32) -> x (9, H, W), sRGB target (3, H, W), motion in px (2, H, W), geometry mask (1, H, W) bool."""

    normal = torch.nan_to_num(frame.normal, nan=0.0)
    geo = normal.norm(dim=0, keepdim=True) > 0.5
    log_z = frame.depth.clamp_min(1e-6).log10()
    median = log_z[geo].median() if geo.any() else log_z.new_tensor(0.0)
    depth = torch.where(geo, (log_z - median) / depth_scale, log_z.new_tensor(3.0))
    h, w = frame.motion.shape[-2:]
    motion = frame.motion * frame.motion.new_tensor([w, h]).view(2, 1, 1)
    x = torch.cat([srgb(frame.albedo), normal * geo, depth, motion / MOTION_SCALE])

    return x, srgb(frame.gt), motion, geo

def history_valid(frame, previous, tolerance: float = 0.02) -> torch.Tensor:
    """(1, H, W) bool: surface visible in `previous` (geometry, on screen, not disoccluded). Same test as
    analysis/temporal.py; the header matrices are stored transposed (OUTPUT_FORMAT.md)."""

    hd = frame.header
    vp, prev_vp = (np.asarray(hd[k], np.float64).T for k in ("viewProjection", "prevViewProjection"))
    _, expected, in_front = reproject(frame.depth, vp, prev_vp, hd["nearPlane"], hd["farPlane"])
    warped_depth, inside = warp_single(previous.depth, frame.motion, mode="nearest")
    moved = (warped_depth[0] - expected).abs() > tolerance * expected.abs()

    return (frame.geometry_mask()[0] & inside & in_front & ~moved)[None]

def warp(image: torch.Tensor, motion: torch.Tensor):
    """Batched: sample previous-frame images (B, C, H, W) at x - motion (px, (B, 2, H, W)).
    Returns the warped image and a (B, 1, H, W) mask of pixels whose source lies inside the image (crop)."""

    _, _, h, w = image.shape
    ys, xs = torch.meshgrid(torch.arange(h, device=image.device), torch.arange(w, device=image.device), indexing="ij")
    sx, sy = xs - motion[:, 0], ys - motion[:, 1]
    grid = torch.stack([(sx + 0.5) / w * 2 - 1, (sy + 0.5) / h * 2 - 1], dim=-1)
    inside = ((sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)).unsqueeze(1)
    
    return F.grid_sample(image, grid.to(image.dtype), mode="bilinear", padding_mode="border", align_corners=False), inside



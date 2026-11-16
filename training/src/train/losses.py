"""Losses on sRGB images (B, 3, H, W). Computed in float32 (outside autocast)."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .inputs import warp

LUMA = (0.2126, 0.7152, 0.0722)

def luminance(rgb):
    return (rgb * rgb.new_tensor(LUMA).view(1, 3, 1, 1)).sum(1, keepdim=True)

def _blur(x, sigma=1.5):
    r = math.ceil(3 * sigma)
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    c = x.shape[1]
    x = F.conv2d(x, k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    return F.conv2d(x, k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)

def ssim(a, b):
    """Mean SSIM of luminance, Gaussian window sigma 1.5: the same definition as the analysis and validation."""
    a, b = luminance(a), luminance(b)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_a, mu_b, aa, bb, ab = _blur(torch.cat([a, b, a * a, b * b, a * b], dim=1)).split(1, dim=1)
    var_a, var_b, cov = aa - mu_a ** 2, bb - mu_b ** 2, ab - mu_a * mu_b
    return (((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))).mean()

def temporal(out, out_prev, gt, gt_prev, motion, mask):
    """The output must change between frames exactly as much as the GT does. Matching the GT's change (instead of
    forcing no change) keeps disocclusions and real changes free of penalty."""
    w_out, inside = warp(out_prev, motion)
    w_gt, _ = warp(gt_prev, motion)
    valid = (inside & mask).float()
    diff = ((out - w_out) - (gt - w_gt)).abs().mean(1, keepdim=True)
    return (diff * valid).sum() / valid.sum().clamp_min(1)




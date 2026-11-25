"""Representation analysis: the activations of a layer projected on their first 3 principal components, as RGB.
Pixels with similar feature vectors get similar colours, so the image shows what the layer tells apart (edges vs flat
areas, geometry vs background, history-valid vs disoccluded...). PCA is fitted on the window's feature vectors at the
layer's own resolution and shown with nearest-neighbour upscaling, so coarse layers look blocky, as they are."""

from __future__ import annotations

import torch
import torch.nn.functional as F

@torch.no_grad()
def pca_rgb(act, window, frame_size):
    """act (1, K, h, w) -> (RGB (3, H_window, W_window) in [0, 1] on the CPU, share of variance of the 3 components)."""

    top, left, height, width = window
    fy, fx = frame_size[0] // act.shape[2], frame_size[1] // act.shape[3]  # 1, 2, 4 or 8; window aligned to 8

    feats = act[0, :, top // fy:(top + height) // fy, left // fx:(left + width) // fx].float().cpu()
    flat = feats.flatten(1).T
    flat = flat - flat.mean(0, keepdim=True)
    _, s, v = torch.pca_lowrank(flat, q=3, center=False)

    proj = (flat @ v).T.reshape(3, *feats.shape[1:])
    lo = torch.quantile(proj.flatten(1), 0.01, dim=1).view(3, 1, 1)
    hi = torch.quantile(proj.flatten(1), 0.99, dim=1).view(3, 1, 1)
    image = ((proj - lo) / (hi - lo).clamp_min(1e-8)).clamp(0, 1)
    
    explained = float((s ** 2).sum() / (flat ** 2).sum().clamp_min(1e-12))
    return F.interpolate(image[None], size=(height, width), mode="nearest")[0], explained



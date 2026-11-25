"""Faithfulness checks of the maps (course section 8).
deletion       input blocks of the window are blurred in the order of the map (most important first). A faithful map
               changes S faster than random orders: higher AOPC = mean |S - S0| / S0 over the steps
               (Samek et al. 2017; |.| because S of a regression output may move either way)
randomisation  cascading randomisation of the weights from the output layer back (Adebayo et al. 2018): a map that
               reflects learned parameters must change, i.e. its rank correlation with the trained map must fall
stability      noise below the 8-bit step on the input colour must not change the map much (correlation stays high)
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from analysis.ops import gaussian_blur

from .case import crop
from .gradcam import VARIANTS, GradCAM

BLUR_CHANNELS = [0, 1, 2, 3, 4, 5, 6, 9, 10, 11]  # colour, normal, depth, warped history; motion and valid mask stay

def _ranks(v):
    _, inverse, counts = torch.unique(v.flatten(), sorted=True, return_inverse=True, return_counts=True)
    ends = counts.cumsum(0).double()
    return (ends - (counts.double() - 1) / 2)[inverse]  # average rank of ties (ReLU maps have many zeros)

def spearman(a, b):
    ra, rb = _ranks(a), _ranks(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = float(ra.norm() * rb.norm())
    return float((ra * rb).sum()) / denom if denom > 0 else 0.0

@torch.no_grad()
def deletion(model, x, score_fn, rank_maps, window, block=8, steps=20, random_orders=5, sigma=2.0):
    top, left, hh, ww = window
    blurred = x.clone()
    blurred[0, BLUR_CHANNELS] = gaussian_blur(x[0, BLUR_CHANNELS], sigma)
    nr, nc = hh // block, ww // block
    n = nr * nc
    cuts = [round(n * i / steps) for i in range(steps + 1)]
    s0 = float(score_fn(model(x), x))

    def curve(order):
        position = torch.empty(n, dtype=torch.long, device=x.device)
        position[order] = torch.arange(n, device=x.device)
        rank_image = position.view(nr, nc).repeat_interleave(block, 0).repeat_interleave(block, 1)
        
        change = []
        for cut in cuts:
            m = torch.zeros(x.shape[-2:], dtype=torch.bool, device=x.device)
            m[top:top + nr * block, left:left + nc * block] = rank_image < cut
            xm = torch.where(m, blurred, x)
            change.append(abs(float(score_fn(model(xm), xm)) - s0) / max(s0, 1e-12))
            
        return change

    result = {"fraction": [c / n for c in cuts], "curves": {}, "aopc": {}}
    for variant, rank_map in rank_maps.items():
        blocks = F.avg_pool2d(rank_map[top:top + nr * block, left:left + nc * block][None, None], block).flatten()
        c = curve(blocks.argsort(descending=True))
        result["curves"][variant], result["aopc"][variant] = c, sum(c[1:]) / steps

    r = torch.tensor([curve(torch.randperm(n).to(x.device)) for _ in range(random_orders)])
    result["curves"]["random"], result["curves"]["randomStd"] = r.mean(0).tolist(), r.std(0).tolist()
    aopc = r[:, 1:].mean(1)
    
    result["aopc"]["random"], result["aopc"]["randomStd"] = float(aopc.mean()), float(aopc.std())
    return result

def randomisation_stages(model):
    """(name, module) from the output back to the input: out, dec1 .. decN-1, encN .. enc1."""

    stages = [("out", model.out)]
    stages += [(f"dec{j + 1}", m) for j, m in enumerate(reversed(model.up))]
    stages += [(f"enc{len(model.down) + 1 - j}", m) for j, m in enumerate(reversed(model.down))]
    return stages + [("enc1", model.inc)]

def cascading_randomisation(model, x, score_fn, layer, window, reference):
    rand = copy.deepcopy(model)
    result = {"stages": [], **{v: [] for v in VARIANTS}}
    for name, module in randomisation_stages(rand):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                m.reset_parameters()  # default init (also replaces the zero init of the output conv)
        
        maps = GradCAM(rand, [layer])(x, score_fn)[0][layer]
        result["stages"].append(name)
        for v in VARIANTS:
            result[v].append(spearman(crop(maps[v], window), crop(reference[v], window)))

    return result

def stability(model, x, score_fn, layer, window, reference, noise=0.002, repeats=3):
    cam = GradCAM(model, [layer])
    corr = {v: [] for v in VARIANTS}
    for _ in range(repeats):
        xn = x.clone()
        xn[:, :3] += torch.randn_like(xn[:, :3]) * noise
        maps = cam(xn, score_fn)[0][layer]
        for v in VARIANTS:
            corr[v].append(spearman(crop(maps[v], window), crop(reference[v], window)))

    return {v: {"mean": float(np.mean(c)), "std": float(np.std(c))} for v, c in corr.items()}



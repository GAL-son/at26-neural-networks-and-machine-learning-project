"""Feature attribution by ablation of input channel groups (a perturbation method, course section 6.2).
Each group of the 13 input channels is replaced by a baseline and the change of the output is measured:
  dS     change of the explained scalar S in R        dPsnr  change of the output PSNR in the window
  map    |lum(output ablated) - lum(output)|: where the group matters
Baselines stay as close as possible to inputs seen in training (course: masking must not create unnatural inputs):
  history -> "no history" (zeros, valid = 0): exactly the first frame of every training sequence
  normal  -> 0 and depth -> 3.0: the background codes      motion -> 0: a static camera
  colour  -> Gaussian blur (sigma 2): there is no neutral colour; blur removes the aliased edges, keeps the content
  'blur' row: each group blurred instead (only its fine detail removed; the history validity mask is kept)."""

from __future__ import annotations

import torch

from analysis.ops import gaussian_blur
from train.metrics import psnr

from .case import crop
from .gradcam import luminance

GROUPS = {"colour": slice(0, 3), "normal": slice(3, 6), "depth": slice(6, 7), "motion": slice(7, 9),
          "history": slice(9, 13)}
NEUTRAL = {"normal": 0.0, "depth": 3.0, "motion": 0.0}

def ablate(x, group, baseline, sigma=2.0):
    x = x.clone()
    if group == "colour" or baseline == "blur":
        channels = slice(9, 12) if group == "history" else GROUPS[group]
        x[0, channels] = gaussian_blur(x[0, channels], sigma)
    elif group == "history":
        x[:, GROUPS["history"]] = 0.0
    else:
        x[:, GROUPS[group]] = NEUTRAL[group]
    return x

@torch.no_grad()
def ablation(model, x, y, out, score_fn, window, baselines=("neutral", "blur"), sigma=2.0):
    s0 = float(score_fn(out, x))
    p0 = psnr(crop(out.clamp(0, 1), window), crop(y, window))
    groups, maps = {}, {}
    for baseline in baselines:
        for group in GROUPS:
            if group == "colour" and baseline != baselines[0]:
                continue  # colour is always blurred: one run is enough
            
            xa = ablate(x, group, baseline, sigma)
            o = model(xa)
            key = f"{group}/{'blur' if group == 'colour' else baseline}"
            groups[key] = {"dS": float(score_fn(o, xa)) - s0,
                           "dPsnr": psnr(crop(o.clamp(0, 1), window), crop(y, window)) - p0}
            maps[key] = crop((luminance(o) - luminance(out)).abs(), window)[0, 0]
    return {"S": s0, "psnr": p0, "groups": groups}, maps



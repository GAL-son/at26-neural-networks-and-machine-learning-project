"""Grad-CAM for the anti-aliasing U-Net, a dense (image-to-image) network.

There is no class logit, so the explained scalar S is defined over a region R of the output
(course: for segmentation, explain the result for a chosen region of the mask):
  correction  S = mean_R |lum(output) - lum(input colour)|  how strongly the network changes the aliased input in R.
              The U-Net is residual (output = input colour + correction): S depends only on the learned correction.
  error       S = mean_R |lum(output) - lum(GT)|            how wrong the output is in R (hard and failure cases)
For the activations A of a layer (K channels, h x w):
  Grad-CAM  (Selvaraju et al. 2017)  map = ReLU(sum_k alpha_k A_k), alpha_k = mean_ij dS/dA_k,ij  (one weight per channel)
  HiResCAM  (Draelos & Carin 2020)   map = ReLU(sum_k dS/dA_k * A_k)                             (element-wise)
With a small R, Grad-CAM's channel weights also light up every place where those channels fire, far outside the
receptive field of R ("where else the network uses the same features"); HiResCAM keeps only locations that contribute
to S. Both are upsampled bilinearly to the frame; positive values = activations that increase S.
Sanity check: at dec1 (right before the linear 1x1 output conv) HiResCAM for 'correction' equals |correction| in R.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

LUMA = (0.2126, 0.7152, 0.0722)
VARIANTS = ("gradcam", "hirescam")

def luminance(img):
    """(B, 3, H, W) -> (B, 1, H, W)."""
    return (img * img.new_tensor(LUMA).view(1, 3, 1, 1)).sum(1, keepdim=True)

def target(kind: str, y, mask):
    """S(output, input) over the region mask (1, 1, H, W) bool. The input is an argument because ablation and the
    deletion test change it: 'correction' is always measured against the colour actually fed to the network."""

    if kind == "correction":
        return lambda out, x: (luminance(out) - luminance(x[:, :3])).abs()[mask].mean()
    if kind == "error":
        return lambda out, x: (luminance(out) - luminance(y)).abs()[mask].mean()
    
    raise ValueError(f"unknown target '{kind}' (correction | error)")

def named_layers(model) -> dict:
    """enc1 (full res) .. encN (deepest = bottleneck), then decN-1 (coarsest decoder) .. dec1 (full res, last block)."""
    layers = {"enc1": model.inc}
    layers.update({f"enc{i + 2}": module for i, module in enumerate(model.down)})
    layers.update({f"dec{len(model.up) - j}": module for j, module in enumerate(model.up)})
    return layers

class GradCAM:
    def __init__(self, model, layers):
        available = named_layers(model)
        unknown = [name for name in layers if name not in available]
        if unknown:
            raise ValueError(f"unknown layer(s) {unknown}; available: {list(available)}")
        
        self.model, self.modules = model, {name: available[name] for name in layers}

    def __call__(self, x, score_fn):
        """One forward + backward pass. Returns ({layer: {"gradcam", "hirescam": (H, W), "alpha": (K,)}}, S, output,
        {layer: activations (1, K, h, w)})."""
        
        acts, grads = {}, {}

        def hook(name):
            def forward_hook(module, inputs, output):
                acts[name] = output
                output.register_hook(lambda grad: grads.__setitem__(name, grad))
            return forward_hook

        handles = [module.register_forward_hook(hook(name)) for name, module in self.modules.items()]
        try:
            with torch.enable_grad():
                xin = x.detach().requires_grad_(True)  # weights are frozen; this makes autograd build the graph
                out = self.model(xin)
                score = score_fn(out, xin)
                score.backward()
        finally:
            for handle in handles:
                handle.remove()

        size = x.shape[-2:]
        upsample = lambda m: F.interpolate(F.relu(m), size=size, mode="bilinear", align_corners=False)[0, 0]
        maps = {}
        for name in self.modules:
            a, g = acts[name].detach(), grads[name]
            alpha = g.mean(dim=(2, 3), keepdim=True)
            maps[name] = {"gradcam": upsample((alpha * a).sum(1, keepdim=True)),
                          "hirescam": upsample((g * a).sum(1, keepdim=True)),
                          "alpha": alpha.flatten()}
        return maps, float(score), out.detach(), {name: a.detach() for name, a in acts.items()}



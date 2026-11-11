"""What the network has to fix: the difference between aliased input and GT, as displayed (sRGB-encoded).

- the "do nothing" baseline: PSNR / SSIM of the input against the GT; a trained network must beat it;
- how many pixels change at all and how the error is spread (per pixel, per 256x256 patch) -> loss weighting
  and patch sampling;
- where the error is: at geometric edges (visible in depth / normals), inside surfaces (texture aliasing from the
  missing mipmaps, sub-pixel geometry the 1-sample G-buffer missed), or on background.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .base import Analysis, Histogram
from .ops import geometric_edges, luminance, srgb, ssim_map

REGIONS = ("geometricEdges", "interior", "background")

def _psnr(mse):
    return 10 * np.log10(1 / np.maximum(mse, 1e-12))

class ResidualAnalysis(Analysis):
    name = "residual"

    def begin_scene(self, scene: dict) -> None:
        super().begin_scene(scene)
        self.frame_mse, self.frame_ssim, self.patch_mse = [], [], []
        self.regions = np.zeros((len(REGIONS), 3))  # pixels, changed pixels, squared error sum
        self.error = Histogram(np.geomspace(1 / 1024, 1, 41))

    def update(self, frame, previous) -> None:
        a, g = srgb(frame.albedo), srgb(frame.gt)
        diff = a - g
        squared = (diff * diff).mean(0)  # (H, W)
        error = diff.abs().amax(0)       # (H, W), largest channel error
        changed = error > self.params.get("changeThreshold", 2 / 255)

        geometry = frame.geometry_mask()
        edges = geometric_edges(frame.depth, frame.normal, geometry,
                                self.params.get("depthTolerance", 0.02), self.params.get("creaseDegrees", 30.0))
        geo = geometry[0]
        masks = (edges, geo & ~edges, ~geo & ~edges)
        regions = [torch.stack([m.sum().float(), (m & changed).sum().float(), torch.where(m, squared, 0.0).sum()])
                   for m in masks]
        patches = F.avg_pool2d(squared[None, None], self.params.get("patchSize", 256))[0, 0].flatten()
        ssim = ssim_map(luminance(a), luminance(g)).mean()

        row = torch.cat([torch.stack(regions).flatten(), squared.mean()[None], ssim[None], patches]).cpu().double().numpy()
        self.regions += row[:9].reshape(3, 3)
        self.frame_mse.append(row[9])
        self.frame_ssim.append(row[10])
        self.patch_mse.extend(row[11:])
        self.error.update(error)

    def end_scene(self) -> dict:
        if not self.frame_mse:
            return {**self.scene, "framesChecked": 0}
        mse, ssim = np.array(self.frame_mse), np.array(self.frame_ssim)
        psnr = _psnr(mse)
        patches = np.sort(np.array(self.patch_mse))[::-1]
        patch_psnr = _psnr(patches)
        top = max(1, len(patches) // 10)
        pixels, changed, squared = self.regions.T
        total_pixels, total_changed, total_squared = self.regions.sum(0)

        return {
            **self.scene,
            "framesChecked": len(mse),
            "baseline": {  # input vs GT, i.e. a network that does nothing
                "psnrMean": psnr.mean(),
                "psnrWorstFrame": psnr.min(),
                "psnrOfMeanMse": _psnr(mse.mean()),
                "ssimMean": ssim.mean(),
                "ssimWorstFrame": ssim.min(),
            },
            "changedPixelFraction": total_changed / total_pixels,
            "pixelErrorQuantiles": {f"p{q}": self.error.quantile(q / 100) for q in (50, 90, 99, 99.9)},
            "regions": {
                name: {
                    "pixelShare": pixels[i] / total_pixels,
                    "changedFraction": changed[i] / max(pixels[i], 1),
                    "shareOfChangedPixels": changed[i] / max(total_changed, 1),
                    "shareOfSquaredError": squared[i] / max(total_squared, 1e-12),
                }
                for i, name in enumerate(REGIONS)
            },
            "patches": {
                "size": self.params.get("patchSize", 256),
                "count": len(patches),
                "flatFraction": float(np.mean(patch_psnr >= self.params.get("flatPatchPsnr", 45.0))),
                "errorShareOfTop10Percent": patches[:top].sum() / max(patches.sum(), 1e-12),
                "psnrQuantiles": {f"p{q}": float(np.percentile(patch_psnr, q)) for q in (10, 50, 90)},
            },
            "psnrPerFrame": psnr.tolist(),
        }

    def describe(self, result: dict) -> str:
        if not result["framesChecked"]:
            return "no frames"
        edges = result["regions"]["geometricEdges"]["shareOfSquaredError"]
        return (f"baseline PSNR {result['baseline']['psnrMean']:.2f} dB, SSIM {result['baseline']['ssimMean']:.4f}, "
                f"{100 * result['changedPixelFraction']:.1f}% pixels change, "
                f"{100 * edges:.0f}% of error at geometric edges, "
                f"{100 * result['patches']['flatFraction']:.0f}% flat patches")





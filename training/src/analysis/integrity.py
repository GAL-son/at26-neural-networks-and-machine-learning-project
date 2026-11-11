"""Integrity scan (plan section 1).
- every frame readable, no NaN / Inf, no missing frames, header consistent and matching the file name;
- value ranges: albedo / GT in [0, 1], unit normals, depth in [near, far], background depth = far,
  background motion = 0, frame 0 without motion;
- per-channel statistics over all pixels and over geometry only (input for normalisation);
- motion vectors vs. a reprojection of depth through the header matrices: confirms the motion convention
  and the matrix layout (M = stored.T per OUTPUT_FORMAT.md, or as stored).
"""

from __future__ import annotations

from collections import defaultdict
import numpy as np
import torch

from common.exr import HEIGHT, LAYERS, WIDTH
from .base import Analysis, ChannelStats
def predict_motion(depth: torch.Tensor, vp: np.ndarray, prev_vp: np.ndarray, near: float, far: float):
    """uv_curr - uv_prev implied by depth and two P*V matrices (clip = M @ [x, y, z, 1]), for a static scene.
    Returns motion (2, H, W) and a (H, W) mask of pixels in front of the previous camera."""
    
    _, height, width = depth.shape
    
    reproject = torch.tensor(prev_vp @ np.linalg.inv(vp), dtype=depth.dtype, device=depth.device)
    
    x = ((torch.arange(width, device=depth.device) + 0.5) / width * 2 - 1).expand(height, width)
    y = ((torch.arange(height, device=depth.device) + 0.5) / height * 2 - 1)[:, None].expand(height, width)
    z = depth[0]
    
    clip = torch.stack([x * z, y * z, far * (z - near) / (far - near), z], dim=-1)  # clip.w = linear Z
    
    previous = clip @ reproject.T
    
    valid = previous[..., 3] > 1e-6
    previous_ndc = previous[..., :2] / previous[..., 3:].clamp_min(1e-6)
    
    return ((torch.stack([x, y], dim=-1) - previous_ndc) * 0.5).permute(2, 0, 1), valid

class IntegrityAnalysis(Analysis):
    name = "integrity"

    def begin_scene(self, scene: dict) -> None:
        super().begin_scene(scene)
        self.checked = 0
        self.unreadable = []
        self.issues = defaultdict(list)  # issue -> frame indices
        self.first_header = None
        self.stats = {region: {layer: ChannelStats(name.split(".")[1] for name in names)
                               for layer, names in LAYERS.items()} for region in ("all", "geometry")}
        self.motion_px = ChannelStats(["magnitude"])
        self.log_depth = ChannelStats(["log10Z"])
        self.reprojection = {"transposed": np.zeros(3), "asStored": np.zeros(3)}  # pixels, error sum, pixels < 0.5 px

    def skip(self, index: int, error: str) -> None:
        self.unreadable.append({"index": index, "error": error})

    def update(self, frame, previous) -> None:
        self.checked += 1
        index, header = frame.index, frame.header
        near, far = header["nearPlane"], header["farPlane"]
        vp = np.asarray(header["viewProjection"], np.float64)
        prev_vp = np.asarray(header["prevViewProjection"], np.float64)
        height, width = frame.depth.shape[-2:]
        geometry = frame.geometry_mask()
        geo, depth = geometry[0], frame.depth[0]
        to_px = torch.tensor([width, height], device=frame.motion.device, dtype=frame.motion.dtype).view(2, 1, 1)
        motion_px = frame.motion * to_px

        for layer in LAYERS:
            self.stats["all"][layer].update(getattr(frame, layer))
            self.stats["geometry"][layer].update(getattr(frame, layer), geometry)
        self.motion_px.update(motion_px.norm(dim=0, keepdim=True), geometry)
        self.log_depth.update(frame.depth.clamp_min(1e-6).log10(), geometry)

        signature = ((height, width), header["ssaaFactor"], near, far)
        self.first_header = self.first_header or signature
        normal_error = torch.where(geo, (frame.normal.norm(dim=0) - 1).abs(), 0.0).amax()
        checks = {
            "unexpectedResolution": (height, width) != (HEIGHT, WIDTH),
            "headerChanged": signature != self.first_header,
            "headerIndexMismatch": header["frameIndex"] != index,
            "albedoOutOfRange": bool(((frame.albedo < 0) | (frame.albedo > 1)).any()),
            "gtOutOfRange": bool(((frame.gt < 0) | (frame.gt > 1)).any()),
            "depthOutOfRange": bool((geo & ((depth < near * 0.999) | (depth > far * 1.001))).any()),
            "backgroundDepthNotFar": bool((~geo & (depth < far * 0.999)).any()),
            "backgroundMotion": bool((~geo & (frame.motion != 0).any(0)).any()),
            "normalNotUnitLength": float(normal_error) > self.params.get("normalTolerance", 0.01),
            "frame0Motion": index == 0 and bool((frame.motion != 0).any()),
            "frame0PrevViewProjectionDiffers": index == 0 and not np.array_equal(vp, prev_vp),
        }
        for issue, failed in checks.items():
            if failed:
                self.issues[issue].append(index)

        if self.params.get("reprojection", True) and index > 0 and not np.array_equal(vp, prev_vp):
            for layout, (m, prev_m) in {"transposed": (vp.T, prev_vp.T), "asStored": (vp, prev_vp)}.items():
                predicted, valid = predict_motion(frame.depth, m, prev_m, near, far)
                error = (predicted * to_px - motion_px).norm(dim=0)
                mask = geo & valid & error.isfinite()
                self.reprojection[layout] += torch.stack([
                    mask.sum().float(), torch.where(mask, error, 0.0).sum(), (mask & (error < 0.5)).sum().float(),
                ]).cpu().double().numpy()

    def end_scene(self) -> dict:
        reprojection = {layout: {"meanErrorPx": total / pixels if pixels else None,
                                 "fractionBelow0.5Px": below / pixels if pixels else None}
                        for layout, (pixels, total, below) in self.reprojection.items()}
        errors = {layout: result["meanErrorPx"] for layout, result in reprojection.items()}
        best = min(errors, key=errors.get) if None not in errors.values() else None
        non_finite = {layer: stats.non_finite for layer, stats in self.stats["all"].items()}

        problems = [f"{len(self.scene['missingFrames'])} missing frame(s)"] if self.scene["missingFrames"] else []
        problems += [f"{len(self.unreadable)} unreadable frame(s)"] if self.unreadable else []
        problems += [f"{issue} in {len(frames)} frame(s)" for issue, frames in self.issues.items()]
        problems += [f"{count} non-finite value(s) in {layer}" for layer, count in non_finite.items() if count]
        if best and errors[best] >= self.params.get("reprojectionThresholdPx", 0.25):
            problems.append("motion vectors don't match the depth reprojection")
        elif best == "asStored":
            problems.append("matrix layout is 'asStored', not 'transposed' as documented")

        return {
            "ok": not problems,
            "problems": problems,
            **self.scene,
            "framesChecked": self.checked,
            "unreadableFrames": self.unreadable,
            "header": dict(zip(("size", "ssaaFactor", "near", "far"), self.first_header or ())),
            "issueFrames": {issue: frames[:20] for issue, frames in self.issues.items()},  # first 20 per issue
            "reprojection": {**reprojection, "bestLayout": best},
            "motionMagnitudePx": self.motion_px.as_dict()["magnitude"],
            "depthLog10": self.log_depth.as_dict()["log10Z"],
            "stats": {region: {layer: stats.as_dict() for layer, stats in layers.items()}
                      for region, layers in self.stats.items()},
        }

    def describe(self, result: dict) -> str:
        best = result["reprojection"]["bestLayout"]
        error = f"{result['reprojection'][best]['meanErrorPx']:.3f}px ({best})" if best else "n/a"
        status = "OK" if result["ok"] else "PROBLEMS: " + "; ".join(result["problems"])
        return (f"{result['framesChecked']} frames, mean motion {result['motionMagnitudePx']['mean']:.2f}px, "
                f"reprojection error {error} | {status}")
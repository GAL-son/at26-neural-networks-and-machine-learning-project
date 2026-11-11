"""Camera motion and how much the previous frame helps (plan items 2.1 and 3.1).

- motion per scene (mean, p50 / p95 / p99 in pixels) and a static / slow / fast category for the split;
- frames where the camera doesn't move: the renderer has no sub-pixel jitter, so such frames repeat the same
  aliased samples and their history adds no new information;
- previous frame warped by the motion vectors: share of pixels without valid history (disoccluded / off screen),
  history error of the GT (how reusable history is) and of the input (flicker caused by aliasing).
  Errors are in sRGB-encoded values, on geometry pixels with valid history. Bilinear resampling alone gives a
  small error, so compare input against GT rather than reading the absolute numbers.
"""
from __future__ import annotations

import numpy as np
import torch

from .base import Analysis, ChannelStats, Histogram
from .ops import reproject, srgb, warp

MOTION_EDGES = np.concatenate([[0.0], np.geomspace(1e-3, 1024, 61)])

def _fmt(value, spec=".2f"):
    return "n/a" if value is None else format(value, spec)

class TemporalAnalysis(Analysis):
    name = "temporal"

    def begin_scene(self, scene: dict) -> None:
        super().begin_scene(scene)
        self.frames = self.pairs = self.static_frames = 0
        self.motion = Histogram(MOTION_EDGES)
        self.motion_stats = ChannelStats(["px"])
        self.frame_motion = []
        self.history = np.zeros(6)  # geometry pixels, disoccluded, off screen, usable, GT error sum, input error sum

    def update(self, frame, previous) -> None:
        self.frames += 1
        header = frame.header
        vp = np.asarray(header["viewProjection"], np.float64)
        prev_vp = np.asarray(header["prevViewProjection"], np.float64)
        self.static_frames += int(np.array_equal(vp, prev_vp))

        height, width = frame.depth.shape[-2:]
        geometry = frame.geometry_mask()
        magnitude = (frame.motion * frame.motion.new_tensor([width, height]).view(2, 1, 1)).norm(dim=0, keepdim=True)
        self.motion.update(magnitude, geometry)
        self.motion_stats.update(magnitude, geometry)
        self.frame_motion.append(float(torch.where(geometry, magnitude, 0.0).sum() / geometry.sum().clamp_min(1)))

        if previous is None:
            return
        self.pairs += 1
        if self.params.get("matrixLayout", "transposed") == "transposed":  # see OUTPUT_FORMAT.md / integrity
            vp, prev_vp = vp.T, prev_vp.T
        _, expected_depth, in_front = reproject(frame.depth, vp, prev_vp, header["nearPlane"], header["farPlane"])
        warped_depth, inside = warp(previous.depth, frame.motion, mode="nearest")

        geo = geometry[0]
        tolerance = self.params.get("occlusionTolerance", 0.02)
        moved = (warped_depth[0] - expected_depth).abs() > tolerance * expected_depth.abs()
        disoccluded = geo & inside & (~in_front | moved)
        off_screen = geo & ~inside
        usable = geo & inside & ~disoccluded
        gt_error = (srgb(warp(previous.gt, frame.motion)[0]) - srgb(frame.gt)).abs().mean(0)
        input_error = (srgb(warp(previous.albedo, frame.motion)[0]) - srgb(frame.albedo)).abs().mean(0)
        self.history += torch.stack([
            geo.sum().float(), disoccluded.sum().float(), off_screen.sum().float(), usable.sum().float(),
            torch.where(usable, gt_error, 0.0).sum(), torch.where(usable, input_error, 0.0).sum(),
        ]).cpu().double().numpy()

    def end_scene(self) -> dict:
        stats = self.motion_stats.as_dict()["px"]
        mean = stats["mean"] if stats["count"] else None
        static_px, fast_px = self.params.get("staticMotionPx", 0.1), self.params.get("fastMotionPx", 4.0)
        category = None if mean is None else "static" if mean < static_px else "slow" if mean < fast_px else "fast"

        geometry, disoccluded, off_screen, usable, gt_error, input_error = self.history
        history = None
        if usable:
            history = {
                "pairs": self.pairs,
                "disoccludedFraction": disoccluded / geometry,
                "offScreenFraction": off_screen / geometry,
                "gtHistoryError": gt_error / usable,
                "inputHistoryError": input_error / usable,
                "flickerRatio": input_error / max(gt_error, 1e-12),  # > 1: input changes more than GT -> shimmer
            }
        notes = []
        if self.static_frames > 1:  # frame 0 is always static
            notes.append(f"{self.static_frames} frame(s) without camera motion: with no sub-pixel jitter their "
                         "history repeats the same samples")
        if not self.pairs:
            notes.append("no consecutive frame pairs: use stride 1, or clip >= 2 when subsampling")

        return {
            **self.scene,
            "framesChecked": self.frames,
            "motionPx": {"mean": mean, "max": stats["max"] if stats["count"] else None,
                         **{f"p{q}": self.motion.quantile(q / 100) for q in (50, 95, 99)}},
            "category": category,
            "staticFrames": self.static_frames,
            "history": history,
            "meanMotionPxPerFrame": self.frame_motion,
            "notes": notes,
        }

    def describe(self, result: dict) -> str:
        motion, history = result["motionPx"], result["history"] or {}
        return (f"motion mean {_fmt(motion['mean'])} px, p95 {_fmt(motion['p95'])} px ({result['category']}), "
                f"{result['staticFrames']} static frame(s), "
                f"disoccluded {_fmt(100 * history['disoccludedFraction'] if history else None, '.1f')}%, "
                f"flicker ratio {_fmt(history.get('flickerRatio'))}")

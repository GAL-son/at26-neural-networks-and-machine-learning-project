"""Visual check of single frames: inputs, target, residual, edge regions and a zoom on the worst spot."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Rectangle
from scipy import ndimage

from common import (DEFAULT_CAPTURES, DEFAULT_REPORTS, SERIES, find_sequences, geometric_edges, load_frame, pick,
                    regions, save_figure, setup_matplotlib, srgb)

MAGNITUDE = LinearSegmentedColormap.from_list("magnitude", ["#fcfcfb", "#86b6ef", "#2a78d6", "#0d366b"])
REGION_COLORS = ListedColormap(["#383835", "#d9d8d3", SERIES[1]])  # background, interior, edge zone


def sheet(plt, f, zoom: int, path: Path) -> None:
    albedo, gt = srgb(f.albedo), srgb(f.gt)
    residual = np.sqrt(((gt - albedo) ** 2).sum(-1))
    _, masks = regions(f, geometric_edges(f))
    region_id = masks["interior"] * 1 + masks["edge"] * 2  # background stays 0

    half = zoom // 2
    energy = ndimage.uniform_filter(residual ** 2, zoom, mode="constant")
    energy[:half] = energy[-half:] = 0
    energy[:, :half] = energy[:, -half:] = 0
    cy, cx = np.unravel_index(np.argmax(energy), energy.shape)
    y0, x0 = cy - half, cx - half
    window = (slice(y0, y0 + zoom), slice(x0, x0 + zoom))

    fg = ~f.background
    log_depth = np.log(np.maximum(f.depth, 1e-6))
    lo, hi = np.percentile(log_depth[fg], [1, 99]) if fg.any() else (0.0, 1.0)
    depth_image = np.where(fg, 1 - np.clip((log_depth - lo) / (hi - lo + 1e-6), 0, 1), 0)
    normals = np.where(fg[..., None], f.normal * 0.5 + 0.5, 0)
    motion = np.linalg.norm(f.motion_px, axis=-1)
    residual_max = np.percentile(residual, 99.5)

    panels = [
        (albedo, {}, "input albedo (1 spp)"),
        (gt, {}, "ground truth (SSAA 4x4)"),
        (residual, {"cmap": MAGNITUDE, "vmin": 0, "vmax": residual_max}, f"|GT - input|, scale max {residual_max:.3f}"),
        (region_id, {"cmap": REGION_COLORS, "vmin": 0, "vmax": 2}, "regions: background / interior / edge zone"),
        (normals, {}, "world normals"),
        (depth_image, {"cmap": "gray"}, "log depth (near = bright)"),
        (motion, {"cmap": MAGNITUDE}, f"|motion| [px], max {motion.max():.1f}"),
        (np.concatenate([albedo[window], gt[window]], axis=1), {}, f"worst {zoom}px window: input | ground truth"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(20, 6.6))
    for ax, (image, kwargs, title) in zip(axes.ravel(), panels):
        ax.imshow(image, interpolation="nearest", **kwargs)
        ax.set_title(title, loc="left", fontsize=9)
        ax.axis("off")
    for ax in (axes[0, 0], axes[0, 2]):
        ax.add_patch(Rectangle((x0, y0), zoom, zoom, fill=False, edgecolor=SERIES[1], lw=1.5))
    fig.suptitle(f"{f.path.parent.name} / {f.path.name}", x=0.01, ha="left")
    save_figure(fig, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, default=DEFAULT_CAPTURES)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--frames", type=int, default=3, help="frames per scene, evenly spaced")
    parser.add_argument("--zoom", type=int, default=64)
    args = parser.parse_args()

    plt = setup_matplotlib()
    for scene, paths in find_sequences(args.captures).items():
        for path in pick(paths, args.frames):
            print(f"[{scene}] {path.name}")
            sheet(plt, load_frame(path), args.zoom, args.out / "contact_sheets" / f"{scene}_{path.stem}.png")


if __name__ == "__main__":
    main()




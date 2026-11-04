"""Pass 1: integrity scan of every captured frame: shapes, headers, gaps, NaNs, I/O cost."""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from common import (DEFAULT_CAPTURES, DEFAULT_REPORTS, LAYERS, block_mean, find_sequences, frame_index,
                    load_frame, save_figure, save_json, scene_colors, setup_matplotlib)

STATIC_THRESHOLD = 0.5 / 255  # mean change of a GT thumbnail below this = camera standing still


def scan_frame(path: Path):
    start = time.perf_counter()
    f = load_frame(path)
    load_s = time.perf_counter() - start

    fg = ~f.background
    normal_len = np.linalg.norm(f.normal[fg], axis=-1)
    motion = np.linalg.norm(f.motion_px, axis=-1)[fg]
    row = {
        "index": f.index,
        "header_index": int(f.header.get("frameIndex", -1)),
        "width": f.width,
        "height": f.height,
        "ssaa": int(f.header.get("ssaaFactor", -1)),
        "near": float(f.header.get("nearPlane", np.nan)),
        "far": f.far,
        "file_mb": path.stat().st_size / 2 ** 20,
        "load_s": load_s,
        "nonfinite": sum(int((~np.isfinite(getattr(f, layer))).sum()) for layer in LAYERS),
        "background_frac": float(1.0 - fg.mean()),
        "normal_len_err": float(np.abs(normal_len - 1.0).max()) if fg.any() else 0.0,
        "motion_mean_px": float(motion.mean()) if fg.any() else 0.0,
        "motion_p99_px": float(np.percentile(motion, 99)) if fg.any() else 0.0,
        "residual_mean": float(np.abs(f.gt - f.albedo).mean()),
        "gt_min": float(f.gt.min()),
        "gt_max": float(f.gt.max()),
    }
    return row, block_mean(f.gt.mean(-1), 16)


def summarize(paths: list[Path], rows: list[dict], thumbs: list[np.ndarray]) -> dict:
    indices = [frame_index(p) for p in paths]
    missing = sorted(set(range(max(indices) + 1)) - set(indices))
    static = sum(float(np.abs(a - b).mean()) < STATIC_THRESHOLD for a, b in zip(thumbs, thumbs[1:]))
    distinct = lambda key: sorted({r[key] for r in rows})
    s = {
        "frames": len(paths),
        "scanned": len(rows),
        "missing_indices": missing,
        "header_index_mismatch": [r["index"] for r in rows if r["header_index"] != r["index"]],
        "resolutions": sorted({(r["width"], r["height"]) for r in rows}),
        "ssaa": distinct("ssaa"),
        "near": distinct("near"),
        "far": distinct("far"),
        "nonfinite_values": sum(r["nonfinite"] for r in rows),
        "max_normal_len_err": max(r["normal_len_err"] for r in rows),
        "gt_range": [min(r["gt_min"] for r in rows), max(r["gt_max"] for r in rows)],
        "near_static_pairs": int(static),
        "mostly_background_frames": sum(r["background_frac"] > 0.5 for r in rows),
        "motion_mean_px": float(np.mean([r["motion_mean_px"] for r in rows])),
        "motion_p99_px_max": max(r["motion_p99_px"] for r in rows),
        "dataset_gb": sum(p.stat().st_size for p in paths) / 2 ** 30,
        "mean_load_s": float(np.mean([r["load_s"] for r in rows])),
    }
    s["full_pass_decode_min"] = s["mean_load_s"] * len(paths) / 60

    warnings = []
    if missing:
        warnings.append(f"{len(missing)} missing frame indices, first {missing[:5]}")
    if s["header_index_mismatch"]:
        warnings.append("frameIndex header differs from file name")
    for key in ("resolutions", "ssaa", "near", "far"):
        if len(s[key]) > 1:
            warnings.append(f"inconsistent {key}: {s[key]}")
    if s["nonfinite_values"]:
        warnings.append(f"{s['nonfinite_values']} NaN/Inf values")
    if s["max_normal_len_err"] > 0.01:
        warnings.append(f"normals not unit length (max error {s['max_normal_len_err']:.3f})")
    if s["gt_range"][0] < 0 or s["gt_range"][1] > 1.001:
        warnings.append(f"ground truth outside [0, 1]: {s['gt_range']}")
    if s["mostly_background_frames"]:
        warnings.append(f"{s['mostly_background_frames']} frames are >50% background (camera outside the scene?)")
    first = next((r for r in rows if r["index"] == 0), None)
    if first and first["motion_mean_px"] > 1e-3:
        warnings.append("frame 0 has non-zero motion")
    s["warnings"] = warnings
    return s


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, default=DEFAULT_CAPTURES)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--stride", type=int, default=1, help="scan every n-th frame")
    args = parser.parse_args()

    sequences = find_sequences(args.captures)
    colors = scene_colors(list(sequences))
    summaries, all_rows = {}, {}
    for scene, paths in sequences.items():
        rows, thumbs = [], []
        scanned = paths[::args.stride]
        for i, path in enumerate(scanned, 1):
            print(f"\r[{scene}] {i}/{len(scanned)} {path.name}", end="", flush=True)
            row, thumb = scan_frame(path)
            rows.append(row)
            thumbs.append(thumb)
        print()

        args.out.mkdir(parents=True, exist_ok=True)
        with open(args.out / f"inventory_{scene}.csv", "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        s = summaries[scene] = summarize(paths, rows, thumbs)
        all_rows[scene] = rows
        print(f"  {s['frames']} frames {s['resolutions']} ssaa {s['ssaa']} | {s['dataset_gb']:.1f} GB, "
              f"{s['mean_load_s']:.2f} s/frame -> {s['full_pass_decode_min']:.1f} min per full pass | "
              f"motion mean {s['motion_mean_px']:.2f} px, p99 max {s['motion_p99_px_max']:.1f} px | "
              f"near-static pairs {s['near_static_pairs']}")
        for warning in s["warnings"]:
            print(f"  WARNING {warning}")

    save_json(args.out / "inventory.json", summaries)

    plt = setup_matplotlib()
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    for scene, rows in all_rows.items():
        x = [r["index"] for r in rows]
        top.plot(x, [r["motion_mean_px"] for r in rows], color=colors[scene], lw=1.5, label=scene)
        bottom.plot(x, [r["background_frac"] for r in rows], color=colors[scene], lw=1.5)
    top.set_ylabel("mean |motion| [px / frame]")
    bottom.set_ylabel("background share")
    bottom.set_xlabel("frame")
    if len(all_rows) > 1:
        top.legend()
    save_figure(fig, args.out / "inventory_timeline.png")


if __name__ == "__main__":
    main()






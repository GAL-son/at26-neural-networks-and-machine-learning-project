"""Pass 3: frame-to-frame structure: redundancy, train/validation gap, motion-vector convention, flicker."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage

from common import (DEFAULT_CAPTURES, DEFAULT_REPORTS, INK_2, SERIES, block_mean, find_sequences, geometric_edges,
                    load_frame, luminance, pick, save_figure, save_json, scene_colors, setup_matplotlib, srgb,
                    warp_previous)

CONVENTIONS = {"+x +y (header)": (1, 1), "-x +y": (-1, 1), "+x -y": (1, -1), "-x -y": (-1, -1), "ignore motion": (0, 0)}


def autocorrelation(paths: list, stride: int, max_lag: int):
    frames = [load_frame(p) for p in paths[::stride]]
    data = np.stack([block_mean(luminance(srgb(f.gt)), 8).ravel() for f in frames])
    data = (data - data.mean(1, keepdims=True)) / (data.std(1, keepdims=True) + 1e-8)
    lags, corr = [0], [1.0]
    for k in range(1, min(len(data), max_lag // stride + 1)):
        lags.append(k * stride)
        corr.append(float((data[:-k] * data[k:]).mean()))
    return np.array(lags), np.array(corr)


def pair_metrics(prev, cur):
    motion = cur.motion_px
    fg = ~cur.background
    if not fg.any() or np.percentile(np.linalg.norm(motion, axis=-1)[fg], 90) < 0.5:
        return None  # camera (almost) standing still: nothing to test
    edges = geometric_edges(cur)
    edge_zone = ndimage.binary_dilation(edges, iterations=1) & fg
    interior = fg & ~ndimage.binary_dilation(edges, iterations=4)
    gt_prev, gt_cur = luminance(srgb(prev.gt)), luminance(srgb(cur.gt))
    in_prev, in_cur = luminance(srgb(prev.albedo)), luminance(srgb(cur.albedo))

    result = {}
    for name, sign in CONVENTIONS.items():
        warped, valid = warp_previous(gt_prev, motion, sign)
        mask = interior & valid
        result[name] = float(np.median(np.abs(warped - gt_cur)[mask])) if mask.any() else np.nan

    warped_gt, valid = warp_previous(gt_prev, motion)
    warped_in, _ = warp_previous(in_prev, motion)
    mask = edge_zone & valid
    if mask.any():
        result["flicker_input"] = float(np.abs(warped_in - in_cur)[mask].mean())
        result["flicker_gt"] = float(np.abs(warped_gt - gt_cur)[mask].mean())
    return result


def first_lag_below(lags, corr, level):
    below = np.nonzero(corr < level)[0]
    return int(lags[below[0]]) if below.size else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, default=DEFAULT_CAPTURES)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--stride", type=int, default=4, help="frame stride for the autocorrelation")
    parser.add_argument("--max-lag", type=int, default=300, help="frames")
    parser.add_argument("--pairs", type=int, default=40, help="consecutive frame pairs per scene")
    args = parser.parse_args()

    sequences = find_sequences(args.captures)
    colors = scene_colors(list(sequences))
    summaries, curves = {}, {}
    for scene, paths in sequences.items():
        print(f"[{scene}] autocorrelation")
        lags, corr = autocorrelation(paths, args.stride, args.max_lag)
        curves[scene] = (lags, corr)

        results, static = [], 0
        positions = pick(list(range(1, len(paths))), args.pairs)
        for i, t in enumerate(positions, 1):
            print(f"\r[{scene}] pair {i}/{len(positions)}", end="", flush=True)
            prev, cur = load_frame(paths[t - 1]), load_frame(paths[t])
            if cur.index - prev.index != 1:
                continue
            metrics = pair_metrics(prev, cur)
            if metrics is None:
                static += 1
            else:
                results.append(metrics)
        print()

        convention = {name: float(np.nanmean([r[name] for r in results])) if results else np.nan for name in CONVENTIONS}
        flicker = [r for r in results if "flicker_input" in r]
        s = summaries[scene] = {
            "split_gap_frames": {"corr<0.5": first_lag_below(lags, corr, 0.5), "corr<0.2": first_lag_below(lags, corr, 0.2)},
            "pairs_tested": len(results),
            "pairs_static": static,
            "motion_convention_median_error": convention,
            "best_convention": min(convention, key=lambda k: np.nan_to_num(convention[k], nan=np.inf)),
            "flicker_input": float(np.mean([r["flicker_input"] for r in flicker])) if flicker else np.nan,
            "flicker_gt": float(np.mean([r["flicker_gt"] for r in flicker])) if flicker else np.nan,
        }
        s["flicker_ratio"] = s["flicker_input"] / max(s["flicker_gt"], 1e-12)
        print(f"  split gap {s['split_gap_frames']} frames | static pairs {static}/{len(positions)}")
        print("  warp error by convention: " + ", ".join(f"{k} {v:.4f}" for k, v in convention.items()))
        if s["best_convention"] != "+x +y (header)":
            print(f"  WARNING motion vectors fit '{s['best_convention']}' better than the header convention")
        print(f"  edge flicker input {s['flicker_input']:.4f} vs GT {s['flicker_gt']:.4f} (x{s['flicker_ratio']:.2f})")

    save_json(args.out / "temporal.json", summaries)

    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for scene, (lags, corr) in curves.items():
        ax.plot(lags, corr, color=colors[scene], label=scene)
    ax.axhline(0.5, color=INK_2, lw=0.8, ls=":")
    ax.set_xlabel("lag [frames]")
    ax.set_ylabel("correlation of GT thumbnails")
    if len(curves) > 1:
        ax.legend()
    save_figure(fig, args.out / "temporal_autocorrelation.png")

    fig, ax = plt.subplots(figsize=(1.8 + 1.6 * len(summaries), 3.2))
    x = np.arange(len(summaries))
    ax.bar(x - 0.2, [s["flicker_input"] for s in summaries.values()], 0.38, color=SERIES[0], label="aliased input")
    ax.bar(x + 0.2, [s["flicker_gt"] for s in summaries.values()], 0.38, color=SERIES[1], label="ground truth")
    ax.set_xticks(x, list(summaries))
    ax.set_ylabel("warped frame difference on edges")
    ax.legend()
    save_figure(fig, args.out / "temporal_flicker.png")


if __name__ == "__main__":
    main()




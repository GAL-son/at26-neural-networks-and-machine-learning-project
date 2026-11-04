"""Pass 2: what the network has to learn, measured on evenly spaced frames of every scene."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib.ticker import PercentFormatter
from scipy import ndimage

from common import (DEFAULT_CAPTURES, DEFAULT_REPORTS, INK_2, SERIES, find_sequences, geometric_edges, load_frame,
                    luminance, pick, psnr, regions, save_figure, save_json, scene_colors, setup_matplotlib, srgb,
                    ssim_map)

REGIONS = ("edge", "interior", "background")
BLUR_SIGMAS = (0.5, 0.75, 1.0)
MAX_DISTANCE = 32  # px; the last distance bin collects everything further away
SAMPLES_PER_FRAME = 50_000


def new_stats() -> dict:
    return {
        "pixels": dict.fromkeys(REGIONS, 0),
        "sq": dict.fromkeys(REGIONS, 0.0),  # squared error in sRGB, summed over RGB
        "l1": dict.fromkeys(REGIONS, 0.0),  # absolute error in linear space
        "baselines": defaultdict(lambda: defaultdict(float)),
        "ssim": defaultdict(lambda: defaultdict(list)),
        "dist_sq": np.zeros(MAX_DISTANCE + 1),
        "angle_sq": np.zeros(46),
        "crop_energy": [],
        "crop_edge": [],
        "samples": defaultdict(list),
    }


def analyze_frame(f, st: dict, rng, crop: int, stride: int) -> None:
    edges = geometric_edges(f)
    distance, masks = regions(f, edges)
    albedo, gt = srgb(f.albedo), srgb(f.gt)
    sq = ((gt - albedo) ** 2).sum(-1)
    l1 = np.abs(f.gt - f.albedo).sum(-1)

    # 1. Where the error lives
    for name, mask in masks.items():
        st["pixels"][name] += int(mask.sum())
        st["sq"][name] += float(sq[mask].sum())
        st["l1"][name] += float(l1[mask].sum())

    # 2. Trivial baselines the network has to beat
    gt_luma = luminance(gt)
    candidates = {"identity": albedo}
    candidates.update({f"blur_{s}": srgb(ndimage.gaussian_filter(f.albedo, (s, s, 0))) for s in BLUR_SIGMAS})
    for name, prediction in candidates.items():
        err = ((prediction - gt) ** 2).sum(-1)
        ssim = ssim_map(luminance(prediction), gt_luma)
        b = st["baselines"][name]
        b["sq_all"] += float(err.sum())
        b["px_all"] += err.size
        st["ssim"][name]["all"].append(float(ssim.mean()))
        for region in ("edge", "interior"):
            mask = masks[region]
            b[f"sq_{region}"] += float(err[mask].sum())
            b[f"px_{region}"] += int(mask.sum())
            if mask.any():
                st["ssim"][name][region].append(float(ssim[mask].mean()))

    # 3. How far from a geometric edge the error reaches
    bins = np.minimum(distance, MAX_DISTANCE).astype(int)
    st["dist_sq"] += np.bincount(bins.ravel(), weights=sq.ravel(), minlength=MAX_DISTANCE + 1)

    # 4. Edge orientation: near-axis edges make long staircases that need wide context
    gy = ndimage.gaussian_filter(gt_luma, 1.0, order=(1, 0))
    gx = ndimage.gaussian_filter(gt_luma, 1.0, order=(0, 1))
    strong = edges & (np.hypot(gx, gy) > 0.01)
    phi = np.degrees(np.arctan2(gy[strong], gx[strong])) % 90.0
    deviation = np.minimum(phi, 90.0 - phi)  # 0 = axis aligned, 45 = diagonal
    weight = ndimage.uniform_filter(sq, 3)[strong]
    st["angle_sq"] += np.bincount(np.round(deviation).astype(int), weights=weight, minlength=46)[:46]

    # 5. What random crops contain (summed-area tables, crops on a stride grid)
    sat = lambda a: np.pad(a.astype(np.float64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sq_sat, edge_sat = sat(sq), sat(masks["edge"])
    y, x = np.meshgrid(np.arange(0, f.height - crop + 1, stride), np.arange(0, f.width - crop + 1, stride),
                       indexing="ij")
    box = lambda s: s[y + crop, x + crop] - s[y, x + crop] - s[y + crop, x] + s[y, x]
    st["crop_energy"].append((box(sq_sat) / crop ** 2).ravel())
    st["crop_edge"].append((box(edge_sat) / crop ** 2).ravel())

    # 6. Raw value samples for distributions and normalisation
    idx = rng.integers(0, sq.size, SAMPLES_PER_FRAME)
    fg = ~f.background.ravel()[idx]
    samples = st["samples"]
    samples["albedo_srgb"].append(albedo.reshape(-1, 3)[idx])
    samples["gt_srgb"].append(gt.reshape(-1, 3)[idx])
    samples["residual_srgb"].append(np.sqrt(sq.ravel()[idx]))
    samples["normal"].append(f.normal.reshape(-1, 3)[idx][fg])
    samples["depth"].append(f.depth.ravel()[idx][fg])
    samples["motion_px"].append(f.motion_px.reshape(-1, 2)[idx][fg])


def describe(x: np.ndarray) -> dict:
    q = np.percentile(x, [0.1, 1, 50, 99, 99.9], axis=0)
    return {"mean": x.mean(0), "std": x.std(0), "min": x.min(0), "max": x.max(0),
            "p0.1": q[0], "p1": q[1], "p50": q[2], "p99": q[3], "p99.9": q[4]}


def step_length(deviation_deg: np.ndarray) -> np.ndarray:
    """Pixels between the steps of the staircase an edge at this angle produces."""
    return 1.0 / np.tan(np.radians(np.maximum(deviation_deg, 0.25)))


def summarize(st: dict) -> tuple[dict, dict]:
    total_px, total_sq, total_l1 = (sum(st[k].values()) for k in ("pixels", "sq", "l1"))
    region_summary = {
        r: {"pixel_share": st["pixels"][r] / total_px,
            "error_share_srgb": st["sq"][r] / total_sq,
            "error_share_linear_l1": st["l1"][r] / total_l1,
            "rmse_srgb": float(np.sqrt(st["sq"][r] / max(3 * st["pixels"][r], 1)))}
        for r in REGIONS}
    density = {r: st["sq"][r] / max(st["pixels"][r], 1) for r in REGIONS}

    baselines = {}
    for name, b in st["baselines"].items():
        baselines[name] = {f"psnr_{r}": psnr(b[f"sq_{r}"] / max(3 * b[f"px_{r}"], 1)) for r in ("all", "edge", "interior")}
        baselines[name].update({f"ssim_{r}": float(np.mean(v)) for r, v in st["ssim"][name].items()})

    distance_cum = np.cumsum(st["dist_sq"]) / st["dist_sq"].sum()
    reach = {f"{int(p * 100)}%": int(np.searchsorted(distance_cum, p)) for p in (0.5, 0.9, 0.99)}

    angle_share = st["angle_sq"] / st["angle_sq"].sum()
    lengths = step_length(np.arange(46))
    long_steps = {f">{n}px": float(angle_share[lengths > n].sum()) for n in (16, 32, 64, 128)}

    crops = np.concatenate(st["crop_energy"])
    crop_edge = np.concatenate(st["crop_edge"])
    ordered = np.sort(crops)[::-1]
    negligible = crops < 0.05 * crops.mean()
    proposal = np.sqrt(crops + 0.01 * crops.mean())
    proposal /= proposal.sum()
    crop_summary = {
        "count": int(crops.size),
        "negligible_share": float(negligible.mean()),
        "top20_error_share": float(ordered[: max(1, len(ordered) // 5)].sum() / ordered.sum()),
        "median_edge_pixel_share": float(np.median(crop_edge)),
        "sqrt_sampling": {
            "negligible_draw_share": float(proposal[negligible].sum()),
            "error_per_crop_vs_uniform": float((proposal * crops).sum() / crops.mean()),
        },
    }

    summary = {
        "regions": region_summary,
        "error_density_edge_vs_interior": density["edge"] / max(density["interior"], 1e-12),
        "baselines": baselines,
        "error_radius_px": reach,  # 32 means "32 px or more"
        "edge_error_on_steps_longer_than": long_steps,
        "crops": crop_summary,
    }
    plot_data = {"distance_cum": distance_cum, "angle_share": angle_share, "crops": crops,
                 "samples": {k: np.concatenate(v) for k, v in st["samples"].items()}}
    return summary, plot_data


def channel_report(samples: dict) -> dict:
    report = {key: describe(value) for key, value in samples.items() if len(value)}
    if len(samples["depth"]):
        report["log_depth"] = describe(np.log(samples["depth"]))
    if len(samples["motion_px"]):
        report["motion_px_magnitude"] = describe(np.linalg.norm(samples["motion_px"], axis=-1))
    return report


def plot_region_share(plt, summaries: dict, path: Path) -> None:
    scenes = list(summaries)
    fig, axes = plt.subplots(1, len(scenes), figsize=(4.6 * len(scenes), 2.6), squeeze=False, sharex=True)
    y = np.arange(len(REGIONS))
    for ax, scene in zip(axes[0], scenes):
        r = summaries[scene]["regions"]
        for offset, key, color, label in ((-0.21, "pixel_share", SERIES[0], "share of pixels"),
                                          (0.21, "error_share_srgb", SERIES[1], "share of squared error (sRGB)")):
            values = [r[k][key] for k in REGIONS]
            ax.barh(y + offset, values, 0.4, color=color, label=label)
            for yy, v in zip(y, values):
                ax.text(v + 0.01, yy + offset, f"{v:.0%}", va="center", fontsize=8)
        ax.set_yticks(y, REGIONS)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.15)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_title(scene, loc="left")
    axes[0][0].legend(loc="lower right")
    save_figure(fig, path)


def plot_lines(plt, series: dict, colors: dict, path: Path, xlabel: str, ylabel: str,
               reference=None, logx=False) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for scene, (x, y) in series.items():
        ax.plot(x, y, color=colors[scene], label=scene)
    if reference is not None:
        ax.plot(*reference, color=INK_2, lw=0.8, ls=":")
    if logx:
        ax.set_xscale("log", base=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    if len(series) > 1:
        ax.legend()
    save_figure(fig, path)


def plot_distributions(plt, plot_data: dict, colors: dict, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    panels = [
        (axes[0, 0], lambda s: np.log10(s["depth"]), "log10 depth [scene units], foreground", False),
        (axes[0, 1], lambda s: np.linalg.norm(s["motion_px"], axis=-1), "|motion| [px / frame], foreground", True),
        (axes[1, 0], lambda s: luminance(s["gt_srgb"]), "ground-truth luminance (sRGB)", False),
        (axes[1, 1], lambda s: s["residual_srgb"], "|GT - input| per pixel (sRGB)", True),
    ]
    for ax, extract, title, log_y in panels:
        for scene, data in plot_data.items():
            values = extract(data["samples"])
            if len(values):
                ax.hist(values, bins=120, density=True, histtype="step", lw=1.5, color=colors[scene], label=scene)
        ax.set_title(title, loc="left")
        if log_y:
            ax.set_yscale("log")
    if len(plot_data) > 1:
        axes[0, 0].legend()
    save_figure(fig, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, default=DEFAULT_CAPTURES)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--frames", type=int, default=40, help="frames per scene, evenly spaced (0 = all)")
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument("--crop-stride", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    sequences = find_sequences(args.captures)
    colors = scene_colors(list(sequences))
    summaries, plot_data = {}, {}
    for scene, paths in sequences.items():
        st = new_stats()
        chosen = pick(paths, args.frames)
        for i, path in enumerate(chosen, 1):
            print(f"\r[{scene}] {i}/{len(chosen)} {path.name}", end="", flush=True)
            analyze_frame(load_frame(path), st, rng, args.crop, args.crop_stride)
        print()
        summaries[scene], plot_data[scene] = summarize(st)
        summaries[scene]["channels"] = channel_report(plot_data[scene]["samples"])

        s = summaries[scene]
        r = s["regions"]
        best = max(s["baselines"], key=lambda k: s["baselines"][k]["psnr_all"])
        print(f"  edge zone {r['edge']['pixel_share']:.1%} of pixels -> {r['edge']['error_share_srgb']:.0%} of error | "
              f"interior {r['interior']['pixel_share']:.1%} -> {r['interior']['error_share_srgb']:.0%} | "
              f"background -> {r['background']['error_share_srgb']:.0%} | "
              f"edge/interior error density x{s['error_density_edge_vs_interior']:.1f}")
        print(f"  identity PSNR {s['baselines']['identity']['psnr_all']:.2f} dB "
              f"(edge {s['baselines']['identity']['psnr_edge']:.2f}) | best {best} "
              f"{s['baselines'][best]['psnr_all']:.2f} dB (edge {s['baselines'][best]['psnr_edge']:.2f})")
        print(f"  error radius {s['error_radius_px']} px | edge error on staircases "
              f"{s['edge_error_on_steps_longer_than']}")
        c = s["crops"]
        print(f"  crops: {c['negligible_share']:.0%} negligible, top 20% hold {c['top20_error_share']:.0%} of error, "
              f"sqrt sampling gives x{c['sqrt_sampling']['error_per_crop_vs_uniform']:.2f} error per crop")
        depth = s["channels"].get("log_depth")
        if depth is not None:
            print(f"  median depth {np.exp(depth['p50']):.2f} units, motion p99 "
                  f"{s['channels']['motion_px_magnitude']['p99']:.2f} px")

    all_samples = {k: np.concatenate([plot_data[s]["samples"][k] for s in plot_data]) for k in ("albedo_srgb", "normal", "motion_px")}
    normalization = {
        "note": "computed on all scenes - recompute on the training split only before training",
        "albedo_srgb": {"mean": all_samples["albedo_srgb"].mean(0), "std": all_samples["albedo_srgb"].std(0)},
        "normal": {"mean": all_samples["normal"].mean(0), "std": all_samples["normal"].std(0)},
        "motion_px": {"std": all_samples["motion_px"].std(0)},
        "median_depth_per_scene": {s: float(np.exp(summaries[s]["channels"]["log_depth"]["p50"]))
                                   for s in summaries if "log_depth" in summaries[s]["channels"]},
    }
    save_json(args.out / "analysis.json", summaries)
    save_json(args.out / "normalization.json", normalization)

    plt = setup_matplotlib()
    plot_region_share(plt, summaries, args.out / "region_share.png")
    plot_lines(plt, {s: (np.arange(MAX_DISTANCE + 1), d["distance_cum"]) for s, d in plot_data.items()}, colors,
               args.out / "error_vs_edge_distance.png",
               "distance to nearest geometric edge [px] (last point = 32+)", "cumulative share of squared error")
    grid = np.geomspace(1, 256, 64)
    lengths = step_length(np.arange(46))
    plot_lines(plt, {s: (grid, [d["angle_share"][lengths > g].sum() for g in grid]) for s, d in plot_data.items()},
               colors, args.out / "edge_step_length.png",
               "staircase step length L [px]", "share of edge error on steps longer than L", logx=True)
    lorenz = {}
    for s, d in plot_data.items():
        ordered = np.sort(d["crops"])[::-1]
        lorenz[s] = (np.arange(1, len(ordered) + 1) / len(ordered), np.cumsum(ordered) / ordered.sum())
    plot_lines(plt, lorenz, colors, args.out / "crop_energy.png",
               f"share of {args.crop}px crops, sorted by error", "cumulative share of squared error",
               reference=([0, 1], [0, 1]))
    plot_distributions(plt, plot_data, colors, args.out / "distributions.png")


if __name__ == "__main__":
    main()




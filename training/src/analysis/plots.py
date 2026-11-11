"""Charts of an analysis run:  python -m analysis.plots [run folder]   (default: the newest run)

Writes <run>/plots/*.png and <run>/plots/summary.csv (one row per scene, the numbers behind the charts).
Scenes are sorted hardest first (lowest baseline PSNR on top); "(!)" marks a scene that fails the
integrity / consistency gate.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt
import numpy as np

from common.config import load_config

from .runner import DEFAULTS, REPO_ROOT

# Colours: validated categorical slots (colour-blind safe) + one-hue ramp for ordered values.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # blue, orange, aqua
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # light -> dark blue
DASHED = (0, (4, 3))

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 9,
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": INK2, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.labelsize": 8, "legend.frameon": False, "legend.fontsize": 8,
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
})

def pick(data, *keys):
    """data[k1][k2]... or None when any level is missing."""
    for key in keys:
        data = data.get(key) if isinstance(data, dict) else None
    return data

def pct(value):
    return None if value is None else 100 * value

def fmt(value):
    return "n/a" if value is None else f"{value:.3g}"

# ---------------------------------------------------------------- data

def latest_run() -> Path:
    config = load_config(REPO_ROOT / "config" / "analysis.json", DEFAULTS, ("dataset", "outputDir"))
    runs = sorted(p for p in Path(config["outputDir"]).iterdir() if (p / "run.json").exists())
    if not runs:
        raise SystemExit(f"No analysis runs in {config['outputDir']}")
    return runs[-1]

def load_run(run: Path) -> dict[str, dict]:
    """{scene: {analysis: result}}"""
    reports = {}
    for name in ("integrity", "consistency", "residual", "temporal"):
        path = run / f"{name}.json"
        if path.exists():
            for scene, result in json.loads(path.read_text("utf-8")).items():
                reports.setdefault(scene, {})[name] = result
    return reports

def summary_row(scene: str, r: dict) -> dict:
    best = pick(r, "integrity", "reprojection", "bestLayout")
    gate = pick(r, "integrity", "ok") is not False and pick(r, "consistency", "ok") is not False
    regions = pick(r, "residual", "regions") or {}
    return {
        "scene": scene,
        "label": scene if gate else f"(!) {scene}",
        "split": pick(r, "integrity", "split"),
        "gate ok": gate,
        "frames": pick(r, "residual", "framesChecked"),
        "reprojection error px": pick(r, "integrity", "reprojection", best, "meanErrorPx") if best else None,
        "clipped input %": pct(pick(r, "consistency", "clippedFraction", "input")),
        "baseline PSNR dB": pick(r, "residual", "baseline", "psnrMean"),
        "worst-frame PSNR dB": pick(r, "residual", "baseline", "psnrWorstFrame"),
        "baseline SSIM": pick(r, "residual", "baseline", "ssimMean"),
        "changed pixels %": pct(pick(r, "residual", "changedPixelFraction")),
        "edge pixels %": pct(pick(regions, "geometricEdges", "pixelShare")),
        "edge error %": pct(pick(regions, "geometricEdges", "shareOfSquaredError")),
        "interior error %": pct(pick(regions, "interior", "shareOfSquaredError")),
        "flat patches %": pct(pick(r, "residual", "patches", "flatFraction")),
        "top-10% patches error %": pct(pick(r, "residual", "patches", "errorShareOfTop10Percent")),
        "motion mean px": pick(r, "temporal", "motionPx", "mean"),
        "motion p95 px": pick(r, "temporal", "motionPx", "p95"),
        "category": pick(r, "temporal", "category"),
        "static frames": pick(r, "temporal", "staticFrames"),
        "disoccluded %": pct(pick(r, "temporal", "history", "disoccludedFraction")),
        "flicker ratio": pick(r, "temporal", "history", "flickerRatio"),
        "log10 depth mean": pick(r, "integrity", "depthLog10", "mean"),
        "log10 depth std": pick(r, "integrity", "depthLog10", "std"),
    }

def write_csv(rows: list[dict], path: Path) -> None:
    columns = [c for c in rows[0] if c != "label"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: f"{v:.4g}" if isinstance(v, float) else v for c, v in row.items()})

# ---------------------------------------------------------------- charts

def save(fig, path: Path, title: str) -> None:
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def scene_axis(ax, rows) -> None:
    ax.set_yticks(np.arange(len(rows)), [row["label"] for row in rows])
    ax.invert_yaxis()  # hardest scene on top

OVERVIEW = [  # column, title, mark, reference line (param, default)
    ("baseline PSNR dB", "Baseline PSNR (dB): score to beat", "dot", None),
    ("baseline SSIM", "Baseline SSIM", "dot", None),
    ("changed pixels %", "Pixels changed by AA (%)", "bar", None),
    ("edge error %", "Error at geometric edges (%)", "bar", None),
    ("clipped input %", "Clipped input pixels (%)", "bar", ("consistency", "clipThreshold", 0.01, 100)),
    ("motion mean px", "Mean motion (px / frame)", "bar", ("temporal", "fastMotionPx", 4.0, 1)),
    ("flicker ratio", "Flicker ratio (input / GT history error)", "bar", 1.0),
    ("disoccluded %", "Disoccluded pixels (%)", "bar", None),
]

def overview(rows, params, out: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(17, 2 * (0.32 * len(rows) + 0.9)), sharey=True)
    for ax, (column, title, mark, ref) in zip(axes.flat, OVERVIEW):
        values = [row[column] for row in rows]
        known = [(i, v) for i, v in enumerate(values) if v is not None]
        if mark == "bar":  # magnitudes: bars from zero
            ax.barh([i for i, _ in known], [v for _, v in known], height=0.6, color=SERIES[0])
            ax.set_xlim(0, max([v for _, v in known] + [1e-9]) * 1.2)
            ax.grid(axis="y", visible=False)
        else:              # values bunched far from zero: dots
            ax.scatter([v for _, v in known], [i for i, _ in known], s=36, color=SERIES[0], zorder=3)
            ax.margins(x=0.25)
        for i, v in enumerate(values):
            if v is None:
                ax.text(0.01, i, "n/a", transform=ax.get_yaxis_transform(), va="center", fontsize=7, color=MUTED)
            else:
                ax.annotate(fmt(v), (v, i), xytext=(4, 0), textcoords="offset points", va="center",
                            fontsize=7, color=INK2)
        if isinstance(ref, tuple):
            analysis, key, default, scale = ref
            ref = scale * params.get(analysis, {}).get(key, default)
        if ref is not None:
            ax.axvline(ref, color=INK2, lw=1, ls=DASHED)
            title += f"  (dashed: {ref:g})"
        ax.set_title(title)
    scene_axis(axes[0, 0], rows)
    save(fig, out / "01_overview.png", "Scene overview")

REGIONS = [("geometricEdges", "geometric edges"), ("interior", "interior (textures, sub-pixel detail)"),
           ("background", "background")]

def error_location(reports, rows, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 1.2 + 0.36 * len(rows)), sharey=True)
    y = np.arange(len(rows))
    for ax, (key, title) in zip(axes, [("pixelShare", "Share of pixels (%)"),
                                       ("shareOfSquaredError", "Share of squared error (%)")]):
        left = np.zeros(len(rows))
        for (region, label), color in zip(REGIONS, SERIES):
            widths = np.array([pct(pick(reports[row["scene"]], "residual", "regions", region, key)) or 0.0
                               for row in rows])
            ax.barh(y, widths, left=left, height=0.6, color=color, edgecolor=SURFACE, linewidth=1.5, label=label)
            for i, (start, width) in enumerate(zip(left, widths)):
                if width >= 8:  # label only segments wide enough to hold it
                    ax.text(start + width / 2, i, f"{width:.0f}", ha="center", va="center", fontsize=7,
                            color="#ffffff")
            left += widths
        ax.set_xlim(0, 100)
        ax.grid(axis="y", visible=False)
        ax.set_title(title)
    scene_axis(axes[0], rows)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.02))
    save(fig, out / "02_error_location.png",
         "Where the aliasing error is: edges -> depth/normal inputs help; interior -> textures / temporal")

def error_distribution(reports, rows, params, out: Path) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 1.2 + 0.36 * len(rows)), sharey=True)
    residual = params.get("residual", {})
    for q, color in zip(("p50", "p90", "p99", "p99.9"), RAMP):
        points = [(255 * v, i) for i, row in enumerate(rows)
                  if (v := pick(reports[row["scene"]], "residual", "pixelErrorQuantiles", q))]
        if points:
            left.scatter(*zip(*points), s=36, color=color, label=q, zorder=3)
    left.axvline(255 * residual.get("changeThreshold", 2 / 255), color=INK2, lw=1, ls=DASHED)
    left.set_xscale("log")
    left.set_xlabel("largest channel error, display levels (x / 255), log scale")
    left.set_title("Per-pixel error quantiles  (dashed: 'changed' threshold)")
    left.legend(loc="lower right", title="quantile", title_fontsize=8)

    for i, row in enumerate(rows):
        q = pick(reports[row["scene"]], "residual", "patches", "psnrQuantiles") or {}
        if q.get("p10") is None:
            continue
        right.plot([q["p10"], q["p90"]], [i, i], color=SERIES[0], lw=2, solid_capstyle="round")
        right.scatter([q["p50"]], [i], s=40, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        if row["flat patches %"] is not None:
            right.annotate(f"{row['flat patches %']:.0f}% flat", (q["p90"], i), xytext=(6, 0),
                           textcoords="offset points", va="center", fontsize=7, color=INK2)
    right.axvline(residual.get("flatPatchPsnr", 45.0), color=INK2, lw=1, ls=DASHED)
    right.margins(x=0.15)
    right.set_xlabel("PSNR of a patch (dB)")
    right.set_title(f"Patch PSNR: p10 - p50 (dot) - p90  (dashed: 'flat' threshold)")
    right.grid(axis="y", visible=False)
    scene_axis(left, rows)
    save(fig, out / "03_error_distribution.png", "How the error is spread: loss weighting and patch sampling")

def per_frame(reports, rows, analysis, key, ylabel, ref, path: Path, title: str) -> None:
    cols = min(5, len(rows))
    nrows = math.ceil(len(rows) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(3.3 * cols, 2.1 * nrows + 0.6), sharey=True, squeeze=False)
    for ax, row in zip(axes.flat, rows):
        values = pick(reports[row["scene"]], analysis, key) or []
        ax.plot(values, color=SERIES[0], lw=1.2)
        if values:
            ax.axhline(float(np.mean(values)), color=INK2, lw=1, ls=DASHED)
        if ref is not None:
            ax.axhline(ref, color=MUTED, lw=1, ls=":")
        ax.set_title(row["label"], fontsize=8)
    for ax in axes.flat[len(rows):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    fig.supxlabel("frame (in processed order)", fontsize=8, color=INK2)
    save(fig, path, title)

CATEGORIES = {"static": SERIES[0], "slow": SERIES[1], "fast": SERIES[2]}

def difficulty_map(rows, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    placed = [r for r in rows if r["motion p95 px"] is not None and r["baseline PSNR dB"] is not None]
    for category, color in CATEGORIES.items():
        group = [r for r in placed if r["category"] == category]
        if group:
            ax.scatter([r["motion p95 px"] for r in group], [r["baseline PSNR dB"] for r in group], s=70,
                       color=color, edgecolor=SURFACE, linewidth=1.5, label=category, zorder=3)
    for r in placed:
        ax.annotate(r["label"], (r["motion p95 px"], r["baseline PSNR dB"]), xytext=(6, 4),
                    textcoords="offset points", fontsize=7, color=INK2)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("motion p95 (px / frame), symlog scale")
    ax.set_ylabel("baseline PSNR (dB), lower = harder")
    ax.legend(title="motion category", title_fontsize=8, loc="best")
    save(fig, out / "06_difficulty_map.png", "Difficulty map: spread every split across both axes")

def luminance(reports, rows, out: Path) -> None:
    cols = min(5, len(rows))
    nrows = math.ceil(len(rows) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(3.3 * cols, 2.1 * nrows + 0.8), sharex=True, squeeze=False)
    handles = []
    for ax, row in zip(axes.flat, rows):
        ax.set_title(row["label"], fontsize=8)
        hist = pick(reports[row["scene"]], "consistency", "luminanceHistogram")
        if not hist or not hist["input"]:
            continue
        edges = np.array(hist["edges"])
        centers = (edges[:-1] + edges[1:]) / 2
        for name, color, label in (("input", SERIES[0], "input"), ("gt", SERIES[1], "GT")):
            fractions = 100 * np.array(hist[name])  # [below 0, 32 bins, >= 1]
            line, = ax.step(centers, fractions[1:-1], where="mid", color=color, lw=1.2, label=label)
            if len(handles) < 2:
                handles.append(line)
        ax.text(0.98, 0.95, f">= 1.0: {100 * hist['input'][-1]:.1f}% / {100 * hist['gt'][-1]:.1f}%",
                transform=ax.transAxes, ha="right", va="top", fontsize=7, color=INK2)
    for ax in axes.flat[len(rows):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("% of pixels")
    fig.supxlabel("linear luminance (32 bins over [0, 1]); corner: share at >= 1.0, input / GT",
                  fontsize=8, color=INK2)
    if handles:
        fig.legend(handles, ["input", "GT"], loc="upper right", ncol=2)
    save(fig, out / "07_luminance.png", "Brightness distribution: input and GT should overlap")

# ---------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m analysis.plots", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", nargs="?", type=Path, help="run folder (default: newest run)")
    args = parser.parse_args(argv)

    run = args.run or latest_run()
    reports = load_run(run)
    if not reports:
        raise SystemExit(f"No reports in {run}")
    params = json.loads((run / "run.json").read_text("utf-8")).get("analyses", {})
    out = run / "plots"
    out.mkdir(exist_ok=True)

    rows = sorted((summary_row(scene, r) for scene, r in reports.items()),
                  key=lambda row: (row["baseline PSNR dB"] is None, row["baseline PSNR dB"] or 0))
    write_csv(rows, out / "summary.csv")
    overview(rows, params, out)
    error_location(reports, rows, out)
    error_distribution(reports, rows, params, out)
    per_frame(reports, rows, "residual", "psnrPerFrame", "PSNR (dB)", None, out / "04_psnr_per_frame.png",
              "Baseline PSNR per frame (dashed: scene mean): dips = frames to inspect")
    per_frame(reports, rows, "temporal", "meanMotionPxPerFrame", "px / frame",
              params.get("temporal", {}).get("fastMotionPx", 4.0), out / "05_motion_per_frame.png",
              "Mean motion per frame (dashed: scene mean, dotted: 'fast' threshold)")
    difficulty_map(rows, out)
    luminance(reports, rows, out)
    print(f"{len(rows)} scene(s): charts and summary.csv written to {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
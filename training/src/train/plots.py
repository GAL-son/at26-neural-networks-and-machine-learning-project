"""Learning curves of a run:  python -m train.plots [run folder]  (default: newest run). Writes <run>/plots/*.png."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common.plotting import INK2, MUTED, SERIES, save  # shared style (moved from analysis/plots.py)

from common.config import load_config

CONFIG = Path(__file__).resolve().parents[2] / "config" / "training.json"
TITLES = {"total": "Total (weighted)", "l1": "L1", "ssim": "1 - SSIM", "temporal": "Temporal"}


def ema(values, alpha=0.05):
    out, acc = [], None
    for v in values:
        acc = v if acc is None else (1 - alpha) * acc + alpha * v
        out.append(acc)
    return out

def load(run: Path):
    train, val = [], []
    for line in (run / "metrics.jsonl").read_text("utf-8").splitlines():
        record = json.loads(line)
        (train if record["kind"] == "train" else val).append(record)
    return train, val

def losses(train, val, out):
    keys = [k for k in TITLES if any(k in r for r in train)]
    with_loss = [r for r in val if all("loss" in m for m in r["scenes"].values())]  # older runs: train only
    fig, axes = plt.subplots(1, len(keys), figsize=(4.5 * len(keys), 3.5), squeeze=False)
    for ax, key in zip(axes[0], keys):
        steps = [r["step"] for r in train if key in r]
        values = [r[key] for r in train if key in r]
        ax.plot(steps, values, color=SERIES[0], alpha=0.25, lw=1)                       # raw
        ax.plot(steps, ema(values), color=SERIES[0], lw=2, label="train (hard crops)")  # smoothed
        if with_loss:
            ax.plot([r["step"] for r in with_loss],
                    [np.mean([m["loss"][key] for m in r["scenes"].values()]) for r in with_loss],
                    color=SERIES[1], lw=2, marker="o", ms=3, label="val (full frames, mean of scenes)")
        ax.set_yscale("log")
        ax.set_title(TITLES[key])
        ax.set_xlabel("step")
    axes[0][0].legend(loc="upper right")
    save(fig, out / "01_losses.png", "Training vs validation loss (faint: raw train, solid: smoothed)")

def validation(val, out):
    names = sorted(val[0]["scenes"])
    steps = [r["step"] for r in val]
    panels = [("psnrGain", "PSNR gain over baseline (dB)", 0.0),
              ("ssimGain", "SSIM gain over baseline", 0.0),
              ("flicker", "Output flicker ratio (target 1.0; dashed: input)", 1.0)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, (key, title, ref) in zip(axes, panels):
        for name, color in zip(names, SERIES):
            ax.plot(steps, [r["scenes"][name][key] for r in val], color=color, lw=2, marker="o", ms=3, label=name)
            if key == "flicker":
                ax.axhline(val[0]["scenes"][name]["flickerInput"], color=color, lw=1, ls=(0, (4, 3)))
        ax.axhline(ref, color=MUTED, lw=1, ls=":")
        ax.set_title(title)
        ax.set_xlabel("step")
    axes[0].legend(loc="lower right")
    for name in names:  # direct label of the latest value
        axes[0].annotate(f"{val[-1]['scenes'][name]['psnrGain']:+.2f}", (steps[-1], val[-1]["scenes"][name]["psnrGain"]),
                         xytext=(4, 0), textcoords="offset points", fontsize=7, color=INK2, va="center")
    save(fig, out / "02_validation.png", "Validation (full frames, val scenes)")

def health(train, out):
    steps = [r["step"] for r in train]
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.5))
    for ax, key, title in [(axes[0], "lr", "Learning rate"), (axes[1], "gradNorm", "Gradient norm (before clipping)"),
                           (axes[2], "dataWait", "Share of time waiting for data")]:
        values = [r[key] for r in train]
        ax.plot(steps, ema(values) if key != "lr" else values, color=SERIES[0], lw=2)
        ax.set_title(title)
        ax.set_xlabel("step")
    axes[2].set_ylim(0, 1)
    axes[2].axhline(0.1, color=MUTED, lw=1, ls=":")  # above ~10%: the GPU is starved, add workers / crops per frame
    save(fig, out / "03_health.png", f"Training health ({np.mean([r['cropsPerSec'] for r in train[-20:]]):.0f} crops/s recently)")

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m train.plots")
    parser.add_argument("run", nargs="?", type=Path, help="run folder")
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args(argv)

    if args.run:
        run = args.run
    else:
        runs_dir = load_config(args.config, {}, ("outputDir", ))["outputDir"]
        runs = sorted(p for p in runs_dir.glob("*") if (p / "metrics.jsonl").exists())

        if not runs:
            raise SystemExit(f"No runs with metrics.jsonl in {runs_dir}")
        run = runs[-1]
    
    train, val = load(run)
    out = run / "plots"
    out.mkdir(exist_ok=True)
    losses(train, val, out)
    if val:
        validation(val, out)
    health(train, out)
    print(f"plots written to {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())



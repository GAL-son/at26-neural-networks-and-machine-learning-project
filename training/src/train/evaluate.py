"""Evaluation of a trained checkpoint on whole sequences (default: the test split).

    python -m train.evaluate <run>/checkpoints/best.pt                  # test split, all frames
    python -m train.evaluate <checkpoint> --split val --frames 120      # quick look

Runs the network recurrently over every scene from frame 0 to the end, exactly like inference: frame 0 (and any
frame after a gap or an unreadable file) has no history, every other frame gets the previous output warped with
the motion vectors. Metrics, all in sRGB, on frames with history:
  psnr / ssim   output and aliased input against the SSAA ground truth; gain = output - input
  psnrHard      PSNR over the 10% of pixels with the largest input error (edges, thin detail)
  flicker       output change between frames / GT change over history-valid pixels (1.0 = as stable as the GT)
  gainQuarters  PSNR gain in each quarter of the sequence: a falling trend = errors building up over time
Writes <run>/eval/<checkpoint>-<split>/: summary.json, <scene>.csv per frame, psnr_over_time.png, preview tiles.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from common.config import load_dataset_config
from common.dataset import find_scenes, iter_frames
from common.device import describe_device, select_device
from common.plotting import SERIES, save

from .inputs import IN_CHANNELS, build_frame, history_valid, warp
from .losses import ssim
from .metrics import ValSet, psnr
from .model import UNet

HARD_FRACTION = 0.10  # share of pixels with the largest input error that count as "hard"

def load_model(path: Path, device):
    # weights_only=False: our own file, and the saved config contains Path objects
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = UNet(IN_CHANNELS, config["model"]["channels"]).to(device)
    if "ema" in ckpt:  # averaged weights; AveragedModel stores them under "module."
        state = {k.removeprefix("module."): v for k, v in ckpt["ema"].items() if k.startswith("module.")}
        source = "EMA"
    else:
        state, source = ckpt["model"], "raw"
    model.load_state_dict(state)
    model.eval()
    return model, config, ckpt.get("step"), source

def hard_psnr(out, inp, y):
    """PSNR of output and input over the HARD_FRACTION of pixels where the input is furthest from the GT."""
    err = (inp - y).abs().mean(1)  # (1, H, W)
    k = max(1, int(err.numel() * HARD_FRACTION))
    mask = (err >= err.flatten().topk(k).values[-1]).unsqueeze(1).expand_as(y)
    return psnr(out[mask], y[mask]), psnr(inp[mask], y[mask])

@torch.no_grad()
def evaluate_scene(model, name, frames, device, amp, depth_scale, limit, workers):
    indices = sorted(frames)[:limit] if limit else sorted(frames)
    tile_at = {indices[int(len(indices) * f)] for f in (0.1, 0.5, 0.9)}
    rows, errors, tiles = [], np.zeros(3), {}  # errors: change vs warped previous for output, input, GT
    prev = previous_frame = None
    for item in iter_frames(name, frames, indices, workers=workers):
        if isinstance(item, tuple):  # unreadable file: the history chain breaks here
            print(f"  skipped: {item[1]}")
            prev = previous_frame = None
            continue
        frame = item.to(device)
        if previous_frame is not None and frame.index != previous_frame.index + 1:
            prev = previous_frame = None  # missing frame(s): no valid history

        x, y, motion, _ = build_frame(frame, depth_scale)
        x, y, motion = x[None], y[None], motion[None]
        if prev is None:
            history, ok = torch.zeros_like(y), torch.zeros_like(y[:, :1])
        else:
            history, inside = warp(prev[0], motion)
            ok = (inside & history_valid(frame, previous_frame)[None]).float()
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            out = model(torch.cat([x, history, ok], dim=1)).float().clamp(0, 1)
        inp = x[:, :3]

        if prev is not None:
            hard_out, hard_in = hard_psnr(out, inp, y)
            rows.append({"frame": frame.index, "psnr": psnr(out, y), "psnrInput": psnr(inp, y),
                         "ssim": float(ssim(out, y)), "ssimInput": float(ssim(inp, y)),
                         "psnrHard": hard_out, "psnrHardInput": hard_in, "historyValid": float(ok.mean())})
            mask = ok[:, 0] > 0.5
            if mask.any():
                for i, (a, b) in enumerate([(out, prev[0]), (inp, prev[1]), (y, prev[2])]):
                    errors[i] += float((a - warp(b, motion)[0]).abs().mean(1)[mask].mean())
        if frame.index in tile_at:
            tiles[frame.index] = ValSet._tile(inp[0], out[0], y[0])
        prev, previous_frame = (out, inp, y), frame
    return rows, errors, tiles

def summarize(rows, errors):
    mean = lambda key: float(np.mean([r[key] for r in rows]))
    s = {key: mean(key) for key in ("psnr", "psnrInput", "ssim", "ssimInput", "psnrHard", "psnrHardInput")}
    gains = [r["psnr"] - r["psnrInput"] for r in rows]
    s.update(psnrGain=s["psnr"] - s["psnrInput"], ssimGain=s["ssim"] - s["ssimInput"],
             psnrHardGain=s["psnrHard"] - s["psnrHardInput"],
             flicker=errors[0] / errors[2], flickerInput=errors[1] / errors[2],
             gainQuarters=[float(np.mean(q)) for q in np.array_split(gains, 4)], frames=len(rows))
    return s

def plot(per_frame, out_dir, title):
    names = list(per_frame)
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 3.5), squeeze=False)
    for ax, name in zip(axes[0], names):
        rows = per_frame[name]
        f = [r["frame"] for r in rows]
        for key, label, color in (("psnr", "output", SERIES[0]), ("psnrInput", "input (aliased)", SERIES[1])):
            ax.plot(f, [r[key] for r in rows], color=color, lw=1.2, label=label)
        ax.set_title(name)
        ax.set_xlabel("frame")
        ax.set_ylabel("PSNR (dB)")
    axes[0][0].legend(loc="lower left")
    save(fig, out_dir / "psnr_over_time.png", title)

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m train.evaluate", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--scenes", nargs="+", help="only these capture folders (default: the whole split)")
    parser.add_argument("--frames", type=int, help="first N frames of each scene (default: all)")
    parser.add_argument("--workers", type=int, default=4, help="EXR decoding processes")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true", help="run the network in float32")
    args = parser.parse_args(argv)

    device = select_device(args.device)
    model, config, step, source = load_model(args.checkpoint, device)
    amp = not args.no_amp and device.type == "cuda"
    dataset = load_dataset_config(config["dataset"])
    scenes = find_scenes(dataset["capturesDir"])
    names = args.scenes or dataset["splits"][args.split]
    missing = [n for n in names if n not in scenes]
    if missing:
        raise SystemExit(f"no captures for {missing} in {dataset['capturesDir']}")

    out_dir = args.checkpoint.resolve().parent.parent / "eval" / f"{args.checkpoint.stem}-{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{describe_device(device)} | {args.checkpoint.name} (step {step}, {source} weights) | "
          f"{args.split}: {', '.join(names)}")

    summary, per_frame = {}, {}
    for name in names:
        tic = time.perf_counter()
        rows, errors, tiles = evaluate_scene(model, name, scenes[name], device, amp,
                                             config["inputs"]["depthScale"], args.frames, args.workers)
        if not rows:
            print(f"{name}: no frames with history, skipped")
            continue
        s = summary[name] = summarize(rows, errors)
        per_frame[name] = rows
        with (out_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        for index, tile in tiles.items():
            plt.imsave(out_dir / f"{name}_{index:06d}.png", tile.permute(1, 2, 0).numpy())
        print(f"{name}: PSNR {s['psnrInput']:.2f} -> {s['psnr']:.2f} dB ({s['psnrGain']:+.2f}), "
              f"hard pixels {s['psnrHardGain']:+.2f} dB, SSIM {s['ssimGain']:+.4f}, "
              f"flicker {s['flicker']:.2f} (input {s['flickerInput']:.2f}), "
              f"gain by quarter {' / '.join(f'{g:+.2f}' for g in s['gainQuarters'])} | "
              f"{s['frames']} frames in {time.perf_counter() - tic:.0f}s")

    keys = ("psnrGain", "psnrHardGain", "ssimGain", "flicker", "flickerInput")
    summary["mean"] = {k: float(np.mean([summary[n][k] for n in per_frame])) for k in keys}
    m = summary["mean"]
    print(f"mean: PSNR {m['psnrGain']:+.2f} dB, hard pixels {m['psnrHardGain']:+.2f} dB, "
          f"flicker {m['flicker']:.2f} (input {m['flickerInput']:.2f})")
    meta = {"checkpoint": str(args.checkpoint.resolve()), "step": step, "weights": source, "split": args.split}
    (out_dir / "summary.json").write_text(json.dumps({**meta, "scenes": summary}, indent=2), "utf-8")
    plot(per_frame, out_dir, f"PSNR over the sequence ({args.split}, step {step})")
    print(f"results in {out_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

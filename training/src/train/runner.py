"""Trains the anti-aliasing U-Net on streamed sequences of crops.

    python -m train                                   # config/training.json
    python -m train --steps 300 --val-every 100       # smoke test
    python -m train --resume runs/<ts>/checkpoints/last.pt

Each training sample is K consecutive frames cropped at one position. The network is unrolled over them exactly like
at inference: step 0 has no history, every later step gets its own previous output warped to the current frame.
Loss = L1 + ssim * (1 - SSIM) on every step (optionally without step 0) + temporal * temporal-stability term.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from common.config import load_config, load_dataset_config
from common.dataset import find_scenes
from common.device import describe_device, select_device

from . import losses
from .data import PatchStream
from .inputs import IN_CHANNELS, warp
from .metrics import ValSet
from .model import UNet

REPO_ROOT = Path(__file__).resolve().parents[2]

def unrolled_loss(model, batch, weights, amp, device):
    """Runs the network over the K frames of each sequence and returns (total loss, {term: value}).

    batch: x (B, K, 9, s, s), y (B, K, 3, s, s), motion (B, K, 2, s, s) in px, valid (B, K, 1, s, s) bool
    (surface visible in the previous frame)."""
    x, y = batch["x"].float(), batch["y"].float()
    motion, valid = batch["motion"].float(), batch["valid"]
    b, k = x.shape[:2]
    size = x.shape[-2:]

    # step 0: no history, like the first frame of a video
    history = x.new_zeros(b, 3, *size)
    history_ok = x.new_zeros(b, 1, *size)

    first = 0 if weights["firstStep"] else 1  # firstStep false = no loss on the frame without history
    terms = {"l1": 0.0, "ssim": 0.0, "temporal": 0.0}
    outs = []
    for s in range(k):
        if s:
            # previous output moved to this frame with the motion vectors; pixels whose source left the crop
            # or was hidden in the previous frame are marked invalid
            history, inside = warp(outs[-1].clamp(0, 1), motion[:, s])
            history_ok = (inside & valid[:, s]).float()
            if weights["detachHistory"]:
                history = history.detach()

        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            out = model(torch.cat([x[:, s], history, history_ok], dim=1))
        out = out.float()  # losses in float32
        outs.append(out)

        if s >= first:
            terms["l1"] = terms["l1"] + (out - y[:, s]).abs().mean() / (k - first)
            terms["ssim"] = terms["ssim"] + (1 - losses.ssim(out, y[:, s])) / (k - first)
        if s:
            terms["temporal"] = terms["temporal"] + losses.temporal(
                out, outs[-2], y[:, s], y[:, s - 1], motion[:, s], history_ok > 0.5) / (k - 1)

    total = sum(weights[name] * value for name, value in terms.items())
    return total, terms

def train(config: dict, resume: Path | None = None) -> Path:
    torch.manual_seed(config["seed"])
    device = select_device(config["device"])
    o, d, w = config["optim"], config["data"], config["loss"]
    amp = o["amp"] and device.type == "cuda"

    # --- data -------------------------------------------------------------------------------------------------
    dataset = load_dataset_config(config["dataset"])
    scenes = find_scenes(dataset["capturesDir"])
    pick = lambda split: {n: scenes[n] for n in dataset["splits"][split]}
    depth_scale = config["inputs"]["depthScale"]

    # --- model, optimiser, schedule (warm-up then cosine) -----------------------------------------------------
    model = UNet(IN_CHANNELS, config["model"]["channels"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=o["lr"], weight_decay=0.0)
    schedule = lambda s: min(1.0, (s + 1) / o["warmup"]) * 0.5 * (1 + math.cos(math.pi * min(s, o["steps"]) / o["steps"]))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, schedule)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    step, best = 0, -float("inf")

    # --- new run or resume ------------------------------------------------------------------------------------
    if resume:
        ckpt = torch.load(resume, map_location=device)
        for obj, key in [(model, "model"), (opt, "opt"), (sched, "sched"), (scaler, "scaler")]:
            obj.load_state_dict(ckpt[key])
        step, best, run = ckpt["step"], ckpt["best"], resume.parent.parent
    else:
        run = config["outputDir"] / datetime.now().strftime("%Y%m%d-%H%M%S")
        for sub in ("checkpoints", "images"):
            (run / sub).mkdir(parents=True)
        (run / "config.json").write_text(json.dumps(config, indent=2, default=str), "utf-8")

    loader = DataLoader(PatchStream(pick("train"), d, depth_scale, config["seed"] + step),
                        batch_size=o["batchSize"], num_workers=d["workers"], pin_memory=True,
                        persistent_workers=True, prefetch_factor=4)
    val = ValSet(pick("val"), config["val"], depth_scale)
    log = (run / "metrics.jsonl").open("a", encoding="utf-8")

    def write(record):
        log.write(json.dumps(record) + "\n")
        log.flush()

    def checkpoint(name):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "step": step, "best": best, "config": config},
                   run / "checkpoints" / name)

    def validate(images: bool):
        nonlocal best
        metrics, tiles = val.evaluate(model, device, amp, previews=images)
        write({"kind": "val", "step": step, "scenes": metrics})
        for name, tile in tiles.items():
            plt.imsave(run / "images" / f"{step:07d}_{name}.png", tile.permute(1, 2, 0).numpy())
        gain = sum(m["psnrGain"] for m in metrics.values()) / len(metrics)
        print(f"\n  val PSNR gain {gain:+.2f} dB | " + ", ".join(
            f"{n}: {m['psnrGain']:+.2f} dB, flicker {m['flicker']:.2f} (input {m['flickerInput']:.2f})"
            for n, m in metrics.items()))
        if step:
            checkpoint("last.pt")
            if gain > best:
                best = gain
                checkpoint("best.pt")

    params = sum(p.numel() for p in model.parameters())
    print(f"{describe_device(device)} | {params / 1e6:.2f}M parameters | run: {run}")

    # step 0: the output conv is zero, so output == input; psnrBaseline and flickerInput must match the analysis
    if step == 0:
        validate(images=True)

    # --- training loop ----------------------------------------------------------------------------------------
    batches, sums, tic, wait = iter(loader), {}, time.perf_counter(), 0.0
    model.train()
    while step < o["steps"]:
        t = time.perf_counter()
        batch = {k: v.to(device, non_blocking=True) for k, v in next(batches).items()}
        wait += time.perf_counter() - t  # time the GPU waited for data

        total, terms = unrolled_loss(model, batch, w, amp, device)

        opt.zero_grad(set_to_none=True)
        scaler.scale(total).backward()
        scaler.unscale_(opt)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), o["gradClip"]))
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1

        for k, v in {**terms, "total": total}.items():
            sums[k] = sums.get(k, 0.0) + float(v)

        if step % config["log"]["every"] == 0:
            n, elapsed = config["log"]["every"], time.perf_counter() - tic
            record = {"kind": "train", "step": step, "lr": sched.get_last_lr()[0], "gradNorm": grad_norm,
                      "cropsPerSec": n * o["batchSize"] / elapsed, "dataWait": wait / elapsed,
                      **{k: v / n for k, v in sums.items()}}
            write(record)
            print(f"\rstep {step}: loss {record['total']:.5f} (l1 {record['l1']:.5f}, ssim {record['ssim']:.4f}, "
                  f"temporal {record['temporal']:.5f}), {record['cropsPerSec']:.0f} seq/s, "
                  f"data wait {100 * record['dataWait']:.0f}%", end="", flush=True)
            sums, tic, wait = {}, time.perf_counter(), 0.0

        if step % config["val"]["every"] == 0 or step == o["steps"]:
            validate(images=step % config["log"]["imagesEvery"] == 0)

    log.close()
    return run

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m train", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config" / "training.json")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--val-every", type=int)
    args = parser.parse_args(argv)
    config = load_config(args.config, {}, ("dataset", "outputDir"))
    if args.steps:
        config["optim"]["steps"] = args.steps
    if args.val_every:
        config["val"]["every"] = args.val_every
    train(config, args.resume)
    return 0




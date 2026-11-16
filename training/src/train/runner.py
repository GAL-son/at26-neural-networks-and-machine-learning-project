"""Trains the anti-aliasing U-Net on streamed crops.

    python -m train                                   # config/training.json
    python -m train --steps 300 --val-every 100       # smoke test
    python -m train --resume runs/<ts>/checkpoints/last.pt
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

def train(config: dict, resume: Path | None = None) -> Path:
    torch.manual_seed(config["seed"])
    device = select_device(config["device"])
    o, d, w = config["optim"], config["data"], config["loss"]
    amp = o["amp"] and device.type == "cuda"

    dataset = load_dataset_config(config["dataset"])
    scenes = find_scenes(dataset["capturesDir"])
    pick = lambda split: {n: scenes[n] for n in dataset["splits"][split]}
    depth_scale = config["inputs"]["depthScale"]

    model = UNet(IN_CHANNELS, config["model"]["channels"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=o["lr"], weight_decay=0.0)
    schedule = lambda s: min(1.0, (s + 1) / o["warmup"]) * 0.5 * (1 + math.cos(math.pi * min(s, o["steps"]) / o["steps"]))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, schedule)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    step, best = 0, -float("inf")

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
    write = lambda record: (log.write(json.dumps(record) + "\n"), log.flush())
    params = sum(p.numel() for p in model.parameters())
    print(f"{describe_device(device)} | {params / 1e6:.2f}M parameters | run: {run}")

    def checkpoint(name):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "step": step, "best": best, "config": config},
                   run / "checkpoints" / name)

    if step == 0:  # zero-initialised output: this must equal the analysis baselines
        write({"kind": "val", "step": 0, "scenes": val.evaluate(model, device, amp)[0]})

    batches, sums, tic, wait = iter(loader), {}, time.perf_counter(), 0.0
    model.train()
    while step < o["steps"]:
        t = time.perf_counter()
        batch = {k: v.to(device, non_blocking=True).float() for k, v in next(batches).items()}
        wait += time.perf_counter() - t

        x, y = batch["x"], batch["y"]
        pairs = "x_prev" in batch
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            out = model(torch.cat([x, batch["x_prev"]]) if pairs else x)
        out = out.float()
        out, out_prev = out.chunk(2) if pairs else (out, None)

        terms = {"l1": (out - y).abs().mean()}
        if w["ssim"]:
            terms["ssim"] = 1 - losses.ssim(out, y)
        if w["temporal"] and pairs:
            terms["temporal"] = losses.temporal(out, out_prev, y, batch["y_prev"], batch["motion"], x[:, MASK:MASK + 1] > 0.5)
        total = sum(w[k] * v for k, v in terms.items())

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
            print(f"\rstep {step}: loss {record['total']:.5f}, {record['cropsPerSec']:.0f} crops/s, "
                  f"data wait {100 * record['dataWait']:.0f}%", end="", flush=True)
            sums, tic, wait = {}, time.perf_counter(), 0.0

        if step % config["val"]["every"] == 0 or step == o["steps"]:
            scenes_metrics = val.evaluate(model, device, amp)
            write({"kind": "val", "step": step, "scenes": scenes_metrics})
            gain = sum(m["psnrGain"] for m in scenes_metrics.values()) / len(scenes_metrics)
            print(f"\n  val PSNR gain {gain:+.2f} dB | " + ", ".join(
                f"{n}: {m['psnrGain']:+.2f} dB, flicker {m['flicker']:.2f}" for n, m in scenes_metrics.items()))
            checkpoint("last.pt")
            if gain > best:
                best = gain
                checkpoint("best.pt")
        if step % config["log"]["imagesEvery"] == 0:
            for name, tile in val.previews(model, device).items():
                plt.imsave(run / "images" / f"{step:07d}_{name}.png", tile.permute(1, 2, 0).numpy())
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




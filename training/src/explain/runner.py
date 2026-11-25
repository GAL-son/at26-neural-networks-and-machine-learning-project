"""Explanations of the trained anti-aliasing U-Net (XAI part of the project).

    python -m explain                                   # all cases of config/explain.json
    python -m explain --cases failure-living-room-dip   # selected cases
    python -m explain --no-checks                       # maps only (faster)

For every case: runs the network recurrently up to the frame (exactly like inference), explains the scalar S of the
region R with Grad-CAM and HiResCAM on the configured layers, projects the layer activations on 3 principal components,
attributes S to the input channel groups by ablation and checks the maps (deletion, randomisation, stability).
Writes <outputDir>/<timestamp>/: params.json (config, checkpoint, library versions), summary.json and one folder per
case (input / output / GT / map / overlay PNGs, figures, maps.npz with the raw maps, summary.json).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import torch

from common.config import load_config, load_dataset_config
from common.dataset import find_scenes
from common.device import describe_device, select_device
from train.evaluate import load_model
from train.metrics import psnr

from . import figures
from .ablation import ablation
from .case import crop, region_box, run_to_frame, select_region, window_around
from .checks import cascading_randomisation, deletion, stability
from .features import pca_rgb
from .gradcam import VARIANTS, GradCAM, target

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = {"device": "auto", "seed": 0, "warmup": None, "workers": 4, "window": 384,
            "layers": ["enc1", "enc2", "enc3", "enc4", "dec3", "dec2", "dec1"], "checkLayer": "dec2",
            "deletion": {"block": 8, "steps": 20, "randomOrders": 5, "sigma": 2.0},
            "stability": {"noise": 0.002, "repeats": 3}}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def versions(device) -> dict:
    import OpenEXR
    return {"python": platform.python_version(), "torch": torch.__version__, "hip": torch.version.hip,
            "numpy": np.__version__, "matplotlib": matplotlib.__version__,
            "openexr": getattr(OpenEXR, "__version__", None), "device": describe_device(device),
            "platform": platform.platform()}

def map_stats(m, region, aliased):
    """Map (window) statistics: its maximum (the normalisation of the figures) and the share of its mass inside R and
    on aliased pixels (10% largest input error), to compare with the share of area they cover."""

    total = float(m.sum())
    share = lambda sel: float(m[sel].sum()) / total if total > 0 else 0.0
    return {"max": float(m.max()), "massInRegion": share(region), "regionArea": float(region.float().mean()),
            "massOnAliased": share(aliased), "aliasedArea": float(aliased.float().mean())}

def explain_case(model, c, scenes, device, depth_scale, cfg, run, with_checks):
    tic = time.perf_counter()
    torch.manual_seed(cfg["seed"])

    x, y, out = run_to_frame(model, c["scene"], scenes[c["scene"]], c["frame"], device, depth_scale, cfg["warmup"], cfg["workers"])
    mask = select_region(c["region"], x, y, out, c.get("size", 64))
    window = window_around(mask, cfg["window"])
    score = target(c["target"], y, mask)
    layers = cfg["layers"]

    maps, s, _, acts = GradCAM(model, layers)(x, score)
    pcas = {layer: pca_rgb(acts[layer], window, x.shape[-2:]) for layer in layers}
    del acts

    abl, abl_maps = ablation(model, x, y, out, score, window)

    err = crop((x[:, :3] - y).abs().mean(1), window)[0]
    aliased = err >= err.flatten().topk(max(1, err.numel() // 10)).values[-1]
    region = crop(mask[0, 0], window)
    stats = {layer: {v: map_stats(crop(maps[layer][v], window), region, aliased) for v in VARIANTS} for layer in layers}

    checks = None
    if with_checks:
        ref, d, st = maps[cfg["checkLayer"]], cfg["deletion"], cfg["stability"]
        checks = {"deletion": deletion(model, x, score, {v: ref[v] for v in VARIANTS}, window,
                                       d["block"], d["steps"], d["randomOrders"], d["sigma"]),
                  "randomisation": cascading_randomisation(model, x, score, cfg["checkLayer"], window, ref),
                  "stability": stability(model, x, score, cfg["checkLayer"], window, ref, st["noise"], st["repeats"])}

    o = out.clamp(0, 1)
    quality = {"frameInput": psnr(x[:, :3], y), "frameOutput": psnr(o, y),
               "windowInput": psnr(crop(x[:, :3], window), crop(y, window)),
               "windowOutput": psnr(crop(o, window), crop(y, window))}

    folder = run / c["name"]
    folder.mkdir()
    head = f"{c['name']}: {c['scene']} frame {c['frame']}, S = {c['target']} = {s:.4g}"
    figures.save_images(folder, x, y, out, window, maps, layers)
    figures.overview(folder / "overview.png", head, x, y, out, mask, window, maps, layers)
    figures.pca_figure(folder / "pca.png", f"{head} | activations, 3 principal components as RGB", pcas, layers)
    figures.ablation_figure(folder / "ablation.png", f"{head} | input channel groups ablated", abl, abl_maps)
    if checks:
        figures.deletion_figure(folder / "deletion.png", f"{head} | deletion, {cfg['checkLayer']}", checks["deletion"])
        figures.randomisation_figure(folder / "randomisation.png", f"{head} | randomisation, {cfg['checkLayer']}",
                                     checks["randomisation"], checks["stability"])
    np.savez_compressed(folder / "maps.npz", window=np.array(window),
                        **{f"{v}_{layer}": crop(maps[layer][v], window).cpu().numpy() for layer in layers for v in VARIANTS})

    summary = {"case": c, "regionBox": region_box(mask), "window": list(window), "regionPixels": int(mask.sum()),
               "S": s, "psnr": quality, "historyValidWindow": float(crop(x[:, 12], window).mean()), "maps": stats,
               "pcaExplainedVariance": {layer: pcas[layer][1] for layer in layers}, "ablation": abl,
               "checks": checks, "seconds": time.perf_counter() - tic}
    (folder / "summary.json").write_text(json.dumps(summary, indent=2, default=str), "utf-8")
    return summary

def describe(name, r, layer):
    p, m = r["psnr"], r["maps"][layer]
    s = max(r["ablation"]["S"], 1e-12)
    line = (f"{name}: S {r['S']:.4g} | window PSNR {p['windowInput']:.2f} -> {p['windowOutput']:.2f} dB, "
            f"history valid {r['historyValidWindow']:.0%} | {layer} mass on aliased pixels: Grad-CAM "
            f"{m['gradcam']['massOnAliased']:.0%}, HiResCAM {m['hirescam']['massOnAliased']:.0%} "
            f"(area {m['gradcam']['aliasedArea']:.0%}) | ablation dS: " +
            ", ".join(f"{k.split('/')[0]} {v['dS'] / s:+.0%}" for k, v in r["ablation"]["groups"].items()
                      if k.endswith("/neutral") or k.startswith("colour")))
    if r["checks"]:
        a, rnd = r["checks"]["deletion"]["aopc"], r["checks"]["randomisation"]
        line += (f" | AOPC Grad-CAM {a['gradcam']:.3f}, HiResCAM {a['hirescam']:.3f}, random {a['random']:.3f}"
                 f" | rho after +out: {rnd['gradcam'][0]:.2f} / {rnd['hirescam'][0]:.2f}")
    return line + f" | {r['seconds']:.0f}s"

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m explain", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config" / "explain.json")
    parser.add_argument("--cases", nargs="+", help="only these case names")
    parser.add_argument("--no-checks", action="store_true", help="skip deletion / randomisation / stability")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, DEFAULTS, ("checkpoint", "outputDir"))
    if cfg["checkLayer"] not in cfg["layers"]:
        raise SystemExit(f"checkLayer '{cfg['checkLayer']}' must be one of layers {cfg['layers']}")
    device = select_device(cfg["device"])
    model, train_cfg, step, source = load_model(cfg["checkpoint"], device)
    model.requires_grad_(False)
    scenes = find_scenes(load_dataset_config(train_cfg["dataset"])["capturesDir"])
    cases = [c for c in cfg["cases"] if not args.cases or c["name"] in args.cases]
    missing = sorted({c["scene"] for c in cases} - set(scenes))
    if missing:
        raise SystemExit(f"no captures for {missing}")

    run = cfg["outputDir"] / datetime.now().strftime("%Y%m%d-%H%M%S")
    run.mkdir(parents=True)
    params = {"config": cfg, "versions": versions(device),
              "checkpoint": {"path": str(cfg["checkpoint"]), "step": step, "weights": source,
                             "sha256": sha256(cfg["checkpoint"])[:16]},
              "model": {"channels": train_cfg["model"]["channels"], "depthScale": train_cfg["inputs"]["depthScale"]}}
    (run / "params.json").write_text(json.dumps(params, indent=2, default=str), "utf-8")
    print(f"{describe_device(device)} | {Path(cfg['checkpoint']).name} (step {step}, {source}) | "
          f"{len(cases)} case(s) -> {run}")

    results = {}
    for c in cases:
        results[c["name"]] = explain_case(model, c, scenes, device, train_cfg["inputs"]["depthScale"], cfg, run,
                                          not args.no_checks)
        print(describe(c["name"], results[c["name"]], cfg["checkLayer"]))
    (run / "summary.json").write_text(json.dumps(results, indent=2, default=str), "utf-8")
    return 0
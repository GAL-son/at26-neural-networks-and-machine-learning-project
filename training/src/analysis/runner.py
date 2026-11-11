"""Streams the selected scenes through the enabled analyses and writes <outputDir>/<timestamp>/<analysis>.json.

    python -m analysis                                    # everything in config/analysis.json
    python -m analysis --scenes sponza_sponza --limit 30  # quick check of one scene
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch

from common.config import SPLITS, load_config, load_dataset_config
from common.dataset import find_scenes, iter_frames, missing_frames, sample_indices
from common.device import describe_device, select_device

from .consistency import ConsistencyAnalysis
from .integrity import IntegrityAnalysis
from .residual import ResidualAnalysis
from .temporal import TemporalAnalysis

REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSES = {cls.name: cls for cls in [IntegrityAnalysis, ConsistencyAnalysis, ResidualAnalysis, TemporalAnalysis]}
DEFAULTS = {"dataset": "train.json", "scenes": "all", "stride": 1, "clip": 1, "limit": None, "device": "auto",
            "workers": 4, "prefetch": 2, "outputDir": "../../reports/analysis", "analyses": {}}

def run(config: dict) -> Path:
    dataset = load_dataset_config(config["dataset"])
    scenes = find_scenes(dataset["capturesDir"])
    selection = config["scenes"]
    names = list(scenes) if selection == "all" else dataset["splits"][selection] if isinstance(selection, str) \
        else selection
    if not names or set(names) - set(scenes):
        raise ValueError(f"Scenes not found in {dataset['capturesDir']}: {sorted(set(names) - set(scenes)) or names}")

    device = select_device(config["device"])
    analyses = [ANALYSES[name](params, device) for name, params in config["analyses"].items()
                if params.get("enabled", True)]
    out_dir = config["outputDir"] / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True)
    print(f"Device: {describe_device(device)} | {len(names)} scene(s) | output: {out_dir}")

    results = {analysis.name: {} for analysis in analyses}
    for name in names:
        frames = scenes[name]
        indices = sample_indices(frames, config["stride"], config["clip"], config["limit"])
        split = next((split for split, members in dataset["splits"].items() if name in members), None)
        scene = {"scene": name, "split": split, "framesOnDisk": len(frames),
                 "missingFrames": missing_frames(frames, dataset["expectedFrames"])}
        for analysis in analyses:
            analysis.begin_scene(scene)

        previous, start = None, time.perf_counter()
        with torch.no_grad():
            for done, item in enumerate(iter_frames(name, frames, indices, config["workers"], config["prefetch"]), 1):
                if isinstance(item, tuple):  # unreadable file: (index, error)
                    for analysis in analyses:
                        analysis.skip(*item)
                    previous = None
                    continue
                frame = item.to(device)
                if previous is not None and previous.index != frame.index - 1:
                    previous = None
                for analysis in analyses:
                    analysis.update(frame, previous)
                previous = frame
                fps = done / (time.perf_counter() - start)
                print(f"\r  {name}: {done}/{len(indices)} frames ({fps:.1f} fps)", end="", flush=True)
        print()

        for analysis in analyses:
            result = results[analysis.name][name] = analysis.end_scene()
            print(f"    [{analysis.name}] {analysis.describe(result)}")

    for analysis in analyses:
        (out_dir / f"{analysis.name}.json").write_text(json.dumps(results[analysis.name], indent=2), "utf-8")
    (out_dir / "run.json").write_text(json.dumps({**config, "device": describe_device(device)}, indent=2,
                                                 default=str), "utf-8")
    print(f"Reports written to {out_dir}")
    return out_dir

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m analysis", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config" / "analysis.json")
    parser.add_argument("--scenes", nargs="+", help=f"'all', one of {list(SPLITS)} or capture folder names")
    parser.add_argument("--limit", type=int, help="max frames per scene")
    args = parser.parse_args(argv)


    config = load_config(args.config, DEFAULTS, ("dataset", "outputDir"))
    if args.scenes:
        single = len(args.scenes) == 1 and args.scenes[0] in ("all", *SPLITS)
        config["scenes"] = args.scenes[0] if single else args.scenes
    if args.limit:
        config["limit"] = args.limit
    run(config)
    return 0



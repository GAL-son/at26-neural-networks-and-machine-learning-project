"""Cut training and validation patches out of the captured EXR sequences.

Reads training/config/data.json, checks that the train/val/test scenes are disjoint, then for every
train and val scene picks frames, derives the network channels, chooses patch positions (error-weighted
plus uniform) and writes one memory-mappable array per scene:

    data/patches/meta.json             channel layout, settings, patch counts
    data/patches/<split>/<scene>.npy   float16 (N, C, S, S)
    data/patches/<split>/<scene>.csv   patch index: frame, position, error energy, sampling kind

Test scenes are only checked here; they are evaluated on full frames later.

    python training/data/build_patches.py --dry-run     # counts and disk estimate only
    python training/data/build_patches.py --workers 6
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training" / "analysis"))  # EXR loading, masks and warping are shared with the analysis

from common import find_sequences, geometric_edges, load_frame, regions, srgb, warp_previous  # noqa: E402

DEFAULT_CONFIG = ROOT / "training" / "config" / "data.json"

BASE_CHANNELS = [("albedo", 3), ("normal", 3), ("log_depth", 1), ("motion", 2), ("edge", 1)]
TEMPORAL_CHANNELS = [("prev_albedo", 3), ("prev_log_depth", 1), ("prev_valid", 1)]
TARGET_CHANNELS = [("target", 3)]

CONVENTIONS = {
    "albedo, target, prev_albedo": "sRGB, 0-1",
    "normal": "view space: x right, y down (image), z towards the camera; zero on background",
    "log_depth, prev_log_depth": "natural log of linear view depth in scene units",
    "motion": "pixels, current - previous, x right, y down",
    "edge": "1 within 1.5 px of a geometric edge",
    "prev_*": "previous frame warped to the current one with the motion vectors (bilinear)",
    "prev_valid": "1 where the warp source lies inside the previous frame",
}


def channel_layout(temporal: bool) -> dict[str, list[int]]:
    layout, start = {}, 0
    for name, count in BASE_CHANNELS + (TEMPORAL_CHANNELS if temporal else []) + TARGET_CHANNELS:
        layout[name] = [start, start + count]
        start += count
    return layout


def load_config(path: Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    owner = {}
    for split, spec in config["splits"].items():
        for scene in spec["scenes"]:
            if scene in owner:
                raise SystemExit(f"Scene '{scene}' is in both '{owner[scene]}' and '{split}': splits must be disjoint")
            owner[scene] = split
    return config


# --------------------------------------------------------------------------- per-frame channels

def camera_axes(header: dict, transpose: bool) -> np.ndarray:
    """Rows: camera right, image down, towards camera - in world space.

    viewProjection = P * V with a Vulkan perspective (y flipped), so its rows are
    row0 ~ V.right, row1 ~ -V.up, row3 = -V.back; scale factors vanish after normalising.
    """
    m = np.asarray(header["viewProjection"], dtype=np.float64).reshape(4, 4)
    m = m.T if transpose else m
    right = m[0, :3] / np.linalg.norm(m[0, :3])
    down = m[1, :3] / np.linalg.norm(m[1, :3])
    back = -m[3, :3] / np.linalg.norm(m[3, :3])
    return np.stack([right, down, back])


def view_normals(frame, transpose: bool) -> np.ndarray:
    return (frame.normal @ camera_axes(frame.header, transpose).T).astype(np.float32)


def detect_matrix_layout(frame) -> tuple[bool, float]:
    """The EXR stores a glm matrix through Imath; pick the reading under which visible surfaces face the camera."""
    foreground = ~frame.background
    scores = {}
    for transpose in (True, False):
        normals = view_normals(frame, transpose)[foreground]
        scores[transpose] = float((normals[:, 2] > 0).mean()) if len(normals) else 0.0
    best = max(scores, key=scores.get)
    return best, scores[best]


def frame_channels(cur, prev, transpose: bool) -> tuple[np.ndarray, np.ndarray]:
    """Full-frame (C, H, W) stack in channel_layout order, and the per-pixel error energy."""
    albedo, target = srgb(cur.albedo), srgb(cur.gt)
    _, masks = regions(cur, geometric_edges(cur))
    parts = [albedo, view_normals(cur, transpose), np.log(np.maximum(cur.depth, 1e-6))[..., None],
             cur.motion_px, masks["edge"][..., None]]
    if prev is not None:
        motion = cur.motion_px
        prev_albedo = srgb(prev.albedo)
        warped_albedo = np.stack([warp_previous(prev_albedo[..., c], motion)[0] for c in range(3)], axis=-1)
        warped_depth, valid = warp_previous(np.log(np.maximum(prev.depth, 1e-6)), motion)
        parts += [warped_albedo, warped_depth[..., None], valid[..., None]]
    parts.append(target)
    stack = np.concatenate([p.astype(np.float32) for p in parts], axis=-1).transpose(2, 0, 1)
    energy = ((target - albedo) ** 2).sum(-1)
    return stack, energy


def choose_positions(energy, size, count, uniform_share, grid, rng) -> list[tuple[int, int, float, str]]:
    """Patch corners on a grid: (1 - uniform_share) drawn with p ~ sqrt(error), the rest uniformly."""
    h, w = energy.shape
    y, x = (a.ravel() for a in np.meshgrid(np.arange(0, h - size + 1, grid), np.arange(0, w - size + 1, grid), indexing="ij"))
    sat = np.pad(energy.astype(np.float64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    crop = (sat[y + size, x + size] - sat[y, x + size] - sat[y + size, x] + sat[y, x]) / size ** 2

    n_uniform = int(round(count * uniform_share))
    n_weighted = count - n_uniform
    weights = np.sqrt(crop + 0.01 * crop.mean() + 1e-12)
    weighted = rng.choice(len(y), n_weighted, replace=False, p=weights / weights.sum())
    uniform = rng.choice(np.setdiff1d(np.arange(len(y)), weighted), n_uniform, replace=False)
    kinds = ["weighted"] * n_weighted + ["uniform"] * n_uniform
    return [(int(y[i]), int(x[i]), float(crop[i]), kind) for i, kind in zip(np.concatenate([weighted, uniform]), kinds)]


def process_frame(job: dict):
    """Worker: one frame -> (slot, patches float16 (K, C, S, S), index rows)."""
    cur = load_frame(job["cur"])
    prev = load_frame(job["prev"]) if job["temporal"] else None
    if prev is not None and cur.index - prev.index != 1:
        raise ValueError(f"{job['cur']}: previous file is frame {prev.index}, expected {cur.index - 1}")
    stack, energy = frame_channels(cur, prev, job["transpose"])
    rng = np.random.default_rng([job["seed"], job["scene_id"], cur.index])
    positions = choose_positions(energy, job["size"], job["count"], job["uniform_share"], job["grid"], rng)
    s = job["size"]
    patches = np.stack([stack[:, y:y + s, x:x + s] for y, x, _, _ in positions]).astype(np.float16)
    rows = [{"frame": cur.index, "y": y, "x": x, "energy": f"{e:.6g}", "kind": kind} for y, x, e, kind in positions]
    return job["slot"], patches, rows


# --------------------------------------------------------------------------- per-scene build

def plan_frames(paths: list[Path], stride: int) -> list[int]:
    return list(range(1, len(paths), stride))  # frame 0 has no previous frame


def candidate_count(height: int, width: int, size: int, grid: int) -> int:
    return len(range(0, height - size + 1, grid)) * len(range(0, width - size + 1, grid))


def build_scene(scene, scene_id, paths, split, spec, config, out, workers) -> dict:
    size, temporal = config["patch_size"], config["temporal"]
    channels = channel_layout(temporal)["target"][1]
    frames = plan_frames(paths, spec["frame_stride"])
    count = spec["patches_per_frame"]

    first = load_frame(paths[frames[0]])
    grid = config["sampling"]["grid_stride"]
    if count > candidate_count(first.height, first.width, size, grid):
        raise SystemExit(f"{scene}: {count} patches per frame do not fit on a {grid}px grid at {first.width}x{first.height}")
    transpose, facing = detect_matrix_layout(first)
    if facing < 0.7:
        print(f"  WARNING {scene}: only {facing:.0%} of normals face the camera - check the viewProjection reading")

    array = np.lib.format.open_memmap(out / split / f"{scene}.npy", mode="w+", dtype=np.float16,
                                      shape=(len(frames) * count, channels, size, size))
    jobs = [{"prev": paths[t - 1], "cur": paths[t], "slot": slot, "transpose": transpose, "temporal": temporal,
             "seed": config["seed"], "scene_id": scene_id, "size": size, "count": count, "grid": grid,
             "uniform_share": config["sampling"]["uniform_share"]} for slot, t in enumerate(frames)]
    rows = [None] * len(frames)

    def store(result, done):
        slot, patches, frame_rows = result
        array[slot * count:(slot + 1) * count] = patches
        rows[slot] = frame_rows
        print(f"\r  [{split}/{scene}] {done}/{len(frames)} frames", end="", flush=True)

    if workers <= 1:  # inline: easier to debug, full tracebacks
        for done, job in enumerate(jobs, 1):
            store(process_frame(job), done)
    else:
        with ProcessPoolExecutor(workers) as pool:
            futures = [pool.submit(process_frame, job) for job in jobs]
            for done, future in enumerate(as_completed(futures), 1):
                store(future.result(), done)
    print()
    array.flush()
    del array

    with open(out / split / f"{scene}.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["patch", "frame", "y", "x", "energy", "kind"])
        writer.writeheader()
        for patch, row in enumerate(r for frame_rows in rows for r in frame_rows):
            writer.writerow({"patch": patch, **row})

    return {"frames": len(frames), "patches": len(frames) * count, "matrix_transposed": transpose,
            "normals_facing_camera": round(facing, 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="print counts and disk estimate, write nothing")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--overwrite", action="store_true", help="replace an existing patch set")
    args = parser.parse_args()

    config = load_config(args.config)
    captures, out = ROOT / config["captures"], ROOT / config["patches"]
    sequences = find_sequences(captures)
    splits = config["splits"]

    missing = [scene for spec in splits.values() for scene in spec["scenes"] if scene not in sequences]
    if missing:
        raise SystemExit(f"Scenes not found in {captures}: {missing}")
    assigned = [scene for spec in splits.values() for scene in spec["scenes"]]
    unused = sorted(set(sequences) - set(assigned))
    if unused:
        print(f"Not assigned to any split (ignored): {unused}")

    size = config["patch_size"]
    layout = channel_layout(config["temporal"])
    patch_bytes = layout["target"][1] * size * size * 2
    print(f"{layout['target'][1]} channels x {size}^2 x float16 = {patch_bytes / 2 ** 20:.2f} MB per patch")
    total = 0
    for split in ("train", "val"):
        spec = splits[split]
        for scene in spec["scenes"]:
            n = len(plan_frames(sequences[scene], spec["frame_stride"])) * spec["patches_per_frame"]
            total += n
            print(f"  {split:<5} {scene:<28} {n:>6} patches  {n * patch_bytes / 2 ** 30:6.1f} GB")
    print(f"  test  {', '.join(splits['test']['scenes'])} (full frames, no patches)")
    print(f"Total {total} patches, {total * patch_bytes / 2 ** 30:.1f} GB in {out}")
    if args.dry_run:
        return

    if (out / "meta.json").exists() and not args.overwrite:
        raise SystemExit(f"{out} already contains a patch set - use --overwrite to replace it")

    start = time.perf_counter()
    meta_splits = {"test": {"scenes": splits["test"]["scenes"]}}
    for split in ("train", "val"):
        (out / split).mkdir(parents=True, exist_ok=True)
        results = {}
        for scene in splits[split]["scenes"]:
            results[scene] = build_scene(scene, assigned.index(scene), sequences[scene], split, splits[split],
                                         config, out, args.workers)
        meta_splits[split] = {"scenes": splits[split]["scenes"],
                              "patches": {scene: r["patches"] for scene, r in results.items()},
                              "details": results}

    meta = {"created": datetime.now().isoformat(timespec="seconds"), "patch_size": size,
            "temporal": config["temporal"], "channels": layout, "conventions": CONVENTIONS,
            "config": config, "splits": meta_splits}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Done in {(time.perf_counter() - start) / 60:.1f} min -> {out / 'meta.json'}")


if __name__ == "__main__":
    main()




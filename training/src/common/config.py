"""JSON configs from config/. Relative paths are resolved against the config file's directory."""
from __future__ import annotations

import json
from pathlib import Path

SPLITS = ("train", "val", "test")

def load_config(path: str | Path, defaults: dict, path_keys: tuple[str, ...] = ()) -> dict:
    """JSON file merged over defaults; the values of path_keys become absolute Paths."""
    
    path = Path(path).resolve()
    config = {**defaults, **json.loads(path.read_text(encoding="utf-8"))}
    
    for key in path_keys:
        config[key] = (path.parent / config[key]).resolve()
    return config

def load_dataset_config(path: str | Path) -> dict:
    """config/dataset.json: capturesDir, expectedFrames, splits {train, val, test: [capture folder names]}."""
    
    config = load_config(path, {"capturesDir": "../data/captures", "expectedFrames": 600, "splits": {}}, ("capturesDir",))
    
    unknown = set(config["splits"]) - set(SPLITS)
    if unknown:
        raise ValueError(f"{path}: unknown split(s) {sorted(unknown)}")
    
    config["splits"] = {split: config["splits"].get(split, []) for split in SPLITS}
    scenes = [scene for names in config["splits"].values() for scene in names]
    
    if len(scenes) != len(set(scenes)):
        raise ValueError(f"{path}: a scene is listed in more than one split")
    return config





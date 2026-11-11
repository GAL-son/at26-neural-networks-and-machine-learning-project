## Setup
Python **3.12**: AMD's Windows PyTorch wheels are built for cp312 only.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
# 1. GPU build of torch first (otherwise pip installs the CPU build from PyPI):
#    AMD Radeon on Windows: follow "PyTorch via PIP installation" in AMD's ROCm Radeon docs
#    (wheels from repo.radeon.com, installed with pip -f ...). On Linux: the rocm index on pytorch.org.
pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ torch torchvision torchaudio

#    Fallback for any DX12 GPU: pip install torch-directml
# 2. The suite itself (editable, makes common / analysis / training importable):
pip install -e .
```

Check the device: `python -c "from common.device import *; print(describe_device(select_device()))"`.

## Loading frames
The format is described in [renderer/OUTPUT_FORMAT.md](../renderer/OUTPUT_FORMAT.md).

```python
from common.config import load_dataset_config
from common.dataset import find_scenes, iter_frames
from common.exr import load_frame

def main():
    dataset = load_dataset_config("config/dataset.json")
    scenes = find_scenes(dataset["capturesDir"])     # {"sponza_sponza": {0: Path, 1: Path, ...}, ...}

    frame = load_frame(scenes["sponza_sponza"][200]).to("cuda")
    frame.albedo, frame.normal, frame.motion, frame.depth, frame.gt   # (C, H, W) float32
    frame.geometry_mask()                                            # (1, H, W) bool
    frame.header["viewProjection"].T                                 # 4x4, clip = M @ [x, y, z, 1]

    frames = scenes["sponza_sponza"]
    for item in iter_frames("sponza_sponza", frames, sorted(frames), workers=4):
        ...                                          # Frame, or (index, error) for an nreadable file

if __name__ == "__main__":  # required on Windows when workers > 0: worker processes re-import this file
    main()
```

- `load_frame` returns CPU tensors in file precision (float16, depth float32). `Frame.to(device)` moves them to the device and converts them to float32.
- A sequence is ~32 GB decoded. Stream it with `iter_frames`; never collect a whole sequence in a list.

## Analysis
```powershell
python -m analysis                                       # everything in config/analysis.json
python -m analysis --scenes sponza_sponza --limit 30     # quick run on one scene
python -m analysis --scenes val                          # one split
```
Each run writes `reports/analysis/<timestamp>/`, with one `<analysis>.json` per analysis (per-scene results) and `run.json` (the config used).

An analysis is a class derived from `Analysis` (`base.py`) and registered in `ANALYSES` in `runner.py`.
It receives each frame once, already on the device, together with the previous frame when that is frame index - 1.
It accumulates statistics in `update()` and returns a JSON-able dict from `end_scene()`.

| Analysis | Plan item | Status |
|---|---|---|
| `integrity` | 1. NaN/Inf, ranges, frame contiguity, header consistency, motion vs. depth reprojection (also checks the matrix layout) | done |
| motion distribution | 2. Mean / p95 motion per scene, fast / slow / static categories | todo |
| aliasing severity | 2. High-frequency energy of aliased vs. GT | todo |
| discontinuities | 2. Depth / normal gradients | todo |
| temporal history | 3. Warped t-1 GT vs. t GT, occlusion error | todo |
| channel correlation | 3. Correlation / PCA of input channels on sampled patches | todo |
| patch inspection | 3. Side-by-side 256x256 crops | todo |
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
    dataset = load_dataset_config("config/train.json")
    scenes = find_scenes(dataset["capturesDir"])     # {"sponza_sponza": {0: Path, 1: Path, ...}, ...}


    frame = load_frame(scenes["sponza_sponza"][200]).to("cuda")
    frame.albedo, frame.normal, frame.motion, frame.depth, frame.gt   # (C, H, W) float32
    frame.geometry_mask()                                            # (1, H, W) bool
    frame.header["viewProjection"].T                                 # 4x4, clip = M @ [x, y, z, 1]


    frames = scenes["sponza_sponza"]
    for item in iter_frames("sponza_sponza", frames, sorted(frames), workers=4):
        ...                                          # Frame, or (index, error) for an unreadable file


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

| Analysis | What it answers | Plan item |
|---|---|---|
| `integrity` | Is every buffer valid? NaN/Inf, ranges, frame gaps, header consistency, motion vs. depth reprojection (also checks the matrix layout) | 1 |
| `consistency` | Is the GT the same picture as the input, only anti-aliased? Brightness ratio, block-average difference, sub-pixel shift, clipping at 1.0, luminance histograms | new |
| `residual` | How much is there to fix, and where? Baseline PSNR / SSIM, share of changed pixels, error at geometric edges vs. inside surfaces vs. background, per-patch error | 2 (aliasing severity, discontinuities) |
| `temporal` | How does the camera move and does history help? Motion mean / p95 and category, static frames, disocclusion, GT history error vs. input flicker | 2 (motion), 3 (temporal history) |
| channel correlation | Correlation / PCA of input channels on sampled patches | 3, todo |
| patch inspection | Side-by-side 256x256 crops | 3, todo |

Metrics in `residual` and `temporal` are computed on sRGB-encoded values (as displayed), so `2/255` means two display levels.
"Interior" error in `residual` is aliasing the G-buffer can't explain: texture shimmer (no mipmaps) and sub-pixel geometry.

### Reading the results
1. `integrity` and `consistency` must be `ok` before anything else is trusted. A GT / input brightness ratio far from 1 means the GT is shaded differently from the input.
2. `residual.baseline` is the score to beat. `regions` shows how much of the error the G-buffer can point at (`geometricEdges`).
   `patches.flatFraction` shows how many random patches would teach nothing.
3. `temporal.category` and `motionPx` help balance the splits. `history.flickerRatio` > 1 means the input shimmers over time more than the GT does.
   Scene groups in `train.json` keep related scenes in one split.



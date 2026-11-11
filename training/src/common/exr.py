"""Reading one renderer frame. The format is described in renderer/OUTPUT_FORMAT.md."""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import OpenEXR
import torch

WIDTH, HEIGHT = 1920, 1080
FRAME_FILE = re.compile(r"frame_(\d+)\.exr")
# Frame field -> EXR channels, in tensor channel order.
LAYERS = {
    "albedo": ("albedo.R", "albedo.G", "albedo.B"),
    "normal": ("normal.X", "normal.Y", "normal.Z"),
    "motion": ("motion.X", "motion.Y"),
    "depth": ("depth.Z",),
    "gt": ("ground-truth.R", "ground-truth.G", "ground-truth.B"),
}
HEADER_KEYS = ("frameIndex", "ssaaFactor", "nearPlane", "farPlane", "viewProjection", "prevViewProjection")

@dataclass
class Frame:
    """Layers are (C, H, W) tensors, row 0 = top of the image.
    From disk they are CPU tensors in file precision (float16, depth float32); call to() before computing."""

    scene: str
    index: int
    albedo: torch.Tensor  # (3, H, W) linear, aliased, 8-bit quantised
    normal: torch.Tensor  # (3, H, W) world space, unit length, (0, 0, 0) on background
    motion: torch.Tensor  # (2, H, W) uv_current - uv_previous, +x right, +y down
    depth: torch.Tensor   # (1, H, W) linear view-space Z, far plane on background
    gt: torch.Tensor      # (3, H, W) linear, SSAA 4x4 box filtered
    header: dict          # raw EXR header; matrices are 4x4 numpy arrays stored transposed (M = header[...].T)

    def to(self, device: torch.device | str, dtype: torch.dtype = torch.float32) -> Frame:
        # Transfer first, convert on the device: half the bytes over PCIe.
        return replace(self, **{name: getattr(self, name).to(device).to(dtype) for name in LAYERS})

    def geometry_mask(self) -> torch.Tensor:
        """(1, H, W) bool, True where geometry was hit. Background normals are exactly zero."""
        return self.normal.float().norm(dim=0, keepdim=True) > 0.5
    
def load_frame(path: str | Path, scene: str | None = None) -> Frame:
    """Read a whole frame. `scene` defaults to the folder name (e.g. "sponza_sponza")."""
    path = Path(path)
    with OpenEXR.File(str(path), separate_channels=True) as exr:
        raw = exr.header()
        channels = exr.channels()
        layers = {layer: torch.from_numpy(np.stack([channels[name].pixels for name in names]))
                  for layer, names in LAYERS.items()}

    missing = [key for key in HEADER_KEYS if key not in raw]
    if missing:
        raise KeyError(f"missing header attribute(s) {missing}")
    # Only the renderer's attributes: the rest (channel list, windows, compression) are OpenEXR objects
    # that can't be pickled, and DataLoader workers pickle every frame they return.
    header = {key: raw[key] for key in HEADER_KEYS}
    return Frame(scene or path.parent.name, int(FRAME_FILE.fullmatch(path.name).group(1)), header=header, **layers)





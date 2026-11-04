"""Shared loading, masks, metrics and plot style for the dataset analysis scripts."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import OpenEXR
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAPTURES = ROOT / "data" / "captures"
DEFAULT_REPORTS = ROOT.resolve().parent / "reports"

LAYERS = {
    "albedo": ("albedo.R", "albedo.G", "albedo.B"),
    "normal": ("normal.X", "normal.Y", "normal.Z"),
    "motion": ("motion.X", "motion.Y"),
    "depth": ("depth.Z",),
    "gt": ("ground-truth.R", "ground-truth.G", "ground-truth.B"),
}
FRAME_RE = re.compile(r"frame_(\d+)\.exr$")
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
EDGE_RADIUS_PX = 1.5  # pixels this close to a geometric edge belong to the "edge zone"

# Plot style: categorical slots in fixed order, recessive axes
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


@dataclass
class Frame:
    path: Path
    index: int
    albedo: np.ndarray  # (H, W, 3) linear, 1 sample per pixel, 8-bit quantised
    normal: np.ndarray  # (H, W, 3) world space, zero on background
    motion: np.ndarray  # (H, W, 2) current - previous in UV units, +x right, +y down
    depth: np.ndarray   # (H, W) linear view depth in scene units, far plane on background
    gt: np.ndarray      # (H, W, 3) linear, SSAA 4x4 box-filtered
    header: dict = field(repr=False)

    @property
    def height(self) -> int:
        return self.depth.shape[0]

    @property
    def width(self) -> int:
        return self.depth.shape[1]

    @property
    def far(self) -> float:
        return float(self.header.get("farPlane", 10000.0))

    @property
    def background(self) -> np.ndarray:
        return self.depth >= 0.999 * self.far

    @property
    def motion_px(self) -> np.ndarray:
        return self.motion * np.array([self.width, self.height], dtype=np.float32)


def frame_index(path: Path) -> int:
    match = FRAME_RE.search(Path(path).name)
    return int(match.group(1)) if match else -1


def find_sequences(root: Path) -> dict[str, list[Path]]:
    """captures/<scene>/frame_000123.exr -> {scene: [frame paths in index order]}."""
    root = Path(root)
    sequences = {}
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        frames = sorted(folder.glob("frame_*.exr"), key=frame_index)
        if frames:
            sequences[folder.name] = frames
    if not sequences:
        raise SystemExit(f"No frame_*.exr sequences found under {root}")
    return sequences


def pick(items: list, count: int) -> list:
    """Evenly spaced subset, order kept (everything when count <= 0)."""
    if count <= 0 or count >= len(items):
        return list(items)
    return [items[i] for i in np.unique(np.linspace(0, len(items) - 1, count).round().astype(int))]


def load_frame(path: Path) -> Frame:
    with OpenEXR.File(str(path), separate_channels=True) as exr:
        header = dict(exr.header())
        channels = exr.channels()
        missing = [name for names in LAYERS.values() for name in names if name not in channels]
        if missing:
            raise KeyError(f"{path}: missing channels {missing}")
        data = {
            layer: np.stack([np.asarray(channels[name].pixels, dtype=np.float32) for name in names], axis=-1)
            for layer, names in LAYERS.items()
        }
    return Frame(Path(path), frame_index(path), data["albedo"], data["normal"], data["motion"],
                 data["depth"][..., 0], data["gt"], header)


def srgb(linear: np.ndarray) -> np.ndarray:
    x = np.clip(linear, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055).astype(np.float32)


def luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb @ LUMA


def block_mean(image: np.ndarray, block: int) -> np.ndarray:
    h, w = image.shape[0] // block * block, image.shape[1] // block * block
    return image[:h, :w].reshape(h // block, block, w // block, block, *image.shape[2:]).mean(axis=(1, 3))


def geometric_edges(frame: Frame, depth_tol: float = 0.02, crease_deg: float = 30.0) -> np.ndarray:
    """Pixels on either side of a silhouette, depth step or crease.

    1/z is linear in screen space across a plane, so its second difference is ~0 on any flat surface,
    however steep, and only discontinuities and folds stand out. Normals catch creases at equal depth,
    the background mask catches silhouettes against the clear colour.
    """
    background = frame.background
    inv_depth = 1.0 / np.maximum(frame.depth, 1e-6)
    cos_limit = np.cos(np.radians(crease_deg))
    edges = np.zeros(background.shape, dtype=bool)
    for axis in (0, 1):
        curvature = np.roll(inv_depth, 1, axis) - 2.0 * inv_depth + np.roll(inv_depth, -1, axis)
        edges |= np.abs(curvature) > depth_tol * inv_depth

        next_normal = np.roll(frame.normal, -1, axis)
        next_background = np.roll(background, -1, axis)
        step = (frame.normal * next_normal).sum(-1) < cos_limit
        step &= ~background & ~next_background
        step |= background != next_background
        edges |= step | np.roll(step, 1, axis)  # mark both pixels of the pair
    edges[[0, -1], :] = False
    edges[:, [0, -1]] = False
    return edges


def regions(frame: Frame, edges: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Distance to the nearest geometric edge plus edge-zone / interior / background masks."""
    distance = ndimage.distance_transform_edt(~edges)
    edge = distance <= EDGE_RADIUS_PX
    background = frame.background & ~edge
    return distance, {"edge": edge, "interior": ~edge & ~background, "background": background}


def psnr(mse: float) -> float:
    return float(10.0 * np.log10(1.0 / max(mse, 1e-12)))


def ssim_map(a: np.ndarray, b: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Per-pixel SSIM of two single-channel images in [0, 1] (Wang et al. 2004, Gaussian window)."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    blur = lambda x: ndimage.gaussian_filter(x, sigma)
    mu_a, mu_b = blur(a), blur(b)
    var_a = blur(a * a) - mu_a ** 2
    var_b = blur(b * b) - mu_b ** 2
    cov = blur(a * b) - mu_a * mu_b
    return ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))


def warp_previous(previous: np.ndarray, motion_px: np.ndarray, sign=(1.0, 1.0)):
    """Sample a single-channel previous frame at p - motion: where each current pixel was one frame ago."""
    h, w = previous.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    src_x = xs - sign[0] * motion_px[..., 0]
    src_y = ys - sign[1] * motion_px[..., 1]
    valid = (src_x >= 0) & (src_x <= w - 1) & (src_y >= 0) & (src_y <= h - 1)
    return ndimage.map_coordinates(previous, [src_y, src_x], order=1, mode="nearest"), valid


def save_json(path: Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK_2,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.0, "legend.frameon": False, "legend.fontsize": 8,
    })
    return plt


def save_figure(fig, path: Path) -> None:
    import matplotlib.pyplot as plt
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def scene_colors(scenes: list[str]) -> dict[str, str]:
    """One fixed colour per scene, identical in every figure."""
    if len(scenes) > len(SERIES):
        raise ValueError("More than 8 scenes: plot them as small multiples instead of colours")
    return {scene: SERIES[i] for i, scene in enumerate(scenes)}



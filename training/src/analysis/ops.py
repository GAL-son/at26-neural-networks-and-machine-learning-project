"""Image operations shared by the analyses. Images are (C, H, W) float tensors on any device, row 0 = top."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

LUMA = (0.2126, 0.7152, 0.0722)  # Rec. 709 weights

def luminance(rgb: torch.Tensor) -> torch.Tensor:
    """(3, H, W) -> (1, H, W)."""
    return (rgb * rgb.new_tensor(LUMA).view(3, 1, 1)).sum(0, keepdim=True)

def srgb(linear: torch.Tensor) -> torch.Tensor:
    """Linear [0, 1] -> sRGB-encoded, i.e. what is displayed."""
    x = linear.clamp(0, 1)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.clamp_min(1e-12) ** (1 / 2.4) - 0.055)

def pixel_grid(height: int, width: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """NDC x and y of the pixel centres, each (H, W). Row 0 is ndc y = -1 (top in Vulkan)."""
    x = ((torch.arange(width, device=device) + 0.5) / width * 2 - 1).expand(height, width)
    y = ((torch.arange(height, device=device) + 0.5) / height * 2 - 1)[:, None].expand(height, width)
    return x, y

def reproject(depth: torch.Tensor, vp: np.ndarray, prev_vp: np.ndarray, near: float, far: float):
    """Where each pixel's surface point was in the previous frame (static scene, only the camera moves).

    vp, prev_vp: mathematical P * V matrices (clip = M @ [x, y, z, 1]).
    Returns motion (2, H, W) = uv_curr - uv_prev, the point's linear depth in the previous view (H, W)
    and a (H, W) mask of points in front of the previous camera.
    """
    _, height, width = depth.shape
    x, y = pixel_grid(height, width, depth.device)
    z = depth[0]
    matrix = torch.tensor(prev_vp @ np.linalg.inv(vp), dtype=depth.dtype, device=depth.device)
    clip = torch.stack([x * z, y * z, far * (z - near) / (far - near), z], dim=-1)  # clip.w = linear Z
    previous = clip @ matrix.T
    w = previous[..., 3]
    ndc = previous[..., :2] / w.clamp_min(1e-6).unsqueeze(-1)
    return ((torch.stack([x, y], dim=-1) - ndc) * 0.5).permute(2, 0, 1), w, w > 1e-6

def warp(image: torch.Tensor, motion: torch.Tensor, mode: str = "bilinear"):
    """Move a previous-frame image into the current frame by sampling it at uv - motion.
    Returns the warped (C, H, W) image and a (H, W) mask of pixels whose source lies inside the image."""
    _, height, width = image.shape
    x, y = pixel_grid(height, width, image.device)
    grid = torch.stack([x - 2 * motion[0], y - 2 * motion[1]], dim=-1)  # ndc of uv_prev
    inside = (grid.abs() <= 1).all(-1)
    warped = F.grid_sample(image[None], grid[None], mode=mode, padding_mode="border", align_corners=False)[0]
    return warped, inside

def gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of a (C, H, W) image, kernel radius ceil(3 sigma), borders reflected."""
    radius = math.ceil(3 * sigma)
    t = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel = torch.exp(-0.5 * (t / sigma) ** 2)
    kernel = kernel / kernel.sum()
    channels = image.shape[0]
    x = F.pad(image[None], (radius, radius, radius, radius), mode="reflect")
    x = F.conv2d(x, kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1), groups=channels)
    x = F.conv2d(x, kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1), groups=channels)
    return x[0]

def ssim_map(a: torch.Tensor, b: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """SSIM map (1, H, W) of two single-channel images (Gaussian window sigma 1.5, constants of Wang et al.)."""
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mu_a, mu_b, aa, bb, ab = gaussian_blur(torch.cat([a, b, a * a, b * b, a * b]), 1.5).split(1)
    var_a, var_b, cov = aa - mu_a ** 2, bb - mu_b ** 2, ab - mu_a * mu_b
    return ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))


def block_mean(image: torch.Tensor, block: int) -> torch.Tensor:
    """Average over non-overlapping block x block tiles (a remainder at the borders is dropped)."""
    return F.avg_pool2d(image[None], block)[0]

def geometric_edges(depth: torch.Tensor, normal: torch.Tensor, geometry: torch.Tensor,
                    depth_tolerance: float = 0.02, crease_degrees: float = 30.0) -> torch.Tensor:
    """(H, W) bool: pixels at a silhouette, depth step or normal crease, dilated by 1 px.
    These are the edges the G-buffer can tell the network about.

    1/z is linear in screen space across any plane, so its second difference is ~0 on a flat surface however
    steep, and only steps and folds stand out.
    """
    geo = geometry[0]
    inv_depth = 1 / depth[0].clamp_min(1e-6)
    cos_limit = math.cos(math.radians(crease_degrees))
    edges = torch.zeros_like(geo)
    for dim in (0, 1):
        curvature = inv_depth.roll(1, dim) - 2 * inv_depth + inv_depth.roll(-1, dim)
        edges |= curvature.abs() > depth_tolerance * inv_depth
        neighbour = geo.roll(1, dim)
        edges |= geo != neighbour
        edges |= geo & neighbour & ((normal * normal.roll(1, dim + 1)).sum(0) < cos_limit)
    edges[0], edges[-1], edges[:, 0], edges[:, -1] = False, False, False, False  # roll wraps around
    return F.max_pool2d(edges[None, None].float(), 3, stride=1, padding=1)[0, 0] > 0





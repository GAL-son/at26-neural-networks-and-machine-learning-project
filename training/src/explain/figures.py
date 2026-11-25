"""Figures of one explanation case. Each map is normalised to its own maximum inside the window (the maximum is in
the panel title) and laid over the greyscale input; R, the explained region, is outlined in orange.
Ablation maps share one scale so the groups can be compared."""

from __future__ import annotations

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from common.plotting import DASHED, MUTED, RAMP, SERIES, SURFACE, save

from .case import crop
from .gradcam import VARIANTS

CMAP = LinearSegmentedColormap.from_list("explain", [SURFACE] + RAMP)  # one hue: light = 0, dark = maximum
LUMA = np.array([0.2126, 0.7152, 0.0722])
NAMES = {"gradcam": "Grad-CAM", "hirescam": "HiResCAM"}

def rgb(t):
    """(1, 3, H, W) or (3, H, W) tensor -> (H, W, 3) numpy in [0, 1]."""
    t = t[0] if t.dim() == 4 else t
    return t.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()

def grey(t):
    """(H, W) tensor -> (H, W, 3) numpy, min-max scaled."""
    t = t.detach().float()
    t = (t - t.min()) / (t.max() - t.min()).clamp_min(1e-6)
    return np.repeat(t.cpu().numpy()[..., None], 3, axis=-1)

def normalise(m):
    m = m.detach().float().cpu().numpy()
    peak = float(m.max())
    return (m / peak if peak > 0 else m), peak

def overlay(image, m01):
    base = np.repeat((image @ LUMA)[..., None], 3, axis=-1) * 0.7 + 0.15
    a = 0.85 * m01[..., None]
    return base * (1 - a) + CMAP(m01)[..., :3] * a

def box_in_window(mask, window):
    m = crop(mask, window)[0, 0]
    rows, cols = m.any(1).nonzero().flatten(), m.any(0).nonzero().flatten()
    if len(rows) == 0:
        return None
    
    r0, r1, c0, c1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    if (r1 - r0) * (c1 - c0) > 0.5 * m.numel():  # scattered region over the whole window ("hard-pixels")
        return None
    
    return c0 - 0.5, r0 - 0.5, c1 - c0, r1 - r0

def show(ax, image, title, box=None, **kw):
    im = ax.imshow(image, interpolation="nearest", **kw)
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    if box:
        ax.add_patch(patches.Rectangle(box[:2], box[2], box[3], fill=False, edgecolor=SERIES[1], lw=1.2))

    return im

def save_images(folder, x, y, out, window, maps, layers):
    image = rgb(crop(x[:, :3], window))
    plt.imsave(folder / "input.png", image)
    plt.imsave(folder / "output.png", rgb(crop(out, window)))
    plt.imsave(folder / "gt.png", rgb(crop(y, window)))
    plt.imsave(folder / "history.png", rgb(crop(x[:, 9:12], window)))
    for layer in layers:
        for v in VARIANTS:
            m01, _ = normalise(crop(maps[layer][v], window))
            plt.imsave(folder / f"map_{v}_{layer}.png", m01, cmap=CMAP, vmin=0, vmax=1)
            plt.imsave(folder / f"overlay_{v}_{layer}.png", overlay(image, m01))

def overview(path, title, x, y, out, mask, window, maps, layers):
    box = box_in_window(mask, window)
    image, output, gt = rgb(crop(x[:, :3], window)), rgb(crop(out, window)), rgb(crop(y, window))
    top = [("input (aliased)", image), ("output", output), ("ground truth (SSAA)", gt),
           ("|output - GT| x8", np.clip(np.abs(output - gt) * 8, 0, 1)),
           ("warped history", rgb(crop(x[:, 9:12], window))), ("history valid", grey(crop(x[0, 12], window))),
           ("normal", rgb(crop(x[:, 3:6] * 0.5 + 0.5, window))), ("depth (normalised log)", grey(crop(x[0, 6], window)))]
    ncols = max(len(top), len(layers))
    fig, axes = plt.subplots(3, ncols, figsize=(2.3 * ncols, 7.6), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    for ax, (t, img) in zip(axes[0], top):
        show(ax, img, t, box)

    for row, v in ((1, "gradcam"), (2, "hirescam")):
        for ax, layer in zip(axes[row], layers):
            m01, peak = normalise(crop(maps[layer][v], window))
            show(ax, overlay(image, m01), f"{NAMES[v]} {layer}\nmax {peak:.2e}", box)

    save(fig, path, title)

def pca_figure(path, title, pcas, layers):
    fig, axes = plt.subplots(1, len(layers), figsize=(2.3 * len(layers), 2.8), squeeze=False)
    for ax, layer in zip(axes[0], layers):
        image, explained = pcas[layer]
        show(ax, image.permute(1, 2, 0).numpy(), f"{layer}: 3 PCs = {explained:.0%} var.")

    save(fig, path, title)

def ablation_figure(path, title, result, maps):
    baselines = ["neutral", "blur"]
    groups = list(dict.fromkeys(k.split("/")[0] for k in maps))
    vmax = max(float(m.max()) for m in maps.values()) or 1.0
    s = max(result["S"], 1e-12)
    fig, axes = plt.subplots(2, len(groups), figsize=(2.3 * len(groups), 5.4), squeeze=False)
    im = None
    for i, b in enumerate(baselines):
        for j, g in enumerate(groups):
            ax, key = axes[i][j], f"{g}/{b}"
            ax.axis("off")
            if key not in maps:
                ax.text(0.5, 0.5, "no neutral\nbaseline", ha="center", va="center", color=MUTED, fontsize=8)
                continue

            r = result["groups"][key]
            im = show(ax, maps[key].cpu().numpy(), f"{g} -> {b}\nS {r['dS'] / s:+.0%}, PSNR {r['dPsnr']:+.2f} dB",
                      cmap=CMAP, vmin=0, vmax=vmax)
            
    fig.colorbar(im, ax=axes, shrink=0.6, label=f"|change of output luminance| (shared scale, max {vmax:.3f})")
    save(fig, path, title)

def deletion_figure(path, title, d):
    f = d["fraction"]
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    mean, std = np.array(d["curves"]["random"]), np.array(d["curves"]["randomStd"])
    ax.fill_between(f, mean - std, mean + std, color=MUTED, alpha=0.2, lw=0)
    ax.plot(f, mean, color=MUTED, lw=1.5, ls=DASHED, label=f"random order (AOPC {d['aopc']['random']:.3f})")
    for v, color in (("gradcam", SERIES[0]), ("hirescam", SERIES[2])):
        ax.plot(f, d["curves"][v], color=color, lw=2, label=f"{NAMES[v]} order (AOPC {d['aopc'][v]:.3f})")

    ax.set_xlabel("share of window blocks blurred (most important first)")
    ax.set_ylabel("|S - S0| / S0")
    ax.legend(loc="lower right")
    save(fig, path, title)

def randomisation_figure(path, title, r, stab):
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    xs = list(range(len(r["stages"])))
    for v, color in (("gradcam", SERIES[0]), ("hirescam", SERIES[2])):
        ax.plot(xs, r[v], color=color, lw=2, marker="o", ms=4,
                label=f"{NAMES[v]} (noise stability rho {stab[v]['mean']:.2f})")
        
    ax.axhline(0, color=MUTED, lw=1, ls=":")
    ax.set_xticks(xs, [f"+{s}" for s in r["stages"]])
    ax.set_ylim(-0.2, 1.05)
    ax.set_xlabel("layers randomised, cumulatively from the output")
    ax.set_ylabel("Spearman rho with the trained map")
    ax.legend(loc="upper right")
    save(fig, path, title)
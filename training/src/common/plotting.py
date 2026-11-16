# Colours: validated categorical slots (colour-blind safe) + one-hue ramp for ordered values.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"] 
DASHED = (0, (4, 3))

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 9,
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": INK2, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.labelsize": 8, "legend.frameon": False, "legend.fontsize": 8,
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
})

def save(fig, path: Path, title: str) -> None:
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

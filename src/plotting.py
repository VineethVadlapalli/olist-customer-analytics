"""Shared chart style so every figure in the project looks the same."""

from pathlib import Path

import matplotlib.pyplot as plt

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
IMAGES = Path(__file__).resolve().parents[1] / "docs" / "images"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "figure.dpi": 110, "savefig.dpi": 160,
})


def clean(ax, grid_axis="y"):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


def title(fig, text, subtitle=None):
    fig.suptitle(text, x=0.02, ha="left", fontsize=13.5, fontweight="bold")
    if subtitle:
        fig.text(0.02, 0.905, subtitle, fontsize=9.5, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.88 if subtitle else 0.94))


def save(fig, name):
    IMAGES.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMAGES / name, facecolor=SURFACE)

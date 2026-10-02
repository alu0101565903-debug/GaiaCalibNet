"""Shared publication style and vector/raster export for evaluation figures."""
from functools import wraps
from pathlib import Path

import matplotlib.pyplot as plt

# Colour-blind-friendly colours, consistent across all evaluation figures.
COLORS = {"gaia": "#0072B2", "huang": "#009E73", "pinn": "#D55E00"}
ARTICLE_RC = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "normal",
    "axes.linewidth": 0.7,
    "axes.axisbelow": True,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.fontsize": 8,
    "legend.frameon": False,
    "grid.color": "0.85",
    "grid.linewidth": 0.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def article_style(function):
    """Apply the style locally without changing the caller's rcParams."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        with plt.rc_context(ARTICLE_RC):
            return function(*args, **kwargs)
    return wrapped


def save_article_figure(figure, output_path: str | Path) -> None:
    """Save a vector PDF and a 300-dpi PNG, preserving other requested formats."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    destinations = {path.with_suffix(".pdf"), path.with_suffix(".png")}
    if path.suffix:
        destinations.add(path)
    for destination in sorted(destinations):
        figure.savefig(destination, dpi=300, bbox_inches="tight", pad_inches=0.04)

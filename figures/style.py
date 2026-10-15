"""The one style block, shared by every figure in `figures/`.

`FORMATTING.md`'s rules are a property of the paper, not of any one figure, so they are
written once here and imported by each directory's `common.py`.
"""

import config
from matplotlib import pyplot as plt


def set_style():
    """Arial, nothing below 12 pt, editable PDF text.

    Also re-reads `config.py` (see `config.reload`). Python caches imported modules, so an
    edit to `config.py` would otherwise not reach a kernel that has already imported it — a
    cell re-run after the edit would silently keep the old thresholds. Every notebook calls
    this in its first cell, so re-running that cell is enough to pick up a config change; no
    kernel restart, and nothing to remember.
    """
    config.reload()
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 12
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["pdf.fonttype"] = 42


def save(fig, filename):
    """Write the figure's PDF next to the notebook, and say so."""
    fig.savefig(filename, bbox_inches="tight", pad_inches=0.01)
    print(f"saved {filename}")


def despine(ax):
    """Drop the top and right spines, as every data plot in the paper does."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

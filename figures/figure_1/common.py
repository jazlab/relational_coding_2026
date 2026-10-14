"""Shared setup for the figure_1 notebooks: style, data, and the pieces every panel reuses.

Each notebook here makes one or two PDFs and nothing else. What they have in common lives
here so it is defined once: the matplotlib style, the ring-task table and its per-trial
bookkeeping, the two monkeys' names, the ring-and-fixation-cross drawing that three of the
figures start from, and the boxed legend three of them end with.

Everything is a plain function with the tuned values as defaults. The figures are tuned, so
the numbers matter — but they are settled, and a knob nobody turns is just something else to
read.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

# The east-west antipodal exclusion, shared by every figure in figures/ (see
# ../trial_filters.py). It is applied to every 2-object analysis without exception.
_FIGURES = Path(__file__).resolve().parent.parent
if str(_FIGURES) not in sys.path:
    sys.path.insert(0, str(_FIGURES))
import config  # noqa: E402  the one place selection criteria are set
from trial_filters import (  # noqa: E402
    drop_east_west,
    drop_east_west_arrays,
    east_west_mask,
)

RING_CSV = Path("../../cache/behavior_processing/ring.csv")

SUBJECTS = ["Perle", "Elgar"]  # Monkey P left, Monkey E right
SUBJECT_LABELS = {"Perle": "Monkey P", "Elgar": "Monkey E"}

# The style block, the PDF writer and `despine` live in `figures/style.py` and are
# re-exported here, so every notebook keeps calling `common.set_style()` while there is only
# one definition of what the style IS. See FORMATTING.md.
from style import despine, save, set_style  # noqa: F401


# ── data ──────────────────────────────────────────────────────────────────────
def wrap_pi(x):
    """Wrap angles to (-pi, pi]."""
    return (np.asarray(x, float) + np.pi) % (2 * np.pi) - np.pi


def load_ring():
    """The processed ring-task table, all subjects and sessions."""
    return pd.read_csv(RING_CSV)


def ring_radius(df):
    """The stimulus ring's radius — one number for the whole dataset."""
    return float(np.nanmedian(df["object_0_r"]))


def prep_1obj(df):
    """Completed 1-object trials, with the signed response error."""
    d = df[(df["num_objects"] == 1) & (df["completed"] == True)].copy()
    d["error"] = wrap_pi(
        d["response_theta"].values - d["object_0_theta"].values
    )
    return d


def prep_2obj(df):
    """Completed 2-object trials, with the target/non-target bookkeeping every 2-object
    figure needs.

    The cue names one object (the **target**, by `target_object_index`) and the other is the
    **non-target**. Adds:

    * `target_theta`, `nontarget_theta` — the two objects' angles, cued first;
    * `error`, `error_deg` — the response relative to the target, wrapped to (-pi, pi];
    * `rel_theta`, `rel_deg` — the non-target relative to the target, in [0, 2pi).

    The east-west antipodal pair is dropped here, so every 2-object figure in this directory
    is over the same trials (see `../trial_filters.py`).
    """
    d = df[(df["num_objects"] == 2) & (df["completed"] == True)].copy()
    d = d.dropna(
        subset=["response_theta", "object_0_theta", "object_1_theta"]
    ).copy()
    d = drop_east_west(d).copy()  # the repeated 0/180 pair, dropped everywhere
    cued_is_0 = d["target_object_index"] == 0
    d["target_theta"] = np.where(
        cued_is_0, d["object_0_theta"], d["object_1_theta"]
    )
    d["nontarget_theta"] = np.where(
        cued_is_0, d["object_1_theta"], d["object_0_theta"]
    )
    d["error"] = wrap_pi(d["response_theta"].values - d["target_theta"].values)
    d["error_deg"] = np.degrees(d["error"])
    d["rel_theta"] = (
        d["nontarget_theta"].values - d["target_theta"].values
    ) % (2 * np.pi)
    d["rel_deg"] = np.degrees(d["rel_theta"])
    return d


def add_correct(df, threshold=None):
    """Add the boolean `correct` column: did the response land on the target?

    Two senses of "correct" are available here and they are not quite the same thing.

    * The **recorded reward**. The task wrote `reward_duration` into every trial of the raw
      NWB files and `behavior_processing/run_cache_data.py` carries it into `ring.csv`,
      so this is what the animal was actually paid for, online.
    * A **post-hoc distance criterion**: the response landed within `threshold` of the cued
      object, in display units. This is scored offline from the single stored
      `response_position`, which comes out of a saccade-detection pass that is slightly
      better than the one the task ran live — the online reward window was applied to the
      running eye trace, under real-time constraints, and so occasionally fired on a transient
      or missed a landing that the offline endpoint places cleanly inside or outside the
      window.

    Passing a `threshold` scores the distance criterion and uses it for `correct`, keeping the
    recorded reward alongside as `rewarded` and the distance itself as `target_distance` so the
    two can be compared (see `correct_vs_reward`). `threshold=None` reads the record when it is
    there and otherwise falls back to `config.REWARD_THRESHOLD`, warning that it did.

    The two labels agree on ~99% of the trials any figure here plots, and the disagreements are
    trials that sit within a hair of the boundary either way, so nothing that depends on the
    overall shape of the response distribution turns on the choice.
    """
    d = df.copy()
    recorded = d["correct"].astype(bool) if "correct" in d else None

    if threshold is None:
        if recorded is not None:
            d["correct"] = recorded
            return d
        # No recorded reward in this table — reconstruct it, and say so, because the distance
        # criterion is not the same label (see the note above).
        warnings.warn(
            "ring.csv has no `correct` column, so reward is being reconstructed from a "
            "distance threshold, which does not reproduce it exactly. Run "
            "`python behavior_processing/run_cache_data.py` to rebuild ring.csv with the "
            "recorded reward.",
            RuntimeWarning,
        )
        threshold = config.REWARD_THRESHOLD

    cued = d["target_object_index"].fillna(0).astype(int).values
    r = np.where(cued == 0, d["object_0_r"].values, d["object_1_r"].values)
    theta = np.where(
        cued == 0, d["object_0_theta"].values, d["object_1_theta"].values
    )
    dx = r * np.cos(theta) - d["response_r"].values * np.cos(
        d["response_theta"].values
    )
    dy = r * np.sin(theta) - d["response_r"].values * np.sin(
        d["response_theta"].values
    )
    d["target_distance"] = np.hypot(dx, dy)
    d["correct"] = d["target_distance"].values < threshold
    if recorded is not None:
        d["rewarded"] = recorded
    return d


def correct_vs_reward(df):
    """How far the post-hoc `correct` label and the online `rewarded` record agree.

    Both columns come from `add_correct(..., threshold=...)`. Returns `(n, agreement,
    correct_only, rewarded_only)`: trials compared, the fraction on which the two labels
    match, and the two directions of disagreement — scored correct but not rewarded (the
    online detection missed a landing that was in fact on target), and rewarded but not scored
    correct (the online window fired on an eye position the stored endpoint does not support).
    """
    correct = np.asarray(df["correct"].values, bool)
    rewarded = np.asarray(df["rewarded"].values, bool)
    return (
        len(correct),
        float((correct == rewarded).mean()),
        int((correct & ~rewarded).sum()),
        int((~correct & rewarded).sum()),
    )


# ── drawing ───────────────────────────────────────────────────────────────────
def draw_ring(ax, radius=1.0):
    """The dashed stimulus ring and the fixation cross, the frame three figures start from."""
    t = np.linspace(0, 2 * np.pi, 300)
    ax.plot(radius * np.cos(t), radius * np.sin(t), "k--", lw=0.8, alpha=0.5)
    ax.plot(
        0,
        0,
        marker="+",
        color="k",
        markersize=10,
        markeredgewidth=1.6,
        zorder=6,
    )


def clean_square(ax, view):
    """Equal aspect, `+/- view` limits, no ticks — the state every ring panel wants."""
    ax.set_aspect("equal")
    ax.set_xlim(-view, view)
    ax.set_ylim(-view, view)
    ax.set_xticks([])
    ax.set_yticks([])


def legend_dot(color, size, edge="none"):
    """A proxy handle for a round marker, for the boxed legends below."""
    return Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markerfacecolor=color,
        markeredgecolor=edge,
        markersize=size,
    )


def boxed_legend(where, handles, labels, ncol=None, **kwargs):
    """A legend in a rounded box, in the one style figure_1 uses everywhere.

    `where` is the figure or the axes to attach it to. The frame's own box-style pad is set
    to zero and the spacing left to `borderpad`, because a box-style pad grows the frame
    outside the extent the legend reports and `bbox_inches='tight'` then clips it off.
    """
    legend = where.legend(
        handles,
        labels,
        ncol=len(handles) if ncol is None else ncol,
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        edgecolor="0.4",
        borderpad=0.3,
        handlelength=0.8,
        handletextpad=0.3,
        columnspacing=1.2,
        **kwargs
    )
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_boxstyle("round", pad=0.0, rounding_size=0.15)
    return legend

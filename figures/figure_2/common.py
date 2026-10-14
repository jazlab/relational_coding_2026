"""Shared setup for the figure_2 notebooks: style, the spike/behavior cache, and firing rates.

Both figures in this directory are built from the same three things — a session's ring-task
behavior, one unit's binned spike counts, and a mean firing rate over a window — so those
live here rather than in each figure's own library.

Spike counts come from `phys_processing/spike_counts.py`, which bins each trial into
`BIN_MS`-wide bins starting `LEAD_IN_MS` before stimulus onset; those two constants are what
turn a time in milliseconds into an index, and they are imported from there.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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

if str(_FIGURES.parent) not in sys.path:
    sys.path.insert(0, str(_FIGURES.parent))
from phys_processing import spike_counts as spike_counts_lib  # noqa: E402

ROOT = Path("../..")
RING_CSV = ROOT / "cache/behavior_processing/ring.csv"

BIN_MS = spike_counts_lib.BIN_MS  # width of a spike-count bin
LEAD_IN_MS = (
    spike_counts_lib.LEAD_IN_MS
)  # the counts start this far before stimulus onset
FR_WINDOW = (
    0,
    2000,
)  # default window for a mean firing rate: stimulus + delay

# The style block, the PDF writer and `despine` live in `figures/style.py` and are
# re-exported here, so every notebook keeps calling `common.set_style()` while there is only
# one definition of what the style IS. See FORMATTING.md.
from style import despine, save, set_style  # noqa: F401


# ── data ──────────────────────────────────────────────────────────────────────
def load_behavior(subject, session, num_objects=None):
    """One session's ring-task trials, optionally only the 1- or 2-object ones.

    The east-west antipodal pair is dropped here, so every figure in this directory — the
    PSTHs and the per-anchor fits alike — is over the same trials
    (see `../trial_filters.py`).
    """
    df = pd.read_csv(RING_CSV)
    mask = (
        (df["subject"] == subject)
        & (df["session"] == session)
        & np.isfinite(df["response_theta"])
    )
    if num_objects is not None:
        mask &= df["num_objects"] == num_objects
    return drop_east_west(df[mask]).copy()


def load_unit(subject, session, unit_id):
    """One unit's per-trial spike counts: `(spike_counts, trials, quality_label)`.

    `spike_counts[i]` is the binned count vector of trial `trials[i]`; a unit carries only
    the trials it was isolated on, which is why everything downstream matches on trial
    number rather than on position. `quality_label` is `'{probe}/{quality}'`.
    """
    spike_counts, trials = spike_counts_lib.unit_spike_counts(
        subject, session, unit_id
    )
    unit = spike_counts_lib.units(subject, session).iloc[unit_id]
    return spike_counts, trials, f"{unit.probe}/{unit.quality}"


def mean_response_time_ms(behavior, trials=None):
    """Mean time of the animal's response, in ms after stimulus onset.

    `ring.csv` stores every event as a session-clock timestamp in seconds, so this is
    `1000 * mean(response_time - time_stimulus_onset)`. `response_time` is when the response
    landed and is what varies trial to trial; `time_response_onset` is a fixed 250 ms after
    the cue and would just redraw a task boundary.
    """
    df = (
        behavior
        if trials is None
        else behavior[behavior["trial_num"].isin(list(trials))]
    )
    dt = (df["response_time"] - df["time_stimulus_onset"]).dropna()
    return float(dt.mean() * 1000.0) if len(dt) else float("nan")


# ── spike counts ──────────────────────────────────────────────────────────────
def bin_times_ms(n_bins):
    """Each cached bin's time in ms relative to stimulus onset — the bin's left edge, which
    is the convention every window mask in this directory is written against.
    """
    return np.arange(n_bins) * BIN_MS - LEAD_IN_MS


def smooth(spike_count, half_window_ms=50):
    """Triangular-kernel smoothing of one trial's spike counts, edge-corrected.

    Dividing by the kernel convolved with ones makes the first and last bins averages of the
    bins that exist rather than of implicit zeros.
    """
    half_bins = int(half_window_ms / BIN_MS)
    ramp = np.linspace(0, 1, 1 + half_bins)[1:]
    kernel = np.concatenate([ramp[:-1], ramp[::-1]])
    kernel /= kernel.sum()
    return np.convolve(spike_count, kernel, mode="same") / np.convolve(
        np.ones_like(spike_count), kernel, mode="same"
    )


def per_trial_mean_fr(
    spike_counts, trials, trial_nums, window=FR_WINDOW, smooth_half_win_ms=50
):
    """`{trial_num: mean firing rate (Hz) over window}`, for the trials that exist.

    The counts are smoothed first and then averaged over the window, so this is the mean of
    the same rate the PSTHs plot. Trials the unit was not recorded on are absent from the
    result rather than NaN, so a condition's mean is over the trials it actually has.
    """
    index = {int(t): i for i, t in enumerate(trials)}
    out = {}
    for trial in trial_nums:
        i = index.get(int(trial))
        if i is None:
            continue
        rate = smooth(spike_counts[i], smooth_half_win_ms)
        time = bin_times_ms(len(rate))
        keep = (time > window[0]) & (time < window[1])
        out[int(trial)] = rate[keep].mean() * (1000 / BIN_MS)
    return out


def circ_dist(a, b):
    """Unsigned circular distance between angles, in [0, pi]."""
    return np.abs(
        (np.asarray(a) - np.asarray(b) + np.pi) % (2 * np.pi) - np.pi
    )

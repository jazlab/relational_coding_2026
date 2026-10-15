"""Per-trial binned spike counts, computed on the fly from the spike-sorting NWB files.

For each unit, a trial's spike counts are a vector of `BIN_MS`-wide bins starting
`LEAD_IN_MS` before stimulus onset and ending at feedback onset. Only completed
trials on which the unit was observed (the NWB `obs_trials` column) are included.

This needs two things on disk:
- The spike-sorting NWB files in `cache/dandi_data/spikesorting/`, downloaded with
  `$ python download_dandi_data.py --modality=spikesorting` from the repo root.
- `cache/behavior_processing/ring.csv`, built by
  `behavior_processing/run_cache_data.py`, for the trial event times.

Binning a whole session takes about a second, so nothing is cached to disk.

Usage:
    from phys_processing import spike_counts as spike_counts_lib
    counts, trials = spike_counts_lib.unit_spike_counts('Perle', '2022-06-11', 100)
"""

import functools
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

_REPO = (
    Path(__file__).resolve().parents[1]
)  # repository root, where paths.py lives
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
import paths  # noqa: E402

SPIKESORTING_DIR = paths.SPIKESORTING_NWB
RING_PATH = paths.RING_CSV

BIN_MS = 10  # Width of a spike-count bin
LEAD_IN_MS = 200  # Counts start this long before stimulus onset
_BIN_S = BIN_MS / 1000
_LEAD_IN_S = LEAD_IN_MS / 1000


def nwb_path(subject: str, session: str) -> Path:
    """Path to a session's spike-sorting NWB file."""
    return (
        SPIKESORTING_DIR
        / f"sub-{subject}"
        / f"sub-{subject}_ses-{session}_spikesorting.nwb"
    )


@functools.lru_cache(maxsize=1)
def _ring() -> pd.DataFrame:
    return pd.read_csv(RING_PATH)


@contextmanager
def _units_table(subject: str, session: str):
    with NWBHDF5IO(
        str(nwb_path(subject, session)), "r", load_namespaces=True
    ) as io:
        yield io.read().processing["ecephys"].data_interfaces["units"]


def sessions() -> list[tuple[str, str]]:
    """Every `(subject, session)` with both spike-sorting and behavior data."""
    ring = _ring()
    have_behavior = set(zip(ring["subject"], ring["session"]))
    out = []
    for path in SPIKESORTING_DIR.glob("sub-*/sub-*_spikesorting.nwb"):
        subject = path.parent.name[len("sub-") :]
        session = path.name.split("_ses-")[1][: -len("_spikesorting.nwb")]
        if (subject, session) in have_behavior:
            out.append((subject, session))
    return sorted(out)


def units(subject: str, session: str) -> pd.DataFrame:
    """One row per unit in a session, with columns `unit`, `probe`, `quality`.

    `probe` is the electrode group (`s0` is DMFC, `vprobe*` are FEF) and `quality`
    is the spike-sorting label, `good` (single unit) or `mua` (multi-unit).
    """
    with _units_table(subject, session) as table:
        return pd.DataFrame(
            {
                "unit": np.arange(len(table)),
                "probe": table.electrodes_group[:],
                "quality": table.quality[:],
            }
        )


def _completed_trials(subject: str, session: str):
    """Trial numbers and `(start, end)` times of a session's completed trials."""
    ring = _ring()
    trials = ring[
        (ring["subject"] == subject)
        & (ring["session"] == session)
        & ring["completed"]
    ]
    if len(trials) == 0:
        raise ValueError(f"No behavior data found for {subject} {session}")
    starts = trials["time_stimulus_onset"].to_numpy() - _LEAD_IN_S
    ends = trials["time_feedback_onset"].to_numpy()
    return trials["trial_num"].to_numpy(), starts, ends


def _bin_spikes(spike_times, obs_trials, trial_nums, starts, ends):
    """Spike counts per observed trial; spikes in `[start, end]` are counted."""
    observed = obs_trials[trial_nums].astype(bool)
    trial_nums, starts, ends = (
        trial_nums[observed],
        starts[observed],
        ends[observed],
    )
    first = np.searchsorted(spike_times, starts, side="left")
    last = np.searchsorted(spike_times, ends, side="right")
    spike_counts = []
    for start, end, i, j in zip(starts, ends, first, last):
        num_bins = 1 + int((end - start) / _BIN_S)
        bin_indices = ((spike_times[i:j] - start) / _BIN_S).astype(int)
        spike_counts.append(
            np.bincount(bin_indices, minlength=num_bins).astype(float)
        )
    return spike_counts, trial_nums


def unit_spike_counts(subject: str, session: str, unit: int):
    """One unit's `(spike_counts, trials)`.

    `spike_counts[i]` is the binned count vector of trial `trials[i]`. A unit
    carries only the trials it was observed on, so downstream code should match on
    trial number rather than position.
    """
    trial_nums, starts, ends = _completed_trials(subject, session)
    with _units_table(subject, session) as table:
        if not 0 <= unit < len(table):
            raise KeyError(f"Unit {unit} not found in {subject} {session}")
        return _bin_spikes(
            table.spike_times_index[unit],
            table.obs_trials_index[unit],
            trial_nums,
            starts,
            ends,
        )


def session_spike_counts(subject: str, session: str) -> dict:
    """`{unit: (spike_counts, trials, probe, quality)}` for every unit in a session."""
    trial_nums, starts, ends = _completed_trials(subject, session)
    out = {}
    with _units_table(subject, session) as table:
        probes = table.electrodes_group[:]
        qualities = table.quality[:]
        for unit in range(len(table)):
            spike_counts, trials = _bin_spikes(
                table.spike_times_index[unit],
                table.obs_trials_index[unit],
                trial_nums,
                starts,
                ends,
            )
            out[unit] = (spike_counts, trials, probes[unit], qualities[unit])
    return out

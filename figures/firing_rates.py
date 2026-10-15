"""Per-trial firing rates — the one table every model-based figure starts from.

Per unit and completed-or-not 1-/2-object trial, spikes are summed over 0-1000 ms
(``stimulus_fr``) and 1000-2000 ms (``delay_fr``) after stimulus onset and converted to Hz.
Each row also carries the trial's object angles and **identities** (``'a'``, ``'b'`` or
``'c'``) and the unit's brain area. One gzipped CSV per session:

    cache/figures/shared_inputs/firing_rates/{subject}_{session}.csv.gz

Figure 3 (``figure_3/model_comparison/stats_lib.py``), figure 4 (``figure_4/data_lib.py``)
and everything built on them read these files, so they all see the same rates.

Inputs: spikes binned by ``phys_processing/spike_counts.py`` from the DANDI spike-sorting
files, and trial metadata from ``cache/behavior_processing/ring.csv``.

    python firing_rates.py            # build every missing session (~2 min, ~70 MB)

Code that needs the rates calls :func:`ensure_all` (or :func:`session_path` for one
session), which builds whatever is missing, so the table is never read half-built.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import paths  # noqa: E402
from phys_processing import spike_counts as spike_counts_lib  # noqa: E402

RING_PATH = paths.RING_CSV
FR_DIR = paths.SHARED_INPUTS / "firing_rates"

STIM_MS = (0, 1000)
DELAY_MS = (1000, 2000)

#: Probe → brain area, from the unit's electrode group in the spike-sorting NWB.
PROBE_TO_AREA = {"s0": "DMFC", "vprobe0": "FEF", "vprobe1": "FEF"}

COLUMNS = [
    "subject",
    "session",
    "unit",
    "brain_area",
    "trial_num",
    "completed",
    "num_objects",
    "object_0_theta",
    "object_1_theta",
    "object_0_id",
    "object_1_id",
    "stimulus_fr",
    "delay_fr",
]


def _to_bin(ms):
    return (ms + spike_counts_lib.LEAD_IN_MS) // spike_counts_lib.BIN_MS


def sessions():
    """Every ``(subject, session)`` with spike-sorting data and 1-/2-object behavior."""
    behavior = pd.read_csv(RING_PATH)
    behavior = behavior[behavior["num_objects"].isin([1, 2])]
    have = set(
        map(
            tuple,
            behavior[["subject", "session"]].drop_duplicates().to_numpy(),
        )
    )
    return [s for s in spike_counts_lib.sessions() if s in have]


def session_path(subject, session):
    """The session's firing-rate file, building it first if it is missing."""
    path = FR_DIR / f"{subject}_{session}.csv.gz"
    if not path.exists():
        build_sessions([(subject, session)])
    return path


def ensure_all():
    """Build every missing session, then return the full session list."""
    all_sessions = sessions()
    missing = [
        s
        for s in all_sessions
        if not (FR_DIR / f"{s[0]}_{s[1]}.csv.gz").exists()
    ]
    if missing:
        print(
            f"Building firing rates for {len(missing)} of {len(all_sessions)} sessions "
            f"into {FR_DIR}"
        )
        build_sessions(missing)
    return all_sessions


def build_sessions(sessions_to_build=None, verbose=True):
    """Write ``FR_DIR/{subject}_{session}.csv.gz`` for each session; returns the paths.

    Sessions whose file already exists are skipped.
    """
    sessions_to_build = sessions_to_build or sessions()
    FR_DIR.mkdir(parents=True, exist_ok=True)
    behavior = pd.read_csv(RING_PATH)
    behavior = behavior[behavior["num_objects"].isin([1, 2])]

    stim_lo, stim_hi = _to_bin(STIM_MS[0]), _to_bin(STIM_MS[1])
    delay_lo, delay_hi = _to_bin(DELAY_MS[0]), _to_bin(DELAY_MS[1])
    stim_s = (STIM_MS[1] - STIM_MS[0]) / 1000.0
    delay_s = (DELAY_MS[1] - DELAY_MS[0]) / 1000.0

    written = []
    for subject, session in sessions_to_build:
        out_path = FR_DIR / f"{subject}_{session}.csv.gz"
        if out_path.exists():
            written.append(out_path)
            continue
        meta = behavior[
            (behavior["subject"] == subject) & (behavior["session"] == session)
        ]
        meta = meta[~meta["trial_num"].duplicated(keep="first")].set_index(
            "trial_num"
        )
        if not len(meta):
            if verbose:
                print(f"{subject}/{session}: no 1/2-object behavior — skipped")
            continue
        rows = []
        units = spike_counts_lib.session_spike_counts(subject, session)
        for unit, (spike_counts, trials, probe, _) in sorted(units.items()):
            area = PROBE_TO_AREA.get(probe, "unknown")
            for sc, trial_num in zip(spike_counts, trials):
                trial_num = int(trial_num)
                if trial_num not in meta.index or len(sc) < delay_hi:
                    continue  # not a 1/2-object trial, or too short
                m = meta.loc[trial_num]
                rows.append(
                    (
                        subject,
                        session,
                        unit,
                        area,
                        trial_num,
                        bool(m["completed"]),
                        int(m["num_objects"]),
                        float(m["object_0_theta"]),
                        float(m["object_1_theta"]),
                        m["object_0_id"],
                        m["object_1_id"],
                        float(np.sum(sc[stim_lo:stim_hi]) / stim_s),
                        float(np.sum(sc[delay_lo:delay_hi]) / delay_s),
                    )
                )
        df = pd.DataFrame(rows, columns=COLUMNS)
        tmp = out_path.with_name(out_path.name + ".tmp")
        df.to_csv(tmp, index=False, compression="gzip")
        tmp.replace(out_path)  # a file exists only once it is complete
        written.append(out_path)
        if verbose:
            print(
                f'{subject}/{session}: {df["unit"].nunique()} units, {len(df):,} rows '
                f"→ {out_path.name}"
            )
    return written


if __name__ == "__main__":
    ensure_all()

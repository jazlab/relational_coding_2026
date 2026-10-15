"""Every shared location in the repository, in one place.

All paths are built from this file's own location, so the repository works wherever it is
cloned and whatever the working directory is. Nothing here is an absolute path.

Python modules put the repository root on ``sys.path`` (from their own ``__file__``) and then
``import paths``. Notebooks find this file by walking up from their working directory, which
Jupyter sets to the notebook's own folder:

    import sys
    from pathlib import Path
    _root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'paths.py').exists())
    sys.path.insert(0, str(_root))
    import paths
"""

from pathlib import Path

#: The repository root (the directory holding this file).
ROOT = Path(__file__).resolve().parent

# ── Code ──────────────────────────────────────────────────────────────────────
FIGURES = ROOT / "figures"
DATA_EXPLORATION = ROOT / "data_exploration"
TASK_VISUALIZATION = DATA_EXPLORATION / "task_visualization"

# ── Data and caches (everything under cache/ is gitignored) ───────────────────
CACHE = ROOT / "cache"

#: The DANDI NWB files, written by `download_dandi_data.py`.
DANDI_DATA = CACHE / "dandi_data"
BEHAVIOR_NWB = DANDI_DATA / "behavior"
SPIKESORTING_NWB = DANDI_DATA / "spikesorting"

#: One row per trial, written by `behavior_processing/run_cache_data.py`.
RING_CSV = CACHE / "behavior_processing" / "ring.csv"

#: Everything the figures build. Each figure keeps its own subdirectory.
FIGURES_CACHE = CACHE / "figures"
#: Inputs read by more than one figure (firing rates, the modulation table, unit quality).
SHARED_INPUTS = FIGURES_CACHE / "shared_inputs"

#: Outputs of the `data_exploration/` scripts.
DATA_EXPLORATION_CACHE = CACHE / "data_exploration"


def behavior_nwb(subject, session):
    """The behavior+task NWB file for one session."""
    return (
        BEHAVIOR_NWB
        / f"sub-{subject}"
        / f"sub-{subject}_ses-{session}_behavior+task.nwb"
    )


def add_to_sys_path(*directories):
    """Put code directories on ``sys.path`` (first argument first), skipping duplicates."""
    import sys

    for d in reversed(directories):
        d = str(d)
        if d not in sys.path:
            sys.path.insert(0, d)

"""Shared setup for the supp_figure_2 notebooks: style, saving, and the split by task phase.

This directory is figures 3 and 4's population results computed **once per task phase**
instead of over the whole trial: once on the firing rate during the **stimulus** period
(0-1000 ms after stimulus onset, objects on screen) and once on the **delay** period
(1000-2000 ms, objects gone). The main figures fit every model to the mean rate over both,
the 2-s stimulus + delay window. The question here is whether either result is carried by one
phase — in particular whether the relational code is present while the objects are visible,
or only once they have to be held in memory.

The same five results as `supp_figure_1` (split by area) and `supp_figure_9` (single units):

* from figure 3 — `model_wins_and_probability`, `rf_orientation`;
* from figure 4 — `figure_model_comparison`, `figure_relationality`, `figure_ks_stats`.

**Unlike those two, this is not a subset of cached fits.** A different window is different
data, so every model is refit per phase. The libraries are the main figures' own — nothing is
reimplemented — and three module-level switches select the window:

* ``stats_lib.FR_COLUMNS`` (figure 3) and ``data_lib.FR_COLUMNS`` (figure 4) — which of the
  cached per-trial rate columns, ``stimulus_fr`` and ``delay_fr``, are averaged into the rate
  that is fit. Both by default;
* ``data_lib.DURATION_S`` — the window length the Poisson likelihood in ``models_lib`` turns a
  rate into a spike count with. 2 s by default; **1 s** for one phase, or every likelihood
  would be scored on doubled counts. Figure 3's estimators are ordinal and need no duration.

:func:`use_phase` sets all three and repoints every cache path at this directory's own tree,
``cache/figures/supp_figure_2/{phase}/``, so a phase fit can never land in, or be read
from, the main figures' caches. `build_cache.py` builds those caches.

.. warning::
   The switches are module globals, so they live in **one process**. A joblib worker imports
   the libraries afresh and sees the defaults — the 2-s window — whatever the parent set.
   That is why `build_cache.py` does its own parallelising, with each worker calling
   :func:`use_phase` itself, and why no notebook here passes ``n_jobs > 1`` to a library.

**The population is not re-selected per phase.** The shared 2-object modulation criterion
(`neuron_selection.py`) is computed once, on the 2-s window, over every neuron, and is applied
first, exactly as in `supp_figure_1`: both phases are drawn over the **same 1,339 neurons** as
the main figures, so a difference between the two panels is a difference between phases, not
between populations. What *is* recomputed within each phase is everything downstream — every
fit, which model wins, the posteriors and elpd differences, the phase-determinacy cut of
`rf_orientation`, the synthetic distributions of the KS test.

.. note::
   As in `supp_figure_1`, `figure_4/data_lib.py` does ``from common import
   drop_east_west_arrays``, and from here *this* module is what ``common`` resolves to, so the
   re-export below is load-bearing.
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_FIGURES = _HERE.parent
_ROOT = _FIGURES.parent

for _p in (
    _FIGURES,
    _FIGURES / "figure_3" / "model_comparison",
    _FIGURES / "figure_4",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import config  # noqa: E402,F401  the one place selection criteria are set
from style import despine, save, set_style  # noqa: E402,F401
from trial_filters import (  # noqa: E402,F401
    drop_east_west,
    drop_east_west_arrays,
    east_west_mask,
)

#: One figure each, in this order.
PHASES = ("stimulus", "delay")

#: The cached per-trial rate column each phase is fit on, and the window it was summed over
#: (ms after stimulus onset; see `figure_4/data_lib.build_firing_rates`).
PHASE_COLUMN = {"stimulus": "stimulus_fr", "delay": "delay_fr"}
PHASE_WINDOW_MS = {"stimulus": (0, 1000), "delay": (1000, 2000)}
PHASE_DURATION_S = 1.0

CACHE_ROOT = _ROOT / "cache/figures/supp_figure_2"

#: The figure-4 models this directory fits: the five the figures compare, the metric's own
#: model, and the reference. `softmax` and `priority_map` are in the main cache but in no
#: figure here, so they are not refit.
FIT_MODELS = (
    "future_field",
    "relational_bias",
    "normalization",
    "inhibition",
    "basis_function",
    "weighted_average",
    "constant",
)

#: Figure 3's fits, at the settings the main caches were built with (read off their
#: `fit_tag` / `param_*` columns): sigma 20 deg, 5 folds, uncapped comparisons and 1000
#: bootstrap replicates for the model comparison; 50 bootstraps and the default comparison
#: caps for the phase fit.
COMPARISON_N_BOOT = 1000


def paths(phase):
    """Every cache path for one phase."""
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}, not {phase!r}")
    root = CACHE_ROOT / phase
    f3 = root / "figure_3"
    return dict(
        root=root,
        figure_3=f3,
        comparison_units=f3 / "comparison_units.csv",
        comparison_parts=f3 / "comparison_parts",
        comparison_boot=f3 / "comparison_boot",
        phase_units=f3 / "phase_units.csv",
        phase_draws=f3 / "phase_draws.csv.gz",
        phase_parts=f3 / "phase_parts",
        figure_4=root / "figure_4",
        fits=root / "figure_4" / "fits",
        fit_parts=root / "figure_4" / "fit_parts",
        bias_recovery=root / "figure_4" / "bias_recovery",
        bias_parts=root / "figure_4" / "bias_parts",
    )


def use_phase(phase, verbose=False):
    """Point the figure-3 and figure-4 libraries at one task phase, in this process.

    Sets the rate column, the likelihood's window length, and every cache path the libraries
    read at call time. Paths a library binds as a *default argument* cannot be reached this
    way and are passed explicitly by the loaders below; nothing here relies on them.
    """
    import comparison_lib as cl
    import data_lib as dl
    import ks_lib
    import stats_lib as sl

    p = paths(phase)
    col = PHASE_COLUMN[phase]

    sl.FR_COLUMNS = (col,)
    sl.CACHE_DIR = p["figure_3"]
    sl.UNITS_PATH = p["phase_units"]
    sl.PHASES_PATH = p["phase_draws"]
    sl.PARTS_DIR = p["phase_parts"]
    cl.CACHE_DIR = p["figure_3"]
    cl.UNITS_PATH = p["comparison_units"]

    dl.FR_COLUMNS = (col,)
    dl.DURATION_S = PHASE_DURATION_S
    # FR_DIR is left alone: the firing-rate cache holds both columns and is shared.
    dl.CACHE_DIR = p["figure_4"]
    dl.FIT_DIR = p["fits"]
    ks_lib.BIAS_CACHE_DIR = p["bias_recovery"]
    if verbose:
        lo, hi = PHASE_WINDOW_MS[phase]
        print(
            f"phase: {phase} ({lo}-{hi} ms, `{col}`, {PHASE_DURATION_S:g} s)"
        )
    return p


# ── Figure 3 ──────────────────────────────────────────────────────────────────
def comparison_units(phase, apply_modulation=True):
    """One phase's sign-only model comparison, through the shared population criterion."""
    import comparison_lib as cl
    import stats_lib as sl

    p = use_phase(phase)
    return sl.select_units(
        cl.load_units(path=p["comparison_units"]),
        apply_modulation=apply_modulation,
    )


def phase_fit(phase, apply_modulation=True):
    """One phase's 2-object phase fit and its bootstrap draws: ``(units, phases)``."""
    import stats_lib as sl

    p = use_phase(phase)
    units = sl.select_units(
        sl.load_units(path=p["phase_units"]), apply_modulation=apply_modulation
    )
    return units, sl.load_phases(path=p["phase_draws"])


# ── Figure 4 ──────────────────────────────────────────────────────────────────
def selected_fits(phase, models, verbose=True):
    """One phase's figure-4 fits: the shared criterion, then restricted to ``models``.

    The same order as `figure_4/common.selected_fits` and `supp_figure_1`: the criterion
    first, the model restriction second, so ``best_model`` is recomputed within the set
    actually drawn.
    """
    import fit_lib
    import neuron_selection

    use_phase(phase)
    summary = fit_lib.load_summary(fit_lib.FitConfig(), verbose=False)
    summary = neuron_selection.filter_units(
        summary, unit_col="unit", verbose=verbose
    )
    summary = fit_lib.restrict_models(summary, models, verbose=False)
    if verbose:
        n = summary[["subject", "session", "unit"]].drop_duplicates().shape[0]
        print(f"{phase}: {n} neurons enter the figure")
    return summary


def selected_fits_by_phase(models, phases=PHASES, verbose=True):
    """``[(phase, that phase's selected fits), ...]`` — the figure-4 notebooks' entry point.

    The fits are plain tables, so reading them needs no phase state. Anything that *computes*
    — `ks_lib.synthetic_biases` — must call :func:`use_phase` for its group first, because
    after this returns the libraries are pointed at the last phase in the list.
    """
    return [(ph, selected_fits(ph, models, verbose=verbose)) for ph in phases]

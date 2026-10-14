"""The v2 population for the ring decoder.

`figure_3/motor_decoding` fits a ring decoder on one population per animal and compares the
2-object decode against its own additive null. Its notebook is explicit that there is **no
wedge**: `pl.candidate_units(subject)` returns every neuron passing the shared 2-object
modulation criterion of `figures/config.py`, and the figure contrasts a decode against a
null rather than one sub-population against another.

So the port is narrower than the titration's. Two changes, and only the first is required:

1. **The pool.** The adopted rule is `figures/config.py`'s ``MODULATION_CRITERION`` and
   nothing else — the same rule `efficient_coding/` uses, which is what stops the two halves of
   figure 3 from describing different populations. (An earlier version of this file used the
   model-defined radial cut instead; that admitted ~2200 neurons against the criterion's 1339,
   so results produced under it are not comparable.)
2. **An optional wedge** (:func:`wedge_pool`), which figure 3 has the machinery for but does
   not use. It is here because "do relational neurons carry the movement better?" is a
   natural follow-up, not because the main figure needs it. Left off by default.

Everything else — `fit_tuning`, `joint_fit`, `decode`, `nearest_max_errors`, the annulus
plotting — is imported from figure 3 unchanged. None of it knows a model exists.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_V2 = _HERE.parent
# `population_lib` and `decoder_lib` are the **model-free** decoder — the read-out, the
# response tables, the trial loading. They lived in `figure_3_deprecated/motor_decoding/` and
# were imported rather than copied so the two versions could not drift; when that directory was
# deleted on 2026-09-11 they moved here, which is the same arrangement with one version left.
for _p in (
    str(_V2),
    str(_V2.parent),
    str(_HERE),
    str(_V2 / "model_comparison"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import comparison_lib as cl  # noqa: E402  the model comparison rho is built from
import decoder_lib as dl  # noqa: E402  model-free
import population_lib as pl  # noqa: E402  model-free: the read-out, not the models
import selection_lib as sel  # noqa: E402
import stats_lib as sl  # noqa: E402

# Re-exported so the notebook imports one module.
sample_units = pl.sample_units
load_units_1obj_trials = pl.load_units_1obj_trials
load_units_2obj_trials = pl.load_units_2obj_trials
_session_trials = pl._session_trials
fit_tuning = pl.fit_tuning if hasattr(pl, "fit_tuning") else dl.fit_tuning


def candidate_units(
    subject="all",
    brain_area="all",
    verbose=True,
    exclusion="config",
    coordinate="prob",
):
    """The v2 pool: neurons past the radial cut, with ``rho`` and ``rho_rank`` attached.

    Signature-compatible with `figure_3/motor_decoding/population_lib.candidate_units` so the
    notebook cell that builds a decoder does not have to change shape.

    ``exclusion='config'`` is the default and is the adopted rule: `figures/config.py`'s
    ``MODULATION_CRITERION`` and nothing else. ``'none'`` applies no population criterion and
    is a diagnostic only.
    """
    units = cl.load_units()
    # 'config' -- the shared MODULATION_CRITERION and nothing else -- is the adopted rule, and
    # must match what efficient_coding/pool_lib.py uses or the two halves of figure 3 would
    # describe different populations again, which is the failure figure_3 exists to fix.
    kept = sel.prepared_pool(
        units, coordinate=coordinate, exclusion=exclusion, verbose=verbose
    )
    if subject not in (None, "all"):
        kept = kept[kept["subject"] == subject]
    if brain_area not in (None, "all"):
        kept = kept[kept["brain_area"] == brain_area]
    out = kept.reset_index(drop=True)
    if verbose:
        print(
            f"  pool: {len(out)} neurons"
            f'{"" if subject in (None, "all") else " for " + sel.subject_title(subject)}'
        )
    return out


def wedge_pool(pool_df, x, verbose=True):
    """The sub-pool at sweep fraction ``x`` of the rank coordinate. Not used by the main
    figure — see the module docstring."""
    pos, (lo, hi) = sel.wedge_positions(pool_df["rho_rank"].to_numpy(float), x)
    out = pool_df.iloc[pos].reset_index(drop=True)
    if verbose:
        print(
            f"  wedge x={x:g} -> rank [{lo:.2f}, {hi:.2f}]: {len(out)} neurons, "
            f'realized mean raw ρ {out["rho"].mean():.3f}'
        )
    return out


def pool_summary(pool_df):
    """A short table for the notebook: size, mean ρ and firing rate per subject and area."""
    df = pool_df.copy()
    rows = []
    for s in sel.ordered_subjects(df):
        for area in ["all"] + sorted(
            df.loc[df["subject"] == s, "brain_area"].unique()
        ):
            m = (df["subject"] == s) & (
                (df["brain_area"] == area) if area != "all" else True
            )
            if not m.any():
                continue
            rows.append(
                dict(
                    subject=sel.subject_title(s),
                    area=area,
                    n=int(m.sum()),
                    mean_rho=float(df.loc[m, "rho"].mean()),
                    median_radius=float(df.loc[m, "radius"].median()),
                )
            )
    return pd.DataFrame(rows)

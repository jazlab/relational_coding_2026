r"""The relationality metric of figure 4, read off one neuron at a time.

Figure 4 measures a neuron's relationality as the fitted ``bias`` of the **relational-bias
model** (`figure_4/models_lib.RelationalBiasModel`): five parameters — baseline, amplitude,
preferred direction ``theta``, ``width`` and ``bias`` — fit jointly to the neuron's 2-object
trials by maximum Poisson likelihood (`figure_4/fit_lib`). ``bias = 0`` is the retinotopic model,
``bias = 1`` the fully relational one.

This module adds the **profile** of that fit over ``bias``: at each value on a grid, ``bias`` is
held fixed and the other four parameters are refit, with the same multi-start optimiser
(`fit_lib._fit`, ``n_theta_starts`` rotations of ``theta``) and the same bounds as the cached fit.
Plotted against ``bias``, the profile's peak is where the free fit lands, so the curve shows how
sharply the data pin relationality down rather than only where.

Goodness of fit is the **all-trial log-likelihood per trial, above a constant-rate model fit to
the same trials** (nats/trial) — figure 4's ΔLL convention, but on the training trials, because
a profile is a statement about the fit, not about held-out prediction.

Everything about the model, its fit and the example neurons is imported from `figure_4/`, not
copied: the example neurons are read from `figure_example_neurons.ipynb`'s own ``UNITS`` list
(uncommented entries only), and the free
fit is the cached all-trial fit (`ring_lib.unit_fits`).

Profiles are cached per neuron in ``cache/figures/supp_figure_5/profiles/``, one row per
``bias`` value, so a finer grid computes only the values not already there.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_FIGURES = _HERE.parent
_ROOT = _FIGURES.parent
FIGURE_4 = _FIGURES / "figure_4"
if str(FIGURE_4) not in sys.path:
    sys.path.insert(1, str(FIGURE_4))

import data_lib as dl  # noqa: E402  figure 4
import fit_lib as fl  # noqa: E402
import models_lib as ml  # noqa: E402
import ring_lib as rl  # noqa: E402

MODEL = "relational_bias"
EXAMPLES_NOTEBOOK = FIGURE_4 / "figure_example_neurons.ipynb"
CACHE_DIR = _ROOT / "cache" / "figures" / "supp_figure_5" / "profiles"


def example_units():
    """``[(subject, session, unit, rotation_deg), ...]`` — `figure_example_neurons.ipynb`'s ``UNITS``,
    read from that notebook so the two figures show the same neurons in the same order.
    """
    return [
        tuple(u) for u in dl.notebook_literal(str(EXAMPLES_NOTEBOOK), "UNITS")
    ]


def _model():
    return ml.build_models([MODEL])[MODEL]


def unit_trials(subject, session, unit_id):
    return dl.session_unit_trials(subject, session, verbose=False)[unit_id]


def _d2(unit):
    return (unit["t0"], unit["t1"], unit["fr2"], fl.unit_contrast(unit))


def constant_ll(unit):
    """Total log-likelihood of the constant-rate model (the trials' mean rate)."""
    fr = unit["fr2"]
    return float(
        np.sum(ml.poisson_ll_per_trial(fr, np.full(len(fr), fr.mean())))
    )


def free_fit(subject, session, unit_id, cfg=None):
    """The cached all-trial fit with every parameter free, and its ΔLL (nats/trial).
    Returns ``(params, dll)``."""
    cfg = cfg or fl.FitConfig()
    p = rl.unit_fits(subject, session, unit_id, cfg, verbose=False)[MODEL]
    unit = unit_trials(subject, session, unit_id)
    ll = -fl._neg_ll([], _model(), (), p, _d2(unit))
    return p, (ll - constant_ll(unit)) / len(unit["fr2"])


def _fit_fixed_bias(model, unit, bias, cfg):
    """The other four parameters refit with ``bias`` held fixed — `fit_all_trials`' multi-start,
    minus ``bias`` from the free set (and so minus its three ``bias`` starting values).
    """
    stats = ml.unit_init_stats(unit)
    p0 = fl._apply_width_policy(
        {**model.init_params(stats), "bias": float(bias)}, cfg
    )
    free = tuple(n for n in model.free_params(cfg.fixed_width) if n != "bias")
    starts = [
        {**p0, "theta": p0["theta"] + 2 * np.pi * k / cfg.n_theta_starts}
        for k in range(cfg.n_theta_starts)
    ]
    return fl._fit(model, free, p0, _d2(unit), starts, cfg)


def profile(subject, session, unit_id, bias_grid, cfg=None, verbose=True):
    """ΔLL (nats/trial) at each ``bias`` in ``bias_grid``, as a DataFrame sorted by ``bias``.

    Cached per neuron, one row per ``bias``; only grid values not already cached are fit.
    """
    cfg = cfg or fl.FitConfig()
    path = CACHE_DIR / f"{subject}_{session}_{unit_id}.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(old) and old["cfg_tag"].iloc[0] != cfg.tag:
        old = pd.DataFrame()  # a different estimator: start over
    grid = np.round(np.asarray(bias_grid, float), 6)
    have = set(np.round(old["bias"], 6)) if len(old) else set()
    todo = [b for b in grid if b not in have]
    if todo:
        model, unit = _model(), unit_trials(subject, session, unit_id)
        ll0, n = constant_ll(unit), len(unit["fr2"])
        rows = []
        for b in todo:
            p, ll = _fit_fixed_bias(model, unit, b, cfg)
            rows.append(
                dict(
                    bias=b,
                    ll=ll,
                    ll_constant=ll0,
                    n_2obj=n,
                    dll=(ll - ll0) / n,
                    **{f"p_{k}": v for k, v in p.items() if k != "bias"},
                )
            )
        new = pd.DataFrame(rows).assign(cfg_tag=cfg.tag)
        df = pd.concat([old, new], ignore_index=True) if len(old) else new
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)  # atomic
        if verbose:
            print(
                f"  {subject} {session} unit {unit_id}: fit {len(todo)} new bias values"
            )
    else:
        df = old
    keep = np.isin(np.round(df["bias"], 6), grid)
    return df[keep].sort_values("bias").reset_index(drop=True)


def predicted_rates(params, phis, sep_deg):
    """The model's 2-object rate for the chords ``phis ± sep/2`` — the convention of
    `figure_2/psth_lib.draw_dense_2obj`, so a model ring and a data ring with the same ``phis``
    are drawn over the same chords."""
    return rl.predicted_2obj(_model(), params, phis, sep_deg)

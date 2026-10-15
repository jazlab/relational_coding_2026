"""A neuron's cached all-trial model fits, and a model's predicted 2-object tuning.

:func:`unit_fits` reads one neuron's all-trial parameter fits from the figure-4 fit cache
(fitting that neuron on the spot if its session is not cached), and :func:`predicted_2obj`
evaluates a model's 2-object rate over a family of same-separation chords. No binning is
needed for a prediction: the model is evaluated at the exact angles.
"""

import models_lib as ml
import numpy as np


def predicted_2obj(model, params, phis, sep_deg):
    """Predicted rate over the pair's midpoint direction at one separation."""
    half = np.radians(sep_deg) / 2
    phis = np.asarray(phis, float)
    return ml.rectify(ml.rate_2obj(model, params, phis - half, phis + half))


def params_from_summary(summary_row):
    """``{parameter: value}`` from a summary/units row's ``p_*`` columns."""
    return {
        k[2:]: float(v)
        for k, v in summary_row.items()
        if k.startswith("p_") and np.isfinite(v)
    }


def unit_fits(subject, session, unit_id, cfg, verbose=True):
    """Every model's **all-trial** parameter fit for one neuron, as
    ``{model: {parameter: value}}``.

    Reads the per-session cache when that session has been fit, and otherwise fits just
    this one neuron on the spot (a few seconds for all the models). The parameters are the
    ones fit to **all** of the neuron's 2-object trials — what ``fit_session`` stores in
    its ``units`` file — which is what a tuning curve should be drawn from; the held-out
    likelihoods come from the per-split fits.
    """
    import fit_lib as fl
    import pandas as pd

    path = fl.cache_dir(cfg) / f"{subject}_{session}_units.csv.gz"
    if path.exists():
        units = pd.read_csv(path)
        row = units[units["unit"] == unit_id]
        if len(row) and set(cfg.models) <= set(row["model"]):
            if verbose:
                print(f"  {subject}/{session} unit {unit_id}: from cache")
            return {
                r["model"]: params_from_summary(r) for _, r in row.iterrows()
            }

    if verbose:
        print(
            f"  {subject}/{session} unit {unit_id}: not in the cache — fitting now"
        )
    _, units_df = fl.fit_session(
        subject,
        session,
        cfg,
        unit_ids=[unit_id],
        n_jobs=1,
        verbose=False,
        cache=False,
    )
    return {r["model"]: params_from_summary(r) for _, r in units_df.iterrows()}

r"""**The model-based measure of relationality: the relational-bias model's fitted ``bias``.**

Each neuron is fit with a five-parameter model — ``baseline``, ``amplitude``, ``theta`` (the
mean, i.e. the preferred direction), ``width``, and ``bias`` — whose rate is

    baseline + amplitude · Σ_objects  von_mises(theta, width, biased object angle)

where the *biased object angles* are the objects' directions measured in a coordinate frame
centred on

    (1 − bias) · fixation  +  bias · (the pair's midpoint).

``bias = 0`` is the retinotopic model exactly; ``bias = 1`` measures each object's direction
from the other, the fully relational frame. The fitted ``bias`` is therefore one interpretable
number per neuron on a fixed [0, 1] scale, and it is what
``figure_relationality.ipynb`` plots and ``figure_ks_stats.ipynb`` tests.

See :class:`models_lib.RelationalBiasModel` for the model and :func:`models_lib.bias_angles`
for the transform.

Why the estimator is held fixed
-------------------------------
Real and synthetic values both come from :func:`fit_lib.fit_all_trials` — the same
multi-start, the same bounds, the same width policy. That is not a detail: ``bias`` is bounded
to [0, 1] and its estimator is biased near those bounds, so the distribution of ``bias`` is as
much a property of the estimator as of the neurons. Comparing distributions across a change of
estimator would measure the estimator.

Where it is blind
-----------------
For an antipodal pair the midpoint *is* fixation, so the reference point does not move for any
``bias`` and the trial carries no information about it. About a fifth of 2-object trials are
more than 160° apart, which costs effective sample size but cannot bias the comparison.
"""

import fit_lib as fl

#: The model whose fitted parameter is the metric, and the parameter's column in the cache.
MODEL = "relational_bias"
P_BIAS = "p_bias"

UNIT_KEYS = ["subject", "session", "unit"]


def biases(summary, model=MODEL, verbose=True):
    """One row per neuron with a ``bias`` column — the schema every table and figure in this
    directory reads.

    Reads the **all-trial** fit from the cache that ``run_cache.py`` builds; it does not fit
    anything itself. A neuron whose row is missing the parameter (the model was never fit for
    it) is dropped and counted.
    """
    sub = summary[summary["model"] == model]
    if len(sub) == 0:
        raise KeyError(
            f"no '{model}' rows in the summary, so there is nothing to measure. The metric's "
            f"model is the measuring instrument, not one of the models under test, so a "
            f"notebook whose model list holds only the models it compares will have had these "
            f"rows dropped by `common.selected_fits`. Pass `MODELS + ('{model}',)` to it. "
            f"(Returning an empty frame here used to let the figure draw and save with no "
            f"data curve at all.)"
        )
    if P_BIAS not in sub.columns:
        raise KeyError(
            f"'{P_BIAS}' is not in the summary: the '{model}' model has not been fit yet. "
            f"Run `python run_cache.py --stage fits`."
        )
    out = sub[
        UNIT_KEYS + ["brain_area", "n_2obj", "dtest_ll_mean", P_BIAS]
    ].copy()
    out = out.rename(columns={P_BIAS: "bias", "dtest_ll_mean": "dll"})
    n = len(out)
    out = out.dropna(subset=["bias"]).reset_index(drop=True)
    over = out["bias"] > 1.0 + 1e-6
    if over.any():
        raise ValueError(
            f"{int(over.sum())} of {len(out)} neurons have bias > 1 (max "
            f'{out["bias"].max():.3f}). Those fits predate the identifiability bound and are '
            f"not comparable with current ones — see models_lib.BIAS_MAX. Refit."
        )
    if verbose:
        print(f"{len(out)} of {n} neurons have a fitted bias")
    return out


def unit_bias(unit, cfg=None, model=None):
    """Fit the model to **one** neuron's trials and return its ``bias``.

    Used for the demo notebook and for anything that needs the metric on trials that are not
    in the cache — synthetic ones, say. Goes through :func:`fit_lib.fit_all_trials` so the
    estimator is identical to the one the cached values came from.
    """
    import models_lib as ml

    cfg = cfg or fl.FitConfig(models=(MODEL, "constant"))
    model = model or ml.build_models([MODEL])[MODEL]
    fit = fl.fit_all_trials(unit, cfg, {MODEL: model})[MODEL]
    return float(fit["bias"])

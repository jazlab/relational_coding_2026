"""The relationality coordinate ρ and the wedges both downstream analyses draw.

`efficient_coding/` and `motor_decoding/` differ from figure 3's versions in exactly one
place: **which neurons go into a sub-sample**. Everything else — the kernel-regression
response tables, the MLP read-out, the ring decoder — is model-free and is imported from
`figures/figure_3/` rather than copied, so the two versions cannot drift and any difference
in the result is attributable to the selection rule alone. This module is that rule.

The coordinate
--------------
Figure 3 places each neuron on a plane whose axes are the two models' cross-validated R² and
uses the polar angle::

    rho = (2/pi) * arctan2(cw_cv_r2+, const_cv_r2+)      0 = retinotopic, 1 = relational

The geometry is kept and the axes are the two models' **held-out excess accuracy** — the share
of held-out trial-pair orderings each model calls correctly, above the coin flip that is the
constant model. Both axes are the same kind of object, goodness of fit above a null, measured
out of sample::

    rho = (2/pi) * arctan2(relational_excess+, retinotopic_excess+)

`P(relational)` would also run 0 to 1, but it is deliberately *not* used here: it collapses
"both models fail" onto the same value as "genuinely ambiguous", and telling those apart is
what the exclusion rule is for.

The axes come from `model_comparison/comparison_lib.py` — the **sign-only** comparison, which
is the only model comparison figure 3 makes. Each model is scored on the orderings it predicts
and on nothing else, so the coordinate inherits that: a neuron's position is decided by how
often each model gets the direction of a rate difference right, never by how confident either
was. Until 2026-09-13 the axes were a graded Bradley-Terry model's per-comparison held-out
dLL; the two coordinates correlate at Spearman 0.63, so this is a real change of position and
not a relabelling, and every sweep and decoder downstream was re-run for it.

Why the exclusion rule is load-bearing
--------------------------------------
`arctan2(0, 0) = 0`, so every neuron that **neither** model fits lands at rho = 0 and is
indistinguishable from a genuinely retinotopic one. That pile is real but it is much smaller on
this axis than on the graded one it replaced: 16.6% of neurons sit at exactly rho = 0 and 14.4%
at exactly rho = 1, against 36.9% and 20.3% before. Shrinking it was not the reason for the
change but it is the clearest benefit of it — the extremes of the wedge are far less of a
dumping ground for failed fits than they were.

The rule used is the **shared model-agnostic criterion** of `figures/config.py`
(:func:`config_pool`), which is what the rest of the paper selects on and does this job harder:
of the neurons it removes, nearly all of the rho = 0 pile goes with them.

A second scheme — a **radial cut** in the plane of the two models' evidence, thresholded in z
so that it read "determinable tuning under at least one model at this level" — was built,
measured against this one, and **removed on 2026-09-11**. It had not been used by a figure for
some time, but it was still `prepared_pool`'s default, so a caller that omitted the argument
got the abandoned rule with no warning. Its sensitivity analysis is reported in the
efficient-coding write-ups; `radius_z` and `radius_excess` survive as diagnostic columns, which
is all anything still reads them for.

Rank for sampling, raw for plotting
-----------------------------------
The wedge is taken on **rho ranked within subject**; the figure is drawn against **raw rho**.
The two animals' raw rho distributions have very different shapes, so a fixed wedge in raw rho
selects a different fraction of each population and the two sweeps are not comparable. Ranking
fixes that. Raw rho is what a reader can interpret, so it is what the axis shows — and the
axis is the *realized* mean rho of the sub-population actually drawn (:func:`realized_rho`),
not the sweep parameter, so the population's own value lands on the same axis by construction
and the "nearly optimal" claim becomes checkable. That is the fix to the reviewer's third
point, and it is a property of this module rather than of the plotting code.

"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

_HERE = Path(__file__).resolve().parent
_FIGURES = _HERE.parent
for _p in (str(_FIGURES), str(_HERE / "model_comparison")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import stats_lib as sl  # noqa: E402  figure_3/model_comparison/stats_lib.py

#: Which radius to cut on. See the module docstring for why 'z' is the default.
RADIUS_SPACE = "z"

SUBJECTS = ["Perle", "Elgar"]
SUBJECT_LABELS = {"Perle": "Monkey P", "Elgar": "Monkey E"}

#: The wedge colours, matching model_comparison's.
RETINOTOPIC_COLOR = sl.MODEL_COLORS["retinotopic"]
RELATIONAL_COLOR = sl.MODEL_COLORS["relational"]


# ── The coordinate ────────────────────────────────────────────────────────────
def _pos(units, col):
    return np.maximum(units[col].to_numpy(float), 0.0)


def rho_angle(units):
    """Raw ρ in [0, 1]: 0 on the retinotopic axis, 1 on the relational axis, 0.5 the diagonal.

    Rectified at zero exactly as figure 3 rectifies its R²: a *negative* excess accuracy means
    the model is worse than a coin, which is not evidence of the other model and must not be
    allowed to swing the angle past the axis.
    """
    return (
        2.0
        * np.arctan2(
            _pos(units, "relational_excess"), _pos(units, "retinotopic_excess")
        )
        / np.pi
    )


PROB_AXIS = ("retinotopic", "relational")

#: Chance is 1/2 and perfect is 1, so doubling the excess over chance puts a model's held-out
#: accuracy on [0, 1]: 0 is "no better than a coin", 1 is "every held-out ordering right".
ACCURACY_SCALE = 2.0


def fit_strength(units, model):
    """One model's held-out performance on a 0-1 scale: ``2 * (accuracy - 1/2)``.

    The comparison scores a model by the fraction of held-out trial-pair orderings it calls
    correctly, so chance is 1/2 and perfect is 1. Doubling the excess over chance puts that on
    [0, 1], where 0 is "no better than a coin" and 1 is "every ordering right" -- readable in
    the same way a fraction-of-variance-explained is, without pretending to be one.

    Multiplying both axes by a constant leaves ``arctan2`` untouched, so the scale changes how
    the plane is *labelled* and not which neurons any wedge selects.

    Not rectified: a negative value means the model did worse than a coin out of sample, and the
    cuts below are entitled to see that. :func:`rho_angle` rectifies separately.
    """
    return units[f"{model}_excess"].to_numpy(float) * ACCURACY_SCALE


def rho_prob(units, models=PROB_AXIS, n_trials=None):
    """Titration coordinate from the comparison's per-neuron posterior over models.

    The probability simplex over the **three** models is 2-dimensional, but a titration wedge
    is 1-dimensional, so an axis has to be chosen. This returns the probability of ``models[1]``
    **conditional on the neuron being one of the two**::

        rho = p_relational / (p_retinotopic + p_relational)

    Two reasons to project this way rather than invent a path through the triangle. It is what
    figure 3 does — `angle_metric(x_model='const', y_model='cw')` puts two of its three models
    on the axis and leaves `ioa` off it — so the two versions stay comparable. And the
    efficient-coding claim is specifically about retinotopic and relational codes being
    complementary; a third direction on the same axis would change what the curve means rather
    than sharpen it.

    The consequence to keep in view is that neurons best explained by **proximity** are not
    excluded, they are placed by whatever their retinotopic-versus-relational split happens to
    be, which for them is a ratio of two small numbers. :func:`prob_axis_report` measures how
    many neurons that is and how ill-determined their position is; ``drop_proximity_best`` in
    :func:`prepared_pool` removes them if that is preferred.

    The probabilities are the comparison's ``<model>_win_frac`` — the fraction of trial
    resamples in which that model has the highest held-out accuracy for this neuron. They sum
    to 1 over the three models by construction, so this is a genuine per-neuron posterior over
    the model set rather than a softmax of an evidence scale. It replaced the graded
    comparison's ``n_trials x dLL`` softmax on 2026-09-13; that quantity was never a calibrated
    posterior and said so. ``n_trials`` is accepted and ignored, so callers that passed it do
    not have to change.
    """
    del n_trials  # the win fractions need no evidence scale
    a = units[f"{models[0]}_win_frac"].to_numpy(float)
    b = units[f"{models[1]}_win_frac"].to_numpy(float)
    tot = a + b
    with np.errstate(all="ignore"):
        out = np.where(tot > 0, b / tot, np.nan)
    return out


def prob_axis_report(units, models=PROB_AXIS, n_trials=None):
    """What the projection onto the two-model axis costs: who is off-axis, and how badly."""
    del n_trials
    third = [m for m in sl.MODELS if m not in models]
    rows = []
    for s in ordered_subjects(units):
        m = (units["subject"] == s).to_numpy()
        on = (
            units.loc[m, [f"{x}_win_frac" for x in models]]
            .to_numpy(float)
            .sum(axis=1)
        )
        row = dict(
            subject=SUBJECT_LABELS.get(s, s),
            n=int(m.sum()),
            median_on_axis_mass=float(np.median(on)),
            frac_on_axis_below_0p5=float(np.mean(on < 0.5)),
        )
        for t in third:
            row[f"best_{t}"] = int((units.loc[m, "best_model"] == t).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def min_fit_cut(units, min_fit, mode="either", verbose=True):
    """Drop neurons whose held-out fit strength is below ``min_fit`` on **both** axes.

    A neuron is kept if **either** model clears the bar, so this asks "is this neuron worth
    placing on the plane at all?" and leaves the retinotopic-versus-relational question
    entirely to the angle. That is what keeps it from being a thumb on the scale: it is
    symmetric in the two models by construction, so it cannot prefer one.

    This is the cut that addresses the coordinate's real weakness. The polar angle is
    scale-free in the interior of the plane, but its **extremes are a dumping ground for
    failed fits**: 31.0% of the pool sits at exactly rho = 0 or 1, and every one of those got
    there by rectification -- one model's held-out accuracy came out at or below chance and was
    clipped -- rather than by a large ratio. A neuron whose retinotopic fit is merely negative
    noise is otherwise placed at "purely relational". Measured consequence: the extreme wedges
    carry about 30% less explainable variance than the middle, which biases the titration
    toward a dip for reasons that have nothing to do with complementarity.

    **The two modes do very different things, and only one fixes the rectification problem.**

    ``mode='either'`` (drop a neuron only when *both* axes are below the bar) keeps any neuron
    one model fits, including the ones whose *other* fit is negative and clipped. It barely
    touches the extremes, and a stricter bar collapses Monkey E's pool long before it cleans
    them up. It is a fit-quality filter, not a fix for the coordinate.

    ``mode='both'`` (require *both* axes to clear the bar) is the one that removes rectified
    neurons: at a threshold of 0 it keeps only neurons where both models beat a coin flip out
    of sample, so every survivor's angle is a ratio of two real fits rather than an artifact of
    clipping.

    Either way the cut is symmetric in the two models, so it cannot prefer one -- which is what
    makes it safe to apply before the angle is taken.
    """
    if min_fit is None:
        return units
    a = fit_strength(units, "retinotopic")
    b = fit_strength(units, "relational")
    if mode == "either":  # drop only if BOTH axes are below the bar
        r = np.maximum(a, b)
    elif mode == "both":  # require BOTH axes to clear it
        r = np.minimum(a, b)
    else:
        raise ValueError(f"mode must be 'either' or 'both', got {mode!r}")
    keep = np.isfinite(r) & (r >= min_fit)
    out = units[keep].copy()
    if verbose:
        which = "at least one axis" if mode == "either" else "BOTH axes"
        print(
            f"min fit strength >= {min_fit:g} on {which}: "
            f"{int(keep.sum())} of {len(units)} kept"
        )
        for s_ in ordered_subjects(units):
            m = (units["subject"] == s_).to_numpy()
            print(
                f"  {SUBJECT_LABELS.get(s_, s_)}: {int(keep[m].sum())} of {int(m.sum())}"
            )
    return out


def sum_fit_cut(units, min_sum, verbose=True):
    """Drop neurons whose two pseudo-R-squareds **sum** to less than ``min_sum``.

    A radial cut in the pseudo-R-squared plane under the L1 norm: it removes the neurons close
    to the *origin* — the ones neither model explains — and keeps everything far from it
    regardless of direction. Being a function of the distance only, it is symmetric in the two
    models by construction and so cannot prefer one, which is what makes it safe to apply
    before the angle is taken.

    It is a different instrument from :func:`min_fit_cut` and the two compose rather than
    substitute. ``min_fit_cut(..., mode='both')`` is a cut on the *weaker* axis: it removes
    neurons whose angle is an artefact of rectification, but keeps a neuron whose two fits are
    both tiny-but-positive, and such a neuron sits at a well-determined angle carrying almost
    no tuning. This cut removes exactly those. Conversely it keeps a neuron with one strong fit
    and one slightly negative one, which ``mode='both'`` drops. Apply both to get a pool that
    is neither rectified nor uninformative.

    :func:`fit_strength` is ``2 * (accuracy - 1/2)`` — how far above a coin flip the model
    calls held-out orderings — so a threshold of 0.02 means "the two models together get at
    least 1% more of the orderings right than chance". Note the scale is not figure 3's
    cross-validated R-squared and the two thresholds are not interchangeable.

    The sum is taken on the **rectified** values, matching :func:`rho_angle`, so a negative fit
    contributes zero rather than cancelling a positive one. Without that a neuron with a strong
    relational fit and a badly negative retinotopic one could be cut for the wrong reason.
    """
    if min_sum is None:
        return units
    r = _pos(units, "retinotopic_excess") + _pos(units, "relational_excess")
    r = r * ACCURACY_SCALE
    keep = np.isfinite(r) & (r >= min_sum)
    out = units[keep].copy()
    if verbose:
        print(
            f"sum of pseudo-R2 >= {min_sum:g}: {int(keep.sum())} of {len(units)} kept"
        )
        for s_ in ordered_subjects(units):
            m = (units["subject"] == s_).to_numpy()
            print(
                f"  {SUBJECT_LABELS.get(s_, s_)}: {int(keep[m].sum())} of {int(m.sum())}"
            )
    return out


def sig_fit_cut(
    units, q, use_fdr=True, models=("retinotopic", "relational"), verbose=True
):
    """Drop neurons that **neither** model fits significantly better than a constant null.

    Each model's ``<model>_beats_null_p`` is a one-sided test that its held-out accuracy is
    above 1/2, taken over the comparison's trial bootstrap. A neuron with no stimulus
    dependence cannot order two trials better than a coin, so chance *is* the constant model
    and a small p means "this model beats a constant null for this neuron". The cut keeps a
    neuron if **either** model clears the bar and drops it only when both fail, which is the
    literal reading of "explained by neither".

    Like :func:`sum_fit_cut` this removes the neurons near the origin of the plane, but it
    asks a different question and will not select the same set. ``sum_fit_cut`` is an
    **effect-size** floor: it wants the two models to account for some minimum share of the
    baseline's uncertainty. This is a **significance** floor: it wants the tuning to be
    resolvable, which a weakly-driven neuron with many trials can be and a strongly-driven one
    with few trials may not. Comparing the two titration curves is therefore a check on which
    kind of "uninformative neuron" was doing the damage, if either.

    ``use_fdr`` applies Benjamini-Hochberg **per model, across the neurons passed in**, at
    level ``q`` — matching `config.py`'s treatment of the modulation criterion, and for the
    same reason: several hundred neurons are tested at once and an uncorrected 5% would admit
    a tenth of that many by chance. Set it False to threshold the raw p directly.

    Note the correction is taken over the pool *as handed to this function*, not over all 3186
    fitted neurons, because that is the family of tests this analysis actually performs. It is
    the less conservative choice of the two; the counts it prints make the difference visible.
    """
    if q is None:
        return units
    import neuron_selection as ns

    keep = np.zeros(len(units), bool)
    per_model = {}
    for m in models:
        pv = units[f"{m}_beats_null_p"].to_numpy(float)
        hit = (
            ns.benjamini_hochberg(pv, q)
            if use_fdr
            else (np.isfinite(pv) & (pv <= q))
        )
        per_model[m] = int(hit.sum())
        keep |= hit
    out = units[keep].copy()
    if verbose:
        how = f"BH-FDR q = {q:g}" if use_fdr else f"raw p <= {q:g}"
        print(
            f"significant vs a constant null under EITHER model ({how}): "
            f"{int(keep.sum())} of {len(units)} kept"
        )
        print(
            "  per model: "
            + ", ".join(f"{m} {n}" for m, n in per_model.items())
        )
        for s_ in ordered_subjects(units):
            m_ = (units["subject"] == s_).to_numpy()
            print(
                f"  {SUBJECT_LABELS.get(s_, s_)}: {int(keep[m_].sum())} of {int(m_.sum())}"
            )
    return out


def config_pool(units, verbose=True):
    """The shared, **model-agnostic** exclusion of `figures/config.py`.

    This is the right partner for :func:`rho_prob`, and it is why the radial cut this module
    used to offer was removed rather than kept as an option. That cut was defined from the two
    models being contrasted (their Wald p-values), so it preferentially kept neurons that one
    of those two models happens to fit — a mild circularity of exactly the kind the reviewer
    objected to in panel 3c. ``config.py``'s criterion is a kernel-smoother test of whether the
    neuron has *any* 2-object tuning, which is independent of the retinotopic-versus-relational
    question.

    It also does the job the radial cut was there for, and does it harder: of the 665 neurons
    neither model explains, this admits **6**. The price is pool size — 1339 neurons against
    2460 — so the wedges are smaller and the titration curve noisier. That is the trade, and it
    is worth taking because the alternative buys its neurons with circularity.

    Using it also puts this figure on the same population as the rest of `figures`.
    """
    return sl.select_units(units, apply_modulation=True, verbose=verbose)


def radius_excess(units):
    """Radial coordinate in excess accuracy — distance from the origin of the accuracy plane."""
    return np.hypot(
        _pos(units, "retinotopic_excess"), _pos(units, "relational_excess")
    )


def radius_z(units):
    """Radial coordinate in z: ``hypot(z_retinotopic, z_relational)``.

    Each model's ``*_beats_null_p`` is a one-sided test that its held-out accuracy beats
    chance, so ``z = norm.isf(p)`` is its evidence for *any* predictable structure. A neuron
    neither model can explain sits near the origin whatever its accuracies happen to be, which
    is what a radial cut is for. (One-sided on a scalar, so the normal quantile -- the graded
    version used a 2-d.o.f. chi-square because its test was on a tuning vector.)
    """
    from scipy.stats import norm

    z = []
    for m in ("retinotopic", "relational"):
        p = np.clip(units[f"{m}_beats_null_p"].to_numpy(float), 1e-300, 1.0)
        with np.errstate(all="ignore"):
            z.append(np.maximum(norm.isf(p), 0.0))
    return np.hypot(*z)


def add_coordinates(units, space=RADIUS_SPACE):
    """Attach ``rho``, ``radius`` and ``radius_z``/``radius_excess`` to a copy of ``units``."""
    u = units.copy()
    u["rho"] = rho_angle(u)
    u["radius_z"] = radius_z(u)
    u["radius_excess"] = radius_excess(u)
    u["radius"] = u[f"radius_{space}"]
    return u


# ── Rank within subject, and the wedge ────────────────────────────────────────
def add_rank(units, col="rho", out="rho_rank"):
    """Rank ``col`` to [0, 1] **within each subject**.

    The wedge is taken on this, not on raw ρ. The animals' raw distributions are shaped very
    differently, so a fixed raw wedge selects a different fraction of each population and the
    sweeps stop being comparable. Ranking makes sweep fraction mean "this quantile of *this*
    animal's population", which is what the titration is actually varying.
    """
    u = units.copy()
    u[out] = np.nan
    for s in u["subject"].unique():
        m = (u["subject"] == s).to_numpy()
        v = u.loc[m, col].to_numpy(float)
        u.loc[m, out] = (
            pd.Series(v).rank(method="average").to_numpy() - 0.5
        ) / len(v)
    return u


def angle_range_from_mean(x):
    """The wedge at sweep fraction ``x`` — figure 3's rule, kept identical.

    ``x < 0.5 -> [0, 2x]`` (retinotopic side), ``x >= 0.5 -> [2x-1, 1]`` (relational side),
    ``x = 0.5 -> [0, 1]`` (everything). Nested, so the sweep runs from a pure-retinotopic
    sample through the whole population to a pure-relational one.
    """
    return (0.0, 2.0 * x) if x < 0.5 else (2.0 * x - 1.0, 1.0)


def wedge_positions(values, x):
    """Row positions inside the wedge at sweep fraction ``x``, and its ``(lo, hi)`` bounds."""
    lo, hi = angle_range_from_mean(x)
    v = np.asarray(values, float)
    return np.where(np.isfinite(v) & (v >= lo) & (v <= hi))[0], (lo, hi)


def realized_rho(rho_raw, positions):
    """Mean **raw** ρ of a drawn sub-population — the figure's x coordinate.

    The sweep fraction is a parameter of the sampling procedure and is not a proportion of
    anything in the brain, which is exactly the reviewer's objection. This is: the average
    relationality of the neurons the decoder actually received. Plotting against it puts the
    whole population's own mean ρ on the same axis by construction, so "the brain sits near
    the optimum" becomes a claim a reader can check off the figure.
    """
    return float(np.mean(np.asarray(rho_raw, float)[np.asarray(positions)]))


def population_rho(units, n_boot=2000, seed=0):
    """The population's own mean raw ρ per subject, with a neuron-level bootstrap CI.

    This is the vertical line the titration figure needs. The CI is over **neurons** — the
    quantity is a property of the recorded population, and the uncertainty that matters is
    which neurons happened to be recorded.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for s in ordered_subjects(units):
        v = units.loc[units["subject"] == s, "rho"].to_numpy(float)
        v = v[np.isfinite(v)]
        if not len(v):
            continue
        boots = np.array(
            [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)]
        )
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append(
            dict(
                subject=SUBJECT_LABELS.get(s, s),
                n=len(v),
                mean_rho=float(v.mean()),
                lo=float(lo),
                hi=float(hi),
            )
        )
    return pd.DataFrame(rows)


# ── Shared bits ───────────────────────────────────────────────────────────────
def ordered_subjects(u):
    present = list(pd.unique(u["subject"]))
    return [s for s in SUBJECTS if s in present] + [
        s for s in present if s not in SUBJECTS
    ]


def subject_title(s):
    return SUBJECT_LABELS.get(s, s)


def _despine(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def coordinate_values(units, coordinate="angle", **kw):
    """The titration coordinate. ``'angle'`` = :func:`rho_angle`, ``'prob'` = :func:`rho_prob`."""
    if coordinate == "angle":
        return rho_angle(units)
    if coordinate == "prob":
        return rho_prob(units, **kw)
    raise ValueError(
        f"coordinate must be 'angle' or 'prob', got {coordinate!r}"
    )


def prepared_pool(
    units,
    coordinate="angle",
    exclusion="config",
    verbose=True,
    min_fit=None,
    min_fit_mode="both",
    min_fit_sum=None,
    sig_q=None,
    sig_use_fdr=True,
    **kw,
):
    """Apply the exclusion, attach ``rho`` and ``rho_rank``. The one entry point.

    ``exclusion='config'`` is the shared model-agnostic criterion of `figures/config.py`
    (see :func:`config_pool` for why that is the better partner for ``coordinate='prob'``);
    ``'none'`` applies neither. The third option, a model-defined **radial cut**, was removed
    on 2026-09-11: no figure had used it for some time, but it was still this function's
    default, so any caller that omitted the argument got the abandoned rule silently. Its
    sensitivity analysis is reported in the efficient-coding write-ups.

    The defaults are now the adopted scheme, for the same reason.

    ``min_fit`` / ``min_fit_mode`` are the per-axis fit cut (:func:`min_fit_cut`),
    ``min_fit_sum`` the joint effect-size floor (:func:`sum_fit_cut`) and ``sig_q`` the
    joint significance floor (:func:`sig_fit_cut`). All default to off and compose when set;
    the last two are alternative answers to the same question and are normally used one at a
    time.
    """
    if exclusion == "config":
        kept = config_pool(units, verbose=verbose)
    elif exclusion == "none":
        kept = units.copy()
    else:
        raise ValueError(
            f"exclusion must be 'config' or 'none', got {exclusion!r}"
        )
    kept = min_fit_cut(kept, min_fit, mode=min_fit_mode, verbose=verbose)
    # After the per-axis cut, so the count it prints is the number of *additional* neurons the
    # joint floor removes rather than an overlap with the one above.
    kept = sum_fit_cut(kept, min_fit_sum, verbose=verbose)
    kept = sig_fit_cut(kept, sig_q, use_fdr=sig_use_fdr, verbose=verbose)
    # The radial coordinates come along as diagnostics. They used to be attached by the radial
    # cut, so removing it took `radius` away from everything that merely *reported* it —
    # `population_v2.pool_summary` among them. They are a description of where a neuron sits in
    # the evidence plane, which is worth having whether or not anything cuts on it.
    kept = add_coordinates(kept)
    kept["rho"] = coordinate_values(kept, coordinate, **kw)
    return add_rank(kept)


def slice_data(
    units,
    sweep_fractions,
    subject,
    coordinate="angle",
    exclusion="config",
    min_fit=None,
    min_fit_mode="both",
):
    """Per requested sweep fraction: which rows are in the wedge, and its bounds.

    Returned rather than plotted so a caller can count and check before drawing. ``pos`` are
    positions into the returned ``g``, not into ``units``.
    """
    g = prepared_pool(
        units,
        coordinate,
        exclusion,
        verbose=False,
        min_fit=min_fit,
        min_fit_mode=min_fit_mode,
    )
    g = g[g["subject"] == subject].reset_index(drop=True)
    g = add_rank(g)  # re-rank within the single subject
    rank, rho = g["rho_rank"].to_numpy(float), g["rho"].to_numpy(float)
    out = []
    for x in sweep_fractions:
        pos, (lo, hi) = wedge_positions(rank, x)
        out.append(
            dict(
                x=x,
                pos=pos,
                rank_lo=lo,
                rank_hi=hi,
                n=len(pos),
                mean_rho=realized_rho(rho, pos) if len(pos) else np.nan,
            )
        )
    return g, out


def plot_slices(
    units,
    sweep_fractions=(0.1, 0.3, 0.5, 0.7, 0.9),
    figsize=None,
    point_size=6,
    show_hist=True,
    unselected_color="0.86",
    log_axes=False,
    coordinate="angle",
    exclusion="config",
    axis_scale=1.0,
    axis_unit="excess accuracy",
    min_fit=None,
    min_fit_mode="both",
):
    """The titration's insets on their own: what each sweep fraction selects.

    Figure 3 embeds these as small insets above the curve, where they are too small to read
    and their selection can only be taken on trust. Drawn full size they answer the question
    the curve cannot: *what is actually in each sub-population?*

    Two rows per subject. The **plane** row shows the two models' held-out excess accuracies
    with the wedge's neurons coloured by rho and everything else left grey, so the slice is
    visible as a geometric wedge from the origin — which is what it is. The **distribution** row
    shows the subject's whole rho histogram with the slice's share shaded, so the extremes being
    nearly pure (rho piles at exactly 0 and 1 whenever one accuracy is rectified to zero) is
    visible rather than surprising.

    ``log_axes`` puts the plane on log scales, which spreads the dense clump near the origin
    where most neurons sit. It changes nothing about the selection.
    """
    subjects = ordered_subjects(units)
    n_col = len(sweep_fractions)
    n_row = len(subjects) * (2 if show_hist else 1)
    figsize = figsize or (1.75 * n_col + 0.8, 1.85 * n_row)
    fig, axes = plt.subplots(
        n_row, n_col, figsize=figsize, squeeze=False, layout="constrained"
    )
    for si, s in enumerate(subjects):
        g, slices = slice_data(
            units,
            sweep_fractions,
            s,
            coordinate=coordinate,
            exclusion=exclusion,
            min_fit=min_fit,
            min_fit_mode=min_fit_mode,
        )
        rho = g["rho"].to_numpy(float)
        x_ret = _pos(g, "retinotopic_excess") * axis_scale
        x_rel = _pos(g, "relational_excess") * axis_scale
        hi_lim = float(np.nanpercentile(np.hypot(x_ret, x_rel), 99.5))
        r_plane = si * (2 if show_hist else 1)
        for ci, sl_ in enumerate(slices):
            ax = axes[r_plane][ci]
            m = np.zeros(len(g), bool)
            m[sl_["pos"]] = True
            ax.scatter(
                x_ret[~m],
                x_rel[~m],
                c=unselected_color,
                s=point_size,
                linewidths=0,
                zorder=1,
            )
            ax.scatter(
                x_ret[m],
                x_rel[m],
                c=rho[m],
                cmap="coolwarm",
                vmin=0,
                vmax=1,
                s=point_size,
                linewidths=0,
                zorder=2,
            )
            if log_axes:
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlim(1e-5, hi_lim)
                ax.set_ylim(1e-5, hi_lim)
            else:
                ax.set_xlim(0, hi_lim)
                ax.set_ylim(0, hi_lim)
            ax.plot([0, hi_lim], [0, hi_lim], color="0.75", lw=0.6, zorder=0)
            lab = "ρ̄" if coordinate == "angle" else "P̄"
            ax.set_title(
                f'x = {sl_["x"]:g}\n{sl_["n"]} units, {lab} = {sl_["mean_rho"]:.2f}',
                fontsize=9,
                pad=3,
            )
            ax.tick_params(labelsize=7)
            if ci == 0:
                ax.set_ylabel(
                    f"{subject_title(s)}\nrelational {axis_unit}", fontsize=9
                )
            if not show_hist:
                ax.set_xlabel(f"retinotopic {axis_unit}", fontsize=8)
            _despine(ax)

            if show_hist:
                ax = axes[r_plane + 1][ci]
                bins = np.linspace(0, 1, 26)
                ax.hist(
                    rho, bins=bins, color=unselected_color, edgecolor="none"
                )
                ax.hist(
                    rho[m],
                    bins=bins,
                    color=(
                        RELATIONAL_COLOR
                        if sl_["x"] > 0.5
                        else (RETINOTOPIC_COLOR if sl_["x"] < 0.5 else "0.55")
                    ),
                    edgecolor="none",
                )
                ax.set_xlim(0, 1)
                ax.set_xticks([0, 0.5, 1])
                ax.tick_params(labelsize=7)
                ax.set_xlabel(
                    "ρ" if coordinate == "angle" else "P(relational)",
                    fontsize=9,
                )
                if ci == 0:
                    ax.set_ylabel("# neurons", fontsize=9)
                _despine(ax)
    return fig

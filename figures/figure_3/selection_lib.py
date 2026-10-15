"""The relationality coordinate ρ, the neuron pool it is defined on, and the wedges.

`efficient_coding/`, `efficient_coding_schematic/` and `motor_decoding/` select their neurons
here, so they all start from the same pool and the same coordinate.

The coordinate
--------------
Each neuron is placed on a plane whose axes are the two models' **held-out excess accuracy**
from the model comparison — the share of held-out trial-pair orderings each model calls
correctly, above the coin flip that is the constant model — and ρ is the polar angle::

    rho = (2/pi) * arctan2(relational_excess+, retinotopic_excess+)    0 = retinotopic, 1 = relational

Both axes are the same kind of object, goodness of fit above a null, measured out of sample.

The axes come from `model_comparison/comparison_lib.py` — the **sign-only** comparison, which
is the only model comparison figure 3 makes. Each model is scored on the orderings it predicts
and on nothing else, so the coordinate inherits that: a neuron's position is decided by how
often each model gets the direction of a rate difference right, never by how confident either
was.

Why the exclusion rule is load-bearing
--------------------------------------
`arctan2(0, 0) = 0`, so every neuron that **neither** model fits lands at rho = 0 and is
indistinguishable from a genuinely retinotopic one. With no criterion at all (the 3186 neurons
with enough 2-object trials), 16.6% sit at exactly rho = 0 and 14.4% at exactly rho = 1.

The rule used is the **shared model-agnostic criterion** of `figures/config.py`
(:func:`config_pool`), which is what the rest of the paper selects on: of the 1339 neurons it
keeps, 7.5% sit at exactly 0 or 1. The titration additionally requires both models to beat
chance out of sample (:func:`min_fit_cut`).

Rank for sampling, raw for plotting
-----------------------------------
The wedge is taken on **rho ranked within subject**; the figure is drawn against **raw rho**.
The two animals' raw rho distributions have very different shapes, so a fixed wedge in raw rho
selects a different fraction of each population and the two sweeps are not comparable. Ranking
fixes that. Raw rho is what a reader can interpret, so it is what the axis shows.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_FIGURES = _HERE.parent
for _p in (str(_FIGURES), str(_HERE / "model_comparison")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import stats_lib as sl  # noqa: E402  figure_3/model_comparison/stats_lib.py

SUBJECTS = ["Perle", "Elgar"]
SUBJECT_LABELS = {"Perle": "Monkey P", "Elgar": "Monkey E"}


# ── The coordinate ────────────────────────────────────────────────────────────
def _pos(units, col):
    return np.maximum(units[col].to_numpy(float), 0.0)


def rho_angle(units):
    """Raw ρ in [0, 1]: 0 on the retinotopic axis, 1 on the relational axis, 0.5 the diagonal.

    Rectified at zero: a *negative* excess accuracy means the model is worse than a coin, which is not evidence of the other model and must not be
    allowed to swing the angle past the axis.
    """
    return (
        2.0
        * np.arctan2(
            _pos(units, "relational_excess"), _pos(units, "retinotopic_excess")
        )
        / np.pi
    )


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


def min_fit_cut(units, min_fit, verbose=True):
    """Keep only neurons whose held-out fit strength is at least ``min_fit`` on **both** axes.

    This is the cut that addresses the coordinate's real weakness. The polar angle is
    scale-free in the interior of the plane, but its **extremes are a dumping ground for
    failed fits**: 31.0% of the fitted neurons (7.5% of the config pool) sit at exactly
    rho = 0 or 1, and every one of those got
    there by rectification -- one model's held-out accuracy came out at or below chance and was
    clipped -- rather than by a large ratio. A neuron whose retinotopic fit is merely negative
    noise is otherwise placed at "purely relational". Measured consequence: the extreme wedges
    carry about 30% less explainable variance than the middle, which biases the titration
    toward a dip for reasons that have nothing to do with complementarity.

    Requiring *both* axes to clear the bar removes the rectified neurons: at a threshold of 0
    it keeps only neurons where both models beat a coin flip out of sample, so every
    survivor's angle is a ratio of two real fits rather than an artifact of clipping. (Asking
    only that *either* model clear it would keep every neuron one model fits, including the
    ones whose other fit is negative and clipped, and barely touch the extremes.)

    The cut is symmetric in the two models, so it cannot prefer one -- which is what makes it
    safe to apply before the angle is taken. ``min_fit=None`` applies no cut.
    """
    if min_fit is None:
        return units
    r = np.minimum(
        fit_strength(units, "retinotopic"), fit_strength(units, "relational")
    )
    keep = np.isfinite(r) & (r >= min_fit)
    out = units[keep].copy()
    if verbose:
        print(
            f"min fit strength >= {min_fit:g} on BOTH axes: "
            f"{int(keep.sum())} of {len(units)} kept"
        )
        for s_ in ordered_subjects(units):
            m = (units["subject"] == s_).to_numpy()
            print(
                f"  {SUBJECT_LABELS.get(s_, s_)}: {int(keep[m].sum())} of {int(m.sum())}"
            )
    return out


def config_pool(units, verbose=True):
    """The shared, **model-agnostic** exclusion of `figures/config.py`: 1339 neurons.

    ``config.py``'s criterion is a kernel-smoother test of whether the neuron has *any*
    2-object tuning, which is independent of the retinotopic-versus-relational question. A cut
    defined from the two models being contrasted would preferentially keep neurons that one of
    those two models happens to fit — a circularity this avoids.

    It is the same criterion every population figure in `figures` starts from.
    """
    return sl.select_units(units, verbose=verbose)


# ── Rank within subject, and the wedge ────────────────────────────────────────
def add_rank(units):
    """Rank ``rho`` to [0, 1] **within each subject**, as ``rho_rank``.

    The wedge is taken on this, not on raw ρ. The animals' raw distributions are shaped very
    differently, so a fixed raw wedge selects a different fraction of each population and the
    sweeps stop being comparable. Ranking makes sweep fraction mean "this quantile of *this*
    animal's population", which is what the titration is actually varying.
    """
    u = units.copy()
    u["rho_rank"] = np.nan
    for s in u["subject"].unique():
        m = (u["subject"] == s).to_numpy()
        v = u.loc[m, "rho"].to_numpy(float)
        u.loc[m, "rho_rank"] = (
            pd.Series(v).rank(method="average").to_numpy() - 0.5
        ) / len(v)
    return u


def angle_range_from_mean(x):
    """The wedge at sweep fraction ``x``.

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


def population_rho(units):
    """The population's own mean raw ρ per subject, with a 2000-draw neuron-level bootstrap CI.

    The CI is over **neurons** — the quantity is a property of the recorded population, and the
    uncertainty that matters is which neurons happened to be recorded.
    """
    n_boot = 2000
    rng = np.random.default_rng(0)
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


def prepared_pool(units, exclusion="config", verbose=True, min_fit=None):
    """Apply the exclusion, attach ``rho`` and ``rho_rank``. The one entry point.

    ``exclusion='config'`` is the shared model-agnostic criterion of `figures/config.py`
    (:func:`config_pool`); ``'none'`` applies no criterion. ``min_fit`` is the per-axis fit
    cut (:func:`min_fit_cut`), off by default. ``rho`` is :func:`rho_angle`.
    """
    if exclusion == "config":
        kept = config_pool(units, verbose=verbose)
    elif exclusion == "none":
        kept = units.copy()
    else:
        raise ValueError(
            f"exclusion must be 'config' or 'none', got {exclusion!r}"
        )
    kept = min_fit_cut(kept, min_fit, verbose=verbose).copy()
    kept["rho"] = rho_angle(kept)
    return add_rank(kept)

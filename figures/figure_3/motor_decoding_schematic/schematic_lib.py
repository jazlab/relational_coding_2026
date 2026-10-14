r"""Which 1-object receptive-field locations does one 2-object stimulus drive?

The question
------------
Fix a single 2-object stimulus ``(θ0, θ1)``. Take a population of real neurons. For each
neuron ask one number:

    ratio  =  (its firing rate for THIS stimulus)  /  (its average 2-object firing rate)

and then plot that number **at the neuron's own 1-object receptive-field direction**. Above
1 means this stimulus drives the neuron harder than its average 2-object stimulus does;
below 1 means it drives it less. Reading the resulting ring answers "for which receptive-field
locations does this stimulus cause an increase in firing?" — which is the question a motor
decoder's bias is downstream of.

The three pieces, and where each is copied from
-----------------------------------------------
1. **Which neurons.** ``candidate_units`` + ``band_positions`` — the candidate pool and the
   retinotopic↔relational **wedge** on the R² plane.
   ``BIAS`` (its ``x`` / ``PERCENTILE_MEAN``) slides the population from purely retinotopic
   (0) through everything (0.5) to purely relational (1). Unlike the decoder notebooks
   nothing is *drawn* from the wedge — **every** neuron in it is used, as asked.

2. **Rotated replicas.** ``replica_specs`` / ``replica_preferred_directions`` — the
   construction ``../motor_decoding/population_lib.py`` also uses. Recorded receptive fields
   are not uniform on the ring, so a plot of "response vs. RF direction" built from them
   alone would have holes and clumps. Each real unit is therefore entered ``num_replicas``
   times, rotated by ``2πk/num_replicas`` (and once more per reflection, optionally), so the
   RF directions tile the ring evenly. A replica is **not** new data: it carries the same
   trials as its unit, evaluated at the correspondingly rotated (and possibly reflected)
   stimulus. Copy ``c`` responds to a stimulus ``θ`` through its unit at ``s_c·θ + offset_c``
   and therefore has effective preferred direction ``μ_c = s_c·(μ_u − offset_c)``.

3. **The 1-object RF direction** ``μ_u``. ``oneobj_fits`` — the ``oneobj_theta_mean`` column
   of ``xval_units.csv``, the same 1-object Poisson-cosine phase
   ``../model_comparison/rf_orientation.ipynb`` measures its 2-object shifts against. It is
   already a **circular** bootstrap mean upstream, and ``oneobj_p``
   (:func:`cosine_fit_pvalue`) says whether that direction is real at all.

The response estimate
---------------------
A neuron's rate for a queried stimulus is the **swap-invariant circular-Gaussian
kernel-weighted mean of its own recorded 2-object trials** (:func:`bump_weights_2obj`) —
"some smoothing" of real data, with no model in between. ``SAMPLE_SIGMA_DEG`` is the kernel
width. Swap-invariance means the estimate does not depend on which object is labelled 0.

Cache dependency
----------------
Everything is read from the shared caches:

    cache/figures/shared_inputs/
        firing_rates/{subject}_{session}.csv.gz   per-trial, per-unit firing rates
        modulation.csv                            per-neuron 2-object modulation metrics

which is the tree ``../motor_decoding`` reads, so this directory and that one select from
exactly the same population.

Self-containedness
------------------
Per ``figures/FORMATTING.md`` this directory owns its code: the pool filters, the wedge,
the trial loading, the kernel weights and the replica construction are **copied** from
``../motor_decoding/population_lib.py`` rather than imported, so the two cannot drift apart
silently. Only the last section — the ratio table and the ring figures — is new.
"""

from __future__ import annotations

import sys
from pathlib import Path
from pathlib import Path as _Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import f as _f_dist

# Resolved from THIS FILE, not from the working directory: `sys.path.append('..')` only works
# when the notebook happens to run from this directory, so importing the module from anywhere
# else failed with ModuleNotFoundError.
# Resolved relatively again since 2026-09-11: it used to be pinned by name at
# `figure_3_deprecated` because two `figure_3` trees existed and one level up would have
# imported the wrong `common` — and a different cache tree with it — silently. There is now one
# figure 3, so the relative path is unambiguous and survives the directory being renamed.
_FIGURE_3 = _Path(__file__).resolve().parent.parent
if str(_FIGURE_3) not in sys.path:
    sys.path.insert(0, str(_FIGURE_3))
import common
from common import drop_east_west_arrays

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = (_HERE / "../../..").resolve()  # repository root

# The **model-free inputs** are shared, not rebuilt: per-trial firing rates and the
# kernel-smoother modulation test know nothing about any model, so every version reads the
# identical files and cannot drift. Only `rho` is this figure's own, and it comes from the model
# comparison's held-out accuracies via `../selection_lib.py` — that substitution is the whole of
# this port.
_SHARED = _ROOT / "cache/figures/shared_inputs"
FR_DIR = _SHARED / "firing_rates"
MODULATION_PATH = _SHARED / "modulation.csv"
CACHE_DIR = _SHARED  # kept: other helpers resolve paths from it
XVAL_UNITS_PATH = (
    _ROOT / "cache/figures/figure_3/model_comparison" / "xval_units.csv"
)

_BUILD_HINT = (
    "build the firing rates with `python firing_rates.py` in figures/ and "
    "`modulation.csv` with `python run_modulation_cache.py` in ../model_comparison/"
)

MODELS = ("const", "cw", "ioa")
KERNEL_SCORES = ("cv_r2", "dev")
R2_KINDS = ("raw", "fraction")
DISPLAY = {
    "const": "Retinotopic",
    "cw": "Relational",
    "ioa": "Proximity",
    "cv_r2": "kernel ceiling",
    "dev": "kernel deviance",
}
SUBJECT_ORDER = ["Perle", "Elgar"]
SUBJECT_TITLE = {"Perle": "Monkey P", "Elgar": "Monkey E"}

MIN_CEILING = 1e-3  # floor for the 'fraction' R² denominator
_ID_COLS = ["subject", "session", "unit_id"]


def subject_title(s):
    return SUBJECT_TITLE.get(s, s)


def circ_dist(a, b):
    """Circular arc distance in [0, π]; broadcasts."""
    d = np.abs(np.asarray(a, float) - np.asarray(b, float)) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Which neurons — the pool and the retinotopic ↔ relational wedge
# ══════════════════════════════════════════════════════════════════════════════
def load_modulation(path=MODULATION_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — {_BUILD_HINT}.")
    return pd.read_csv(path)


def r2_column(model):
    """The ``modulation.csv`` column holding ``model``'s cross-validated variance
    explained (or the kernel score itself, for ``'cv_r2'`` / ``'dev'``)."""
    if model in KERNEL_SCORES:
        return model
    if model not in MODELS:
        raise ValueError(
            f"model must be one of {MODELS + KERNEL_SCORES} (got {model!r})"
        )
    return f"{model}_cv_r2"


def r2_values(df, model, r2_kind="raw"):
    """``model``'s per-unit R² for every row of ``df``. ``'raw'`` is the cross-validated
    variance explained as cached; ``'fraction'`` divides it by the kernel-smoother ceiling
    ``cv_r2`` (the total explainable variance)."""
    if r2_kind not in R2_KINDS:
        raise ValueError(
            f"r2_kind must be one of {R2_KINDS} (got {r2_kind!r})"
        )
    v = df[r2_column(model)].to_numpy(float)
    if r2_kind == "raw" or model in KERNEL_SCORES:
        return v
    return v / np.maximum(df["cv_r2"].to_numpy(float), MIN_CEILING)


def r2_label(model, r2_kind="raw"):
    name = DISPLAY.get(model, model)
    return (
        f"{name} CV R²"
        if (model in KERNEL_SCORES or r2_kind == "raw")
        else f"{name} R² / ceiling"
    )


def cosine_fit_pvalue(r2, n_trials):
    """Significance of a neuron's 1-object cosine tuning fit, from its R² and trial count.

    ``r = b + c1·cos θ + c2·sin θ`` is an ordinary linear model whose two directional
    coefficients are what "is this neuron tuned?" asks about, so under the null the
    regression F statistic ``F = (R²/2)/((1 − R²)/(n − 3))`` is F(2, n − 3) and the p-value
    is its survival function."""
    r2 = np.clip(np.asarray(r2, float), 0.0, 1.0 - 1e-12)
    df2 = np.maximum(np.asarray(n_trials, float) - 3.0, 1.0)
    F = (r2 / 2.0) / np.maximum((1.0 - r2) / df2, 1e-300)
    return _f_dist.sf(F, 2.0, df2)


_ONEOBJ_CACHE = {}


# `oneobj_fits` used to be defined here too, a second copy of
# `motor_decoding/population_lib.py`'s with one column renamed. It is imported now: the two were
# the same function reading the same cache, which is the arrangement that lets a fix land in one
# of them and not the other. The import also carries the port to the pairwise 1-object fits made
# on 2026-09-11, when the parametric `xval_units.csv` was deleted with `figure_3_deprecated/`.
import sys as _sys

_sys.path.insert(
    0, str(_Path(__file__).resolve().parent.parent / "motor_decoding")
)
from population_lib import oneobj_fits  # noqa: E402


def candidate_units(
    subject="all", brain_area="all", mod_df=None, verbose=True
):
    """The candidate-unit pool: every neuron passing the shared **2-object modulation
    criterion** (``figures/neuron_selection.py``), joined to its 1-object cosine fit.

    The 1-object fit is still joined on — every dot in this figure is *positioned* by
    that direction — but it is no longer a selection criterion, and neither are the
    kernel-ceiling, shuffle-p and trial-count filters this used to apply.
    """
    import sys
    from pathlib import Path

    figures = Path(__file__).resolve().parent.parent.parent
    if str(figures) not in sys.path:
        sys.path.insert(0, str(figures))
    import neuron_selection

    mod = load_modulation() if mod_df is None else mod_df.copy()
    keep = _ID_COLS + [
        "brain_area",
        r2_column("const"),
        r2_column("cw"),
        "cv_r2",
        "p",
        "n_2obj",
    ]
    df = (
        mod[keep]
        .rename(columns={"p": "mod_p"})
        .merge(oneobj_fits(), on=_ID_COLS, how="inner")
    )
    if subject != "all":
        df = df[df["subject"] == subject]
    if brain_area != "all":
        df = df[df["brain_area"] == brain_area]
    df = neuron_selection.filter_units(df, unit_col="unit_id", verbose=verbose)
    return df.reset_index(drop=True)


def angle_metric(df, x_model="const", y_model="cw", r2_kind="raw"):
    """Per-unit angle on the retinotopic-vs-relational plane, normalised to ``[0, 1]``:
    0 = on the retinotopic axis, 1 = on the relational axis, 0.5 = the diagonal.

    **This is the ported quantity.** Figure 3 builds the plane out of the two models'
    cross-validated R-squared; v2 builds it out of their held-out per-comparison dLLs, which
    are the same kind of object — goodness of fit above a null, cross-validated — and takes the
    same polar angle. If ``df`` already carries a ``rho`` column (which
    :func:`v2_candidate_units` puts there, straight from ``../selection_lib.py``), that column
    *is* the metric and is returned unchanged; the R-squared route below is kept only so the
    unported figure-3 tables still work.
    """
    if "rho" in getattr(df, "columns", ()):
        return np.asarray(df["rho"], float)
    x = np.maximum(r2_values(df, x_model, r2_kind), 0.0)
    y = np.maximum(r2_values(df, y_model, r2_kind), 0.0)
    return 2.0 * np.arctan2(y, x) / np.pi


def v2_candidate_units(subject="all", brain_area="all", verbose=True, **kw):
    """:func:`candidate_units`, but carrying **v2's** ``rho`` and its radial cut.

    The pool and the 1-object fits that position every dot are unchanged — they are model-free
    and shared. What changes is the relationality coordinate, which is taken from
    ``../selection_lib.py`` exactly as `efficient_coding/` and `motor_decoding/` take it, so all
    three v2 analyses select on the same number.
    """
    import sys as _s
    from pathlib import Path as _P

    _v2 = _P(__file__).resolve().parent.parent
    for _q in (str(_v2), str(_v2 / "model_comparison")):
        if _q not in _s.path:
            _s.path.insert(0, _q)
    import comparison_lib as _cl  # the model comparison rho is built from
    import selection_lib as sel

    pool = candidate_units(
        subject=subject, brain_area=brain_area, verbose=verbose
    )
    units = sel.add_rank(
        sel.add_coordinates(sel.config_pool(_cl.load_units(), verbose=verbose))
    )
    cols = [
        c
        for c in (
            "subject",
            "session",
            "unit_id",
            "rho",
            "rho_rank",
            "radius_z",
        )
        if c in units.columns
    ]
    out = pool.merge(
        units[cols], on=[c for c in _ID_COLS if c in cols], how="inner"
    )
    if verbose:
        print(
            f"  rho joined: {len(out)} of {len(pool)} pool neurons have a comparison fit"
        )
    return out.reset_index(drop=True)


def angle_range_from_mean(x):
    """The ``[0, 1]`` angle interval the band covers at bias ``x``: ``x < 0.5 → [0, 2x]``
    (the retinotopic side), ``x ≥ 0.5 → [2x − 1, 1]`` (the relational side), ``0.5 → [0, 1]``
    (everything). The wedges are **nested**, so the mid-sweep bands hold far more neurons
    than the ends."""
    return (0.0, 2.0 * x) if x < 0.5 else (2.0 * x - 1.0, 1.0)


def fit_strength(df, x_model="const", y_model="cw", r2_kind="raw"):
    """Joint fit strength ``sqrt(x_r2² + y_r2²)`` — the radial coordinate of the plane whose
    angular coordinate is :func:`angle_metric`."""
    return np.hypot(
        r2_values(df, x_model, r2_kind), r2_values(df, y_model, r2_kind)
    )


def band_positions(
    pool_df, bias, r2_min=0.0, x_model="const", y_model="cw", r2_kind="raw"
):
    """Row positions of ``pool_df`` inside the band: the :func:`angle_range_from_mean` wedge
    at ``bias``, **and** at least ``r2_min`` from the origin.

    The two cuts are orthogonal. The angle is a *ratio*, so it only means anything once the
    neuron is explained by something: near the origin both R² are near 0 and the angle is
    noise, which would fill a supposedly "retinotopic" wedge with unexplained neurons.

    Returns ``(positions, (lo, hi))``."""
    lo, hi = angle_range_from_mean(bias)
    a = angle_metric(pool_df, x_model, y_model, r2_kind)
    rad = fit_strength(pool_df, x_model, y_model, r2_kind)
    positions = np.where(
        np.isfinite(a)
        & (a >= lo)
        & (a <= hi)
        & np.isfinite(rad)
        & (rad >= r2_min)
    )[0]
    if not len(positions):
        raise ValueError(
            f"No neurons in the wedge [{lo:.2f}, {hi:.2f}] with joint R² ≥ "
            f"{r2_min:g} at bias={bias}."
        )
    return positions, (lo, hi)


# ══════════════════════════════════════════════════════════════════════════════
# 2. The recorded trials
# ══════════════════════════════════════════════════════════════════════════════
def session_unit_trials(subject, session, completed_only=True):
    """One session's firing-rate cache → per unit, its cleaned 2-object trials.

    The firing rate is the mean of the stimulus and delay windows. The
    over-represented east–west antipodal pair (one object ≈ 0°, the other ≈ 180°, repeated
    ~1500×) is dropped, so it cannot dominate the kernel-weighted response estimate.
    """
    path = FR_DIR / f"{subject}_{session}.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — {_BUILD_HINT}.")
    df = pd.read_csv(path)
    if completed_only:
        df = df[df["completed"].astype(bool)]
    df = df[df["num_objects"] == 2]
    df = df.assign(
        _fr=df[["stimulus_fr", "delay_fr"]].astype(float).mean(axis=1)
    )
    out = {}
    for uid, g in df.groupby("unit"):
        fr = g["_fr"].to_numpy(float)
        t0 = g["object_0_theta"].to_numpy(float) % (2 * np.pi)
        t1 = g["object_1_theta"].to_numpy(float) % (2 * np.pi)
        ok = np.isfinite(fr) & np.isfinite(t0) & np.isfinite(t1)
        t0, t1, fr = t0[ok], t1[ok], fr[ok]
        # the repeated east-west pair, dropped in every analysis (see figures/trial_filters.py)
        t0, t1, fr = drop_east_west_arrays(t0, t1, fr)
        out[int(uid)] = (t0, t1, fr)
    return out


def load_units_2obj_trials(units_df, verbose=True):
    """``[(t0, t1, fr), ...]``, one per row of ``units_df`` and in row order. Each session's
    cache is read **once** and shared across that session's units."""
    cache = {}
    out = []
    for _, row in units_df.iterrows():
        key = (row["subject"], row["session"])
        if key not in cache:
            cache[key] = session_unit_trials(
                *key,
            )
        out.append(cache[key][int(row["unit_id"])])
    if verbose:
        n_trials = int(np.median([len(fr) for _, _, fr in out])) if out else 0
        print(
            f"loaded {len(out)} units from {len(cache)} sessions "
            f"(median {n_trials} 2-object trials each)"
        )
    return out


def bump_weights_2obj(t0, t1, query_t0, query_t1, sigma_deg):
    """``[n_query, n_trials]`` **swap-invariant** circular-Gaussian weights of each recorded
    trial for each query pair; rows sum to 1.

    The distance between a query and a trial is the smaller of the direct and
    object-swapped per-object arc-distance sums, so the estimate does not depend on which
    object is labelled 0 — matching the task's own order symmetry."""
    query_t0 = np.atleast_1d(np.asarray(query_t0, float))
    query_t1 = np.atleast_1d(np.asarray(query_t1, float))
    d_direct = circ_dist(query_t0[:, None], t0[None, :]) + circ_dist(
        query_t1[:, None], t1[None, :]
    )
    d_swap = circ_dist(query_t0[:, None], t1[None, :]) + circ_dist(
        query_t1[:, None], t0[None, :]
    )
    e = -0.5 * (np.minimum(d_direct, d_swap) / np.radians(sigma_deg)) ** 2
    e -= e.max(axis=1, keepdims=True)  # numerically stable per query row
    w = np.exp(e)
    return w / w.sum(axis=1, keepdims=True)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Rotated (and reflected) replicas
# ══════════════════════════════════════════════════════════════════════════════
def replica_specs(num_replicas, include_reflections=False):
    """Per-unit copies as ``(offset, reflect)`` pairs: ``num_replicas`` evenly spaced
    rotations in ``[0, 2π)``, plus — with ``include_reflections`` — the same rotations of the
    **reflected** neuron (query angles negated before the offset). A unit thus yields
    ``num_replicas`` copies, or ``2·num_replicas``."""
    offsets = 2 * np.pi * np.arange(num_replicas) / num_replicas
    specs = [(off, False) for off in offsets]
    if include_reflections:
        specs += [(off, True) for off in offsets]
    return specs


def replica_preferred_direction(mu, offset, reflect):
    """Effective 1-object preferred direction of one copy. The copy responds to a stimulus
    ``θ`` through its unit at ``s·θ + offset`` (``s = −1`` if reflected), which peaks where
    ``s·θ + offset = μ``, i.e. at ``θ = s·(μ − offset)``."""
    s = -1.0 if reflect else 1.0
    return (s * (mu - offset)) % (2 * np.pi)


# ══════════════════════════════════════════════════════════════════════════════
# 3b. Idealized neurons — the alternative to real data
# ══════════════════════════════════════════════════════════════════════════════
# TWO families, and they share one receptive field. Both have the SAME 1-object tuning
#
#       rate_1obj(θ)  =  b + a·g(θ − μ)
#
# with the same bump ``g`` and the same preferred direction ``μ``. They differ in exactly one
# thing: **which vector gets fed into that receptive field when there are two objects.**
#
#   retinotopic   the vector from FIXATION to each object.  The 2-object receptive field is
#                 therefore literally the 1-object one, integrated over both objects:
#                     b + a·[g(θ0 − μ) + g(θ1 − μ)]
#
#   relational    the vector from the OTHER OBJECT to each object — equivalently, from the
#                 pair's midpoint, since  A − midpoint = (A − B)/2  points the same way:
#                     b + a·[g(r_AB − μ) + g(r_BA − μ)],   r_AB = ∠(A − B),  r_BA = r_AB + π
#                 The receptive field is unchanged and is oriented parallel to the 1-object
#                 preferred direction; only the reference point moved. This is the α = 1 end
#                 of the relational-bias model of ../../figure_4/neuron_models.
#
# Both are exchange-symmetric, ``r(θ0, θ1) = r(θ1, θ0)``, like the task.
#
# Useful closed form for the relational family. With objects on the unit ring,
#     A − B  ∝  (−sin S, cos S)·2·sin((θ0 − θ1)/2),      S = (θ0 + θ1)/2,
# so ``r_AB = S ± π/2`` and the drive collapses to
#     b + a·[g(S + π/2 − μ) + g(S − π/2 − μ)],
# a function of the pair's midpoint direction S with **two lobes, at μ ± 90°**. A relational
# unit is therefore driven hardest when the object-to-object axis points along μ — which is
# the same sentence as "the vector between the objects is fed into the receptive field",
# and is what the schematic is there to make checkable.
IDEAL_KINDS = ("retinotopic", "relational")
IDEAL_MIN_SEP_DEG = 80.0  # no recorded 2-object trial is closer than this


def kappa_from_width(width_deg):
    """von Mises concentration whose bump ``exp(κ(cos x − 1))`` is at half height at
    ``width_deg``. Small width = sharp; ``width_deg → 90°`` is cosine-like."""
    w = np.radians(float(width_deg))
    return float(np.log(2.0) / max(1.0 - np.cos(w), 1e-9))


def vm_bump(x, width_deg):
    """Circular bump with peak 1 at ``x = 0`` and half-height half-width ``width_deg``.
    This is ``g`` — the receptive field itself, shared by both families and by both the
    1- and the 2-object conditions."""
    return np.exp(kappa_from_width(width_deg) * (np.cos(x) - 1.0))


def relative_direction(t0, t1):
    """``∠(A − B)`` for objects at angles ``t0``, ``t1`` on the unit ring — the direction of
    object 0 *seen from object 1* (equivalently, seen from the pair's midpoint).

    Computed from the Cartesian difference rather than from the closed form, so it stays
    correct at every configuration including the antipodal one."""
    t0 = np.asarray(t0, float)
    t1 = np.asarray(t1, float)
    return np.arctan2(np.sin(t0) - np.sin(t1), np.cos(t0) - np.cos(t1))


def ideal_raw_rate(spec, t0, t1):
    """The unit's **2-object** rate, before normalisation. Hz.

    The receptive field ``g(· − μ)`` is the same object as in :func:`ideal_raw_rate_1obj`;
    only the vector fed into it changes with ``kind``."""
    kind, mu, w = spec["kind"], spec["mu"], spec["width_deg"]
    b, a = spec["baseline"], spec["amplitude"]
    t0 = np.asarray(t0, float)
    t1 = np.asarray(t1, float)
    if kind == "retinotopic":
        # Fixation → object, for each object. The 1-object RF, integrated over both.
        return b + a * (vm_bump(t0 - mu, w) + vm_bump(t1 - mu, w))
    if kind == "relational":
        # Other object → object, for each object. Same RF, moved reference point.
        r01 = relative_direction(t0, t1)
        return b + a * (vm_bump(r01 - mu, w) + vm_bump(r01 + np.pi - mu, w))
    raise ValueError(f"kind must be one of {IDEAL_KINDS} (got {kind!r})")


def ideal_stimulus_moments(spec, n_grid=72, min_sep_deg=IDEAL_MIN_SEP_DEG):
    """``(mean, sd)`` of a unit's rate over **the task's stimulus distribution**: ``θ0``
    uniform and the separation uniform on ``[min_sep_deg, 180°]``.

    That floor is not cosmetic. No recorded 2-object trial has the objects closer than
    ≈80°, so averaging an idealized unit over *all* separations would compare it against
    stimuli the animal never saw — and the families differ most exactly there.
    """
    g = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    a, b = np.meshgrid(g, g, indexing="ij")
    sep = np.degrees(circ_dist(a, b))
    w = (sep >= min_sep_deg - 1e-9).astype(float)
    w /= w.sum()
    r = ideal_raw_rate(spec, a, b)
    mean = float(np.sum(w * r))
    sd = float(np.sqrt(max(np.sum(w * r**2) - mean**2, 0.0)))
    return mean, sd


def ideal_stimulus_mean(spec, n_grid=72, min_sep_deg=IDEAL_MIN_SEP_DEG):
    """The unit's stimulus-distribution mean **after** normalisation — the mean of exactly
    the rate :func:`ideal_rate` returns.

    :func:`ideal_stimulus_moments` deliberately reports the *raw* moments, because that is
    what the normalisation is calibrated from; anything comparing against the plotted rate
    wants this one instead."""
    m, _ = ideal_stimulus_moments(spec, n_grid, min_sep_deg)
    return float(spec["offset"] + spec["gain"] * m)


IDEAL_NORMALIZATIONS = ("moments", "peak", "range", None)


def ideal_units(
    n_units,
    composition=None,
    width_deg=None,
    baseline=2.0,
    amplitude=5.0,
    normalize="moments",
    mean_rate=10.0,
    mod_sd=3.0,
    peak_rate=20.0,
    floor_rate=2.0,
    peak_stimulus=None,
    n_grid=72,
    min_sep_deg=IDEAL_MIN_SEP_DEG,
    min_rate=0.05,
):
    """A population of idealized neurons → ``(units_df, specs)``.

    ``composition`` is ``{kind: fraction}`` over :data:`IDEAL_KINDS` (normalised to sum to
    1), so a pure population is ``{'retinotopic': 1.0}`` and a mixture is the idealized
    analogue of the real path's ``BIAS``. Within each kind the preferred directions **tile
    the full circle**, so every sub-population stays unbiased over orientation however the
    split falls.

    ``normalize`` puts the families on a common rate budget — an affine rescale
    ``r → offset + gain·r`` that always fixes the **stimulus-mean rate** at ``mean_rate`` and
    differs in what it matches on top of that:

    ``'moments'``  a common stimulus **SD** of the rate, ``mod_sd``. Equalises how much each
                   unit varies on average, which is the right control when comparing how much
                   *information* a family carries.
    ``'peak'``     a common **maximum** rate, ``peak_rate``. Equalises how hard each family
                   can be driven, which is the right control when the figure is read as "how
                   bright does this get" — two families with the same rate SD can still have
                   different peaks, because a bump with two lobes spreads the same variance
                   over twice as much of the ring.

                   ``peak_stimulus=(θ0, θ1)`` is what makes that match hold **in the
                   figure**: the maximum is then taken over the rotations of that one pair —
                   exactly the set of responses the ring plots — rather than over every
                   supported stimulus. The two are not the same. A retinotopic unit's global
                   best stimulus is the closest pair the task allows (both objects in one
                   RF), so calibrating on the whole set leaves it short of ``peak_rate`` at
                   any wider separation, and the two families come out ~10% apart on a figure
                   drawn at 90°.

                   With ``peak_stimulus`` set the scale is calibrated on the rotations of
                   that one pair and is only guaranteed sane there — at other separations the
                   larger gain can push a rate negative. Pair it with ``baseline='rotations'``
                   so the denominator comes from that same set and nothing in the figure
                   depends on a stimulus the scale was not calibrated for.
    ``'range'``    a common **floor and peak**, ``floor_rate`` and ``peak_rate``: each
                   family's response over the calibration set is mapped onto exactly that
                   interval. The strictest of the three, and the one to use when two families
                   are drawn **on one colour scale in Hz** — they then span the same range, so
                   the colours compare pattern rather than dynamic range, and the rate cannot
                   go negative by construction. Note it does *not* equalise the mean, so the
                   ``'ratio'`` view's denominator differs between families.
    ``None``       no rescale; each family keeps its raw ``baseline`` and ``amplitude``.

    It matters most whenever the population is **mixed**, but also whenever two families are
    drawn on one colour scale. The same offset and gain are applied to the 1-object rate, so
    a unit is one neuron in both conditions.

    ``units_df`` carries a ``theta`` column — each unit's 1-object receptive-field direction,
    the coordinate every dot is positioned by — alongside ``subject`` / ``session`` /
    ``unit_id``, so the rest of the pipeline cannot tell it from a real pool. **``theta`` is
    ``μ`` and means the same thing for both families**: the preferred direction of the one
    receptive field they share.

    One asymmetry worth knowing. A relational unit's *2-object* response is invariant under
    ``μ → μ + 180°`` (the two lobes at ``μ ± 90°`` simply swap), while its 1-object response
    is not — the same identifiability note the fitted relational model carries. Preferred
    directions are tiled over the full circle regardless, because ``μ`` and ``μ + 180°`` are
    genuinely different neurons in the 1-object condition, which is the condition the figure
    positions dots by.
    """
    composition = composition or {"retinotopic": 1.0}
    width_deg = {
        **{"retinotopic": 60.0, "relational": 60.0},
        **(width_deg or {}),
    }
    bad = set(composition) - set(IDEAL_KINDS)
    if bad:
        raise ValueError(
            f"unknown kind(s) {sorted(bad)}; composition keys must be "
            f"{IDEAL_KINDS}"
        )
    total = float(sum(composition.values()))
    if total <= 0:
        raise ValueError(
            "composition fractions must sum to something positive"
        )

    counts = {
        k: int(round(n_units * v / total))
        for k, v in composition.items()
        if v > 0
    }
    if counts:  # give the rounding remainder to the largest group
        counts[max(counts, key=counts.get)] += n_units - sum(counts.values())
    counts = {k: n for k, n in counts.items() if n > 0}

    specs, rows = [], []
    for kind, n in counts.items():
        mus = 2 * np.pi * np.arange(n) / n
        for i in range(n):
            spec = dict(
                kind=kind,
                mu=float(mus[i]),
                width_deg=float(width_deg[kind]),
                baseline=float(baseline),
                amplitude=float(amplitude),
                gain=1.0,
                offset=0.0,
            )
            if normalize not in IDEAL_NORMALIZATIONS and normalize is not True:
                raise ValueError(
                    f"normalize must be one of {IDEAL_NORMALIZATIONS} "
                    f"(got {normalize!r})"
                )
            if normalize:
                mode = "moments" if normalize is True else normalize
                m, sd = ideal_stimulus_moments(spec, n_grid, min_sep_deg)
                # Everything below is measured over the SUPPORTED separations only: the task
                # never goes below min_sep_deg, so a peak or a minimum outside that range is
                # a stimulus the animal never saw and must not set the scale.
                gg = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
                aa, bb = np.meshgrid(gg, gg, indexing="ij")
                sup = np.degrees(circ_dist(aa, bb)) >= min_sep_deg - 1e-9
                raw = ideal_raw_rate(spec, aa, bb)[sup]
                # `check` is the set the rates must stay positive on, and it is the same
                # set the scale is calibrated from — with `peak_stimulus` that is the
                # rotations of the shown pair (all the figure ever queries), otherwise every
                # supported stimulus.
                check, where = raw, f"separations >= {min_sep_deg:g} deg"
                if mode in ("peak", "range") and peak_stimulus is not None:
                    off = np.linspace(0, 2 * np.pi, 720, endpoint=False)
                    check = ideal_raw_rate(
                        spec,
                        (peak_stimulus[0] + off) % (2 * np.pi),
                        (peak_stimulus[1] + off) % (2 * np.pi),
                    )
                    where = "the rotations of peak_stimulus"
                if mode == "range":
                    span = float(check.max()) - float(check.min())
                    spec["gain"] = (peak_rate - floor_rate) / max(span, 1e-12)
                    spec["offset"] = floor_rate - spec["gain"] * float(
                        check.min()
                    )
                else:
                    if mode == "peak":
                        spec["gain"] = (peak_rate - mean_rate) / max(
                            float(check.max()) - m, 1e-12
                        )
                    else:
                        spec["gain"] = mod_sd / max(sd, 1e-12)
                    spec["offset"] = mean_rate - spec["gain"] * m
                lo = float((spec["offset"] + spec["gain"] * check).min())
                if lo < min_rate:
                    raise ValueError(
                        f"normalised {kind} rates reach {lo:.3g} Hz < min_rate={min_rate} "
                        f"over {where} — lower "
                        + (
                            "peak_rate"
                            if mode in ("peak", "range")
                            else "mod_sd"
                        )
                        + " or raise mean_rate"
                        + (
                            " (or use normalize='range', which fixes the floor too)"
                            if mode == "peak"
                            else ""
                        )
                    )
            specs.append(spec)
            rows.append(
                dict(
                    subject="ideal",
                    session=kind,
                    unit_id=len(specs) - 1,
                    kind=kind,
                    theta=spec["mu"],
                    width_deg=spec["width_deg"],
                )
            )
    return pd.DataFrame(rows), specs


def ideal_rate(spec, t0, t1):
    """The unit's 2-object rate after normalisation — what the figure actually uses."""
    return spec["offset"] + spec["gain"] * ideal_raw_rate(spec, t0, t1)


# ── The schematic: what each family feeds into its receptive field ────────────


# ══════════════════════════════════════════════════════════════════════════════
# 4. The ratio table — this directory's own computation
# ══════════════════════════════════════════════════════════════════════════════
BASELINES = ("trials", "rotations", "grid")


def kernel_responders(units_trials, sigma_deg=10.0):
    """Turn recorded trials into **responders** — one callable per neuron, mapping query
    angles to rates.

    ``respond(q0, q1)`` is the swap-invariant kernel-weighted mean of that neuron's own
    2-object trials. ``respond.trial_rates`` carries the raw recorded rates, which is what
    the ``'trials'`` baseline averages; it is the one thing an idealized unit cannot supply.
    """
    out = []
    for t0, t1, fr in units_trials:

        def respond(q0, q1, t0=t0, t1=t1, fr=fr):
            with np.errstate(all="ignore"):
                return bump_weights_2obj(t0, t1, q0, q1, sigma_deg) @ fr

        respond.trial_rates = fr
        out.append(respond)
    return out


def ideal_responders(specs, min_sep_deg=IDEAL_MIN_SEP_DEG):
    """The same interface for idealized units — ``respond(q0, q1)`` evaluates the analytic
    rate. ``respond.trial_rates`` is ``None``: an idealized unit has no trials, so the
    ``'trials'`` baseline falls back to the task's stimulus-distribution mean (see
    :func:`ratio_table`)."""
    out = []
    for spec in specs:

        def respond(q0, q1, spec=spec):
            return ideal_rate(spec, q0, q1)

        respond.trial_rates = None
        respond.spec = spec
        respond.min_sep_deg = min_sep_deg
        out.append(respond)
    return out


def _baseline_rate(respond, kind, theta0, theta1, specs, n_grid):
    """One neuron's reference rate, the denominator of the ratio. See :func:`ratio_table`."""
    if kind == "trials":
        if getattr(respond, "trial_rates", None) is not None:
            return float(np.mean(respond.trial_rates))
        # An idealized unit has no trials. The honest analogue of "its average 2-object
        # firing rate" is its mean over the stimulus distribution the task actually used —
        # theta0 uniform, separation uniform on [min_sep_deg, 180]. It must be the mean of
        # the rate the NUMERATOR uses, i.e. after normalisation, or the ratio is off by the
        # normalising gain.
        return ideal_stimulus_mean(
            respond.spec,
            n_grid,
            getattr(respond, "min_sep_deg", IDEAL_MIN_SEP_DEG),
        )
    if kind == "rotations":
        offs = np.array([off for off, _ in specs])
        signs = np.array([-1.0 if refl else 1.0 for _, refl in specs])
        q0 = (signs * theta0 + offs) % (2 * np.pi)
        q1 = (signs * theta1 + offs) % (2 * np.pi)
        return float(np.mean(respond(q0, q1)))
    if kind == "grid":
        g = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
        a, b = np.meshgrid(g, g, indexing="ij")
        return float(np.mean(respond(a.ravel(), b.ravel())))
    raise ValueError(f"baseline must be one of {BASELINES} (got {kind!r})")


def ratio_table(
    units_df,
    responders,
    theta0,
    theta1,
    num_replicas=32,
    include_reflections=False,
    baseline="trials",
    grid_n=36,
    verbose=True,
):
    """One row per **(neuron, replica)**: where its receptive field points, what this
    stimulus does to it, and the ratio of the two.

    ``responders`` is a list of callables, one per row of ``units_df``, built by
    :func:`kernel_responders` from recorded trials or by :func:`ideal_responders` from
    idealized units. Everything downstream of this call is identical either way — the
    replica arithmetic, the ring, the annulus and the profile do not know which they were
    given.

    Columns
    -------
    ``subject``, ``session``, ``unit_id``  the neuron the copy came from
    ``unit_row``                           its row in ``units_df``
    ``offset``, ``reflect``                which copy of it this is
    ``mu``                                 the copy's 1-object RF direction, radians in [0, 2π)
    ``rate``                               its rate for ``(theta0, theta1)``, Hz
    ``baseline``                           its reference rate, Hz
    ``ratio``                              ``rate / baseline``

    The numerator is the responder evaluated at the rotated query
    ``(s·θ0 + offset, s·θ1 + offset)`` — evaluating a rotated copy at the fixed stimulus is
    the same as evaluating the neuron at the counter-rotated stimulus, which is the whole
    trick that lets a few hundred receptive fields cover the ring.

    The denominator is chosen by ``baseline``:

    ``'trials'``     the mean of the neuron's **recorded** 2-object firing rates. Literally
                     "its average 2-object firing rate", and free of any extrapolation —
                     but it is an average over the stimulus set the task happened to
                     present, which is not uniform. All copies of a unit share it, so the
                     ratio varies across the ring only through the numerator. For an
                     **idealized** unit there are no trials, so this becomes the mean over
                     the task's stimulus distribution (θ0 uniform, separation uniform on
                     [``IDEAL_MIN_SEP_DEG``, 180°]) — the same idea, computed rather than
                     sampled.
    ``'rotations'``  the mean of that neuron's response over **all the rotations plotted** —
                     the average of its own dots. This makes the ratio purely a statement
                     about *which orientation* of this stimulus suits the neuron best, with
                     each neuron's overall drive divided out exactly.
    ``'grid'``       the mean over a uniform ``grid_n × grid_n`` grid of ``(θ0, θ1)``.
                     Rotation-invariant like ``'rotations'``, but it includes object
                     separations the task never presented (nothing under ~80°), which is
                     kernel extrapolation for real neurons and a stimulus the animal never
                     saw for idealized ones.
    """
    specs = replica_specs(num_replicas, include_reflections)
    offs = np.array([off for off, _ in specs])
    refl = np.array([bool(r) for _, r in specs])
    signs = np.where(refl, -1.0, 1.0)
    q0 = (signs * float(theta0) + offs) % (2 * np.pi)
    q1 = (signs * float(theta1) + offs) % (2 * np.pi)

    rows = []
    for u, (respond, (_, meta)) in enumerate(
        zip(responders, units_df.iterrows())
    ):
        rates = np.asarray(respond(q0, q1), float)
        base = _baseline_rate(respond, baseline, theta0, theta1, specs, grid_n)
        mu_u = float(meta["theta"])
        rows.append(
            pd.DataFrame(
                {
                    "subject": meta["subject"],
                    "session": meta["session"],
                    "unit_id": int(meta["unit_id"]),
                    "unit_row": u,
                    "offset": offs,
                    "reflect": refl,
                    "mu": [
                        replica_preferred_direction(mu_u, o, r)
                        for o, r in specs
                    ],
                    "rate": rates,
                    "baseline": base,
                }
            )
        )
    out = pd.concat(rows, ignore_index=True)
    out["ratio"] = out["rate"] / np.where(
        np.abs(out["baseline"]) > 1e-9, out["baseline"], np.nan
    )
    if verbose:
        print(
            f"{len(responders)} units × {len(specs)} copies = {len(out)} dots"
        )
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 5. Figures
# ══════════════════════════════════════════════════════════════════════════════
def diverging_norm(values, vmin=None, vmax=None, center=1.0, percentile=98.0):
    """A norm centred on ``center`` (no change). Unset limits are taken from a symmetric
    percentile of the data, so a handful of extreme ratios cannot flatten the whole map.
    """
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if vmin is None or vmax is None:
        half = np.percentile(np.abs(v - center), percentile) if len(v) else 1.0
        half = max(float(half), 1e-6)
        vmin = center - half if vmin is None else vmin
        vmax = center + half if vmax is None else vmax
    return TwoSlopeNorm(
        vcenter=center,
        vmin=min(vmin, center - 1e-9),
        vmax=max(vmax, center + 1e-9),
    )


def _polar_circle(r_center, theta_center, radius, n=181):
    """A circle of Cartesian radius ``radius`` centred at polar ``(r_center, theta_center)``,
    returned as ``(theta, r)`` so it can be drawn on a polar axes. Drawing the patch in
    Cartesian and converting is the only way to get a marker that is actually round on a
    polar plot."""
    t = np.linspace(0, 2 * np.pi, n)
    cx, cy = r_center * np.cos(theta_center), r_center * np.sin(theta_center)
    x, y = cx + radius * np.cos(t), cy + radius * np.sin(t)
    return np.arctan2(y, x), np.hypot(x, y)


def plot_rf_ring(
    table,
    theta0,
    theta1,
    ax=None,
    figsize=(4.6, 4.6),
    cmap="coolwarm",
    vmin=None,
    vmax=None,
    center=1.0,
    clip_percentile=98.0,
    jitter=0.35,
    r_outer=1.0,
    view=1.35,
    seed=0,
    point_size=6,
    point_alpha=0.55,
    stim_color="k",
    stim_lw=1.6,
    stim_radius=0.2,
    stim_ls=(0, (2, 2)),
    show_stim=True,
    show_ring=True,
    value="ratio",
    ring_color="0.75",
    ring_lw=0.8,
    ring_ls="--",
    title=None,
    colorbar=True,
    cbar_label="Rate / mean rate",
    cbar_shrink=0.72,
    cbar_pad=0.12,
    sort_by_abs=True,
    norm=None,
    angle_labels=False,
):
    """The main figure: **one dot per (neuron, replica)**, placed at its 1-object receptive
    field direction, coloured by its firing-rate ratio.

    The angle of a dot is the only thing that carries information — the radius is pure
    ``jitter``, so that the thousands of dots at nearly the same angle are visible as a
    density rather than overplotted into one mark. The band of dots ends **exactly on
    ``r_outer``** and is ``2·jitter`` wide inwards, so with ``r_outer`` and ``view`` set to
    the stimulus panel's ring radius and axis limit the two panels' rings are the same
    circle at the same size.

    ``show_ring`` draws the dashed stimulus ring and ``show_stim`` the two dotted object
    circles — the two annotations this panel shares with the stimulus panel. Turn them off
    for a bare density.

    ``value`` picks the column coloured: ``'ratio'`` (rate ÷ that unit's mean rate, so
    ``center = 1`` is "no change") or ``'rate'`` (Hz, so ``center`` should be the population's
    mean rate). The two differ only in units — the ratio's denominator is one number per
    unit — but the colorbar's label has to match whichever is chosen.

    The colour is diverging around ``center``: **warm = this stimulus drives the neuron
    above that reference, cool = below**. Limits default to a symmetric ``clip_percentile`` of
    the data. ``sort_by_abs`` draws the dots nearest 1 first, so the strong ones are not
    buried under the null-coloured majority.

    The two black spokes are the stimulus itself, ``theta0`` and ``theta1``.
    """
    if ax is None:
        fig, ax = plt.subplots(
            figsize=figsize, subplot_kw=dict(projection="polar")
        )
    else:
        fig = ax.figure

    mu = table["mu"].to_numpy(float)
    ratio = table[value].to_numpy(
        float
    )  # 'ratio' (dimensionless) or 'rate' (Hz)
    ok = np.isfinite(mu) & np.isfinite(ratio)
    mu, ratio = mu[ok], ratio[ok]
    rng = np.random.default_rng(seed)
    r_center = r_outer - jitter
    r = r_center + rng.uniform(-jitter, jitter, size=len(mu))
    if sort_by_abs:
        order = np.argsort(np.abs(ratio - center))
        mu, ratio, r = mu[order], ratio[order], r[order]

    if norm is None:
        norm = diverging_norm(ratio, vmin, vmax, center, clip_percentile)
    sc = ax.scatter(
        mu,
        r,
        c=ratio,
        cmap=cmap,
        norm=norm,
        s=point_size,
        alpha=point_alpha,
        linewidths=0,
        zorder=3,
    )

    if show_stim:
        # The two objects, as open dotted circles ON TOP of the dots and centred on the ring
        # itself — the same marker at the same place as the discs in the stimulus panel.
        for th in (float(theta0), float(theta1)):
            t, rr = _polar_circle(r_outer, th, stim_radius)
            ax.plot(
                t,
                rr,
                ls=stim_ls,
                color=stim_color,
                lw=stim_lw,
                zorder=10,
                solid_capstyle="butt",
            )
    if show_ring:
        ax.plot(
            np.linspace(0, 2 * np.pi, 361),
            np.full(361, r_outer),
            ls=ring_ls,
            color=ring_color,
            lw=ring_lw,
            zorder=1,
        )

    ax.set_ylim(0, view)
    ax.set_yticks([])
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["0°", "90°", "180°", "270°"] if angle_labels else [])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)
    if title:
        ax.set_title(title)
    if colorbar:
        cb = fig.colorbar(sc, ax=ax, shrink=cbar_shrink, pad=cbar_pad)
        cb.set_label(cbar_label)
        cb.outline.set_visible(False)
    return fig, ax, sc


def plot_stimulus_ring(
    theta0,
    theta1,
    ax=None,
    figsize=(2.2, 2.2),
    radius=1.0,
    disc_radius=0.2,
    disc_color="0.6",
    disc_edge="k",
    disc_edge_lw=0.8,
    ring_color="0.75",
    ring_lw=1.0,
    ring_ls="--",
    cross_size=8,
    cross_lw=1.3,
    view=1.35,
    title=None,
):
    """The stimulus itself: two discs on a dashed ring, with the fixation cross at its
    centre.

    Drawn in the style of ``../efficient_coding_schematic/schematic_lib.plot_stimuli`` (one
    panel of it, without the reference circles and arrows that figure needs to compare
    stimuli), so a stimulus looks the same wherever in figure 3 it is shown.

    The objects are plain discs with **no chord between them**: the panel shows the stimulus
    as the animal sees it — two objects at two ring positions — rather than asserting the
    pair's relative vector, which is one code's reading of the stimulus and not a property
    of it. ``disc_radius`` is in data units where the ring has radius ``radius``.

    Drawn on a **polar** axes, so that this panel and the ring panels beside it are the same
    kind of axes: they then get the same box from the layout, the same ring at the same size
    for the same ``view``, and their titles land on one line without any nudging. (A plain
    axes with ``aspect='equal'`` either shrinks its box to a square — dropping its title
    below the others — or needs ``adjustable='datalim'``, which conflicts with fixed limits
    and warns.)
    """
    if ax is None:
        fig, ax = plt.subplots(
            figsize=figsize, subplot_kw=dict(projection="polar")
        )
    else:
        fig = ax.figure
    ax.plot(
        np.linspace(0, 2 * np.pi, 361),
        np.full(361, radius),
        color=ring_color,
        lw=ring_lw,
        ls=ring_ls,
        zorder=1,
    )
    ax.plot(
        [0],
        [0],
        marker="+",
        color="k",
        markersize=cross_size,
        markeredgewidth=cross_lw,
        zorder=2,
    )
    for th in (float(theta0), float(theta1)):
        t, r = _polar_circle(radius, th, disc_radius)
        ax.fill(
            t,
            r,
            facecolor=disc_color,
            edgecolor=disc_edge,
            lw=disc_edge_lw,
            zorder=4,
        )
    ax.set_ylim(0, view)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)
    if title:
        ax.set_title(title)
    return fig, ax


def plot_band_scatter(
    ax,
    pool_df,
    positions,
    band=None,
    x_model="const",
    y_model="cw",
    r2_kind="raw",
    s=10,
    point_alpha=0.3,
    band_size=16,
    band_color="tab:red",
    band_alpha=0.9,
    wedge_color="tab:blue",
    wedge_alpha=0.13,
    wedge_inner=0.0,
):
    """Which neurons the band holds, on the retinotopic-vs-relational R² plane — grey for
    the whole pool, ``band_color`` for the ones in use, and the wedge itself shaded.
    """
    x = r2_values(pool_df, x_model, r2_kind)
    y = r2_values(pool_df, y_model, r2_kind)
    lim = float(np.nanmax([np.nanmax(np.abs(x)), np.nanmax(np.abs(y))])) * 1.05
    if band is not None:
        lo, hi = band
        th = np.linspace(lo * np.pi / 2, hi * np.pi / 2, 60)
        rr = lim * np.sqrt(2)
        ax.fill(
            np.concatenate([[0], rr * np.cos(th)]),
            np.concatenate([[0], rr * np.sin(th)]),
            color=wedge_color,
            alpha=wedge_alpha,
            lw=0,
            zorder=0,
        )
        if wedge_inner > 0:
            circ = np.linspace(0, np.pi / 2, 60)
            ax.fill(
                wedge_inner * np.cos(circ),
                wedge_inner * np.sin(circ),
                color="white",
                lw=0,
                zorder=1,
            )
    ax.scatter(
        x, y, s=s, color="0.5", alpha=point_alpha, linewidths=0, zorder=2
    )
    ax.scatter(
        x[positions],
        y[positions],
        s=band_size,
        color=band_color,
        alpha=band_alpha,
        linewidths=0,
        zorder=3,
    )
    ax.axhline(0, color="0.8", lw=0.6, zorder=1)
    ax.axvline(0, color="0.8", lw=0.6, zorder=1)
    ax.set_xlabel(r2_label(x_model, r2_kind))
    ax.set_ylabel(r2_label(y_model, r2_kind))
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return len(positions)

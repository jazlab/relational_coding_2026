"""The real recorded population behind the fitted-lobe ring decoder.

Self-contained (per `figures/FORMATTING.md`): nothing here imports code from
`new_exploration/`, and nothing reads a cache generated there. Everything is read
straight from the shared caches under

    cache/figures/shared_inputs/
        firing_rates/{subject}_{session}.csv.gz   per-trial, per-unit firing rates
        modulation.csv                            per-neuron kernel modulation metrics

This module picks which real units enter the decoder, loads their real 1- and
2-object trials, and estimates the population's response to any query stimulus as a
**kernel-weighted mean of those real trials** — with `num_replicas` rotated (and
optionally reflected) copies per unit, so a handful of recorded neurons tile the whole
ring.

What comes from where
---------------------
* **Which neurons** — `modulation.csv`'s cross-validated variance explained of each
  2-object firing-rate model on held-out trials, `const_cv_r2` (retinotopic) and
  `cw_cv_r2` (relational), plus the model-agnostic kernel-smoother ceiling `cv_r2` and
  its shuffle-null `p`. The selection itself is a fit-strength filter on the const-cw
  plane followed by an angle wedge (or percentile band) — :func:`select_units_titration`,
  with :func:`subsample_units` for neuron-count matching.
* **The decoding lobe's center μ** and each unit's 1-object cosine R² — `xval_units.csv`'s
  `oneobj_theta_mean` / `oneobj_r2_mean`, already averaged **circularly** over bootstraps
  upstream (an arithmetic mean would put a neuron whose per-bootstrap angles straddle the
  0°/360° seam on the opposite side of the ring). Those columns are in **degrees**;
  :func:`oneobj_fits` converts μ to radians.
* **Trials** — the firing-rate cache, one read per session shared across that session's
  units (:func:`session_unit_trials`, :func:`_session_trials`). The firing rate is the
  mean of the stimulus and delay windows.

The whole stimulus + delay window is always used, so there is no `phase` knob, and there
are no synthetic populations anywhere — every response here is estimated from real
recorded trials.
"""

import sys
from pathlib import Path
from pathlib import Path as _Path

import numpy as np
import pandas as pd
from scipy.stats import f as _f_dist

# Resolved from THIS FILE, not from the working directory: `sys.path.append('..')` only works
# when the notebook happens to run from this directory, so importing the module from anywhere
# else failed with ModuleNotFoundError.
_FIGURE_3 = _Path(__file__).resolve().parent.parent
if str(_FIGURE_3) not in sys.path:
    sys.path.insert(0, str(_FIGURE_3))
from common import drop_east_west_arrays

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = (_HERE / "../../..").resolve()  # repository root

# All three moved on 2026-09-11, when `figure_3_deprecated/` and its cache were deleted. The
# firing rates and the modulation table were byte-different but content-identical copies of the
# shared inputs every other figure reads — a second copy of the paper's trial data, which is the
# kind of thing that silently stops matching. The 1-object fits now come from the **phase
# fit's** cache: see :func:`oneobj_fits`.
_SHARED = _ROOT / "cache/figures/shared_inputs"
FR_DIR = _SHARED / "firing_rates"
MODULATION_PATH = _SHARED / "modulation.csv"
ONEOBJ_PATH = _ROOT / "cache/figures/figure_3/model_comparison/phase_units.csv"

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
SUBJECT_TITLE = {"Perle": "Monkey P", "Elgar": "Monkey E"}

# Below this kernel ceiling a unit has essentially no explainable 2-object structure,
# so 'fraction' (model R² / ceiling) is meaningless; floored here as a backstop.
MIN_CEILING = 1e-3

_ID_COLS = ["subject", "session", "unit_id"]


def circ_dist(a, b):
    """Circular arc distance in [0, π]; broadcasts."""
    d = np.abs(np.asarray(a, float) - np.asarray(b, float)) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


# ── Per-unit real trials, straight from the firing-rate cache ─────────────────
def session_unit_trials(subject, session, completed_only=True):
    """One session's firing-rate cache → per unit, its cleaned 1- and 2-object trial
    arrays. The firing rate is the mean of the stimulus and delay windows.

    The over-represented east-west antipodal stimulus pair (one object at 0 deg and the
    other at 180 deg) is always dropped -- see `figures/trial_filters.py` for why and
    for the exact rule; there is no option to keep it.

    """
    path = FR_DIR / f"{subject}_{session}.csv.gz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — build it with `python firing_rates.py` in figures/."
        )
    df = pd.read_csv(path)
    if completed_only:
        df = df[df["completed"].astype(bool)]
    df = df.assign(
        _fr=df[["stimulus_fr", "delay_fr"]].astype(float).mean(axis=1)
    )
    out = {}
    for uid, g in df.groupby("unit"):
        g1 = g[g["num_objects"] == 1]
        fr1 = g1["_fr"].to_numpy(float)
        th1 = g1["object_0_theta"].to_numpy(float) % (2 * np.pi)
        ok1 = np.isfinite(fr1) & np.isfinite(th1)
        th1, fr1 = th1[ok1], fr1[ok1]

        g2 = g[g["num_objects"] == 2]
        fr2 = g2["_fr"].to_numpy(float)
        t0 = g2["object_0_theta"].to_numpy(float) % (2 * np.pi)
        t1 = g2["object_1_theta"].to_numpy(float) % (2 * np.pi)
        ok2 = np.isfinite(fr2) & np.isfinite(t0) & np.isfinite(t1)
        t0, t1, fr2 = t0[ok2], t1[ok2], fr2[ok2]
        # the repeated east-west pair, dropped in every analysis (see figures/trial_filters.py)
        t0, t1, fr2 = drop_east_west_arrays(t0, t1, fr2)

        out[int(uid)] = dict(
            t0=t0,
            t1=t1,
            fr2=fr2,
            theta1=th1,
            fr1=fr1,
            brain_area=str(g["brain_area"].iloc[0]),
        )
    return out


def _session_trials(units_df):
    """``{(subject, session): session_unit_trials(...)}`` for every session represented
    in ``units_df``, read **once** each. That loader returns a unit's cleaned 1- *and*
    2-object trials together, so both public loaders below share this one pass.
    """
    cache = {}
    for subject, session in (
        units_df[["subject", "session"]].drop_duplicates().to_numpy()
    ):
        cache[(subject, session)] = session_unit_trials(subject, session)
    return cache


def load_units_1obj_trials(units_df, session_cache=None):
    """One ``(theta, fr)`` of real 1-object trials per row of ``units_df`` (same
    order). Raw non-negative firing rates (mean of the stimulus and delay windows).
    """
    cache = (
        session_cache
        if session_cache is not None
        else _session_trials(units_df)
    )
    return [
        (
            cache[(r["subject"], r["session"])][int(r["unit_id"])]["theta1"],
            cache[(r["subject"], r["session"])][int(r["unit_id"])]["fr1"],
        )
        for _, r in units_df.iterrows()
    ]


def load_units_2obj_trials(units_df, session_cache=None):
    """One ``(t0, t1, fr)`` of real 2-object trials per row of ``units_df`` (same
    order). The east-west exclusion is applied when the session is read, so pass
    ``session_cache`` only if it was built with the same setting."""
    cache = (
        session_cache
        if session_cache is not None
        else _session_trials(units_df)
    )
    out = []
    for _, r in units_df.iterrows():
        d = cache[(r["subject"], r["session"])][int(r["unit_id"])]
        out.append((d["t0"], d["t1"], d["fr2"]))
    return out


# ── Cross-validated R² accessors (modulation.csv) ─────────────────────────────
def load_modulation(path=MODULATION_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — build it with "
            f"`python run_modulation_cache.py` in ../model_comparison/."
        )
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
    """``model``'s per-unit R² for every row of ``df``, in the ``r2_kind`` flavor.

    ``'raw'`` is the cross-validated variance explained as cached; ``'fraction'``
    divides it by the kernel-smoother ceiling ``cv_r2`` (the total explainable
    variance), giving the fraction of *explainable* variance that model captures.
    The kernel scores (``'cv_r2'`` / ``'dev'``) are always returned raw."""
    if r2_kind not in R2_KINDS:
        raise ValueError(
            f"r2_kind must be one of {R2_KINDS} (got {r2_kind!r})"
        )
    v = df[r2_column(model)].to_numpy(float)
    if r2_kind == "raw" or model in KERNEL_SCORES:
        return v
    return v / np.maximum(df["cv_r2"].to_numpy(float), MIN_CEILING)


def r2_label(model, r2_kind="raw"):
    """Axis label for ``(model, r2_kind)``."""
    name = DISPLAY.get(model, model)
    if model in KERNEL_SCORES or r2_kind == "raw":
        return f"{name} CV R²"
    return f"{name} R² / ceiling"


# ── The 1-object cosine fit (the decoding lobe's center μ) ────────────────────
def cosine_fit_pvalue(r2, n_trials):
    """Significance of a neuron's 1-object cosine tuning fit, from its R² and trial count.

    The fit ``r = b + c1·cos θ + c2·sin θ`` is an ordinary linear model whose **two**
    directional coefficients are what "is this neuron tuned?" asks about. Under the null
    of no directional tuning the usual regression F statistic

        ``F = (R²/2) / ((1 − R²)/(n − 3))``

    is F-distributed with ``(2, n − 3)`` degrees of freedom, so the p-value is that
    distribution's survival function — an exact test given Gaussian residuals, and cheap
    enough to evaluate for the whole cache at once (no refitting, no permutation).

    Note ``R²`` in the cache is the bootstrap **mean** of the per-resample training R²,
    which stands in for the full-sample R² here.
    """
    r2 = np.clip(np.asarray(r2, float), 0.0, 1.0 - 1e-12)
    df2 = np.maximum(np.asarray(n_trials, float) - 3.0, 1.0)
    F = (r2 / 2.0) / np.maximum((1.0 - r2) / df2, 1e-300)
    return _f_dist.sf(F, 2.0, df2)


_ONEOBJ_CACHE = {}


def oneobj_fits(path=ONEOBJ_PATH):
    """One row per neuron with its **1-object receptive-field direction**.

    Columns: ``theta`` (preferred direction μ in ``[0, 2π)``), ``theta_R`` (the circular
    resultant length of that direction across bootstraps — 1 = every bootstrap agreed, near 0 =
    it is not reliably determined), ``oneobj_p`` (is there a direction at all) and ``n_1obj``
    (the trials it was measured on).

    **Ported to the pairwise estimator on 2026-09-11.** It read `xval_units.csv`, the parametric
    1-object cosine fit, and that cache went with `figure_3_deprecated/`. The replacement is the
    same quantity measured by the estimator the paper now uses: `phase_units.csv` holds
    ``oneobj_phase_deg`` — the preferred direction from the pairwise comparison with no anchor —
    with ``oneobj_sd_deg`` its bootstrap circular SD and ``oneobj_p`` a Wald test against "no
    direction" rather than the R²-derived p the cosine fit needed. ``theta_R`` is recovered from
    the circular SD by inverting ``sd = sqrt(-2·ln R)``, exactly as before.

    Three columns are gone with the cosine fit and had no consumer: ``r_squared``,
    ``oneobj_amplitude`` and ``oneobj_baseline``.

    Only these few columns are read from the (large) cache, and the result is memoized per path.
    """
    path = Path(path)
    key = str(path)
    if key in _ONEOBJ_CACHE:
        return _ONEOBJ_CACHE[key].copy()
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — build it with "
            f"`python run_phase_cache.py` in ../model_comparison/."
        )
    cols = _ID_COLS + [
        "oneobj_phase_deg",
        "oneobj_sd_deg",
        "oneobj_p",
        "n_1obj",
    ]
    df = pd.read_csv(path, usecols=cols)
    sd = np.radians(df["oneobj_sd_deg"].to_numpy(float))
    out = pd.DataFrame(
        {
            "theta": np.radians(df["oneobj_phase_deg"].to_numpy(float))
            % (2 * np.pi),
            "theta_R": np.exp(-0.5 * sd**2),  # invert sd = sqrt(-2 ln R)
            "oneobj_p": df["oneobj_p"].to_numpy(float),
            "n_1obj": df["n_1obj"].to_numpy(int),
        }
    )
    out = pd.concat([df[_ID_COLS], out], axis=1)
    _ONEOBJ_CACHE[key] = out
    return out.copy()


# ── Candidate neurons ─────────────────────────────────────────────────────────
def candidate_units(
    subject="all", brain_area="all", mod_df=None, verbose=True
):
    """The candidate-unit pool: every neuron passing the shared **2-object modulation
    criterion**, joined to its 1-object cosine fit.

    The criterion (``figures/neuron_selection.py``) is model-agnostic and is the same one
    every other population figure uses. It replaces the filters this took before — the
    1-object fit's p-value, the kernel ceiling, the shuffle-null p and the trial floor.

    **The 1-object fit is still joined on**, because the decoder centres a lobe on each
    neuron's preferred direction and the figures position marks by it; it is simply no
    longer a *selection* criterion. ``theta_R`` is carried so a neuron whose direction is
    poorly determined can be identified downstream rather than silently dropped here.

    ``modulation.csv``'s own ``p`` is renamed ``mod_p`` on the way in, because the 1-object
    fit contributes a ``p`` of its own and the two answer different questions.
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
    """Per-unit angle on the retinotopic-vs-relational plane, normalized to ``[0, 1]``:
    ``2·arctan(y_r2 / x_r2)/π``. ``0`` = on the retinotopic axis, ``1`` = on the relational
    axis, ``0.5`` = the ``y = x`` diagonal. It is the actual angle *value* (not a rank), so
    an interval of it is a fixed geometric wedge of the plane."""
    x = np.maximum(r2_values(df, x_model, r2_kind), 0.0)
    y = np.maximum(r2_values(df, y_model, r2_kind), 0.0)
    return 2.0 * np.arctan2(y, x) / np.pi


def angle_range_from_mean(x):
    """The ``[0, 1]`` angle interval the neurons are drawn from, at bias ``x``:
    ``x < 0.5 → [0, 2x]`` (the retinotopic side), ``x ≥ 0.5 → [2x − 1, 1]`` (the relational
    side), ``0.5 → [0, 1]`` (everything). The wedges are nested and centered on ``x``, so
    sweeping ``x`` from 0 to 1 slides the population from purely retinotopic to purely
    relational."""
    return (0.0, 2.0 * x) if x < 0.5 else (2.0 * x - 1.0, 1.0)


def fit_strength(df, x_model="const", y_model="cw", r2_kind="raw"):
    """Joint const-cw fit strength ``sqrt(x_r2² + y_r2²)`` — the distance from the origin
    in the R² plane, i.e. how well the unit is explained by *either* 2-object model. The
    **radial** coordinate whose angular coordinate is :func:`angle_metric`."""
    return np.hypot(
        r2_values(df, x_model, r2_kind), r2_values(df, y_model, r2_kind)
    )


def band_positions(
    pool_df,
    percentile_mean,
    r2_min=0.0,
    x_model="const",
    y_model="cw",
    r2_kind="raw",
):
    """Row positions of ``pool_df`` inside the selection region: the
    :func:`angle_range_from_mean` wedge at ``percentile_mean``, **and** at least
    ``r2_min`` from the origin.

    The two cuts are orthogonal and answer different questions. ``angle_metric`` is a
    *ratio*, so it only means anything once the neuron is explained by something: near the
    origin both R² are near 0 and the angle is noise, which would fill a supposedly
    "retinotopic" wedge with neurons that are simply unexplained. The radial cut
    (:func:`fit_strength`) asks "is this neuron's 2-object response explained **at all**",
    and the wedge then asks "**by which** model".

    Returns ``(positions, (lo, hi))``."""
    lo, hi = angle_range_from_mean(percentile_mean)
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
            f"No neurons in the wedge [{lo:.2f}, {hi:.2f}] with "
            f"‖const,cw‖ >= {r2_min:g} at percentile_mean={percentile_mean}."
        )
    return positions, (lo, hi)


def sample_units(pool_df, positions, num_neurons, rng):
    """Draw ``num_neurons`` neurons **with replacement** from ``pool_df.iloc[positions]``.

    One bootstrap of the population: a neuron may be drawn more than once (and many are
    drawn not at all), so the decoder's input dimension is exactly ``num_neurons`` however
    large or small the wedge is. Re-drawing with a different ``rng`` gives another
    bootstrap.

    Returns ``(units_df, counts)`` — the drawn neurons in draw order (``num_neurons`` rows,
    with repeats), and how many times each *pool* row was drawn (aligned to ``positions``),
    which is what the selection scatter sizes its markers by."""
    positions = np.asarray(positions)
    draws = rng.integers(0, len(positions), size=int(num_neurons))
    counts = np.bincount(draws, minlength=len(positions))
    return pool_df.iloc[positions[draws]].reset_index(drop=True), counts


# ── Replicas ──────────────────────────────────────────────────────────────────
def replica_offsets(num_replicas):
    """``num_replicas`` evenly spaced rotation offsets in ``[0, 2π)``."""
    return 2 * np.pi * np.arange(num_replicas) / num_replicas


def _replica_specs(num_replicas, include_reflections):
    """Per-unit copies as ``(offset, reflect)`` pairs: ``num_replicas`` evenly spaced
    rotations, plus — if ``include_reflections`` — the same rotations of the
    **reflected** neuron (query angles negated, a mirror image across the x-axis, before
    the rotation offset), giving all orientations of the reflection. A unit thus yields
    ``num_replicas`` copies, or ``2·num_replicas`` with reflections. The response
    functions and :func:`replica_preferred_directions` iterate this same list, so their
    columns line up."""
    offsets = replica_offsets(num_replicas)
    specs = [(off, False) for off in offsets]
    if include_reflections:
        specs += [(off, True) for off in offsets]
    return specs


def replica_preferred_directions(
    mu_units, num_replicas, include_reflections=False
):
    """``[n_inputs]`` effective preferred direction of every (unit, copy) column, in the
    **same order** as :func:`population_response_1obj` / :func:`population_response_2obj`.
    A copy responds to a stimulus ``θ`` via its unit at ``s·θ + offset`` (``s = −1`` for
    a reflected copy, else ``+1``), which peaks at ``θ = s·(μ_u − offset)`` — that is its
    effective preferred direction. Non-reflected copies give ``μ_u − offset``; reflected
    copies give ``offset − μ_u``. Use these to build the decoder so each column's lobe is
    centered on the direction it is tuned to."""
    specs = _replica_specs(num_replicas, include_reflections)
    out = []
    for mu in np.asarray(mu_units, float):
        for offset, reflect in specs:
            s = -1.0 if reflect else 1.0
            out.append((s * (mu - offset)) % (2 * np.pi))
    return np.array(out)


# ── Kernel-weighted-mean population response (raw rates, replicas) ────────────
def _bump_weights_1obj(trial_theta, query_theta, sigma_deg):
    """``[n_query, n_trials]`` circular-Gaussian weights of each real 1-object trial for
    each query angle; rows sum to 1."""
    query_theta = np.atleast_1d(np.asarray(query_theta, float))
    d = circ_dist(query_theta[:, None], trial_theta[None, :])
    sigma = np.radians(sigma_deg)
    e = -0.5 * (d / sigma) ** 2
    e -= e.max(axis=1, keepdims=True)  # numerically stable per query row
    w = np.exp(e)
    w /= w.sum(axis=1, keepdims=True)
    return w


def _bump_weights_2obj(t0, t1, target_t0, target_t1, sigma_deg):
    """``[n_query, n_trials]`` **swap-invariant** circular-Gaussian weights of each real
    2-object trial for each query pair; rows sum to 1. Distance is the smaller of the
    direct and object-swapped per-object arc-distance sums, so the estimate does not
    depend on which object is labeled 0 vs. 1."""
    target_t0 = np.atleast_1d(np.asarray(target_t0, float))
    target_t1 = np.atleast_1d(np.asarray(target_t1, float))
    d_direct = circ_dist(target_t0[:, None], t0[None, :]) + circ_dist(
        target_t1[:, None], t1[None, :]
    )
    d_swapped = circ_dist(target_t0[:, None], t1[None, :]) + circ_dist(
        target_t1[:, None], t0[None, :]
    )
    d = np.minimum(d_direct, d_swapped)
    sigma = np.radians(sigma_deg)
    e = -0.5 * (d / sigma) ** 2
    e -= e.max(axis=1, keepdims=True)
    w = np.exp(e)
    w /= w.sum(axis=1, keepdims=True)
    return w


def population_response_1obj(
    units_1obj_trials,
    query_theta,
    num_replicas,
    sigma_deg,
    include_reflections=False,
):
    """``[n_query, n_inputs]`` expected 1-object response: kernel-weighted mean of each
    unit's own real 1-object trials, one column per (unit, copy). Copy
    ``(offset, reflect)`` looks the unit up at ``s·query + offset`` (``s = −1`` if
    reflected, else ``+1``). Columns are neuron-major, ordered by
    :func:`_replica_specs`."""
    query_theta = np.atleast_1d(np.asarray(query_theta, float))
    specs = _replica_specs(num_replicas, include_reflections)
    R = np.empty((len(query_theta), len(units_1obj_trials) * len(specs)))
    for u_idx, (theta, fr) in enumerate(units_1obj_trials):
        for j, (offset, reflect) in enumerate(specs):
            s = -1.0 if reflect else 1.0
            q = (s * query_theta + offset) % (2 * np.pi)
            with np.errstate(all="ignore"):
                R[:, u_idx * len(specs) + j] = (
                    _bump_weights_1obj(theta, q, sigma_deg) @ fr
                )
    return R


def _sample_rows(p, rng):
    """Sample one column index per row of the row-stochastic matrix ``p`` ``[m, n]`` via
    vectorized inverse-CDF. Returns ``[m]`` sampled indices."""
    c = np.cumsum(p, axis=1)
    c[:, -1] = 1.0  # guard float drift so a draw always lands
    u = rng.random((p.shape[0], 1))
    return (u < c).argmax(axis=1)


def population_sample_1obj(
    units_1obj_trials,
    query_theta,
    num_replicas,
    sigma_deg,
    rng,
    include_reflections=False,
):
    """``[n_query, n_inputs]`` **single-trial** 1-object response: the sampled
    counterpart of :func:`population_response_1obj`. For each (unit, copy) and each
    query, instead of the kernel-weighted *mean* of the unit's trials, **sample one** of
    its real 1-object trials by weighted nearest neighbors — draw a trial with
    probability proportional to the same Gaussian proximity kernel
    (:func:`_bump_weights_1obj`, width ``sigma_deg``) in stimulus angle around the copy's
    query ``s·query + offset`` — and take that trial's firing rate. Nearby trials are
    likelier but not certain, so the draw is stochastic (``rng``); its expectation over
    draws is exactly :func:`population_response_1obj`. Same column layout / copy
    convention."""
    query_theta = np.atleast_1d(np.asarray(query_theta, float))
    specs = _replica_specs(num_replicas, include_reflections)
    R = np.empty((len(query_theta), len(units_1obj_trials) * len(specs)))
    for u_idx, (theta, fr) in enumerate(units_1obj_trials):
        fr = np.asarray(fr, float)
        for j, (offset, reflect) in enumerate(specs):
            s = -1.0 if reflect else 1.0
            q = (s * query_theta + offset) % (2 * np.pi)
            with np.errstate(all="ignore"):
                p = _bump_weights_1obj(
                    theta, q, sigma_deg
                )  # [n_query, n_trials], rows sum to 1
            R[:, u_idx * len(specs) + j] = fr[_sample_rows(p, rng)]
    return R


def population_response_2obj(
    units_2obj_trials,
    target_t0,
    target_t1,
    num_replicas,
    sigma_deg,
    include_reflections=False,
):
    """``[n_query, n_inputs]`` expected 2-object response: swap-invariant
    kernel-weighted mean of each unit's real 2-object trials, one column per (unit,
    copy). A reflected copy negates **both** query angles before the shared rotation
    offset. Columns are neuron-major, ordered by :func:`_replica_specs`."""
    target_t0 = np.atleast_1d(np.asarray(target_t0, float))
    target_t1 = np.atleast_1d(np.asarray(target_t1, float))
    specs = _replica_specs(num_replicas, include_reflections)
    R = np.empty((len(target_t0), len(units_2obj_trials) * len(specs)))
    for u_idx, (t0, t1, fr) in enumerate(units_2obj_trials):
        for j, (offset, reflect) in enumerate(specs):
            s = -1.0 if reflect else 1.0
            q0 = (s * target_t0 + offset) % (2 * np.pi)
            q1 = (s * target_t1 + offset) % (2 * np.pi)
            with np.errstate(all="ignore"):
                R[:, u_idx * len(specs) + j] = (
                    _bump_weights_2obj(t0, t1, q0, q1, sigma_deg) @ fr
                )
    return R

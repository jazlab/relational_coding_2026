"""The real recorded population behind the fitted-lobe ring decoder.

The trials are read from the shared firing-rate cache,
``cache/figures/shared_inputs/firing_rates/{subject}_{session}.csv.gz``, and the pool from the
model comparison's per-neuron table through `../selection_lib.py`.

This module picks which real units enter the decoder, loads their real 1- and
2-object trials, and estimates the population's response to any query stimulus as a
**kernel-weighted mean of those real trials** — with `num_replicas` rotated copies of each
unit plus the same rotations of its mirror image, so a handful of recorded neurons tile the
whole ring.

What comes from where
---------------------
* **Which neurons** — every neuron passing the shared 2-object modulation criterion of
  `figures/config.py` (:func:`candidate_units`, through `../selection_lib.py`), from which
  :func:`sample_units` draws one bootstrap population.
* **Trials** — the firing-rate cache, one read per session shared across that session's
  units (:func:`session_unit_trials`, :func:`_session_trials`). The firing rate is the
  mean of the stimulus and delay windows.

The whole stimulus + delay window is always used, so there is no `phase` knob, and there
are no synthetic populations anywhere — every response here is estimated from real
recorded trials.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_FIGURE_3 = _HERE.parent
_FIGURES = _FIGURE_3.parent
_ROOT = _FIGURES.parent  # repository root, where paths.py lives
for _p in (_FIGURE_3, _FIGURE_3 / "model_comparison", _FIGURES, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import comparison_lib as cl  # noqa: E402  the model comparison rho is built from
import selection_lib as sel  # noqa: E402
from trial_filters import drop_east_west_arrays  # noqa: E402

import paths  # noqa: E402

FR_DIR = paths.SHARED_INPUTS / "firing_rates"
SUBJECT_TITLE = {"Perle": "Monkey P", "Elgar": "Monkey E"}


def circ_dist(a, b):
    """Circular arc distance in [0, π]; broadcasts."""
    d = np.abs(np.asarray(a, float) - np.asarray(b, float)) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


# ── Per-unit real trials, straight from the firing-rate cache ─────────────────
def session_unit_trials(subject, session):
    """One session's firing-rate cache → per unit, its cleaned 1- and 2-object trial
    arrays (completed trials only). The firing rate is the mean of the stimulus and delay
    windows.

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

        out[int(uid)] = dict(t0=t0, t1=t1, fr2=fr2, theta1=th1, fr1=fr1)
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


def load_units_1obj_trials(units_df, session_cache):
    """One ``(theta, fr)`` of real 1-object trials per row of ``units_df`` (same
    order), from ``session_cache`` (:func:`_session_trials`). Raw non-negative firing rates
    (mean of the stimulus and delay windows).
    """
    return [
        (
            session_cache[(r["subject"], r["session"])][int(r["unit_id"])][
                "theta1"
            ],
            session_cache[(r["subject"], r["session"])][int(r["unit_id"])][
                "fr1"
            ],
        )
        for _, r in units_df.iterrows()
    ]


def load_units_2obj_trials(units_df, session_cache):
    """One ``(t0, t1, fr)`` of real 2-object trials per row of ``units_df`` (same
    order), from ``session_cache`` (:func:`_session_trials`), which has the east-west
    exclusion applied."""
    out = []
    for _, r in units_df.iterrows():
        d = session_cache[(r["subject"], r["session"])][int(r["unit_id"])]
        out.append((d["t0"], d["t1"], d["fr2"]))
    return out


# ── Candidate neurons ─────────────────────────────────────────────────────────
# The pool is the neurons passing the shared modulation criterion (1339), the same as figure 3's
# model comparison. `../efficient_coding/` applies the same criterion plus a fit-strength cut
# (`min_fit=0.0` on both models), so its titration pool is a 1239-neuron subset of this one.
def candidate_units(subject):
    """The decoder's pool for one subject: neurons passing `figures/config.py`'s
    ``MODULATION_CRITERION``, with ``rho`` and ``rho_rank`` attached.
    """
    kept = sel.prepared_pool(
        cl.load_units(), exclusion="config", verbose=False
    )
    return kept[kept["subject"] == subject].reset_index(drop=True)


def sample_units(pool_df, positions, num_neurons, rng):
    """Draw ``num_neurons`` neurons **with replacement** from ``pool_df.iloc[positions]``.

    One bootstrap of the population: a neuron may be drawn more than once (and many are
    drawn not at all), so the decoder's input dimension is exactly ``num_neurons`` however
    large or small the pool is. Re-drawing with a different ``rng`` gives another
    bootstrap.

    Returns ``(units_df, counts)`` — the drawn neurons in draw order (``num_neurons`` rows,
    with repeats), and how many times each *pool* row was drawn (aligned to ``positions``).
    """
    positions = np.asarray(positions)
    draws = rng.integers(0, len(positions), size=int(num_neurons))
    counts = np.bincount(draws, minlength=len(positions))
    return pool_df.iloc[positions[draws]].reset_index(drop=True), counts


# ── Replicas ──────────────────────────────────────────────────────────────────
def _replica_specs(num_replicas):
    """Per-unit copies as ``(offset, reflect)`` pairs: ``num_replicas`` evenly spaced
    rotations in ``[0, 2π)``, plus the same rotations of the **reflected** neuron (query angles
    negated, a mirror image across the x-axis, before the rotation offset), giving all
    orientations of the reflection. A unit thus yields ``2·num_replicas`` copies. The response
    functions and :func:`replica_preferred_directions` iterate this same list, so their
    columns line up."""
    offsets = 2 * np.pi * np.arange(num_replicas) / num_replicas
    return [(off, False) for off in offsets] + [(off, True) for off in offsets]


def replica_preferred_directions(mu_units, num_replicas):
    """``[n_inputs]`` effective preferred direction of every (unit, copy) column, in the
    **same order** as :func:`population_response_1obj` / :func:`population_response_2obj`.
    A copy responds to a stimulus ``θ`` via its unit at ``s·θ + offset`` (``s = −1`` for
    a reflected copy, else ``+1``), which peaks at ``θ = s·(μ_u − offset)`` — that is its
    effective preferred direction. Non-reflected copies give ``μ_u − offset``; reflected
    copies give ``offset − μ_u``. Use these to build the decoder so each column's lobe is
    centered on the direction it is tuned to."""
    specs = _replica_specs(num_replicas)
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
    units_1obj_trials, query_theta, num_replicas, sigma_deg
):
    """``[n_query, n_inputs]`` expected 1-object response: kernel-weighted mean of each
    unit's own real 1-object trials, one column per (unit, copy). Copy
    ``(offset, reflect)`` looks the unit up at ``s·query + offset`` (``s = −1`` if
    reflected, else ``+1``). Columns are neuron-major, ordered by
    :func:`_replica_specs`."""
    query_theta = np.atleast_1d(np.asarray(query_theta, float))
    specs = _replica_specs(num_replicas)
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
    units_1obj_trials, query_theta, num_replicas, sigma_deg, rng
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
    specs = _replica_specs(num_replicas)
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
    units_2obj_trials, target_t0, target_t1, num_replicas, sigma_deg
):
    """``[n_query, n_inputs]`` expected 2-object response: swap-invariant
    kernel-weighted mean of each unit's real 2-object trials, one column per (unit,
    copy). A reflected copy negates **both** query angles before the shared rotation
    offset. Columns are neuron-major, ordered by :func:`_replica_specs`."""
    target_t0 = np.atleast_1d(np.asarray(target_t0, float))
    target_t1 = np.atleast_1d(np.asarray(target_t1, float))
    specs = _replica_specs(num_replicas)
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

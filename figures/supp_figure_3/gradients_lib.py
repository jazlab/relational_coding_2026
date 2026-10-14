r"""The direction of best local discriminability, over the whole two-object stimulus space.

`../figure_3/efficient_coding_schematic/` asks which of two directions each sub-population tells apart
better at ONE stimulus (A, antipodal north/south): a rotation of the pair, `(1, 1)`, or a change
of its separation, `(1, -1)`. This module asks the same question at every stimulus, with one of
two measures.

THE POPULATIONS
---------------
Every neuron is assigned to whichever model fits it better (relational vs retinotopic fit
strength), and each population is the top K of its neurons (:func:`population_units`).
Optionally each neuron is replicated at ``n_replicas`` rotations tiling the circle
(:func:`rotated_trials`): copy ``r`` is the neuron with both objects' angles in every one of its
trials turned by ``r · 360° / n_replicas`` — the same neuron, its tuning rigidly rotated.

THE MEASURES
------------
Both reduce to a 2 × 2 matrix ``J(s)`` per stimulus, with ``uᵀ J u`` the squared d' (per radian²)
of a small displacement along the unit direction ``u``; its **top eigenvector** is the direction
the population tells apart best and ``sqrt`` of its top eigenvalue the d' per radian along it.
The eigenvector is an axis, so it is drawn as a centred segment with no head.

* ``'fisher'`` — Poisson Fisher information, ``J = Σ_i ∇c_i ∇c_iᵀ / c_i``, from each neuron's
  smoothed rate surface (:func:`smoothed_tables`). Local by construction.
* ``'decoding'`` — the schematic's read-out (``decode_lib.decode_pair_real``): a linear
  classifier on the animal's **recorded** trials, cross-validated over trials, telling apart two
  stimuli ``step`` degrees apart along each of several directions through ``s``. Each held-out
  accuracy becomes ``d' = 2 Φ⁻¹(acc)``, and ``d'²`` over the directions is fit by
  ``a + b cos 2φ + c sin 2φ`` — exactly the form ``uᵀ J u`` takes — which gives ``J``
  (:func:`decode_field`). No Poisson model and no derivative, at the cost of a finite step.

THE SMOOTHING
-------------
The schematic's eligibility rule (``decode_lib.trial_distance_deg``): a trial counts for a
stimulus when **both** objects are within ``width`` degrees of the stimulus objects they match,
under whichever pairing is better. ``kernel='box'`` is that rule exactly — the rate is the plain
mean over the eligible trials, the window the decoder's classes are built from. ``'gaussian'``
weights every trial by ``exp(-½ (d / width)²)`` on the same distance instead.

WHAT IS NOT FIXED
-----------------
Only *relative* segment length means anything: lengths are normalised to the longest on the
panel, and the absolute scale depends on the counting window (Fisher) or on the trial counts and
step (decoding). Nothing finer than the smoothing width is resolvable. The two populations
differ in everything else that covaries with the model assignment — firing rate above all.
"""

import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent  # figures/supp_figure_3
_F3 = _HERE.parent / "figure_3"  # the selection, pool and decoder live there
for _p in (
    str(_F3),
    str(_F3.parent),
    str(_F3 / "model_comparison"),
    str(_F3 / "efficient_coding"),
    str(_F3 / "efficient_coding_schematic"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import decode_lib as dl  # noqa: E402  ../figure_3/efficient_coding_schematic — selection, decoder
import pool_lib as pool  # noqa: E402  ../figure_3/efficient_coding — the pool and its trials

#: Per-point decoding results, one JSON file each, so an interrupted run resumes.
DECODE_CACHE = _HERE.parent.parent / "cache/figures/supp_figure_3/decode"

SIDES = ("retinotopic", "relational")


# ── The populations ───────────────────────────────────────────────────────────
def population_units(
    subjects=("Perle", "Elgar"), k=100, rank_within_animal=True, verbose=True
):
    """``{side: units_df}``: the top ``k`` neurons of each population.

    Every neuron is first assigned to whichever model fits it better (relational vs
    retinotopic fit strength, ``selection_lib.fit_strength``), so the two populations are
    disjoint, and each population is then the ``k`` of its neurons with the strongest fit
    (``decode_lib.top_n_positions``, the schematic's rule).

    ``rank_within_animal=True`` ranks each animal separately and takes ``k / len(subjects)``
    from each (the remainder going to the first animals), so the animals contribute evenly.
    ``False`` ranks all of them together and takes the top ``k`` outright — fit strength runs
    systematically higher in Monkey P, so that population comes almost entirely from Monkey P.
    """
    import pandas as pd

    pools = [
        pool.subject_pool(s, verbose=False, with_tables=False)["pop_df"]
        for s in subjects
    ]
    if rank_within_animal:
        ks = [
            k // len(subjects) + (i < k % len(subjects))
            for i in range(len(subjects))
        ]
        parts = [
            (df, dl.top_n_positions(df, n_per_side=ki, verbose=verbose))
            for df, ki in zip(pools, ks)
        ]
    else:
        df = pd.concat(pools, ignore_index=True)
        parts = [(df, dl.top_n_positions(df, n_per_side=k, verbose=verbose))]
    return {
        side: pd.concat(
            [df.iloc[pos[side]] for df, pos in parts], ignore_index=True
        )
        for side in SIDES
    }


def load_trials(units):
    """``{side: [(t0, t1, fr), ...]}``, every 2-object trial of each neuron, in row order."""
    return {side: pool.load_pool_trials(df) for side, df in units.items()}


def _rotations(n_replicas):
    n_replicas = int(n_replicas)
    if n_replicas < 1:
        raise ValueError(f"n_replicas must be at least 1 (got {n_replicas})")
    return 2 * np.pi * np.arange(n_replicas) / n_replicas


def rotated_trials(unit_trials, n_replicas=1):
    """Each unit plus ``n_replicas - 1`` rotated copies tiling the circle, replica-major.

    Copy ``r`` has both objects' angles in every trial turned by ``δ_r = r · 360° / n``, which
    makes it the neuron with its tuning rigidly rotated by ``δ_r``: its rate at ``s`` is the
    original's at ``s − δ_r``. Copy 0 is the neuron as recorded. Any ``n_replicas`` is exact —
    nothing is interpolated.
    """
    return [
        (t0 + d, t1 + d, fr)
        for d in _rotations(n_replicas)
        for (t0, t1, fr) in unit_trials
    ]


# ── Smoothing ─────────────────────────────────────────────────────────────────
def _pair_distance_deg(t0, t1, q0, q1):
    """``[n_query, n_trials]``: ``decode_lib.trial_distance_deg`` for many queries at once —
    the larger of the two objects' arc distances, under the better of the two pairings.
    """
    c = dl.circ_dist
    q0, q1 = np.asarray(q0, float)[:, None], np.asarray(q1, float)[:, None]
    t0, t1 = np.asarray(t0, float)[None, :], np.asarray(t1, float)[None, :]
    direct = np.maximum(c(q0, t0), c(q1, t1))
    swap = np.maximum(c(q0, t1), c(q1, t0))
    return np.degrees(np.minimum(direct, swap))


def smooth_rates(t0, t1, fr, q0, q1, width_deg=15.0, kernel="box"):
    """One neuron's mean rate at each query stimulus ``(q0, q1)`` (radians).

    ``'box'``: the mean over trials whose two objects are both within ``width_deg`` of the
    query's (the schematic's eligibility rule); NaN where no trial qualifies. ``'gaussian'``:
    every trial weighted by ``exp(-½ (d / width_deg)²)`` on the same distance.
    """
    d = _pair_distance_deg(t0, t1, q0, q1)
    if kernel == "box":
        w = (d <= float(width_deg)).astype(float)
    elif kernel == "gaussian":
        w = np.exp(-0.5 * (d / float(width_deg)) ** 2)
    else:
        raise ValueError(
            f"kernel must be 'box' or 'gaussian' (got {kernel!r})"
        )
    n = w.sum(1)
    with np.errstate(all="ignore"):
        return np.where(n > 0, (w @ np.asarray(fr, float)) / n, np.nan)


def _kernel_weights(d, width_deg, kernel):
    if kernel == "box":
        return (d <= float(width_deg)).astype(float)
    if kernel == "gaussian":
        return np.exp(-0.5 * (d / float(width_deg)) ** 2)
    raise ValueError(f"kernel must be 'box' or 'gaussian' (got {kernel!r})")


def local_linear(
    t0, t1, fr, q0, q1, width_deg=15.0, kernel="box", min_trials=None, degree=1
):
    """``(rate, d_rate/dθ0, d_rate/dθ1)`` at each query, by **local polynomial** regression.

    Within the same window as :func:`smooth_rates`, a polynomial in ``Δ`` — each trial's offset
    from the query, under whichever object pairing matched it — is fit to the trials by weighted
    least squares (``degree=1`` a plane, ``degree=2`` a quadric), and the gradient is read off
    its linear terms. A plain window mean is biased at the **edge of the stimulus set**: a
    window half outside it holds trials on one side only, so moving the query toward the edge
    barely changes what it averages and the gradient across the edge comes out too small. That
    edge is the 80° separation floor, and across it is exactly the separation-change direction,
    so the bias pulls every population toward rotation there. A fitted slope removes the bias
    to first order; ``degree=2`` also removes the curvature term that is left at the very edge,
    at the price of more variance.

    NaN where fewer than ``min_trials`` trials carry weight (default: twice the number of
    coefficients) or the fit is degenerate. Gradients are per radian.

    Also returns ``cov``, ``[n_query, 2, 2]``: the sampling covariance of the gradient estimate
    (sandwich form, residual variance from the fit). :func:`fisher_from_gradients` subtracts it,
    because Fisher information sums **squared** gradients and the square of a noisy estimate is
    biased upward by exactly its variance — most where the window is thinnest, at the edge.
    """
    c = dl.circ_dist
    q0a, q1a = np.asarray(q0, float)[:, None], np.asarray(q1, float)[:, None]
    t0a, t1a = np.asarray(t0, float)[None, :], np.asarray(t1, float)[None, :]
    direct = np.maximum(c(q0a, t0a), c(q1a, t1a))
    swap = np.maximum(c(q0a, t1a), c(q1a, t0a))
    use_swap = swap < direct
    w = _kernel_weights(
        np.degrees(np.minimum(direct, swap)), width_deg, kernel
    )
    wrap = lambda x: (x + np.pi) % (2 * np.pi) - np.pi  # noqa: E731
    d0 = wrap(np.where(use_swap, t1a, t0a) - q0a)
    d1 = wrap(np.where(use_swap, t0a, t1a) - q1a)
    cols = [np.ones_like(d0), d0, d1]
    if degree == 2:
        cols += [d0 * d0, d0 * d1, d1 * d1]
    elif degree != 1:
        raise ValueError(f"degree must be 1 or 2 (got {degree!r})")
    X = np.stack(cols, -1)  # [Q, T, p]
    p = X.shape[-1]
    min_trials = 2 * p if min_trials is None else int(min_trials)
    y = np.asarray(fr, float)
    with np.errstate(all="ignore"):
        Xw = X * w[..., None]
        A = np.einsum("qtp,qtr->qpr", Xw, X)
        rhs = np.einsum("qtp,t->qp", Xw, y)
        ok = ((w > 0).sum(1) >= min_trials) & (np.linalg.cond(A) < 1e10)
        coef = np.full(rhs.shape, np.nan)
        coef[ok] = np.linalg.solve(A[ok], rhs[ok][..., None])[..., 0]
        resid = y[None, :] - np.einsum("qtp,qp->qt", X, np.nan_to_num(coef))
        dof = np.maximum(w.sum(1) - p, 1.0)
        sigma2 = (w * resid**2).sum(1) / dof
        B = np.einsum("qtp,qtr->qpr", Xw, Xw)
        cov = np.full((len(coef), 2, 2), np.nan)
        Ainv = np.linalg.inv(A[ok])
        cov[ok] = (sigma2[ok, None, None] * (Ainv @ B[ok] @ Ainv))[:, 1:3, 1:3]
    return coef[:, 0], coef[:, 1], coef[:, 2], cov


def local_linear_tables(
    unit_trials,
    n_grid=36,
    width_deg=15.0,
    kernel="box",
    n_replicas=1,
    degree=1,
):
    """``(rate, d0, d1, cov)``, each ``[n_units * n_replicas, G, G, ...]``, from
    :func:`local_linear`. Rotated copies as in :func:`smoothed_tables`."""
    g = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    g0, g1 = np.meshgrid(g, g, indexing="ij")
    q0, q1 = g0.ravel(), g1.ravel()

    def one(trials, delta=0.0):
        r = [
            local_linear(
                t0 + delta,
                t1 + delta,
                fr,
                q0,
                q1,
                width_deg,
                kernel,
                degree=degree,
            )
            for (t0, t1, fr) in trials
        ]
        return [
            np.stack(
                [x[k].reshape((n_grid, n_grid) + x[k].shape[1:]) for x in r]
            )
            for k in range(4)
        ]

    base = one(unit_trials)
    out = [[], [], [], []]
    for d in _rotations(n_replicas):
        cells = d / (2 * np.pi / n_grid)
        if np.isclose(cells, round(cells)):
            m = int(round(cells))
            parts = [np.roll(b, (m, m), axis=(1, 2)) for b in base]
        else:
            parts = one(unit_trials, d)
        for k in range(4):
            out[k].append(parts[k])
    return tuple(np.concatenate(o) for o in out)


def fisher_from_gradients(
    rate, d0, d1, cov=None, min_rate=0.5, duration_s=1.0
):
    """:func:`fisher_field` from rates and gradients given directly (e.g. local linear).

    With ``cov`` (each gradient's sampling covariance), each neuron contributes
    ``(g gᵀ − cov) / c`` instead of ``g gᵀ / c``: the unbiased estimate of its true
    ``∇r ∇rᵀ / c``. One neuron's term can then be indefinite; the population sum is what is
    used, and a negative top eigenvalue is clipped to 0 by :func:`top_direction`.
    """
    T = float(duration_s)
    c = np.maximum(np.asarray(rate, float), float(min_rate)) * T
    d0, d1 = np.asarray(d0, float) * T, np.asarray(d1, float) * T
    terms = [d0 * d0 / c, d0 * d1 / c, d1 * d1 / c]
    if cov is not None:
        cov = np.asarray(cov, float) * T * T
        terms = [
            terms[0] - cov[..., 0, 0] / c,
            terms[1] - cov[..., 0, 1] / c,
            terms[2] - cov[..., 1, 1] / c,
        ]
    ok = np.isfinite(terms[0]) & np.isfinite(terms[1]) & np.isfinite(terms[2])
    j00, j01, j11 = [
        np.where(ok.any(0), np.where(ok, t, 0.0).sum(0), np.nan) for t in terms
    ]
    return np.stack([np.stack([j00, j01], -1), np.stack([j01, j11], -1)], -2)


def smoothed_tables(
    unit_trials, n_grid=36, width_deg=15.0, kernel="box", n_replicas=1
):
    """``[n_units * n_replicas, G, G]`` rate surfaces on the grid, replica-major.

    The recorded neuron is smoothed once; a copy rotated by a whole number of grid cells is that
    table shifted (identical to smoothing its rotated trials, since the grid is rotation-closed),
    and any other rotation is smoothed from its rotated trials.
    """
    g = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    g0, g1 = np.meshgrid(g, g, indexing="ij")
    q0, q1 = g0.ravel(), g1.ravel()
    base = np.stack(
        [
            smooth_rates(t0, t1, fr, q0, q1, width_deg, kernel).reshape(
                n_grid, n_grid
            )
            for (t0, t1, fr) in unit_trials
        ]
    )
    out = []
    for d in _rotations(n_replicas):
        cells = d / (2 * np.pi / n_grid)
        if np.isclose(cells, round(cells)):
            m = int(round(cells))
            out.append(np.roll(base, (m, m), axis=(1, 2)))
        else:
            out.append(
                np.stack(
                    [
                        smooth_rates(
                            t0 + d, t1 + d, fr, q0, q1, width_deg, kernel
                        ).reshape(n_grid, n_grid)
                        for (t0, t1, fr) in unit_trials
                    ]
                )
            )
    return np.concatenate(out)


def ideal_rate(model, t0, t1, phase_deg=90.0, baseline=5.0, amplitude=2.0):
    """Noise-free rate of an ideal neuron — the retinotopic and relational forms of
    `../figure_3/model_comparison/ideal_neurons.ipynb`, copied rather than imported (that notebook keeps
    them inline).

    Retinotopic: ``b + a·(cos(t0 − μ) + cos(t1 − μ))``, a receptive field at ``phase_deg``
    applied to each object. Relational: ``b + a·cos(t0 + t1 − χ)``, tuned to the pair's sum,
    with preferred axis ``phase_deg`` (``χ = 2·phase − 180°``).
    """
    t0, t1 = np.asarray(t0, float), np.asarray(t1, float)
    if model == "retinotopic":
        mu = np.radians(phase_deg)
        return baseline + amplitude * (np.cos(t0 - mu) + np.cos(t1 - mu))
    if model == "relational":
        chi = np.radians(2.0 * phase_deg - 180.0)
        return baseline + amplitude * np.cos(t0 + t1 - chi)
    raise ValueError(f"unknown model {model!r}")


def ideal_tables(n_neurons=36, n_grid=36, baseline=5.0, amplitude=None):
    """``{side: [n_neurons, G, G]}``: ideal populations whose phases tile the ring evenly.

    ``amplitude`` is one number for both, or a ``{side: a}`` dict; ``None`` takes
    `ideal_neurons.ipynb`'s 2.0 for both. Note that at equal ``a`` the retinotopic neuron's
    rate swings over ``±2a`` (two cosines) and the relational one's over ``±a``.
    """
    if amplitude is None or np.isscalar(amplitude):
        amplitude = {
            s: 2.0 if amplitude is None else float(amplitude) for s in SIDES
        }
    g = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    t0, t1 = np.meshgrid(g, g, indexing="ij")
    phases = np.linspace(0.0, 360.0, n_neurons, endpoint=False)
    return {
        s: np.stack(
            [ideal_rate(s, t0, t1, p, baseline, amplitude[s]) for p in phases]
        )
        for s in SIDES
    }


def synthetic_trials(
    unit_trials,
    units,
    model,
    phases="fitted",
    rates="matched",
    modulation=0.5,
    baseline=5.0,
    amplitude=2.0,
    noise="poisson",
    duration_s=1.0,
    seed=0,
):
    """Each recorded neuron's trials with its firing rates replaced by an **ideal** neuron's.

    The stimuli are kept exactly — every trial's ``(t0, t1)`` — so the synthetic population is
    sampled exactly as the real one was, edge, density and trial counts included, and any
    analysis run on both sees the same stimuli. Only ``fr`` changes, to
    :func:`ideal_rate` of ``model`` (``'retinotopic'`` or ``'relational'``).

    ``phases``: ``'fitted'`` gives each synthetic neuron its recorded counterpart's fitted
    phase under ``model`` (``retinotopic_phase_deg``, the receptive-field direction; or
    ``relational_axis_deg``, the preferred axis — the parameter :func:`ideal_rate` takes), so
    the population's directions are distributed as the real ones are. ``'tile'`` spaces them
    evenly instead: retinotopic directions over 360°, relational axes over 180° (their period).

    ``rates``: ``'matched'`` sets each neuron's baseline to its counterpart's mean recorded rate
    and its peak-to-trough swing to ``modulation`` × that baseline (so ``a = modulation·b/2`` for
    relational and ``modulation·b/4`` for retinotopic, whose two cosines swing over ``±2a``).
    ``'fixed'`` uses ``baseline`` and ``amplitude`` for every neuron, as
    `../figure_3/model_comparison/ideal_neurons.ipynb` does.

    ``noise``: ``'poisson'`` draws each trial's count from Poisson(rate · ``duration_s``) and
    returns it as a rate, so the estimators face trial-to-trial noise of the right kind;
    ``None`` returns the noise-free rate.
    """
    t_list = list(unit_trials)
    n = len(t_list)
    if phases == "fitted":
        col = (
            "retinotopic_phase_deg"
            if model == "retinotopic"
            else "relational_axis_deg"
        )
        ph = units[col].to_numpy(float)
        if not np.isfinite(ph).all():
            raise ValueError(
                f"{(~np.isfinite(ph)).sum()} unit(s) have no fitted {col}"
            )
    elif phases == "tile":
        # A relational neuron's preferred AXIS has period 180° (axis p and p + 180° are the same
        # neuron), so its tiling spans 180°; a retinotopic direction spans the full 360°.
        ph = np.linspace(
            0.0, 180.0 if model == "relational" else 360.0, n, endpoint=False
        )
    else:
        raise ValueError(f"phases must be 'fitted' or 'tile' (got {phases!r})")
    rng = np.random.default_rng(seed)
    out = []
    for (t0, t1, fr), p in zip(t_list, ph):
        if rates == "matched":
            b = float(np.mean(fr))
            a = modulation * b / (4.0 if model == "retinotopic" else 2.0)
        elif rates == "fixed":
            b, a = float(baseline), float(amplitude)
        else:
            raise ValueError(
                f"rates must be 'matched' or 'fixed' (got {rates!r})"
            )
        lam = np.maximum(ideal_rate(model, t0, t1, p, b, a), 0.0)
        if noise == "poisson":
            y = rng.poisson(lam * float(duration_s)) / float(duration_s)
        elif noise is None:
            y = lam
        else:
            raise ValueError(
                f"noise must be 'poisson' or None (got {noise!r})"
            )
        out.append((t0, t1, np.asarray(y, float)))
    return out


# ── Fisher information ────────────────────────────────────────────────────────
def _central_derivative(tables, axis):
    """d/dθ, per radian, by wrapped central differences along ``axis``."""
    n = tables.shape[axis]
    return (np.roll(tables, -1, axis=axis) - np.roll(tables, 1, axis=axis)) / (
        4 * np.pi / n
    )


def fisher_field(tables, min_rate=0.5, duration_s=1.0):
    """``[G, G, 2, 2]``: the population's Fisher matrix at every grid stimulus.

    ``min_rate`` (Hz) floors the Poisson denominator, where a near-silent table cell would
    otherwise dominate the sum through estimation noise alone. A unit that is NaN at a cell or
    either neighbour (a box window with no trials) is left out of the sum there; a cell no unit
    reaches is NaN.
    """
    tables = np.asarray(tables, float)
    c = np.maximum(tables, float(min_rate)) * float(duration_s)
    d0 = _central_derivative(tables, 1) * float(duration_s)
    d1 = _central_derivative(tables, 2) * float(duration_s)
    terms = [d0 * d0 / c, d0 * d1 / c, d1 * d1 / c]
    ok = np.isfinite(terms[0]) & np.isfinite(terms[2])
    j00, j01, j11 = [
        np.where(ok.any(0), np.where(ok, t, 0.0).sum(0), np.nan) for t in terms
    ]
    return np.stack([np.stack([j00, j01], -1), np.stack([j01, j11], -1)], -2)


def top_direction(J):
    """``(u, dprime)``: the top eigenvector ``[..., 2]`` (as ``(dθ0, dθ1)``) and ``sqrt`` of its
    eigenvalue, the d' per radian along it. ``u`` is sign-normalised to ``dθ0 >= 0``; the sign
    carries no meaning."""
    J = np.asarray(J, float)
    bad = ~np.isfinite(J).all((-2, -1))
    w, v = np.linalg.eigh(np.where(bad[..., None, None], 0.0, J))  # ascending
    u = v[..., :, -1]
    u = np.where(u[..., :1] < 0, -u, u)
    dp = np.sqrt(np.maximum(w[..., -1], 0.0))
    return np.where(bad[..., None], np.nan, u), np.where(bad, np.nan, dp)


# ── Decoding ──────────────────────────────────────────────────────────────────
def plotted_points(mask, step=3):
    """``[G, G]`` bool: every ``step``-th grid stimulus where ``mask`` holds — the points
    :func:`plot_field` draws, and so the only ones :func:`decode_field` needs.
    """
    sel = np.zeros_like(mask)
    sel[::step, ::step] = True
    return sel & mask


def _decode_point(trials, s_rad, cfg):
    """Every population's accuracy along each direction through one stimulus.

    A unit enters only if it has ``n_splits`` eligible trials for both classes of **every**
    direction here (otherwise it cannot be cross-validated), and the larger population is then
    subsampled to the smaller, so the two are compared at equal n.
    """
    half = 0.5 * np.radians(cfg["step_deg"])
    phis = np.pi * np.arange(cfg["n_directions"]) / cfg["n_directions"]
    pairs = [
        (
            s_rad - half * np.array([np.cos(p), np.sin(p)]),
            s_rad + half * np.array([np.cos(p), np.sin(p)]),
        )
        for p in phis
    ]
    ok = {}
    for side, ut in trials.items():
        good = np.ones(len(ut), bool)
        for a, b in pairs:
            ia, ib = dl.trial_pools(
                ut, a, b, tol_deg=cfg["tol_deg"], exclusive=True
            )
            good &= np.array(
                [
                    len(x) >= cfg["n_splits"] and len(y) >= cfg["n_splits"]
                    for x, y in zip(ia, ib)
                ]
            )
        ok[side] = np.flatnonzero(good)
    n = min(len(v) for v in ok.values())
    out = dict(n_units=int(n), acc={})
    if n < cfg["min_units"]:
        return out
    rng = np.random.default_rng(cfg["seed"])
    for side, ut in trials.items():
        keep = np.sort(rng.choice(ok[side], n, replace=False))
        sub = [ut[i] for i in keep]
        out["acc"][side] = [
            dl.decode_pair_real(
                sub,
                a,
                b,
                n_trials=cfg["n_trials"],
                n_test=cfg["n_test"],
                tol_deg=cfg["tol_deg"],
                exclusive=True,
                n_splits=cfg["n_splits"],
                seed=cfg["seed"],
                C=cfg["C"],
            )["accuracy"]
            for a, b in pairs
        ]
    return out


def _decode_point_cached(trials, ij, s_rad, cfg, cache_dir):
    import warnings

    path = (
        None
        if cache_dir is None
        else Path(cache_dir) / f"{ij[0]}_{ij[1]}.json"
    )
    if path is not None and path.exists():
        return json.loads(path.read_text())
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter(
            "ignore"
        )  # BLAS's spurious matmul RuntimeWarnings
        res = _decode_point(trials, s_rad, cfg)
    if path is not None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(res))
        tmp.replace(path)  # atomic, so a killed run leaves no half file
    return res


def accuracy_to_J(acc, step_deg, n_test, n_splits):
    """``[2, 2]`` from accuracies along ``len(acc)`` directions evenly covering [0°, 180°).

    ``d' = 2 Φ⁻¹(acc)`` (equal-variance, balanced classes), clipped at chance and one
    held-out error short of perfect so it stays finite; ``d'²`` over the directions is fit by
    ``a + b cos 2φ + c sin 2φ``, the exact form of ``uᵀ J u``, so ``J = [[a+b, c], [c, a−b]]``,
    divided by the step² to put it per radian² like the Fisher measure.
    """
    from scipy.stats import norm

    acc = np.asarray(acc, float)
    top = 1.0 - 0.5 / (2 * n_test * n_splits)
    dp2 = (2 * norm.ppf(np.clip(acc, 0.5, top))) ** 2
    phis = np.pi * np.arange(len(acc)) / len(acc)
    X = np.column_stack(
        [np.ones_like(phis), np.cos(2 * phis), np.sin(2 * phis)]
    )
    a, b, c = np.linalg.lstsq(X, dp2, rcond=None)[0]
    return np.array([[a + b, c], [c, a - b]]) / np.radians(step_deg) ** 2


def decode_field(
    trials,
    points,
    step_deg=20.0,
    n_directions=6,
    tol_deg=15.0,
    n_trials=200,
    n_test=100,
    n_splits=5,
    C=1.0,
    seed=0,
    min_units=10,
    n_jobs=6,
    cache=True,
    verbose=True,
):
    """``({side: [G, G, 2, 2]}, info)``: the decoding analogue of :func:`fisher_field`.

    At each stimulus ``s`` where ``points`` holds, and along ``n_directions`` directions evenly
    covering [0°, 180°), the two stimuli ``s ± (step_deg / 2)·u`` are told apart by
    ``decode_lib.decode_pair_real`` — the schematic's classifier, on recorded trials within
    ``tol_deg`` of each stimulus — and the accuracies are turned into ``J``
    (:func:`accuracy_to_J`). ``trials`` is ``{side: [(t0, t1, fr), ...]}``, rotated copies
    included. Elsewhere ``J`` is NaN, as it is at a point where fewer than ``min_units`` units
    per population have enough trials (``info['n_units']`` says how many did).

    Each point is cached as it finishes (``DECODE_CACHE``, keyed on the units and every
    setting), so an interrupted run resumes. ``n_jobs`` points run in parallel, one BLAS thread
    each.
    """
    from joblib import Parallel, delayed, parallel_config

    cfg = dict(
        step_deg=float(step_deg),
        n_directions=int(n_directions),
        tol_deg=float(tol_deg),
        n_trials=int(n_trials),
        n_test=int(n_test),
        n_splits=int(n_splits),
        C=float(C),
        seed=int(seed),
        min_units=int(min_units),
    )
    if cfg["n_directions"] < 3:
        raise ValueError(
            "n_directions must be at least 3 to fit the cos 2φ form"
        )
    n_grid = points.shape[0]
    cache_dir = None
    if cache:
        h = hashlib.md5()
        h.update(json.dumps(cfg, sort_keys=True).encode())
        h.update(str(n_grid).encode())
        for side in sorted(trials):
            for t0, t1, fr in trials[side]:
                for a in (t0, t1, fr):
                    h.update(np.ascontiguousarray(a, float).tobytes())
        cache_dir = DECODE_CACHE / h.hexdigest()[:16]
        cache_dir.mkdir(parents=True, exist_ok=True)
    idx = list(zip(*np.nonzero(points)))
    g = 2 * np.pi / n_grid
    with parallel_config(backend="loky", inner_max_num_threads=1):
        res = Parallel(n_jobs=n_jobs, verbose=5 if verbose else 0)(
            delayed(_decode_point_cached)(
                trials,
                (int(i), int(j)),
                np.array([i * g, j * g]),
                cfg,
                cache_dir,
            )
            for i, j in idx
        )
    J = {side: np.full((n_grid, n_grid, 2, 2), np.nan) for side in trials}
    n_units = np.full((n_grid, n_grid), np.nan)
    acc = {
        side: np.full((n_grid, n_grid, cfg["n_directions"]), np.nan)
        for side in trials
    }
    for (i, j), r in zip(idx, res):
        n_units[i, j] = r["n_units"]
        for side, a in r["acc"].items():
            acc[side][i, j] = a
            J[side][i, j] = accuracy_to_J(a, step_deg, n_test, n_splits)
    return J, dict(n_units=n_units, acc=acc, cache_dir=cache_dir)


# ── Linear Fisher information with kernel-weighted means ──────────────────────
def kernel_weights_euclid(t0, t1, q0, q1, sigma_deg):
    """``[n_query, n_trials]`` Gaussian weights ``exp(-d² / 2σ²)``, where ``d²`` is the squared
    distance between trial ``(t0, t1)`` and query ``(q0, q1)`` in the ``(θ0, θ1)`` plane —
    circular differences, Euclidean, under whichever object pairing is closer. This is
    ``decoding_orientations.ipynb``'s kernel (not :func:`smooth_rates`'s ``'gaussian'``, which
    is on the larger of the two objects' distances)."""
    c = dl.circ_dist
    q0, q1 = np.asarray(q0, float)[:, None], np.asarray(q1, float)[:, None]
    t0, t1 = np.asarray(t0, float)[None, :], np.asarray(t1, float)[None, :]
    d2 = np.minimum(
        c(q0, t0) ** 2 + c(q1, t1) ** 2, c(q0, t1) ** 2 + c(q1, t0) ** 2
    )
    return np.exp(-0.5 * d2 / np.radians(sigma_deg) ** 2)


def kernel_dprime2(wa, wb, r):
    """Bias-corrected ``d'²`` between two kernel-weighted means of one neuron's trials.

    ``wa``, ``wb``: ``[..., n]`` weights for the two stimuli; ``r``: ``[n]`` rates. Each mean is
    ``Σ w r / Σ w``; the variance is the matching weighted variance (corrected by
    ``n_eff = (Σw)² / Σw²``), pooled over the two. The kernels overlap, so the noise in the
    difference of means is ``σ² Σ_t c_t²`` with ``c_t = w_b(t)/Σw_b − w_a(t)/Σw_a``; that is the
    bias subtracted, giving ``(Δμ)² / σ² − Σ c_t²`` — zero on average for a neuron with no
    signal. Exactly ``decoding_orientations.ipynb``'s ``linear_fisher_kernel``.
    """
    pa = wa / wa.sum(-1, keepdims=True)
    pb = wb / wb.sum(-1, keepdims=True)
    ma, mb = (pa * r).sum(-1), (pb * r).sum(
        -1
    )  # not `@`: Accelerate's matmul warns spuriously
    na, nb = 1.0 / (pa**2).sum(-1), 1.0 / (pb**2).sum(-1)
    va = (pa * (r - ma[..., None]) ** 2).sum(-1) * na / (na - 1)
    vb = (pb * (r - mb[..., None]) ** 2).sum(-1) * nb / (nb - 1)
    vp = (na * va + nb * vb) / (na + nb)
    return (mb - ma) ** 2 / vp - ((pb - pa) ** 2).sum(-1)


def linear_fisher_field(
    trials, points, step_deg=25.0, n_directions=6, sigma_deg=20.0
):
    """``({side: [G, G, 2, 2]}, info)``: :func:`decode_field` with the classifier replaced by
    linear Fisher information from kernel-weighted means.

    At each stimulus ``s`` where ``points`` holds, and along ``n_directions`` directions ``φ``
    evenly covering [0°, 180°), ``d'²`` between ``s ± (step_deg / 2)·u`` is
    :func:`kernel_dprime2` summed over every unit (rotated copies included; units are
    independent, recorded in different sessions), divided by ``step²``. That is ``uᵀ J u`` in
    finite-difference form, so ``a + b cos 2φ + c sin 2φ`` fit over the directions gives ``J``
    exactly as :func:`accuracy_to_J` does for accuracies. Every unit enters every point — the
    kernel never runs out of trials, it only gets wider in effect where they are sparse
    (``info['n_eff_min']``, the smallest effective trial count of any unit and stimulus there).
    """
    n_grid = points.shape[0]
    g = 2 * np.pi / n_grid
    half = 0.5 * np.radians(step_deg)
    phis = np.pi * np.arange(n_directions) / n_directions
    U = np.column_stack([np.cos(phis), np.sin(phis)])  # [D, 2]
    idx = np.argwhere(points)  # [P, 2]
    S = idx * g  # [P, 2] radians
    qa = (S[:, None, :] - half * U[None]).reshape(-1, 2)  # [P*D, 2]
    qb = (S[:, None, :] + half * U[None]).reshape(-1, 2)
    X = np.column_stack(
        [np.ones_like(phis), np.cos(2 * phis), np.sin(2 * phis)]
    )
    J, dp2_all = {}, {}
    n_eff_min = np.full(len(idx), np.inf)
    for side, ut in trials.items():
        dp2 = np.zeros(len(qa))
        for t0, t1, fr in ut:
            r = np.asarray(fr, float)
            wa = kernel_weights_euclid(t0, t1, qa[:, 0], qa[:, 1], sigma_deg)
            wb = kernel_weights_euclid(t0, t1, qb[:, 0], qb[:, 1], sigma_deg)
            dp2 += kernel_dprime2(wa, wb, r)
            ne = np.minimum(
                wa.sum(-1) ** 2 / (wa**2).sum(-1),
                wb.sum(-1) ** 2 / (wb**2).sum(-1),
            )
            n_eff_min = np.minimum(n_eff_min, ne.reshape(len(idx), -1).min(1))
        dp2 = dp2.reshape(len(idx), n_directions) / np.radians(step_deg) ** 2
        coef = np.linalg.lstsq(X, dp2.T, rcond=None)[0]  # [3, P]
        a, b, c = coef
        Jp = np.stack(
            [np.stack([a + b, c], -1), np.stack([c, a - b], -1)], -2
        )  # [P, 2, 2]
        J[side] = np.full((n_grid, n_grid, 2, 2), np.nan)
        J[side][idx[:, 0], idx[:, 1]] = Jp
        dp2_all[side] = dp2
    info_n = np.full((n_grid, n_grid), np.nan)
    info_n[idx[:, 0], idx[:, 1]] = n_eff_min
    return J, dict(n_eff_min=info_n, dprime2_per_rad2=dp2_all, phis=phis)


def sampled_mask(n_grid, min_sep_deg=80.0):
    """``[G, G]`` True where a stimulus exists: the two objects at least ``min_sep_deg`` apart
    around the ring (no 2-object trial is closer, so the band around the diagonal is empty).
    """
    g = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    return np.degrees(dl.circ_dist(g[:, None], g[None, :])) >= float(
        min_sep_deg
    )


def summary(fields, mask):
    """How each population's best direction sits between rotation and separation change.

    ``|cos|`` with ``(1, 1)/√2``: 1 is pure rotation, 0 pure separation change, 0.71 halfway.
    """
    import pandas as pd

    rows = []
    for side, (u, dp) in fields.items():
        m = mask & np.isfinite(dp)
        cos = np.abs(u[m][:, 0] * dl.U_SUM[0] + u[m][:, 1] * dl.U_SUM[1])
        rows.append(
            dict(
                population=side,
                n_points=int(m.sum()),
                median_abs_cos_rotation=float(np.median(cos)),
                frac_closer_to_rotation=float((cos > np.sqrt(0.5)).mean()),
                median_dprime_per_rad=float(np.median(dp[m])),
            )
        )
    return pd.DataFrame(rows).set_index("population")


# ── The figure ────────────────────────────────────────────────────────────────
def plot_field(
    fields,
    mask,
    step=3,
    figsize=(3.4, 3.4),
    colors=None,
    max_len_deg=None,
    lw=1.6,
    alpha=0.9,
    order=SIDES,
    xlabel=r"$\theta_0$ (°)",
    ylabel=r"$\theta_1$ (°)",
    ticks=range(0, 361, 90),
    legend=True,
    legend_labels=None,
    legend_kw=None,
):
    """One segment per population at every ``step``-th grid stimulus where ``mask`` holds.

    The segment's angle is the population's best local direction and its length is
    proportional to that direction's d'. Lengths share **one** scale across both populations,
    with the longest segment on the panel ``max_len_deg`` long (default: 90% of the spacing
    between grid points, so neighbours never touch). Populations are drawn in ``order``, the
    last on top.
    """
    colors = colors or {"retinotopic": "tab:blue", "relational": "tab:orange"}
    legend_labels = legend_labels or {
        "retinotopic": "Retinotopic",
        "relational": "Relational",
    }
    n_grid = mask.shape[0]
    spacing = 360.0 / n_grid * step
    max_len = 0.9 * spacing if max_len_deg is None else float(max_len_deg)

    sel = plotted_points(mask, step)
    for _u, dp in fields.values():
        sel &= np.isfinite(
            dp
        )  # no estimate there (no trials, or too few units)
    i0, i1 = np.nonzero(sel)
    x, y = i0 * 360.0 / n_grid, i1 * 360.0 / n_grid
    top = max(float(dp[sel].max()) for _u, dp in fields.values())

    from matplotlib.collections import LineCollection

    fig, ax = plt.subplots(figsize=figsize)
    for k, side in enumerate(order):
        u, dp = fields[side]
        half = (
            0.5 * (dp[sel] / top * max_len)[:, None] * u[sel]
        )  # [n, 2], centred
        c = np.stack([x, y], -1)
        ax.add_collection(
            LineCollection(
                np.stack([c - half, c + half], 1),
                colors=colors[side],
                linewidths=lw,
                alpha=alpha,
                capstyle="round",
                zorder=3 + k,
            )
        )
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 360)
    ax.set_aspect("equal")
    ax.set_xticks(list(ticks))
    ax.set_yticks(list(ticks))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if legend:
        from matplotlib.lines import Line2D

        handles = [
            Line2D([], [], color=colors[s], lw=2.0, label=legend_labels[s])
            for s in order
        ]
        kw = dict(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=2,
            frameon=False,
            handlelength=1.2,
            columnspacing=1.0,
        )
        kw.update(legend_kw or {})
        ax.legend(handles=handles, **kw)
    fig.tight_layout()
    return fig, ax

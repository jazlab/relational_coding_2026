r"""The direction of best local discriminability, over the whole two-object stimulus space.

`../figure_3/efficient_coding_schematic/` asks which of two directions each sub-population tells
apart better at ONE stimulus (A, antipodal north/south): a rotation of the pair, `(1, 1)`, or a
change of its separation, `(1, -1)`. This module asks the same question at every stimulus.

THE POPULATIONS
---------------
Every neuron is assigned to whichever model fits it better (relational vs retinotopic fit
strength), and each population is the top K of its neurons, ranked over both animals together
(:func:`population_units`).

THE MEASURE
-----------
**Linear Fisher information from kernel-weighted means** (:func:`linear_fisher_field`). A
neuron's mean rate at a stimulus is a weighted average over *all* its 2-object trials, each
weighted by ``exp(-d² / 2σ²)`` with ``d² = Δθ0² + Δθ1²`` under the closer object pairing
(:func:`kernel_weights_euclid`). At each stimulus and along several directions ``u``, the
bias-corrected ``d'²`` between two stimuli a fixed step apart along ``u``
(:func:`kernel_dprime2`) is summed over units and divided by the step²; that is ``uᵀ J u`` in
finite-difference form, and fitting ``a + b cos 2φ + c sin 2φ`` over the directions gives a 2 × 2
matrix ``J(s)``. Its **top eigenvector** is the direction the population tells apart best and
``sqrt`` of its top eigenvalue the d' per radian along it (:func:`top_direction`). The
eigenvector is an axis, so it is drawn as a centred segment with no head.

The same analysis runs on **synthetic** neurons (:func:`synthetic_trials`): each selected
neuron's recorded stimuli with an ideal neuron's Poisson rates in place of its own.

WHAT IS NOT FIXED
-----------------
Only *relative* segment length means anything: lengths are normalised to the longest on the
panel, and the absolute scale depends on the kernel width and the step. Nothing finer than the
kernel width is resolvable. The two populations differ in everything else that covaries with the
model assignment — firing rate above all.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

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

import decode_lib as dl  # noqa: E402  ../figure_3/efficient_coding_schematic — selection
import pool_lib as pool  # noqa: E402  ../figure_3/efficient_coding — the pool and its trials

SIDES = ("retinotopic", "relational")


# ── The populations ───────────────────────────────────────────────────────────
def population_units(subjects, k):
    """``{side: units_df}``: the top ``k`` neurons of each population.

    Every neuron is first assigned to whichever model fits it better (relational vs
    retinotopic fit strength, ``selection_lib.fit_strength``), so the two populations are
    disjoint, and each population is then the ``k`` of its neurons with the strongest fit
    (``decode_lib.top_n_positions``, the schematic's rule). All of ``subjects`` are ranked
    together, so — fit strength running systematically higher in Monkey P — the populations
    come mostly from Monkey P.
    """
    df = pd.concat(
        [
            pool.subject_pool(s, verbose=False, with_tables=False)["pop_df"]
            for s in subjects
        ],
        ignore_index=True,
    )
    pos = dl.top_n_positions(df, n_per_side=k)
    return {side: df.iloc[pos[side]].reset_index(drop=True) for side in SIDES}


def load_trials(units):
    """``{side: [(t0, t1, fr), ...]}``, every 2-object trial of each neuron, in row order."""
    return {side: pool.load_pool_trials(df) for side, df in units.items()}


# ── Synthetic neurons ─────────────────────────────────────────────────────────
def ideal_rate(model, t0, t1, phase_deg, baseline, amplitude):
    """Noise-free rate of an ideal neuron — the retinotopic and relational forms of
    `../figure_3/model_comparison/ideal_neurons.ipynb`, copied rather than imported (that
    notebook keeps them inline).

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


def synthetic_trials(unit_trials, model, modulation, duration_s, seed):
    """Each recorded neuron's trials with its firing rates replaced by an **ideal** neuron's.

    The stimuli are kept exactly — every trial's ``(t0, t1)`` — so the synthetic population is
    sampled exactly as the real one was, edge, density and trial counts included, and any
    analysis run on both sees the same stimuli. Only ``fr`` changes, to a Poisson draw of
    :func:`ideal_rate` of ``model`` (``'retinotopic'`` or ``'relational'``):

    * **phases** are spaced evenly: retinotopic directions over 360°, relational axes over 180°
      (their period);
    * **rates**: each neuron's baseline is its counterpart's mean recorded rate and its
      peak-to-trough swing is ``modulation`` × that baseline (so ``a = modulation·b/2`` for
      relational and ``modulation·b/4`` for retinotopic, whose two cosines swing over ``±2a``);
    * **noise**: each trial's count is drawn from Poisson(rate · ``duration_s``) and returned as
      a rate, so the estimators face trial-to-trial noise of the right kind.
    """
    t_list = list(unit_trials)
    # A relational neuron's preferred AXIS has period 180° (axis p and p + 180° are the same
    # neuron), so its tiling spans 180°; a retinotopic direction spans the full 360°.
    ph = np.linspace(
        0.0,
        180.0 if model == "relational" else 360.0,
        len(t_list),
        endpoint=False,
    )
    rng = np.random.default_rng(seed)
    out = []
    for (t0, t1, fr), p in zip(t_list, ph):
        b = float(np.mean(fr))
        a = modulation * b / (4.0 if model == "retinotopic" else 2.0)
        lam = np.maximum(ideal_rate(model, t0, t1, p, b, a), 0.0)
        y = rng.poisson(lam * float(duration_s)) / float(duration_s)
        out.append((t0, t1, np.asarray(y, float)))
    return out


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


def plotted_points(mask, step):
    """``[G, G]`` bool: every ``step``-th grid stimulus where ``mask`` holds — the points
    :func:`plot_field` draws, and so the only ones :func:`linear_fisher_field` needs.
    """
    sel = np.zeros_like(mask)
    sel[::step, ::step] = True
    return sel & mask


# ── Linear Fisher information with kernel-weighted means ──────────────────────
def kernel_weights_euclid(t0, t1, q0, q1, sigma_deg):
    """``[n_query, n_trials]`` Gaussian weights ``exp(-d² / 2σ²)``, where ``d²`` is the squared
    distance between trial ``(t0, t1)`` and query ``(q0, q1)`` in the ``(θ0, θ1)`` plane —
    circular differences, Euclidean, under whichever object pairing is closer.
    """
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
    signal.
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


def linear_fisher_field(trials, points, step_deg, n_directions, sigma_deg):
    """``({side: [G, G, 2, 2]}, info)``: each population's ``J`` at every grid stimulus where
    ``points`` holds (NaN elsewhere), by linear Fisher information from kernel-weighted means.

    At each such stimulus ``s``, and along ``n_directions`` directions ``φ`` evenly covering
    [0°, 180°), ``d'²`` between ``s ± (step_deg / 2)·u`` is :func:`kernel_dprime2` summed over
    every unit (units are independent, recorded in different sessions), divided by ``step²``.
    That is ``uᵀ J u`` in finite-difference form, so ``a + b cos 2φ + c sin 2φ`` fit over the
    directions gives ``J = [[a+b, c], [c, a−b]]``. Every unit enters every point — the kernel
    never runs out of trials, it only gets wider in effect where they are sparse
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
    J = {}
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
    info_n = np.full((n_grid, n_grid), np.nan)
    info_n[idx[:, 0], idx[:, 1]] = n_eff_min
    return J, dict(n_eff_min=info_n)


def sampled_mask(n_grid, min_sep_deg):
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
def plot_field(fields, mask, step, figsize, colors, lw, alpha, order):
    """One segment per population at every ``step``-th grid stimulus where ``mask`` holds.

    The segment's angle is the population's best local direction and its length is
    proportional to that direction's d'. Lengths share **one** scale across both populations,
    with the longest segment on the panel 90% of the spacing between plotted grid points, so
    neighbours never touch. Populations are drawn in ``order``, the last on top, in ``colors``
    (``{side: colour}``); the legend sits above the axes.
    """
    n_grid = mask.shape[0]
    max_len = 0.9 * (
        360.0 / n_grid * step
    )  # 90% of the spacing between plotted points

    sel = plotted_points(mask, step)
    for _u, dp in fields.values():
        sel &= np.isfinite(
            dp
        )  # no estimate there (no trials, or too few units)
    i0, i1 = np.nonzero(sel)
    x, y = i0 * 360.0 / n_grid, i1 * 360.0 / n_grid
    top = max(float(dp[sel].max()) for _u, dp in fields.values())

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
    ticks = range(0, 361, 90)
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 360)
    ax.set_aspect("equal")
    ax.set_xticks(list(ticks))
    ax.set_yticks(list(ticks))
    ax.set_xlabel(r"$\theta_0$ (°)")
    ax.set_ylabel(r"$\theta_1$ (°)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    labels = {"retinotopic": "Retinotopic", "relational": "Relational"}
    handles = [
        Line2D([], [], color=colors[s], lw=2.0, label=labels[s]) for s in order
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=False,
        handlelength=1.2,
        columnspacing=1.0,
    )
    fig.tight_layout()
    return fig, ax

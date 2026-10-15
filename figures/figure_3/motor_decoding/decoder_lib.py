"""The **jointly-fit** bump decoder: lobes, the population fit, the read-out, the
additive-code control, and the plots.

Pure NumPy/SciPy/matplotlib on top of :mod:`population_lib` in this directory, touching no
cache. Firing rates are used exactly as recorded — nothing here normalizes, rescales or
baseline-corrects them.

## The decoder

A **linear** map from a population firing-rate vector ``r`` to a heatmap of object location
over the ring, ``D(φ) = Σ_c W(φ, c)·r_c``, built from **1-object trials only** and then
applied to 2-object trials. The replica construction of :mod:`population_lib` makes it
rotation-equivariant, so its weights are forced to the form ``W(φ, c) = g_u(φ − μ_c)``.
Here that lobe is a bump of a single global width, scaled by one amplitude per neuron::

    g_u(Δ) = a_u · L(Δ ; w)          μ_c = s_c·(ψ_u − offset_c)

leaving exactly two numbers per neuron — an amplitude ``a_u`` (**signed**) and a phase
``ψ_u``. Both are fit **jointly across the whole population** by :func:`joint_fit`: one least-squares problem whose residual is the decode's
own error, so every parameter is chosen in the context of every other neuron's. That is
what allows an amplitude to come out negative — a neuron can be given a negative weight to
cancel structure another neuron's lobe puts in the wrong place.

## The fit

:func:`joint_fit` alternates two steps against the target ``D*(φ) = L(φ − θ ; w)`` — the
decoder's own lobe, at the true object location — over a grid of 1-object training
conditions:

* **amplitudes** — with the phases held, the decode is *linear* in ``a``, so all of them
  come from one ordinary least-squares solve (:func:`amplitude_step`);
* **phases** — with the amplitudes held, each ``ψ_u`` is re-chosen on a grid against the
  *population* residual, one neuron at a time (:func:`phase_step`).

Both steps run in the harmonic domain, where the whole thing is cheap: a lobe centered on
``μ`` has coefficients ``L̂_k·e^{−ikμ}``, and summing a unit's copies gives
``ζ_{u,k}(ψ) = e^{−ikψ}·A_{u,k} + e^{+ikψ}·B_{u,k}`` with ``A``/``B`` precomputed once from
the rotated and reflected copies (:func:`replica_harmonics`). Harmonic ``k = 0`` is left out
of the objective: it is a φ-constant, it cannot move a maximum, and including it would spend
amplitudes matching a pedestal.

## The read-out

:func:`decode` applies ``W``; :func:`local_maxima` finds the decode's circular peaks (with a
height cut that drops the low ripples a lobe read-out leaves far from any object); and
:func:`nearest_max_errors` turns those into the signed angular error of the nearest peak to
each true object, which :func:`bin_error_summary` bins by inter-object angle.

## The additive-code control

:func:`additive_2obj_response` builds the response a strictly **additive** code would give to
``(θ₀, θ₁)`` out of the same population's own 1-object responses, ``[f(θ₀) + f(θ₁)]/2``, and
it goes through the identical decoder. Because the decoder is linear, that decode is exactly
``[D₁(θ₀) + D₁(θ₁)]/2``. It is additive by construction, so it **cannot** be
repelled: any bias it shows is the decoder's own blur, and it is the yardstick the real
response has to be read against.
"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import population_lib as pl
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Circle
from scipy.signal import find_peaks

HSV_VALUE = 0.95  # brightness of every bin color (see :func:`bin_colors`)

#: Ring points every decode is evaluated at — 2 deg, finer than any feature of a lobe this
#: wide, so it is a resolution rather than a modelling choice and is not exposed in the notebook.
N_DECODE = 180
#: The phase search's resolution in :func:`phase_step` — 4 deg, well under any lobe width used.
N_PHASE_GRID = 90


def decode_grid():
    """The ring angles (rad) a decode is evaluated at: ``N_DECODE`` points, evenly spaced."""
    return np.linspace(0, 2 * np.pi, N_DECODE, endpoint=False)


# ── The bump lobes ───────────────────────────────────────────────────────────


def _lobe_weights(means, grid_phi, width_deg):
    """``[n_phi, n_means]`` non-negative cosine lobes of half-width ``width_deg``
    centered on each preferred direction in ``means`` (the core of
    :func:`decoder_weights_from_means`)."""
    mu = np.asarray(means, float)
    half = np.radians(width_deg)
    d = (np.asarray(grid_phi, float)[:, None] - mu[None, :] + np.pi) % (
        2 * np.pi
    ) - np.pi
    return np.cos(np.clip(d, -half, half) * (np.pi / 2 / half))


def decoder_weights_from_means(means, grid_phi, width_deg, amplitudes):
    """Truncated-cosine decoding weights for an array of preferred directions
    ``means`` (rad) — the real units and their rotated/reflected replicas
    (``population_lib.replica_preferred_directions``), each column scaled by its entry of
    ``amplitudes`` (one per mean)."""
    return (
        _lobe_weights(means, grid_phi, width_deg)
        * np.asarray(amplitudes, float)[None, :]
    )


def fit_tuning(units_1obj):
    """Least-squares cosine tuning ``f(θ) = b + A·cos(θ − μ)`` per unit, from that unit's
    own real 1-object trials only.

    Returns ``dict(mu=...)``: each unit's preferred direction μ (rad), ``[n_units]``.
    """
    mu = []
    for theta, fr in units_1obj:
        theta = np.asarray(theta, float)
        fr = np.asarray(fr, float)
        X = np.column_stack(
            [np.ones_like(theta), np.cos(theta), np.sin(theta)]
        )
        _, c1, c2 = np.linalg.lstsq(X, fr, rcond=None)[0]
        mu.append(float(np.arctan2(c2, c1) % (2 * np.pi)))
    return dict(mu=np.asarray(mu))


# ── The joint population fit ─────────────────────────────────────────────────


def lobe_harmonics(width_deg, k_max, n_grid=2048):
    """Fourier coefficients ``L̂_k`` for ``k = 0 … k_max`` of the lobe centered at 0, in the
    convention ``L(Δ) = Σ_k L̂_k·e^{ikΔ}`` (so a lobe centered on ``μ`` contributes
    ``L̂_k·e^{−ikμ}``). The lobe is even, so every ``L̂_k`` is real."""
    phi = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    L = np.asarray(_lobe_weights([0.0], phi, width_deg), float).ravel()
    k = np.arange(k_max + 1)
    with np.errstate(all="ignore"):  # benign stale BLAS flags in matmul
        coef = (np.exp(-1j * k[:, None] * phi[None, :]) @ L) / n_grid
    return coef.real.copy()


def replica_harmonics(R, n_units, num_replicas, k_max):
    """The two phase-independent halves of each unit's harmonic response.

    With ``μ_c = s_c·(ψ_u − o_c)``, a unit's copies contribute
    ``ζ_{u,k}(ψ) = Σ_c r_c·e^{−ikμ_c} = e^{−ikψ}·A_{u,k} + e^{+ikψ}·B_{u,k}`` where
    ``A`` collects the rotated copies (``s = +1``) and ``B`` the reflected ones
    (``s = −1``). Neither depends on ``ψ``, so they are computed once and the phase search
    becomes a table lookup.

    ``R`` is ``[n_samples, n_inputs]`` in the column order of
    :func:`population_lib.population_response_1obj`. Returns ``A``, ``B``, each
    ``[n_samples, n_units, k_max+1]`` complex."""
    specs = pl._replica_specs(num_replicas)
    R = np.asarray(R, float).reshape(len(R), n_units, len(specs))
    k = np.arange(k_max + 1)
    off = np.array([o for o, _ in specs])
    refl = np.array([s for _, s in specs], bool)
    # non-reflected: e^{-ikψ}·Σ r·e^{+iko}; reflected: e^{+ikψ}·Σ r·e^{-iko}
    E_rot = np.exp(1j * k[None, :] * off[:, None])
    E_ref = np.exp(-1j * k[None, :] * off[:, None])
    with np.errstate(all="ignore"):
        A = R[:, :, ~refl] @ E_rot[~refl]
        B = R[:, :, refl] @ E_ref[refl]
    return A, B


def _zeta(A, B, psi, k):
    """``ζ_{u,k}(ψ)`` from the precomputed halves — ``[n_samples, n_units, k_max+1]``."""
    e = np.exp(-1j * k[None, None, :] * np.asarray(psi, float)[None, :, None])
    return A * e + B * np.conj(e)


def _design(A, B, psi, L_hat, k_use):
    """``[n_samples, n_units, len(k_use)]`` complex decode contribution per unit and
    harmonic: ``L̂_k·ζ_{u,k}(ψ_u)``."""
    k = np.arange(A.shape[2])
    Z = _zeta(A, B, psi, k)[:, :, k_use]
    return Z * L_hat[None, None, k_use]


def _stack(M):
    """Complex ``[n_samples, n_units, n_k]`` design → real least-squares rows: one row per
    (sample, harmonic) with the units as columns, real parts stacked over imaginary. The
    transpose matters — it puts the rows in ``(sample, k)`` order, matching how the target
    is raveled."""
    flat = M.transpose(0, 2, 1).reshape(-1, M.shape[1])
    return np.vstack([flat.real, flat.imag])


def amplitude_step(A, B, psi, L_hat, target, k_use):
    """The amplitudes, all of them at once: ordinary least squares of the decode's
    harmonics on the target's, with the phases held fixed. The decode is linear in ``a``,
    so this is exact and unconstrained — **amplitudes may come out negative**, which is
    the joint fit's whole point.

    Nothing is regularized. The solve goes through the pseudo-inverse, so a rank-deficient
    population — which sampling with replacement guarantees, since duplicate neurons are
    exactly collinear — resolves to the minimum-norm solution rather than failing.
    """
    X = _stack(_design(A, B, psi, L_hat, k_use))
    y = np.concatenate(
        [target[:, k_use].real.ravel(), target[:, k_use].imag.ravel()]
    )
    with np.errstate(all="ignore"):
        a = np.linalg.lstsq(X, y, rcond=None)[0]
    return a


def phase_step(A, B, psi, amps, L_hat, target, k_use):
    """One sweep of coordinate descent on the phases: each ``ψ_u`` is re-chosen on an
    ``N_PHASE_GRID``-point grid to minimize the **population** residual with every other
    neuron held fixed. Because ``ζ`` is a closed form in ``ψ``, a whole grid is evaluated
    with two exponentials and a matmul.

    Returns the updated phases (a copy)."""
    k = np.arange(A.shape[2])
    psi = np.asarray(psi, float).copy()
    grid = np.linspace(0, 2 * np.pi, N_PHASE_GRID, endpoint=False)
    contrib = (
        _design(A, B, psi, L_hat, k_use)
        * np.asarray(amps, float)[None, :, None]
    )
    resid = target[:, k_use] - contrib.sum(axis=1)
    Lk = L_hat[k_use]
    for u in np.argsort(-np.abs(amps)):  # strongest neurons first
        if amps[u] == 0:
            continue
        r_u = resid + contrib[:, u, :]  # residual with this neuron removed
        e = np.exp(-1j * grid[:, None] * k[None, k_use])  # [n_grid, n_k]
        cand = (
            amps[u]
            * Lk[None, None, :]
            * (
                A[:, u, k_use][None, :, :] * e[:, None, :]
                + B[:, u, k_use][None, :, :] * np.conj(e)[:, None, :]
            )
        )
        err = np.abs(r_u[None, :, :] - cand) ** 2
        j = int(err.sum(axis=(1, 2)).argmin())
        psi[u] = grid[j]
        contrib[:, u, :] = cand[j]
        resid = r_u - cand[j]
    return psi


def fit_cost(A, B, psi, amps, L_hat, target, k_use):
    """The fraction of the target's harmonic power the objective
    ``Σ_samples Σ_{k∈k_use} |decode_k − target_k|²`` leaves unexplained (``1 − R²``, so lower
    is better).
    """
    D = (
        _design(A, B, psi, L_hat, k_use)
        * np.asarray(amps, float)[None, :, None]
    ).sum(axis=1)
    err = float((np.abs(D - target[:, k_use]) ** 2).sum())
    tot = float((np.abs(target[:, k_use]) ** 2).sum())
    return err / tot if tot > 0 else np.nan


def joint_fit(
    R_train,
    theta_train,
    n_units,
    num_replicas,
    width_deg,
    psi_init,
    k_max,
    n_iter,
):
    """Fit **every neuron's amplitude and phase at once** against the population decoding
    error on 1-object conditions.

    The target for training condition ``θ_i`` is ``L(φ − θ_i ; w)`` — the decoder's own
    lobe, placed at the true location.
    The objective is the squared error between the decode and that target, summed over
    conditions and over harmonics ``k = 1 … k_max``; ``k = 0`` is excluded because a
    φ-constant cannot move a maximum.

    Alternates :func:`amplitude_step` (exact, all amplitudes at once) with
    :func:`phase_step` (coordinate descent, one neuron at a time against the population
    residual) for ``n_iter`` rounds, starting from the phases ``psi_init``. The phase search's
    resolution is ``N_PHASE_GRID`` points (4 deg), well under any width the lobe is used at.

    Returns ``(amps, psi, info)``; ``info['unexplained']`` is the unexplained fraction after
    the first amplitude solve and after every round."""
    A, B = replica_harmonics(R_train, n_units, num_replicas, k_max)
    L_hat = lobe_harmonics(width_deg, k_max)
    k = np.arange(k_max + 1)
    k_use = np.arange(1, k_max + 1)
    target = L_hat[None, :] * np.exp(
        -1j * k[None, :] * np.asarray(theta_train, float)[:, None]
    )

    psi = np.asarray(psi_init, float).copy()
    amps = amplitude_step(A, B, psi, L_hat, target, k_use)
    unexplained = [fit_cost(A, B, psi, amps, L_hat, target, k_use)]
    for _ in range(n_iter):
        psi = phase_step(A, B, psi, amps, L_hat, target, k_use)
        amps = amplitude_step(A, B, psi, L_hat, target, k_use)
        unexplained.append(fit_cost(A, B, psi, amps, L_hat, target, k_use))
    return amps, psi, dict(unexplained=unexplained)


# ── Applying the decoder, and reading its peaks ──────────────────────────────


def decode(W, r):
    """Apply the linear decoder: ``D = W @ r`` (``[n_phi]``), unrectified."""
    with np.errstate(all="ignore"):  # benign stale BLAS flags in matmul
        return np.asarray(W, float) @ np.asarray(r, float)


#: A decode maximum must rise by this fraction of the curve's peak-to-peak range above its
#: surroundings to count, and a curve whose peak-to-peak is below ``FLAT_FRAC`` of its maximum
#: magnitude has no maxima at all.
MIN_PROMINENCE_FRAC = 0.05
FLAT_FRAC = 0.01


def local_maxima(grid_phi, values, min_height_frac=0.0):
    """Angles (rad) of the **circular** local maxima of ``values`` over the ring
    ``grid_phi``. The ring is treated as periodic (tiled ×3 so a peak straddling
    the 0°/360° seam is found). A peak must rise by at least
    ``MIN_PROMINENCE_FRAC`` of the data's peak-to-peak range above its
    surroundings to count (so negligible ripples aren't reported), and if the
    whole curve is essentially flat — peak-to-peak below ``FLAT_FRAC`` of its
    maximum magnitude (or below a tiny absolute floor) — there are no meaningful
    maxima and an empty array is returned. Peaks are sorted by decreasing height.

    ``min_height_frac`` additionally drops peaks that are *low*, keeping only those whose
    height reaches that fraction of the way from the curve's minimum to its maximum,
    ``(v − min) / (max − min) ≥ min_height_frac``. Prominence alone cannot do this: the
    decode is a Fourier partial sum, so it rings, and a ripple far from any object can be
    locally prominent while sitting near the floor. The two populations separate cleanly —
    on this decoder the object bumps score ≥ 0.97 and every ringing lobe ≤ 0.22 — so
    anything in between (0.5 by default in the notebook) removes the ripples and nothing
    else."""
    values = np.asarray(values, float)
    n = len(values)
    ptp = float(values.max() - values.min())
    if ptp <= max(1e-9, FLAT_FRAC * float(np.max(np.abs(values)))):
        return np.empty(0)
    prom = max(MIN_PROMINENCE_FRAC * ptp, 1e-12)
    tiled = np.concatenate([values, values, values])
    idx, _ = find_peaks(tiled, prominence=prom)
    idx = idx[(idx >= n) & (idx < 2 * n)] - n
    if min_height_frac > 0 and len(idx):
        idx = idx[(values[idx] - values.min()) / ptp >= min_height_frac]
    idx = idx[np.argsort(values[idx])[::-1]]
    return np.asarray(grid_phi, float)[idx]


def nearest_max_errors(grid_phi, conds, decodes, min_height_frac=0.0):
    """Angular error of the decode's **nearest local maximum** to each true object,
    for a set of 2-object ("vertical") conditions.

    For every condition ``(t0, t1)`` and its decode ``D`` over ``grid_phi``, find the
    decode's circular local maxima (:func:`local_maxima`) and, for each of the two
    objects, the maximum nearest that object. Returns per-object point arrays
    ``(inter_object_angle_deg, signed_error_deg)`` in the same convention as the
    behavior notebook's error-vs-inter-object-angle plot:

    - **inter-object angle** = non-target minus target, wrapped to ``[0, 360)``;
    - **signed error** = (nearest local max − target), wrapped to ``(-180, 180]``.

    Both objects of a condition contribute a point (each once as the target), so a
    vertically aligned sweep ``(θ0, −θ0)`` fills the whole ``[0, 360)`` x-axis.
    Conditions whose decode is flat (no maxima) are skipped."""
    rel_deg, err_deg = [], []
    for (t0, t1), D in zip(conds, decodes):
        peaks = local_maxima(grid_phi, D, min_height_frac=min_height_frac)
        if len(peaks) == 0:
            continue
        for target, nontarget in ((t0, t1), (t1, t0)):
            d = np.abs((peaks - target + np.pi) % (2 * np.pi) - np.pi)
            nearest = peaks[int(np.argmin(d))]
            err = (nearest - target + np.pi) % (2 * np.pi) - np.pi
            rel = (nontarget - target) % (2 * np.pi)
            rel_deg.append(np.degrees(rel))
            err_deg.append(np.degrees(err))
    return np.asarray(rel_deg), np.asarray(err_deg)


def bin_error_summary(rel_deg, err_deg, bin_centers_deg):
    """Per-bin ``(centers, mean, sd, count)`` of a nearest-max error cloud, binned to the
    nearest of ``bin_centers_deg`` (points beyond the outer edges dropped), for overlaying
    the real and control bin means on one axis."""
    rel_deg = np.asarray(rel_deg, float)
    err_deg = np.asarray(err_deg, float)
    centers = np.asarray(bin_centers_deg, float)
    half = (
        float(np.median(np.diff(centers))) if len(centers) > 1 else 360.0
    ) / 2
    keep = (rel_deg >= centers[0] - half) & (rel_deg <= centers[-1] + half)
    rel_deg, err_deg = rel_deg[keep], err_deg[keep]
    idx = np.abs(rel_deg[:, None] - centers[None, :]).argmin(axis=1)
    out_c, out_m, out_s, out_n = [], [], [], []
    for k, c in enumerate(centers):
        m = idx == k
        if not m.any():
            continue
        out_c.append(c)
        out_m.append(float(err_deg[m].mean()))
        out_s.append(float(err_deg[m].std(ddof=1)) if m.sum() > 1 else 0.0)
        out_n.append(int(m.sum()))
    return (
        np.asarray(out_c),
        np.asarray(out_m),
        np.asarray(out_s),
        np.asarray(out_n),
    )


def additive_2obj_response(units_1obj, t0, t1, num_replicas, sigma_deg):
    """The population response a strictly **additive** code would give to ``(t0, t1)``,
    built from that population's own real 1-object responses: ``[f(t0) + f(t1)]/2``.
    """
    r0 = pl.population_response_1obj(units_1obj, t0, num_replicas, sigma_deg)
    r1 = pl.population_response_1obj(units_1obj, t1, num_replicas, sigma_deg)
    return 0.5 * (r0 + r1)


def bin_colors(n):
    """``n`` RGBA bin colors evenly spaced in hue (red → yellow → green → cyan → blue),
    with their HSV *value* (brightness) set to :data:`HSV_VALUE`.

    The palette of the behavioral response-error figure, reproduced here so the decoder's
    error-vs-inter-object-angle plot is directly comparable to it bin-for-bin. Note the hue
    only spans ``[0, 2/3]`` — it stops at blue rather than wrapping back through magenta to
    red, so the first and last bins stay distinct.
    """
    rgba = plt.cm.hsv(np.linspace(0, 2 / 3, n))
    hsv = mcolors.rgb_to_hsv(rgba[:, :3])
    hsv[:, 2] = HSV_VALUE
    rgba[:, :3] = mcolors.hsv_to_rgb(hsv)
    return rgba


MARK_PAD = 0.16  # how far outside the annulus a stimulus mark sits
R_INNER, R_OUTER = 0.62, 1.0  # the annulus' inner and outer radius


def plot_annulus(
    ax,
    grid_phi,
    values,
    center,
    cmap,
    mark,
    mark_color,
    mark_size,
    stim_ring_color,
    stim_ring_lw,
    stim_ring_alpha,
    peaks,
    peak_color,
    peak_marker,
    peak_size,
    peak_edge,
):
    """Draw ``values`` (one per ``grid_phi`` angle) as a colored **annulus** — a heatmap
    wrapped around the ring — centered at ``(center, 0)``.

    ``mark`` angles are drawn in ``mark_color`` as a dot of area ``mark_size`` just outside
    the ring — the true stimulus objects — on a dashed circle through those dots, at the
    stimulus radius: the same dashed stimulus ring `../efficient_coding_schematic/`'s figures
    draw, and a way to say where an object could have been when the dots alone only say where
    two were. ``peaks`` angles (e.g. from :func:`local_maxima`) are drawn on the annulus band
    in ``peak_color``, with marker ``peak_marker`` at area ``peak_size`` and stroke
    ``peak_edge`` — the decoder's inferred locations. A *filled* peak marker keeps a thin black
    edge, like the stimulus dots, so it stays legible against either end of the colormap; an
    unfilled one ('x', '+') is drawn in ``peak_color`` alone, with ``peak_edge`` as its weight,
    which is what lets it sit over an object dot without hiding it.

    The axis limits are left alone, so several rings can share **one** axis (see
    :func:`annulus_row_limits`), which is the only way to control the gap between them freely.
    The marks and peaks are drawn unclipped, so a dot is never cut off by an axis edge, however
    tight the limits are."""
    values = np.asarray(values, float)
    cx, cy = center, 0.0
    n = len(grid_phi)
    dphi = 2 * np.pi / n
    edges = np.concatenate(
        [np.asarray(grid_phi, float) - dphi / 2, [grid_phi[-1] + dphi / 2]]
    )  # [n+1]
    RR, PHI = np.meshgrid([R_INNER, R_OUTER], edges, indexing="ij")  # [2, n+1]
    X, Y = cx + RR * np.cos(PHI), cy + RR * np.sin(PHI)
    # edgecolors='face' closes the hairline seams between the flat-shaded cells, which
    # are otherwise visible on the annulus at print DPI.
    ax.pcolormesh(
        X,
        Y,
        values[None, :],
        cmap=cmap,
        shading="flat",
        edgecolors="face",
        linewidth=0,
    )
    r_mark = R_OUTER + MARK_PAD
    # the stimulus ring itself: dashed, through the dots, drawn under everything
    ax.add_patch(
        Circle(
            (cx, cy),
            r_mark,
            fill=False,
            color=stim_ring_color,
            lw=stim_ring_lw,
            ls="--",
            alpha=stim_ring_alpha,
            zorder=1,
        )
    )
    # clip_on=False on the dot: the margin is in data units but `mark_size` is an area in
    # POINTS, so how far a dot reaches past r_mark depends on the figure's scale, which the
    # limits cannot know — a dot at the top or bottom of a wide, short row would otherwise
    # come out with a flat edge. Unclipped it is drawn in full and `bbox_inches='tight'`
    # takes it into the saved page.
    for t in mark:
        ax.scatter(
            [cx + r_mark * np.cos(t)],
            [cy + r_mark * np.sin(t)],
            color=mark_color,
            edgecolor="k",
            linewidth=0.5,
            s=mark_size,
            zorder=6,
            clip_on=False,
        )
    r_mid = 0.5 * (R_INNER + R_OUTER)
    # An unfilled marker ('x', '+') is a stroke and nothing else: `scatter` draws it in its
    # FACE colour and ignores any edgecolor, so `peak_color` goes there and `peak_edge` is
    # the weight of the whole marker. A filled one ('o', '*') keeps the black outline the
    # stimulus dots have.
    peak_face = (
        dict(facecolor=peak_color, edgecolor="k")
        if MarkerStyle(peak_marker).is_filled()
        else dict(color=peak_color)
    )
    for t in peaks:  # the decoder's inferred locations
        ax.scatter(
            [cx + r_mid * np.cos(t)],
            [cy + r_mid * np.sin(t)],
            marker=peak_marker,
            s=peak_size,
            linewidths=peak_edge,
            zorder=7,
            clip_on=False,
            **peak_face,
        )
    ax.set_aspect("equal")
    ax.axis("off")


def annulus_row_centers(n_panels, spacing):
    """Ring centers for a row of ``n_panels`` annuli on one axis, ``spacing`` apart in units
    of the ring radius (``2.0`` = rings exactly touching)."""
    return spacing * np.arange(n_panels, dtype=float)


def annulus_row_limits(ax, n_panels, spacing, margin):
    """Frame a row of annuli drawn at :func:`annulus_row_centers` on a single axis: equal
    aspect, no spines, and just ``margin`` of blank space around the row. The row's own
    aspect ratio is then ``(2·lim + (n−1)·spacing) : 2·lim``, so the figure only sets the
    scale — save with ``bbox_inches='tight'`` and the leftover white space is cropped.

    The axis is given the whole figure. The row carries no ticks, labels or title, so the
    default subplot margins would only shrink it: with them the saved PDF comes out ~22 %
    narrower than ``figsize``, and with them gone its width *is* ``figsize[0]``.
    """
    lim = R_OUTER + margin
    centers = annulus_row_centers(n_panels, spacing)
    ax.set_xlim(centers[0] - lim, centers[-1] + lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_position([0.0, 0.0, 1.0, 1.0])

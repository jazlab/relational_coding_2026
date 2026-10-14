"""The **jointly-fit** bump decoder: lobes, the population fit, the read-out, the
additive-code control, and the plots.

Self-contained (per `figures/FORMATTING.md`): pure NumPy/SciPy/matplotlib on top of
:mod:`population_lib` in this directory, touching no cache and no other figure directory.
Firing rates are used exactly as recorded — nothing here normalizes, rescales or
baseline-corrects them.

## The decoder

A **linear** map from a population firing-rate vector ``r`` to a heatmap of object location
over the ring, ``D(φ) = Σ_c W(φ, c)·r_c``, built from **1-object trials only** and then
applied to 2-object trials. The replica construction of :mod:`population_lib` makes it
rotation-equivariant, so its weights are forced to the form ``W(φ, c) = g_u(φ − μ_c)``.
Here that lobe is a bump of a single global width, scaled by one amplitude per neuron::

    g_u(Δ) = a_u · L(Δ ; w)          μ_c = s_c·(ψ_u − offset_c)

leaving exactly two numbers per neuron — an amplitude ``a_u`` (**signed**) and a phase
``ψ_u``. Unlike the independently-fit decoder, both are fit **jointly across the whole
population** by :func:`joint_fit`: one least-squares problem whose residual is the decode's
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
each true object, which :func:`plot_error_vs_interangle` and :func:`bin_error_summary`
display against the inter-object angle.

## The additive-code control

:func:`additive_2obj_response` builds the response a strictly **additive** code would give to
``(θ₀, θ₁)`` out of the same population's own 1-object responses, ``f(θ₀) + f(θ₁) − b``, and
it goes through the identical decoder. Because the decoder is linear, that decode is exactly
``D₁(θ₀) + D₁(θ₁)`` plus a φ-constant. It is additive by construction, so it **cannot** be
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

N_DECODE = (
    180  # ring points every decode is evaluated at — 2 deg, finer than any
)
# feature of a lobe this wide, so it is a resolution rather than a
# modelling choice and is not exposed in the notebook


def decode_grid(n=N_DECODE):
    """The ring angles (rad) a decode is evaluated at: ``n`` points, evenly spaced."""
    return np.linspace(0, 2 * np.pi, n, endpoint=False)


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


def decoder_weights_from_means(
    means, grid_phi, width_deg=90.0, amplitudes=None
):
    """Truncated-cosine decoding weights for an array of preferred directions
    ``means`` (rad) — the real units and their rotated/reflected replicas, whose
    preferred directions come from the fitted 1-object cosine tuning
    (``neural_lib.replica_preferred_directions``). Each lobe peaks at 1 unless ``amplitudes`` (one per
    mean) is given, in which case each column is scaled by its amplitude (e.g.
    proportional to a neuron's SNR)."""
    W = _lobe_weights(means, grid_phi, width_deg)
    if amplitudes is not None:
        W = W * np.asarray(amplitudes, float)[None, :]
    return W


def fit_tuning(units_1obj):
    """Least-squares cosine tuning ``f(θ) = b + A·cos(θ − μ)`` per unit, from that unit's
    own real 1-object trials only.

    Returns a dict of ``[n_units]`` arrays: ``baseline`` b, ``amplitude`` A ≥ 0,
    ``mu`` (preferred direction, rad), ``r2`` (fraction of firing-rate variance the
    tuning explains), ``sigma2`` (residual variance — the per-neuron noise level the
    Gaussian weighting needs), ``mean_rate``, and ``n_trials``.
    """
    out = {
        k: []
        for k in (
            "baseline",
            "amplitude",
            "mu",
            "r2",
            "sigma2",
            "mean_rate",
            "n_trials",
        )
    }
    for theta, fr in units_1obj:
        theta = np.asarray(theta, float)
        fr = np.asarray(fr, float)
        X = np.column_stack(
            [np.ones_like(theta), np.cos(theta), np.sin(theta)]
        )
        b, c1, c2 = np.linalg.lstsq(X, fr, rcond=None)[0]
        with np.errstate(all="ignore"):  # benign stale BLAS flags in matmul
            pred = X @ np.array([b, c1, c2])
        resid = fr - pred
        ss_tot = float(((fr - fr.mean()) ** 2).sum())
        out["baseline"].append(float(b))
        out["amplitude"].append(float(np.hypot(c1, c2)))
        out["mu"].append(float(np.arctan2(c2, c1) % (2 * np.pi)))
        out["r2"].append(
            0.0 if ss_tot < 1e-12 else float(1 - (resid**2).sum() / ss_tot)
        )
        # residual variance with the 3 fitted d.o.f. removed
        out["sigma2"].append(float((resid**2).sum() / max(len(fr) - 3, 1)))
        out["mean_rate"].append(float(fr.mean()))
        out["n_trials"].append(int(len(fr)))
    return {k: np.asarray(v) for k, v in out.items()}


def _per_copy(values, n_copies):
    """A per-real-neuron array repeated to the per-(unit, copy) column layout."""
    return np.repeat(np.asarray(values, float), n_copies)


# ── The joint population fit ─────────────────────────────────────────────────


def lobe_harmonics(lobe_fn, width_deg, k_max, n_grid=2048):
    """Fourier coefficients ``L̂_k`` for ``k = 0 … k_max`` of the lobe centered at 0, in the
    convention ``L(Δ) = Σ_k L̂_k·e^{ikΔ}`` (so a lobe centered on ``μ`` contributes
    ``L̂_k·e^{−ikμ}``). The lobe is even, so every ``L̂_k`` is real."""
    phi = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    L = np.asarray(lobe_fn([0.0], phi, width_deg), float).ravel()
    k = np.arange(k_max + 1)
    with np.errstate(all="ignore"):  # benign stale BLAS flags in matmul
        coef = (np.exp(-1j * k[:, None] * phi[None, :]) @ L) / n_grid
    return coef.real.copy()


def replica_harmonics(
    R, n_units, n_copies, num_replicas, include_reflections, k_max
):
    """The two phase-independent halves of each unit's harmonic response.

    With ``μ_c = s_c·(ψ_u − o_c)``, a unit's copies contribute
    ``ζ_{u,k}(ψ) = Σ_c r_c·e^{−ikμ_c} = e^{−ikψ}·A_{u,k} + e^{+ikψ}·B_{u,k}`` where
    ``A`` collects the rotated copies (``s = +1``) and ``B`` the reflected ones
    (``s = −1``). Neither depends on ``ψ``, so they are computed once and the phase search
    becomes a table lookup.

    ``R`` is ``[n_samples, n_inputs]`` in the column order of
    :func:`population_lib.population_response_1obj`. Returns ``A``, ``B``, each
    ``[n_samples, n_units, k_max+1]`` complex."""
    specs = pl._replica_specs(num_replicas, include_reflections)
    assert len(specs) == n_copies, (len(specs), n_copies)
    R = np.asarray(R, float).reshape(len(R), n_units, n_copies)
    k = np.arange(k_max + 1)
    off = np.array([o for o, _ in specs])
    refl = np.array([s for _, s in specs], bool)
    # non-reflected: e^{-ikψ}·Σ r·e^{+iko}; reflected: e^{+ikψ}·Σ r·e^{-iko}
    E_rot = np.exp(1j * k[None, :] * off[:, None])
    E_ref = np.exp(-1j * k[None, :] * off[:, None])
    with np.errstate(all="ignore"):
        A = R[:, :, ~refl] @ E_rot[~refl]
        B = R[:, :, refl] @ E_ref[refl] if refl.any() else np.zeros_like(A)
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


def phase_step(A, B, psi, amps, L_hat, target, k_use, n_phase_grid=180):
    """One sweep of coordinate descent on the phases: each ``ψ_u`` is re-chosen on an
    ``n_phase_grid``-point grid to minimize the **population** residual with every other
    neuron held fixed. Because ``ζ`` is a closed form in ``ψ``, a whole grid is evaluated
    with two exponentials and a matmul.

    Returns the updated phases (a copy)."""
    k = np.arange(A.shape[2])
    psi = np.asarray(psi, float).copy()
    grid = np.linspace(0, 2 * np.pi, n_phase_grid, endpoint=False)
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
    """The objective: ``Σ_samples Σ_{k∈k_use} |decode_k − target_k|²``, and the fraction of
    the target's harmonic power it leaves unexplained (``1 − R²``, so lower is better).
    """
    D = (
        _design(A, B, psi, L_hat, k_use)
        * np.asarray(amps, float)[None, :, None]
    ).sum(axis=1)
    err = float((np.abs(D - target[:, k_use]) ** 2).sum())
    tot = float((np.abs(target[:, k_use]) ** 2).sum())
    return err, (err / tot if tot > 0 else np.nan)


def joint_fit(
    R_train,
    theta_train,
    n_units,
    n_copies,
    num_replicas,
    include_reflections,
    lobe_fn,
    width_deg,
    psi_init=None,
    k_max=8,
    n_iter=5,
    fit_phases=True,
    n_phase_grid=90,
    verbose=True,
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
    residual) for ``n_iter`` rounds. ``n_phase_grid`` is the phase search's resolution — 90
    points is 4 deg, well under any width the lobe is used at, so it is left as a default
    rather than exposed as a setting. With ``fit_phases=False`` only the amplitudes are fit
    and ``psi_init`` is kept — the pure "joint amplitudes on fixed preferred directions"
    model.

    Returns ``(amps, psi, info)``; ``info`` holds the cost trace, the unexplained fraction,
    and the initial phases."""
    A, B = replica_harmonics(
        R_train, n_units, n_copies, num_replicas, include_reflections, k_max
    )
    L_hat = lobe_harmonics(lobe_fn, width_deg, k_max)
    T_hat = L_hat
    k = np.arange(k_max + 1)
    k_use = np.arange(1, k_max + 1)
    target = T_hat[None, :] * np.exp(
        -1j * k[None, :] * np.asarray(theta_train, float)[:, None]
    )

    psi = (
        np.zeros(n_units)
        if psi_init is None
        else np.asarray(psi_init, float).copy()
    )
    trace = []
    amps = amplitude_step(A, B, psi, L_hat, target, k_use)
    trace.append(fit_cost(A, B, psi, amps, L_hat, target, k_use))
    for it in range(n_iter if fit_phases else 0):
        psi = phase_step(
            A, B, psi, amps, L_hat, target, k_use, n_phase_grid=n_phase_grid
        )
        amps = amplitude_step(A, B, psi, L_hat, target, k_use)
        trace.append(fit_cost(A, B, psi, amps, L_hat, target, k_use))
        if verbose:
            print(
                f"  iter {it + 1}: unexplained {trace[-1][1]:.4f}  "
                f"(was {trace[-2][1]:.4f})"
            )
    return (
        amps,
        psi,
        dict(
            cost=[c for c, _ in trace],
            unexplained=[u for _, u in trace],
            psi_init=(
                None
                if psi_init is None
                else np.asarray(psi_init, float).copy()
            ),
            k_use=k_use,
            lobe_hat=L_hat,
            target_hat=T_hat,
        ),
    )


# ── Fitting the lobe width ───────────────────────────────────────────────────


_WIDTH_SCORES = {
    "mean": np.mean,
    "median": np.median,
    "p90": lambda e: np.percentile(e, 90),
}


# ── Applying the decoder, and reading its peaks ──────────────────────────────


def decode(W, r, rectify=True):
    """Apply the linear decoder: ``D = W @ r`` (``[n_phi]``). ``rectify`` clips the
    read-out at 0 so it reads as a non-negative location heatmap (the raw cosine
    read-out dips negative opposite the decoded location)."""
    with np.errstate(all="ignore"):  # benign stale BLAS flags in matmul
        d = np.asarray(W, float) @ np.asarray(r, float)
    return np.clip(d, 0.0, None) if rectify else d


def local_maxima(
    grid_phi,
    values,
    min_prominence_frac=0.05,
    flat_frac=0.01,
    min_height_frac=0.0,
):
    """Angles (rad) of the **circular** local maxima of ``values`` over the ring
    ``grid_phi``. The ring is treated as periodic (tiled ×3 so a peak straddling
    the 0°/360° seam is found). A peak must rise by at least
    ``min_prominence_frac`` of the data's peak-to-peak range above its
    surroundings to count (so negligible ripples aren't reported), and if the
    whole curve is essentially flat — peak-to-peak below ``flat_frac`` of its
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
    if ptp <= max(1e-9, flat_frac * float(np.max(np.abs(values)))):
        return np.empty(0)
    prom = max(min_prominence_frac * ptp, 1e-12)
    tiled = np.concatenate([values, values, values])
    idx, _ = find_peaks(tiled, prominence=prom)
    idx = idx[(idx >= n) & (idx < 2 * n)] - n
    if min_height_frac > 0 and len(idx):
        idx = idx[(values[idx] - values.min()) / ptp >= min_height_frac]
    idx = idx[np.argsort(values[idx])[::-1]]
    return np.asarray(grid_phi, float)[idx]


def nearest_max_errors(
    grid_phi, conds, decodes, min_prominence_frac=0.05, min_height_frac=0.0
):
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
        peaks = local_maxima(
            grid_phi,
            D,
            min_prominence_frac=min_prominence_frac,
            min_height_frac=min_height_frac,
        )
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
    nearest of ``bin_centers_deg`` (points beyond the outer edges dropped) — the numbers
    behind :func:`plot_error_vs_interangle`, for overlaying the real and control
    bin means on one axis."""
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


def additive_2obj_response(
    units_1obj,
    t0,
    t1,
    num_replicas,
    sigma_deg,
    include_reflections=False,
    baseline=None,
    n_copies=None,
    mode="sum_drive",
):
    """The population response a strictly **additive** code would give to ``(t0, t1)``,
    built from that population's own real 1-object responses.

    - ``'sum_drive'`` (default) — ``f(t0) + f(t1) − b``: each object contributes its own
      drive on top of a **single** shared baseline, matching the retinotopic model
      ``r = b + A·[cos(θ0−φ) + cos(θ1−φ)]``. Needs ``baseline`` and ``n_copies``.
    - ``'mean'`` — ``[f(t0) + f(t1)]/2``, the other common convention.

    The two differ by a per-neuron constant and a positive scale. Under the replica
    construction a constant-across-copies vector decodes to a φ-**constant**, so both
    modes give the same decode up to an affine transform — identical local maxima, and
    therefore an identical bias measurement. The mode is exposed only so that can be
    checked rather than asserted.
    """
    r0 = pl.population_response_1obj(
        units_1obj, t0, num_replicas, sigma_deg, include_reflections
    )
    r1 = pl.population_response_1obj(
        units_1obj, t1, num_replicas, sigma_deg, include_reflections
    )
    if mode == "mean":
        return 0.5 * (r0 + r1)
    if mode != "sum_drive":
        raise ValueError(f"mode must be 'sum_drive' or 'mean' (got {mode!r})")
    if baseline is None or n_copies is None:
        raise ValueError("mode='sum_drive' needs baseline and n_copies")
    return r0 + r1 - _per_copy(baseline, n_copies)[None, :]


def bin_colors(n, value=None):
    """``n`` RGBA bin colors evenly spaced in hue (red → yellow → green → cyan → blue),
    with their HSV *value* (brightness) set to ``value`` (default :data:`HSV_VALUE`).

    The palette of the behavioral response-error figure, reproduced here so the decoder's
    error-vs-inter-object-angle plot is directly comparable to it bin-for-bin. Note the hue only spans ``[0, 2/3]`` — it stops at blue rather than
    wrapping back through magenta to red, so the first and last bins stay distinct.
    """
    rgba = plt.cm.hsv(np.linspace(0, 2 / 3, n))
    hsv = mcolors.rgb_to_hsv(rgba[:, :3])
    hsv[:, 2] = HSV_VALUE if value is None else value
    rgba[:, :3] = mcolors.hsv_to_rgb(hsv)
    return rgba


MARK_PAD = 0.16  # how far outside r_outer a stimulus mark sits
ANNULUS_MARGIN = 0.3  # blank ring of data units kept around a single annulus


def plot_annulus(
    ax,
    grid_phi,
    values,
    r_inner=0.62,
    r_outer=1.0,
    cmap="magma",
    rectify=True,
    symmetric=False,
    vmin=None,
    vmax=None,
    mark=(),
    mark_color="0.5",
    mark_size=42,
    spoke_lw=2.2,
    stim_ring=False,
    stim_ring_color="k",
    stim_ring_lw=0.7,
    stim_ring_ls="--",
    stim_ring_alpha=0.35,
    peaks=(),
    peak_color="0.5",
    peak_marker="*",
    peak_size=150,
    peak_edge=0.7,
    title=None,
    center=0.0,
    set_lim=True,
    margin=ANNULUS_MARGIN,
):
    """Draw ``values`` (one per ``grid_phi`` angle) as a colored **annulus** — a
    heatmap wrapped around the ring. ``grid_phi`` may be the decode grid or the
    neurons' preferred directions (for a firing-rate heatmap).

    ``mark`` angles are drawn in ``mark_color`` as a dot of area ``mark_size`` just outside
    the ring — the true stimulus objects — with a spoke of width ``spoke_lw`` running out to
    each from the annulus (``spoke_lw=None`` or ``0`` draws the dots alone). ``stim_ring``
    adds the dashed circle *through* those dots, at the stimulus radius: the same dashed
    stimulus ring `../efficient_coding/`'s figures and this directory's schematics draw, and
    a way to say where an object could have been when the dots alone only say where two
    were. ``peaks`` angles (e.g. from :func:`local_maxima`)
    are drawn on the annulus band in ``peak_color``, with marker ``peak_marker`` at area
    ``peak_size`` and stroke ``peak_edge`` — the decoder's inferred locations. A *filled*
    peak marker keeps a thin black edge, like the stimulus dots, so it stays legible against
    either end of the colormap; an unfilled one ('x', '+') is drawn in ``peak_color`` alone,
    with ``peak_edge`` as its weight, which is what lets it sit over an object dot without
    hiding it.

    Explicit ``vmin``/``vmax`` fix the color scale (e.g. shared across a row);
    otherwise ``symmetric`` centers it at 0 (for signed quantities like the raw
    weights). ``rectify`` clips negatives to 0 (for location heatmaps).

    ``center`` is the ring's center — a scalar x offset, or an ``(x, y)`` pair — and
    ``set_lim=False`` leaves the axis limits alone. Together they put several rings on
    **one** axis (see :func:`annulus_row_limits`), which is the only way to control the gap
    between them freely: with one subplot per ring the gap cannot go below the blank
    ``margin`` each ring keeps inside its own axes. The marks and peaks are drawn unclipped,
    so ``margin`` sets the blank space around the row rather than what fits inside it — a dot
    is never cut off by an axis edge, however tight the limits are."""
    values = np.asarray(values, float)
    cx, cy = (center, 0.0) if np.isscalar(center) else center
    v = np.clip(values, 0.0, None) if rectify else values
    n = len(grid_phi)
    dphi = 2 * np.pi / n
    edges = np.concatenate(
        [np.asarray(grid_phi, float) - dphi / 2, [grid_phi[-1] + dphi / 2]]
    )  # [n+1]
    RR, PHI = np.meshgrid([r_inner, r_outer], edges, indexing="ij")  # [2, n+1]
    X, Y = cx + RR * np.cos(PHI), cy + RR * np.sin(PHI)
    if vmin is None and vmax is None and symmetric:
        m = np.abs(v).max() or 1.0
        vmin, vmax = -m, m
    # edgecolors='face' closes the hairline seams between the flat-shaded cells, which
    # are otherwise visible on the annulus at print DPI.
    mesh = ax.pcolormesh(
        X,
        Y,
        v[None, :],
        cmap=cmap,
        shading="flat",
        vmin=vmin,
        vmax=vmax,
        edgecolors="face",
        linewidth=0,
    )
    r_mark = r_outer + MARK_PAD
    if stim_ring:
        # the stimulus ring itself: dashed, through the dots, drawn under everything
        ax.add_patch(
            Circle(
                (cx, cy),
                r_mark,
                fill=False,
                color=stim_ring_color,
                lw=stim_ring_lw,
                ls=stim_ring_ls,
                alpha=stim_ring_alpha,
                zorder=1,
            )
        )
    # clip_on=False on the dot: `margin` is in data units but `mark_size` is an area in
    # POINTS, so how far a dot reaches past r_mark depends on the figure's scale, which the
    # limits cannot know — a dot at the top or bottom of a wide, short row would otherwise
    # come out with a flat edge. Unclipped it is drawn in full and `bbox_inches='tight'`
    # takes it into the saved page.
    for t in mark:
        if spoke_lw:
            ax.plot(
                [cx + r_inner * np.cos(t), cx + r_mark * np.cos(t)],
                [cy + r_inner * np.sin(t), cy + r_mark * np.sin(t)],
                color=mark_color,
                lw=spoke_lw,
                zorder=5,
            )
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
    r_mid = 0.5 * (r_inner + r_outer)
    # An unfilled marker ('x', '+') is a stroke and nothing else: `scatter` draws it in its
    # FACE colour and ignores any edgecolor, so `peak_color` goes there and `peak_edge` is
    # the weight of the whole marker. A filled one ('o', '*') keeps the black outline the
    # stimulus dots have. Same rule as ../efficient_coding/task_lib.py, so a peak marker
    # means the same thing in both figures.
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
    if set_lim:
        lim = r_outer + margin
        ax.set_xlim(cx - lim, cx + lim)
        ax.set_ylim(cy - lim, cy + lim)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title)
    return mesh


def annulus_row_centers(n_panels, spacing):
    """Ring centers for a row of ``n_panels`` annuli on one axis, ``spacing`` apart in units
    of the ring radius (``2.0`` = rings exactly touching)."""
    return spacing * np.arange(n_panels, dtype=float)


def annulus_row_limits(
    ax, n_panels, spacing, r_outer=1.0, margin=ANNULUS_MARGIN, fill_figure=True
):
    """Frame a row of annuli drawn at :func:`annulus_row_centers` on a single axis: equal
    aspect, no spines, and just ``margin`` of blank space around the row. The row's own
    aspect ratio is then ``(2·lim + (n−1)·spacing) : 2·lim``, so the figure only sets the
    scale — save with ``bbox_inches='tight'`` and the leftover white space is cropped.

    ``fill_figure`` gives the axis the whole figure. The row carries no ticks, labels or
    title, so the default subplot margins would only shrink it: with them the saved PDF
    comes out ~22 % narrower than ``figsize``, and with them gone its width *is*
    ``figsize[0]``."""
    lim = r_outer + margin
    centers = annulus_row_centers(n_panels, spacing)
    ax.set_xlim(centers[0] - lim, centers[-1] + lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")
    if fill_figure:
        ax.set_position([0.0, 0.0, 1.0, 1.0])
    return centers


# ── Schematics ────────────────────────────────────────────────────────────────
# Data-free diagrams that state what each analysis is *about*, in the same visual language
# as every ring figure in this directory and in `../efficient_coding/`: a dashed stimulus
# ring, a fixation cross at its centre, grey dots for objects and a green star for a
# read-out. Nothing is fit or measured here — every angle is an argument — so these belong
# next to the plotting code rather than in an analysis notebook.
#
# Every size is a keyword argument, and the ones in *data* units are expressed as fractions
# of `ring_radius`, so the whole diagram scales with the ring rather than with the figure.

SCHEMATIC_GREEN = (
    0.15,
    0.70,
    0.15,
)  # the response/peak green used by every ring figure

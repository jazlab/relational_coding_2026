r"""Which 1-object receptive-field locations does one 2-object stimulus drive?

The question
------------
Fix a single 2-object stimulus ``(θ0, θ1)``. Take a population of idealized neurons, all with
the same receptive-field shape and preferred directions tiling the ring. For each neuron ask
one number — its firing rate for THIS stimulus — and plot it **at the neuron's own 1-object
receptive-field direction**. Reading the resulting ring answers "for which receptive-field
locations does this stimulus cause an increase in firing?" — which is the question a motor
decoder's bias is downstream of.

The two families
----------------
Both have the SAME 1-object tuning

      rate_1obj(θ)  =  b + a·g(θ − μ)

with the same bump ``g`` and the same preferred direction ``μ``. They differ in exactly one
thing: **which vector gets fed into that receptive field when there are two objects.**

  retinotopic   the vector from FIXATION to each object.  The 2-object receptive field is
                therefore literally the 1-object one, integrated over both objects:
                    b + a·[g(θ0 − μ) + g(θ1 − μ)]

  relational    the vector from the OTHER OBJECT to each object — equivalently, from the
                pair's midpoint, since  A − midpoint = (A − B)/2  points the same way:
                    b + a·[g(r_AB − μ) + g(r_BA − μ)],   r_AB = ∠(A − B),  r_BA = r_AB + π
                The receptive field is unchanged and is oriented parallel to the 1-object
                preferred direction; only the reference point moved. This is the bias = 1
                end of the relational-bias model of ../../figure_4/models_lib.py.

Both are exchange-symmetric, ``r(θ0, θ1) = r(θ1, θ0)``, like the task.

Useful closed form for the relational family. With objects on the unit ring,
    A − B  ∝  (−sin S, cos S)·2·sin((θ0 − θ1)/2),      S = (θ0 + θ1)/2,
so ``r_AB = S ± π/2`` and the drive collapses to
    b + a·[g(S + π/2 − μ) + g(S − π/2 − μ)],
a function of the pair's midpoint direction S with **two lobes, at μ ± 90°**. A relational
unit is therefore driven hardest when the object-to-object axis points along μ — which is
the same sentence as "the vector between the objects is fed into the receptive field",
and is what the schematic is there to make checkable.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

_ROOT = (
    Path(__file__).resolve().parents[3]
)  # repository root, where paths.py lives
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import paths  # noqa: E402

#: The shared-input cache directory; the notebook prints it.
CACHE_DIR = paths.SHARED_INPUTS

IDEAL_KINDS = ("retinotopic", "relational")
IDEAL_MIN_SEP_DEG = 80.0  # no recorded 2-object trial is closer than this


def circ_dist(a, b):
    """Circular arc distance in [0, π]; broadcasts."""
    d = np.abs(np.asarray(a, float) - np.asarray(b, float)) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Idealized neurons
# ══════════════════════════════════════════════════════════════════════════════
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

    The receptive field ``g(· − μ)`` is the same for both kinds; only the vector fed into it
    changes with ``kind``."""
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


def ideal_stimulus_mean(spec, n_grid, min_sep_deg):
    """The unit's mean rate, after normalisation (:func:`ideal_rate`), over **the task's
    stimulus distribution**: ``θ0`` uniform and the separation uniform on
    ``[min_sep_deg, 180°]``, on an ``n_grid × n_grid`` grid.

    That floor is not cosmetic. No recorded 2-object trial has the objects closer than
    ≈80°, so averaging an idealized unit over *all* separations would compare it against
    stimuli the animal never saw — and the families differ most exactly there.
    """
    g = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    a, b = np.meshgrid(g, g, indexing="ij")
    sep = np.degrees(circ_dist(a, b))
    w = (sep >= min_sep_deg - 1e-9).astype(float)
    w /= w.sum()
    m = float(np.sum(w * ideal_raw_rate(spec, a, b)))
    return float(spec["offset"] + spec["gain"] * m)


def ideal_units(
    n_units,
    kind,
    width_deg,
    baseline,
    amplitude,
    floor_rate,
    peak_rate,
    peak_stimulus,
    min_rate=0.05,
):
    """A pure population of ``n_units`` idealized neurons of one ``kind`` →
    ``(units_df, specs)``. The preferred directions **tile the full circle**, so the
    population is unbiased over orientation.

    Each unit is put on a common rate budget by an affine rescale ``r → offset + gain·r`` that
    maps its response over the rotations of ``peak_stimulus = (θ0, θ1)`` — exactly the set of
    responses the ring plots — onto ``[floor_rate, peak_rate]``. Two families drawn **on one
    colour scale in Hz** then span the same range, so the colours compare pattern rather than
    dynamic range, and the rate cannot go negative by construction. The same offset and gain
    are applied to the 1-object rate, so a unit is one neuron in both conditions.

    ``units_df`` carries a ``theta`` column — each unit's 1-object receptive-field direction,
    the coordinate every dot is positioned by. **``theta`` is ``μ`` and means the same thing
    for both families**: the preferred direction of the one receptive field they share.

    One asymmetry worth knowing. A relational unit's *2-object* response is invariant under
    ``μ → μ + 180°`` (the two lobes at ``μ ± 90°`` simply swap), while its 1-object response
    is not — the same identifiability note the fitted relational model carries. Preferred
    directions are tiled over the full circle regardless, because ``μ`` and ``μ + 180°`` are
    genuinely different neurons in the 1-object condition, which is the condition the figure
    positions dots by.
    """
    if kind not in IDEAL_KINDS:
        raise ValueError(f"kind must be one of {IDEAL_KINDS} (got {kind!r})")
    off = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    mus = 2 * np.pi * np.arange(n_units) / n_units
    specs, rows = [], []
    for i in range(n_units):
        spec = dict(
            kind=kind,
            mu=float(mus[i]),
            width_deg=float(width_deg),
            baseline=float(baseline),
            amplitude=float(amplitude),
        )
        check = ideal_raw_rate(
            spec,
            (peak_stimulus[0] + off) % (2 * np.pi),
            (peak_stimulus[1] + off) % (2 * np.pi),
        )
        span = float(check.max()) - float(check.min())
        spec["gain"] = (peak_rate - floor_rate) / max(span, 1e-12)
        spec["offset"] = floor_rate - spec["gain"] * float(check.min())
        lo = float((spec["offset"] + spec["gain"] * check).min())
        if lo < min_rate:
            raise ValueError(
                f"normalised {kind} rates reach {lo:.3g} Hz < min_rate={min_rate} "
                f"over the rotations of peak_stimulus — lower peak_rate"
            )
        specs.append(spec)
        rows.append(
            dict(
                subject="ideal",
                session=kind,
                unit_id=i,
                kind=kind,
                theta=spec["mu"],
                width_deg=spec["width_deg"],
            )
        )
    return pd.DataFrame(rows), specs


def ideal_rate(spec, t0, t1):
    """The unit's 2-object rate after normalisation — what the figure actually uses."""
    return spec["offset"] + spec["gain"] * ideal_raw_rate(spec, t0, t1)


# ══════════════════════════════════════════════════════════════════════════════
# 2. The rate table
# ══════════════════════════════════════════════════════════════════════════════
def rate_table(units_df, specs, theta0, theta1, n_grid, min_sep_deg):
    """One row per neuron: where its receptive field points, and what this stimulus does to
    it.

    Columns
    -------
    ``subject``, ``session``, ``unit_id``  the neuron
    ``unit_row``                           its row in ``units_df``
    ``mu``                                 its 1-object RF direction, radians in [0, 2π)
    ``rate``                               its rate for ``(theta0, theta1)``, Hz
    ``baseline``                           its mean rate over the task's stimulus
                                           distribution (:func:`ideal_stimulus_mean`), Hz
    """
    q0, q1 = np.array([float(theta0)]), np.array([float(theta1)])
    rows = []
    for u, (spec, (_, meta)) in enumerate(zip(specs, units_df.iterrows())):
        rows.append(
            dict(
                subject=meta["subject"],
                session=meta["session"],
                unit_id=int(meta["unit_id"]),
                unit_row=u,
                mu=float(meta["theta"]) % (2 * np.pi),
                rate=float(ideal_rate(spec, q0, q1)[0]),
                baseline=ideal_stimulus_mean(spec, n_grid, min_sep_deg),
            )
        )
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Figures
# ══════════════════════════════════════════════════════════════════════════════
def diverging_norm(values, center, percentile):
    """A norm centred on ``center``. The limits are a symmetric ``percentile`` of the data's
    distance from ``center``, so a handful of extreme rates cannot flatten the whole map.
    """
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    half = np.percentile(np.abs(v - center), percentile) if len(v) else 1.0
    half = max(float(half), 1e-6)
    return TwoSlopeNorm(
        vcenter=center,
        vmin=min(center - half, center - 1e-9),
        vmax=max(center + half, center + 1e-9),
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
    ax,
    norm,
    center,
    cmap,
    view,
    jitter,
    seed,
    point_size,
    point_alpha,
    title,
):
    """The main figure: **one dot per neuron**, placed at its 1-object receptive field
    direction, coloured by its firing rate.

    The angle of a dot is the only thing that carries information — the radius is pure
    ``jitter``, so that the hundreds of dots at nearly the same angle are visible as a
    density rather than overplotted into one mark. The band of dots ends **exactly on the
    unit circle** and is ``2·jitter`` wide inwards, so with ``view`` set to the stimulus
    panel's axis limit the two panels' rings are the same circle at the same size.

    The colour is diverging around ``center`` under ``norm``: **warm = this stimulus drives
    the neuron above that reference, cool = below**. The dots nearest ``center`` are drawn
    first, so the strong ones are not buried under the null-coloured majority.

    Returns the scatter, for a colorbar.
    """
    mu = table["mu"].to_numpy(float)
    rate = table["rate"].to_numpy(float)
    ok = np.isfinite(mu) & np.isfinite(rate)
    mu, rate = mu[ok], rate[ok]
    rng = np.random.default_rng(seed)
    r = (1.0 - jitter) + rng.uniform(-jitter, jitter, size=len(mu))
    order = np.argsort(np.abs(rate - center))
    mu, rate, r = mu[order], rate[order], r[order]

    sc = ax.scatter(
        mu,
        r,
        c=rate,
        cmap=cmap,
        norm=norm,
        s=point_size,
        alpha=point_alpha,
        linewidths=0,
        zorder=3,
    )
    ax.set_ylim(0, view)
    ax.set_yticks([])
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels([])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)
    ax.set_title(title)
    return sc


def plot_stimulus_ring(
    theta0, theta1, ax, disc_radius, disc_color, view, title
):
    """The stimulus itself: two discs on a dashed unit ring, with the fixation cross at its
    centre.

    Drawn in the style of ``../efficient_coding_schematic/decode_lib.plot_stimuli`` (one panel
    of it, without the reference circles that figure needs to compare stimuli), so a stimulus
    looks the same wherever in figure 3 it is shown.

    The objects are plain discs with **no chord between them**: the panel shows the stimulus
    as the animal sees it — two objects at two ring positions — rather than asserting the
    pair's relative vector, which is one code's reading of the stimulus and not a property
    of it. ``disc_radius`` is in units of the ring radius.

    Drawn on a **polar** axes, so that this panel and the ring panels beside it are the same
    kind of axes: they then get the same box from the layout, the same ring at the same size
    for the same ``view``, and their titles land on one line without any nudging. (A plain
    axes with ``aspect='equal'`` either shrinks its box to a square — dropping its title
    below the others — or needs ``adjustable='datalim'``, which conflicts with fixed limits
    and warns.)
    """
    ax.plot(
        np.linspace(0, 2 * np.pi, 361),
        np.full(361, 1.0),
        color="0.75",
        lw=1.0,
        ls="--",
        zorder=1,
    )
    ax.plot(
        [0],
        [0],
        marker="+",
        color="k",
        markersize=8,
        markeredgewidth=1.3,
        zorder=2,
    )
    for th in (float(theta0), float(theta1)):
        t, r = _polar_circle(1.0, th, disc_radius)
        ax.fill(t, r, facecolor=disc_color, edgecolor="k", lw=0.8, zorder=4)
    ax.set_ylim(0, view)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)
    ax.set_title(title)

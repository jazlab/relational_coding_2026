r"""Anchor angle against conditional preferred direction — the bottom row of the `*_anchor`
notebooks.

What the panel is
-----------------
Take one object of a 2-object stimulus as the **anchor**, hold it near an angle, and ask where
the tuning curve on the *other* object peaks. Plotting that peak against the anchor's own angle
turns the whole retinotopic-versus-relational question into a slope:

* a **retinotopic** unit's preferred direction does not move with the anchor — a horizontal
  line;
* a **relational** unit's runs the opposite way around the ring — a line of slope **-1**.

It is the same panel ``figure_2/rf_per_anchor.ipynb`` draws beside its rings, and it replaces
the firing-rate-against-midpoint line row of ``tuning_plots.ipynb`` / ``figure_tuning.ipynb``.
Those line plots show *how strongly* a unit is driven; this shows *where the drive sits*, which
is the variable the models actually disagree about.

Two sources, one estimator
--------------------------
:func:`data_anchor_fits` measures it from a unit's trials and :func:`model_anchor_fits` from a
fitted model's predicted rates, so the data figure and the model figure can be laid side by
side. **Both end in the same cosine fit** (:func:`fit_cosine`) over the same conditional
domain, because a comparison between them is only about the neuron and the model if the
estimator is held fixed. Where they cannot be identical is stated in each function.

Provenance
----------
:func:`fit_cosine`, :func:`hue_rgb`, :func:`anchor_angles` and :func:`draw_anchor_panel` are
this directory's copies of ``figures/figure_2/rf_lib.py``, so the two figures' panels look
and mean the same. They are copied rather than imported because ``rf_lib`` pulls figure 2's
``common`` (its own ``FR_WINDOW``, loaders and smoother), which would collide with this
directory's. ``test_anchor_lib.py`` asserts numerically that the copies still agree.
"""

import colorsys

import models_lib as ml
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

# ── the tuned look, matching figure_2/rf_lib ──────────────────────────────────
HSV_VALUE = 0.85
POINT_SIZE, LINEWIDTH = 46, 1.8
MARKER, MARKER_FILLED = "o", True
EDGE_LW, OPEN_EDGE_LW = 0.4, 1.6
TICKS = (0, 180, 360)
XLABEL, YLABEL = "Anchor angle (°)", "Preferred direction (°)"

#: The nine translations that tile the panel. Everything is drawn once per offset and then
#: clipped to the axes, which is what makes the panel a torus: a marker or a line leaving one
#: edge appears at the opposite one because its neighbouring copy is standing there.
_TILE_OFFSETS = [
    (dx, dy) for dx in (-360.0, 0.0, 360.0) for dy in (-360.0, 0.0, 360.0)
]


# ── shared helpers ────────────────────────────────────────────────────────────
def circ_dist(a, b):
    """Circular arc distance in [0, pi]; broadcasts."""
    d = np.abs(np.asarray(a, float) - np.asarray(b, float)) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


def fit_cosine(theta, fr):
    """Least-squares ``baseline + amplitude·cos(θ − mean)``; returns ``mean`` in radians, or
    ``None`` if the fit is degenerate."""
    theta, fr = np.asarray(theta, float), np.asarray(fr, float)
    if len(fr) < 3 or not np.isfinite(fr).all():
        return None
    A = np.column_stack([np.ones_like(theta), np.cos(theta), np.sin(theta)])
    b, c1, c2 = np.linalg.lstsq(A, fr, rcond=None)[0]
    if not np.isfinite([c1, c2]).all() or np.hypot(c1, c2) < 1e-12:
        return None  # a flat curve has no preferred direction
    return float(np.arctan2(c2, c1) % (2 * np.pi))


def hue_rgb(angle, value=HSV_VALUE, hue_offset_rad=0.0):
    """RGB for an anchor at ``angle`` (rad): full-saturation rainbow hue at HSV ``value``."""
    hue = ((angle - hue_offset_rad) % (2 * np.pi)) / (2 * np.pi)
    return colorsys.hsv_to_rgb(hue, 1.0, value)


def anchor_angles(n_anchors, delta_theta_deg=0.0):
    """``n_anchors`` evenly spaced anchors (rad), all rotated by ``delta_theta_deg``."""
    return (
        np.radians(delta_theta_deg)
        + 2 * np.pi * np.arange(n_anchors) / n_anchors
    )


# ── measuring it, from trials ─────────────────────────────────────────────────
def data_anchor_fits(
    unit_trials,
    *,
    n_anchors=6,
    delta_theta_deg=5.0,
    window_deg=30.0,
    min_trials=5,
    orientation_deg=0.0
):
    """Conditional preferred direction per anchor, from one unit's **2-object trials**.

    ``unit_trials`` is a ``data_lib.session_unit_trials`` entry — anything with ``t0``, ``t1``
    and ``fr2``. A trial joins an anchor when *either* object lies within ``window_deg`` of it,
    and the *other* object's angle and rate then enter that anchor's tuning curve; taking
    either object keeps the measurement symmetric in an ordering the task does not define.

    ``orientation_deg`` rotates both axes, so a column drawn here lines up with the same
    unit's rings above it, which ``tuning_grid`` rotates by the same angle. Only the drawing
    moves — the trials, the binning and the fits are untouched.

    Returns ``[(anchor_rad, preferred_rad or None), ...]`` in the rotated frame, with ``None``
    for an anchor holding fewer than ``min_trials`` trials.
    """
    t0 = np.asarray(unit_trials["t0"], float)
    t1 = np.asarray(unit_trials["t1"], float)
    fr = np.asarray(unit_trials["fr2"], float)
    window = np.radians(window_deg)
    rot = np.radians(orientation_deg)

    out = []
    for a in anchor_angles(n_anchors, delta_theta_deg):
        raw = a - rot  # the anchor in the unrotated data
        near0, near1 = (
            circ_dist(t0, raw) <= window,
            circ_dist(t1, raw) <= window,
        )
        other = np.concatenate([t1[near0], t0[near1]])
        other_fr = np.concatenate([fr[near0], fr[near1]])
        mean = (
            fit_cosine(other, other_fr)
            if len(other_fr) >= min_trials
            else None
        )
        out.append((a, None if mean is None else mean + rot))
    return out


# ── measuring it, from a fitted model ─────────────────────────────────────────
def model_anchor_fits(
    model,
    params,
    *,
    n_anchors=6,
    delta_theta_deg=5.0,
    min_sep_deg=80.0,
    n_dense=180,
    orientation_deg=0.0,
    dctr=None
):
    """The same quantity, predicted by a fitted model instead of measured from trials.

    For each anchor the model's rate is evaluated with one object pinned there and the other
    swept around the ring, and a cosine is fit to that curve — the identical estimator
    :func:`data_anchor_fits` applies to real trials, so the two figures' bottom rows are
    comparable rather than merely similar.

    Two differences from the data are unavoidable and worth knowing when reading the pair:

    * the anchor is **exact** here, where a measured anchor is a ``window_deg`` bin. A bin
      blurs the conditional curve slightly, so a measured slope is if anything a little
      shallower than the model's.
    * the sweep is restricted to ``min_sep_deg`` away from the anchor, because the task never
      shows two objects closer than 80° and the models are only constrained where it has
      data. Sweeping the full circle would score every model on an extrapolation.

    Returns the same ``[(anchor_rad, preferred_rad or None), ...]`` as the data version.
    """
    rot = np.radians(orientation_deg)
    span = np.radians(180.0 - min_sep_deg)  # half-width of the allowed sweep
    out = []
    for a in anchor_angles(n_anchors, delta_theta_deg):
        raw = a - rot
        other = raw + np.pi + np.linspace(-span, span, n_dense)
        rate = ml.rate_2obj(
            model, params, np.full(other.shape, raw), other, dctr
        )
        mean = fit_cosine(other, np.asarray(rate, float))
        out.append((a, None if mean is None else mean + rot))
    return out


# ── the panel ─────────────────────────────────────────────────────────────────
def draw_anchor_panel(
    ax,
    pairs,
    *,
    value=HSV_VALUE,
    hue_offset_rad=0.0,
    point_size=POINT_SIZE,
    linewidth=LINEWIDTH,
    marker=MARKER,
    filled=MARKER_FILLED,
    edge_linewidth=None,
    ticks=None,
    xlabel=XLABEL,
    ylabel=YLABEL,
    show_xticklabels=True,
    show_yticklabels=True
):
    """Anchor angle against preferred direction, as a coloured line with a point per anchor.

    ``pairs`` is what :func:`data_anchor_fits` / :func:`model_anchor_fits` return. Colour
    codes the anchor angle, the same wheel the rings above use.

    Both axes are angles on [0, 360), so the panel is a **torus** and is drawn as one:
    consecutive anchors are joined the shorter way round in each coordinate, and the whole
    picture — every marker and every segment — is repeated at each of the nine offsets in
    :data:`_TILE_OFFSETS` and clipped to the axes. A marker or line leaving one edge therefore
    appears at the opposite one. The anchors go all the way round, so the last is joined back
    to the first and a perfectly relational unit draws as one unbroken slope-(-1) line rather
    than a set of fragments. Nothing is drawn across the middle to bridge a wrap.

    Anchors with no fit contribute no point; the line joins their neighbours across the gap.
    """
    ok = [(a, m) for a, m in pairs if m is not None]
    if not ok:
        ax.text(
            0.5,
            0.5,
            "no anchor could be fit",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="0.5",
            fontsize=8,
        )
        ax.set_xlim(0, 360)
        ax.set_ylim(0, 360)
        return ax
    a_rad = np.array([a for a, _ in ok])
    x = np.degrees(a_rad) % 360.0
    y = np.degrees(np.array([m for _, m in ok])) % 360.0
    order = np.argsort(x)
    a_rad, x, y = a_rad[order], x[order], y[order]
    colors = [hue_rgb(a, value, hue_offset_rad) for a in a_rad]

    n = len(x)
    pairs_idx = [(i, i + 1) for i in range(n - 1)] + (
        [(n - 1, 0)] if n > 2 else []
    )
    segs, seg_colors = [], []
    for i, j in pairs_idx:
        step = ((a_rad[j] - a_rad[i] + np.pi) % (2 * np.pi)) - np.pi
        seg_colors.append(
            hue_rgb(a_rad[i] + 0.5 * step, value, hue_offset_rad)
        )
        # The short way round in each coordinate, left UNCUT — it may run outside [0, 360).
        # The tiling below brings the overhang back in on the far edge.
        dx = ((x[j] - x[i] + 180.0) % 360.0) - 180.0
        dy = ((y[j] - y[i] + 180.0) % 360.0) - 180.0
        segs.append([(x[i], y[i]), (x[i] + dx, y[i] + dy)])

    if edge_linewidth is None:
        edge_linewidth = EDGE_LW if filled else OPEN_EDGE_LW
    for ox, oy in _TILE_OFFSETS:
        ax.add_collection(
            LineCollection(
                [[(px + ox, py + oy) for px, py in seg] for seg in segs],
                colors=seg_colors,
                linewidths=linewidth,
                zorder=2,
            )
        )
        if filled:
            ax.scatter(
                x + ox,
                y + oy,
                s=point_size,
                color=colors,
                edgecolors="k",
                marker=marker,
                linewidths=edge_linewidth,
                zorder=3,
            )
        else:
            ax.scatter(
                x + ox,
                y + oy,
                s=point_size,
                facecolors="none",
                edgecolors=colors,
                marker=marker,
                linewidths=edge_linewidth,
                zorder=3,
            )

    # The limits are what make the tiling a torus rather than nine plots.
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 360)
    ticks = TICKS if ticks is None else ticks
    ax.set_xticks(list(ticks))
    ax.set_yticks(list(ticks))
    if not show_xticklabels:
        ax.set_xticklabels([])
    if not show_yticklabels:
        ax.set_yticklabels([])
    # Equal aspect: both axes are the same quantity in the same units, so a degree has to be
    # the same length on each — which is what makes a relational unit fall at exactly 45°.
    ax.set_aspect("equal", adjustable="box")
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    # A full box: the content wraps at all four edges, so all four are real boundaries.
    for spine in ax.spines.values():
        spine.set_visible(True)
    return ax


def reference_slopes(ax, color="0.75", lw=0.8, ls=(0, (3, 3)), zorder=1):
    """The two hypotheses as guides: a horizontal line (retinotopic) and slope -1
    (relational), drawn through every wrap. Off by default in both notebooks — they are an
    interpretation aid, not data."""
    for c in (0, 90, 180, 270, 360):
        (
            ax.axhline(c, color=color, lw=lw, ls=ls, zorder=zorder)
            if c == 180
            else None
        )
    for b in (0, 360, 720):
        ax.plot(
            [0, 360], [b, b - 360], color=color, lw=lw, ls=ls, zorder=zorder
        )
    return ax

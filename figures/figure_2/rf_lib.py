"""The receptive-field-per-anchor figures of `rf_per_anchor.ipynb`.

On a 2-object trial one object is treated as the **anchor** and the other as the variable.
Conditioning on the anchor being near each of `n_anchors` positions leaves, for each anchor,
a tuning curve over where the *other* object was — so the question the figures ask is whether
this unit's receptive field moves with the anchor.

Three figures come out of one pass over a unit's trials:

* **the line plots** — one panel per anchor, the raw trials and a cosine fit, with a dot at
  the anchor's angle and a star at the fitted receptive-field centre;
* **the rings** — those same centres as arrows from anchor to centre, and optionally an
  *idealized* ring beside them showing what a perfectly retinotopic or perfectly relational
  version of this unit would look like;
* **the key**, drawn once for all units rather than repeated under each figure.

The styling constants below are the tuned values; they are settled, and the notebook does not
set them.
"""

import colorsys
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

_FIGURES = Path(__file__).resolve().parent.parent
if str(_FIGURES) not in sys.path:
    sys.path.insert(0, str(_FIGURES))
import config  # noqa: E402  the one place selection criteria are set
from common import FR_WINDOW
from common import circ_dist as _circ_dist
from common import load_behavior, load_unit, per_trial_mean_fr, smooth

# ── the tuned look ────────────────────────────────────────────────────────────
HSV_VALUE = 0.85  # brightness of the anchor colours
POINT_SIZE, POINT_ALPHA = 10, 0.3  # the raw per-trial dots
FIT_LINEWIDTH, BAND_ALPHA = (
    2.2,
    0.12,
)  # the cosine fit, and the shaded anchor window
LINE_DOT_SIZE, LINE_DOT_Y = (
    70,
    0.93,
)  # the anchor dot in a line panel, and its height
LINE_STAR_SIZE, LINE_STAR_Y = 170, 0.82  # the RF-centre star, and its height
RING_DOT_SIZE, RING_STAR_SIZE = 150, 180
# The fixation cross at the ring's centre. The stroke is quoted against the size it was tuned
# at, so a caller that changes only the size gets a cross with the same proportions rather than
# a spindly one -- see `_cross_lw`.
RING_CROSS_SIZE, RING_CROSS_LW = 12, 1.8
RING_COLOR, RING_LINEWIDTH = "0.8", 1.2
RING_ARROW_LW, RING_ARROW_HEAD, RING_ARROW_SHRINK = 1.7, 20, 7
RING_VIEW_MAX, RING_SCALE, RING_Y_SHIFT = 1.22, 1.4, -0.04
RING_TITLE_GAP = (
    0.05  # gap between a ring's title and the circle, in axes fractions
)
LEGEND_BAND, LEGEND_Y = 0.16, -0.02
CURVE_POINT_SIZE, CURVE_LINEWIDTH = (
    46,
    1.8,
)  # the anchor-vs-preferred-direction panel
CURVE_MARKER, CURVE_MARKER_FILLED = (
    "o",
    True,
)  # any matplotlib marker code; open or filled
CURVE_EDGE_LW = 0.4  # black outline of a FILLED marker
CURVE_OPEN_EDGE_LW = (
    1.6  # coloured outline of an OPEN one — it is all the colour there is
)
# The curve panel's axis labels carry the marker each axis refers to, so the panel can be read
# against the ring beside it without the key: the anchor is a dot there, the RF centre a star.
# Mathtext rather than the literal characters U+25CF / U+2605 — Arial has no black-star glyph,
# so the unicode form renders as a fallback box (and warns) under this paper's font.
CURVE_TICKS = (
    0,
    180,
    360,
)  # both axes; they are the same quantity, so one setting
CURVE_XLABEL = "Anchor angle (°)"
CURVE_YLABEL = "Preferred dir (°)"


# ── Per-trial firing rates + object angles ────────────────────────────────────
def two_obj_arrays(
    behavior_2obj,
    spike_counts,
    trials,
    window=FR_WINDOW,
    smooth_half_win_ms=50,
):
    """(t0, t1, fr) over the 2-object trials that have spikes: the two object angles
    (rad) and the per-trial mean firing rate."""
    fr = per_trial_mean_fr(
        spike_counts,
        trials,
        behavior_2obj["trial_num"],
        window,
        smooth_half_win_ms,
    )
    rows = behavior_2obj[behavior_2obj["trial_num"].isin(fr)]
    t0 = rows["object_0_theta"].to_numpy() % (2 * np.pi)
    t1 = rows["object_1_theta"].to_numpy() % (2 * np.pi)
    frv = np.array([fr[int(t)] for t in rows["trial_num"]])
    return t0, t1, frv


# ── Symmetric anchor conditioning ─────────────────────────────────────────────
def condition_on_anchor(t0, t1, fr, anchor, window_rad):
    """Trials with one object within ``window_rad`` of ``anchor``; returns the OTHER
    object's angle and firing rate (symmetric in object order)."""
    near0 = _circ_dist(t0, anchor) <= window_rad
    near1 = _circ_dist(t1, anchor) <= window_rad
    other_theta = np.concatenate([t1[near0], t0[near1]])
    other_fr = np.concatenate([fr[near0], fr[near1]])
    return other_theta, other_fr


def fit_cosine(theta, fr):
    """Least-squares ``baseline + amplitude·cos(θ − mean)`` fit."""
    A = np.column_stack([np.ones_like(theta), np.cos(theta), np.sin(theta)])
    b, c1, c2 = np.linalg.lstsq(A, fr, rcond=None)[0]
    pred = b + c1 * np.cos(theta) + c2 * np.sin(theta)
    ss_res = np.sum((fr - pred) ** 2)
    ss_tot = np.sum((fr - np.mean(fr)) ** 2)
    r2 = 0.0 if ss_tot < 1e-12 else float(1 - ss_res / ss_tot)
    return dict(
        mean=float(np.arctan2(c2, c1) % (2 * np.pi)),
        amplitude=float(np.hypot(c1, c2)),
        baseline=float(b),
        r2=r2,
    )


# ── Anchor colors ─────────────────────────────────────────────────────────────
def hue_rgb(angle, value=1.0, hue_offset_rad=0.0):
    """RGB for an anchor at ``angle`` (rad): full-saturation rainbow hue with the given
    HSV ``value`` (brightness) — lower ``value`` = darker.

    ``hue_offset_rad`` rotates the color wheel with the anchors, so the first anchor is
    always red however far the anchor set is rotated (pass the anchors' rotation).
    """
    hue = ((angle - hue_offset_rad) % (2 * np.pi)) / (2 * np.pi)
    return colorsys.hsv_to_rgb(hue, 1.0, value)


def anchor_angles(n_anchors, delta_theta_deg):
    """``n_anchors`` evenly spaced anchors (rad), all rotated by ``delta_theta_deg``."""
    return (
        np.radians(delta_theta_deg)
        + 2 * np.pi * np.arange(n_anchors) / n_anchors
    )


def _axvspan_wrapped(ax, center_deg, half_deg, xmax=360.0, **kwargs):
    """Shade ±``half_deg`` around ``center_deg``, wrapping across the 0/xmax seam."""
    if 2 * half_deg >= xmax:
        ax.axvspan(0, xmax, **kwargs)
        return
    lo, hi = (center_deg - half_deg) % xmax, (center_deg + half_deg) % xmax
    if lo <= hi:
        ax.axvspan(lo, hi, **kwargs)
    else:
        ax.axvspan(lo, xmax, **kwargs)
        ax.axvspan(0, hi, **kwargs)


# ── The figure ────────────────────────────────────────────────────────────────
def anchor_fits(t0, t1, fr, anchors, window_rad, min_trials):
    """Per anchor: the conditioned data and (if enough trials) its cosine fit.

    Returns a list of ``(anchor, other_theta, other_fr, fit_or_None)``.
    """
    out = []
    for a in anchors:
        oth, ofr = condition_on_anchor(t0, t1, fr, a, window_rad)
        fit = fit_cosine(oth, ofr) if len(ofr) >= min_trials else None
        out.append((a, oth, ofr, fit))
    return out


def idealized_fits(fits, kind):
    """Replace every anchor's fitted RF centre with the one an *ideal* unit of ``kind``
    would have, keeping everything else about ``fits`` untouched.

    ``'retinotopic'`` — every RF centre at the **same** direction, taken as the circular
    mean of this unit's own fitted centres. An ideal retinotopic unit's RF does not care
    where the anchor is.

    ``'relational'`` — every RF centre **mirrored** from its anchor about a fixed axis:
    ``centre = chi - anchor``. The axis is estimated from the unit's own data as the
    circular mean of ``anchor + centre``, which is exactly the quantity figure 3's
    relational model is tuned to (`cos((theta_0 + theta_1) - chi)`), so `chi` is the mean
    axis in that model's parameterisation and the mirror line itself points along
    ``chi / 2``. Taking the mean of the *sum* sidesteps the 180-degree ambiguity that
    estimating the mirror line directly would carry.

    Anchors with no fit in the data stay unfitted, so the idealized panel has exactly the
    same anchors marked as the data panel and the two are read side by side.

    Returns ``(fits, param)`` where ``param`` is the fitted centre (retinotopic) or ``chi``
    (relational), in radians.
    """
    fitted = [(a, f["mean"]) for a, _, _, f in fits if f is not None]
    if not fitted:
        return fits, np.nan
    anchors = np.array([a for a, _ in fitted], float)
    means = np.array([m for _, m in fitted], float)

    if kind == "retinotopic":
        param = float(np.angle(np.mean(np.exp(1j * means))))
        centre_of = lambda a: param
    elif kind == "relational":
        param = float(np.angle(np.mean(np.exp(1j * (anchors + means)))))
        centre_of = lambda a: param - a
    else:
        raise ValueError(
            f"idealized must be 'retinotopic', 'relational' or None, "
            f"got {kind!r}"
        )

    out = [
        (
            a,
            oth,
            ofr,
            None if f is None else {**f, "mean": float(centre_of(a))},
        )
        for a, oth, ofr, f in fits
    ]
    return out, param


def _draw_line_panel(
    ax,
    anchor,
    oth,
    ofr,
    fit,
    ymax,
    window_deg,
    color,
    *,
    point_size,
    point_alpha,
    fit_linewidth,
    band_alpha,
    anchor_dot_size,
    anchor_dot_y_frac,
    star_size,
    star_y_frac,
    xticks=(0, 90, 180, 270, 360),
):
    """One anchor's line plot: firing rate vs. the non-anchor object's angle.

    Raw trials (dots) + cosine fit (solid), with the anchor (big dot) and the fitted RF
    center (star) marked near the top of the panel at their angles — the same two symbols
    the ring panel uses.
    """
    adeg = np.degrees(anchor) % 360.0
    _axvspan_wrapped(
        ax, adeg, window_deg, color=color, alpha=band_alpha, zorder=0
    )

    if len(ofr):
        ax.scatter(
            np.degrees(oth),
            ofr,
            s=point_size,
            alpha=point_alpha,
            color=color,
            linewidths=0,
            zorder=1,
        )
    if fit is not None:
        grid = np.linspace(0, 2 * np.pi, 361)
        ax.plot(
            np.degrees(grid),
            fit["baseline"] + fit["amplitude"] * np.cos(grid - fit["mean"]),
            lw=fit_linewidth,
            color=color,
            zorder=4,
        )

    # Anchor: a big dot at the anchor's angle, ``anchor_dot_y_frac`` up the y-range (drawn
    # just inside the axes so `bbox_inches='tight'` keeps it — artists outside their axes
    # are dropped from the tight bounding box). Wrapped, so a 0°/360° anchor shows on both
    # edges.
    dot_y = anchor_dot_y_frac * ymax
    for xa in (adeg - 360.0, adeg, adeg + 360.0):
        if -0.5 <= xa <= 360.5:
            ax.scatter(
                [xa],
                [dot_y],
                s=anchor_dot_size,
                color=color,
                edgecolors="k",
                linewidths=0.6,
                zorder=6,
            )

    # Conditional RF center: star at the cosine peak's angle, ``star_y_frac`` up the
    # y-range (just below the anchor dot). Unclipped, so a peak near 0°/360° still shows as
    # a whole star instead of being cut by the panel's edge.
    if fit is not None:
        ax.scatter(
            [np.degrees(fit["mean"])],
            [star_y_frac * ymax],
            marker="*",
            s=star_size,
            color=color,
            edgecolors="k",
            linewidths=0.5,
            zorder=7,
            clip_on=False,
        )

    ax.set_xlim(0, 360)
    ax.set_ylim(0, ymax)
    ax.set_xticks(list(xticks))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _cross_lw(size, linewidth=None):
    """Stroke width for a fixation cross of ``size``, keeping the tuned proportions.

    ``markersize`` and ``markeredgewidth`` are independent in matplotlib, so scaling only the
    first gives a large cross drawn with a hairline. Unless a width is named outright, it is
    scaled from the pair the look was tuned at.
    """
    if linewidth is not None:
        return linewidth
    return RING_CROSS_LW * float(size) / RING_CROSS_SIZE


def _draw_ring_panel(
    ax,
    fits,
    *,
    value,
    hue_offset_rad,
    ring_color,
    ring_linewidth,
    r_anchor,
    anchor_dot_size,
    star_size,
    arrow_linewidth,
    arrow_head_scale,
    arrow_shrink,
    view_max,
    cross_size=None,
    cross_linewidth=None,
):
    """All anchors on the ring, each with an arrow to its conditional RF center and a
    star at that center."""
    circ = np.linspace(0, 2 * np.pi, 400)
    ax.plot(
        r_anchor * np.cos(circ),
        r_anchor * np.sin(circ),
        color=ring_color,
        lw=ring_linewidth,
        zorder=1,
    )
    for anchor, _oth, _ofr, fit in fits:
        color = hue_rgb(anchor, value, hue_offset_rad)
        if fit is not None:  # anchor → RF center
            m = fit["mean"]
            ax.annotate(
                "",
                xy=(r_anchor * np.cos(m), r_anchor * np.sin(m)),
                xytext=(r_anchor * np.cos(anchor), r_anchor * np.sin(anchor)),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=arrow_linewidth,
                    alpha=0.9,
                    shrinkA=arrow_shrink,
                    shrinkB=arrow_shrink,
                    mutation_scale=arrow_head_scale,
                ),
                zorder=3,
            )
            ax.scatter(
                r_anchor * np.cos(m),
                r_anchor * np.sin(m),
                marker="*",
                s=star_size,
                color=color,
                edgecolors="k",
                linewidths=0.5,
                zorder=5,
            )
        ax.scatter(
            r_anchor * np.cos(anchor),
            r_anchor * np.sin(anchor),
            s=anchor_dot_size,
            color=color,
            edgecolors="k",
            linewidths=0.6,
            zorder=4,
        )
    cross_size = RING_CROSS_SIZE if cross_size is None else cross_size
    ax.plot(
        0,
        0,
        marker="+",
        color="k",
        markersize=cross_size,
        markeredgewidth=_cross_lw(cross_size, cross_linewidth),
        zorder=6,
    )
    ax.set_xlim(-view_max, view_max)
    ax.set_ylim(-view_max, view_max)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.patch.set_alpha(0.0)  # transparent, so a blown-up ring panel never
    for spine in ax.spines.values():  #   hides a neighbor's tick labels
        spine.set_visible(False)  # schematic panel: no box


#: The nine translations that tile the panel. Everything is drawn once per offset and then
#: clipped to the axes, which is what makes the panel a torus: a marker or a line leaving one
#: edge appears at the opposite one because its neighbouring copy is standing there.
_TILE_OFFSETS = [
    (dx, dy) for dx in (-360.0, 0.0, 360.0) for dy in (-360.0, 0.0, 360.0)
]


def _draw_curve_panel(
    ax,
    fits,
    *,
    value,
    hue_offset_rad,
    point_size,
    linewidth,
    edge_linewidth,
    marker="o",
    filled=True,
    ticks=None,
):
    """Anchor angle against the conditional preferred direction, as a coloured line with a
    point per anchor.

    This is the ring panel's content in Cartesian form, and it is what makes the shape of the
    dependence legible: a **retinotopic** unit's preferred direction does not move with the
    anchor, so its points lie on a horizontal line, while a **relational** unit's preferred
    direction runs the opposite way around the ring, giving a line of slope -1. The ring shows
    the same thing as a fan of arrows, where a slope is hard to read off.

    Both axes are angles wrapped to [0, 360), so the panel is a **torus**, and it is drawn as
    one: consecutive anchors are joined by the shorter way round in each coordinate, and the
    whole picture — every marker and every segment — is then repeated at each of the nine
    offsets in :data:`_TILE_OFFSETS` and clipped to the axes. A marker or a line that leaves
    one edge therefore appears at the opposite one, because its neighbouring copy is standing
    there. That includes the step from the last anchor back to the first, since the anchors go
    all the way round, so a perfectly relational unit draws as one unbroken slope-(-1) line
    that wraps rather than as a set of fragments, and points near an edge show their other half
    across the seam instead of being cut off. Nothing is ever drawn across the middle of the
    plot to bridge a wrap.

    Each segment is coloured for the anchor angle **midway between its two endpoints**, taken
    the short way round the colour wheel, so the colour code means the same thing on a line as
    it does on a point and stays saturated across the red-to-red seam.

    Anchors with too few trials to fit (``fit is None``) contribute no point, exactly as they
    leave no arrow on the ring; the line then joins their neighbours across the gap.

    ``marker`` is any matplotlib marker code and ``filled`` chooses a solid marker with a black
    outline or an **open** one. They are not the same edit: colour codes the anchor angle, so
    an open marker has to carry that colour in its *outline* rather than its face, and a
    hairline outline would lose it — hence :data:`CURVE_OPEN_EDGE_LW` rather than
    :data:`CURVE_EDGE_LW` in the open case.

    ``ticks`` sets both axes' ticks (they are the same quantity, so one setting covers both);
    ``None`` uses :data:`CURVE_TICKS`.
    """
    pts = [(a, f["mean"]) for a, _oth, _ofr, f in fits if f is not None]
    if not pts:
        ax.text(
            0.5,
            0.5,
            "no anchor had enough trials",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="0.5",
        )
        return ax
    a_rad = np.array([a for a, _ in pts])
    x = np.degrees(a_rad) % 360.0
    y = np.degrees(np.array([m for _, m in pts])) % 360.0
    order = np.argsort(x)  # `anchor_angles` need not come out sorted
    a_rad, x, y = a_rad[order], x[order], y[order]
    colors = [hue_rgb(a, value, hue_offset_rad) for a in a_rad]

    # Consecutive pairs, closing the loop: the anchors go all the way round, so the last one
    # is the first one's neighbour. Skipped below two points (nothing to join) and at exactly
    # two (the closing pair would redraw the same connection backwards).
    n = len(x)
    pairs = [(i, i + 1) for i in range(n - 1)] + (
        [(n - 1, 0)] if n > 2 else []
    )

    segs, seg_colors = [], []
    for i, j in pairs:
        # The colour of the anchor angle halfway between the two, going the short way — which
        # for adjacent anchors is the hue halfway between their two hues.
        step = ((a_rad[j] - a_rad[i] + np.pi) % (2 * np.pi)) - np.pi
        mid_color = hue_rgb(a_rad[i] + 0.5 * step, value, hue_offset_rad)
        # The short way round in each coordinate, left UNCUT — it may run outside [0, 360).
        # The tiling below is what brings the overhanging part back in on the far edge, so
        # cutting it at the seam here would be doing that same job twice.
        dx = ((x[j] - x[i] + 180.0) % 360.0) - 180.0
        dy = ((y[j] - y[i] + 180.0) % 360.0) - 180.0
        segs.append([(x[i], y[i]), (x[i] + dx, y[i] + dy)])
        seg_colors.append(mid_color)

    # Nine copies of everything, clipped to the axes: see `_TILE_OFFSETS`.
    for ox, oy in _TILE_OFFSETS:
        if segs:
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
            # Open: the colour moves to the outline, and the face is punched out so the line
            # behind stays visible through the marker.
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
    ticks = CURVE_TICKS if ticks is None else ticks
    ax.set_xticks(list(ticks))
    ax.set_yticks(list(ticks))
    # Equal aspect: both axes are the same quantity in the same units, so a degree has to be
    # the same length on each. It is what makes the two hypotheses readable as *angles* —
    # a retinotopic unit is a horizontal line and a relational one falls at exactly 45 deg —
    # rather than as slopes whose steepness is an artefact of the panel's shape.
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(CURVE_XLABEL)
    ax.set_ylabel(CURVE_YLABEL)
    # A full box, unlike every other panel in this directory: the content wraps at all four
    # edges, so all four are meaningful boundaries rather than just axes to read values off.
    for spine in ax.spines.values():
        spine.set_visible(True)
    return ax


def curve_fits(
    t0, t1, fr, *, n_points, delta_theta_deg, window_deg, min_trials
):
    """Per-anchor fits on a **finer** anchor set than the ring uses, for the curve panel.

    The ring carries one arrow per anchor and is unreadable past a handful; the curve is a
    line and wants as many points as the data support. Both are the same computation
    (:func:`anchor_fits`) on the same trials, differing only in how many anchors are asked
    for, so a point on the curve means exactly what an arrow on the ring means.

    ``delta_theta_deg`` is the same rotation the ring uses, so anchor 0 of the curve is anchor
    0 of the ring and the colour wheel lines up between the two panels.

    Note that the anchors **overlap**: each conditions on an object within ``window_deg`` of
    it, and past ``360 / (2 * window_deg)`` points those windows share trials. That is a
    sliding window rather than a partition, which is what makes a smooth curve possible — but
    it also means neighbouring points are not independent, and the curve should not be read as
    if they were.
    """
    anchors = anchor_angles(n_points, delta_theta_deg)
    return anchor_fits(
        t0,
        t1,
        fr,
        anchors,
        window_rad=np.radians(window_deg),
        min_trials=min_trials,
    )


def ring_curve_figure(
    fits,
    curve,
    hue_offset,
    *,
    figsize=(5.4, 2.6),
    width_ratios=(1.0, 1.0),
    point_size=None,
    curve_height_ratio=1.0,
    curve_gap_in=None,
    curve_y_in=0.0,
    cross_size=None,
    marker=None,
    filled=None,
    edge_linewidth=None,
    ticks=None,
):
    """The two-panel figure: the data ring on the left, anchor against preferred direction on
    the right.

    The left panel is the same ring :func:`ring_figure` draws for the data (no idealization
    beside it); the right is :func:`_draw_curve_panel` over ``curve``, which is normally a
    finer anchor set. Sharing ``hue_offset`` is what ties them together — a colour means the
    same anchor angle in both panels, and in the line and ring figures too.

    Both panels are square — the ring because it is a circle, the curve because its two axes
    are the same quantity in the same units — so ``width_ratios`` is 1:1 and the figure is
    sized to fit two squares side by side. Widening it only adds white space.

    ``point_size`` is the area of a curve dot in **points squared**, matplotlib's ``s``, so it
    scales as the square of the diameter: doubling it makes a dot about 1.4x wider, not 2x.
    ``None`` uses :data:`CURVE_POINT_SIZE`. It is a parameter rather than one of the settled
    constants because the right size depends on ``n_curve_points`` — 36 points want smaller
    dots than 12 — and that is the caller's choice.

    ``curve_height_ratio`` sizes the curve panel against the **ring's circle**, not against
    the ring's axes: ``1.0`` makes the curve exactly as tall as the circle is wide. Those are
    different targets because the ring axes runs to :data:`RING_VIEW_MAX` while the circle has
    radius 1, so the circle fills only ``1 / RING_VIEW_MAX`` of the axes and matching the axes
    would leave the curve visibly the taller of the two. Below 1 the curve is smaller than the
    circle, above 1 larger.

    ``curve_gap_in`` is the horizontal space between the panels, in **inches**, measured from
    the ring circle's right edge to the curve's left spine. It has to be given as a distance
    rather than as the gridspec's ``wspace``: both panels are repositioned by hand after the
    layout engine has run (they are sized against each other, which the engine cannot do), so
    whatever ``wspace`` the engine chose has already been overwritten by the time the figure
    is drawn. Measuring from the *circle* rather than from the ring's axes matches how the
    height is set, and is what makes the number mean what it looks like — the ring axes runs
    out to :data:`RING_VIEW_MAX` while the circle stops at 1, so a gap measured from the axes
    edge would look narrower than it reads. Note the curve's y tick labels and axis label sit
    inside this gap, to the left of the spine. ``None`` leaves the panel where the layout
    engine put it and only resizes it.

    ``curve_y_in`` slides the curve panel **vertically**, in inches, positive upwards. ``0``
    centres it on the ring, which is the alignment the two panels want when they are the same
    size — but the ring's drawn content is not centred in its own axes (it carries
    :data:`RING_Y_SHIFT`, and its arrows crowd the top), so a curve panel much smaller than the
    ring can look low even when it is exactly centred. This is the correction for that, and it
    is deliberately a free number rather than something inferred: where the panel *looks* right
    depends on the unit being drawn.

    ``marker`` and ``filled`` choose the scatter's marker (any matplotlib code) and whether it
    is solid or open; ``None`` uses :data:`CURVE_MARKER` / :data:`CURVE_MARKER_FILLED`.
    ``edge_linewidth`` defaults per case — thin and black around a filled marker, thicker and
    coloured around an open one, where the outline is the only thing carrying the colour.
    ``ticks`` sets both of the curve's axes, defaulting to :data:`CURVE_TICKS`.

    Returns ``(fig, (ax_ring, ax_curve))``.
    """
    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=list(width_ratios))

    ax_ring = fig.add_subplot(gs[0, 0])
    _draw_ring_panel(
        ax_ring,
        fits,
        value=HSV_VALUE,
        hue_offset_rad=hue_offset,
        ring_color=RING_COLOR,
        ring_linewidth=RING_LINEWIDTH,
        r_anchor=1.0,
        anchor_dot_size=RING_DOT_SIZE,
        star_size=RING_STAR_SIZE,
        arrow_linewidth=RING_ARROW_LW,
        arrow_head_scale=RING_ARROW_HEAD,
        arrow_shrink=RING_ARROW_SHRINK,
        view_max=RING_VIEW_MAX,
        cross_size=cross_size,
    )

    ax_curve = fig.add_subplot(gs[0, 1])
    filled = CURVE_MARKER_FILLED if filled is None else filled
    if edge_linewidth is None:
        edge_linewidth = CURVE_EDGE_LW if filled else CURVE_OPEN_EDGE_LW
    _draw_curve_panel(
        ax_curve,
        curve,
        value=HSV_VALUE,
        hue_offset_rad=hue_offset,
        point_size=CURVE_POINT_SIZE if point_size is None else point_size,
        linewidth=CURVE_LINEWIDTH,
        edge_linewidth=edge_linewidth,
        marker=CURVE_MARKER if marker is None else marker,
        filled=filled,
        ticks=ticks,
    )

    # Size the curve against the ring's circle. This has to happen after the layout engine
    # has run — only then do the axes have positions to measure — and the engine is then
    # switched off, or it would recompute at draw time and undo the placement. Same pattern,
    # and the same reason, as the ring blow-up in `ring_figure`.
    fig.canvas.draw()
    fig.set_layout_engine("none")
    ring_pos, curve_pos = ax_ring.get_position(), ax_curve.get_position()
    w_in, h_in = fig.get_size_inches()
    # The ring axes spans [-RING_VIEW_MAX, RING_VIEW_MAX] and the circle has radius 1, so the
    # circle's diameter is this fraction of the axes height.
    circle_h = ring_pos.height / RING_VIEW_MAX
    target_h = curve_height_ratio * circle_h
    target_w = (
        target_h * h_in / w_in
    )  # the curve panel is square (equal aspect)
    if curve_gap_in is None:
        left = (
            curve_pos.x0 + 0.5 * curve_pos.width - 0.5 * target_w
        )  # keep the engine's x
    else:
        # Measured from the CIRCLE's right edge, not the ring axes': the axes runs out to
        # RING_VIEW_MAX while the circle stops at 1, so the two differ by a visible margin.
        circle_right = (
            ring_pos.x0
            + 0.5 * ring_pos.width
            + 0.5 * ring_pos.width / RING_VIEW_MAX
        )
        left = circle_right + curve_gap_in / w_in
    # Centred on the ring, so the two panels align, then nudged by `curve_y_in`. The nudge is
    # in inches like `curve_gap_in`, and for the same reason: both panels are repositioned by
    # hand after the layout engine has run, so a fraction-of-figure offset would mean something
    # different at every `figsize`.
    cy = ring_pos.y0 + 0.5 * ring_pos.height + (curve_y_in or 0.0) / h_in
    ax_curve.set_position([left, cy - 0.5 * target_h, target_w, target_h])
    return fig, (ax_ring, ax_curve)


def legend_handles(
    *,
    line_anchor_dot_size,
    line_star_size,
    point_size,
    fit_linewidth,
    color="0.5",
):
    """Proxy handles for the shared legend, in neutral grey (the real colors code the
    anchor's angle). Marker sizes are in points, so they are converted from the ``s``
    (points²) values used in the plots."""
    pt = lambda s: np.sqrt(s)
    return [
        (
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=color,
                markeredgecolor="none",
                markersize=pt(4 * point_size),
            ),
            "2-object trials",
        ),
        (
            Line2D([0], [0], linestyle="-", color=color, lw=fit_linewidth),
            "Cosine fit",
        ),
        (
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=color,
                markeredgecolor="k",
                markeredgewidth=0.6,
                markersize=pt(line_anchor_dot_size),
            ),
            "Anchor",
        ),
        (
            Line2D(
                [0],
                [0],
                marker="*",
                linestyle="none",
                markerfacecolor=color,
                markeredgecolor="k",
                markeredgewidth=0.5,
                markersize=pt(line_star_size),
            ),
            "RF center",
        ),
    ]


def _unit_fits(
    subject,
    session,
    unit_id,
    *,
    n_anchors,
    delta_theta_deg,
    window_deg,
    min_trials,
    smooth_half_win_ms,
    ymax_pad,
    verbose,
):
    """Everything the figures are drawn from: the per-anchor fits, the shared y limit, the
    colour-wheel offset, the trial arrays and the resolved anchor rotation. Computed once so
    the line, ring and curve figures cannot disagree about a unit."""
    if delta_theta_deg is None:
        delta_theta_deg = 180.0 / n_anchors
    behavior_2obj = load_behavior(subject, session, num_objects=2)
    spike_counts, trials, quality_label = load_unit(subject, session, unit_id)
    t0, t1, fr = two_obj_arrays(
        behavior_2obj,
        spike_counts,
        trials,
        smooth_half_win_ms=smooth_half_win_ms,
    )
    if verbose:
        print(
            f"{subject} {session}  unit {unit_id} @ {quality_label}  |  "
            f"{len(fr)} 2-obj trials  |  anchors rotated by {delta_theta_deg:g}°"
        )
    anchors = anchor_angles(n_anchors, delta_theta_deg)
    fits = anchor_fits(
        t0,
        t1,
        fr,
        anchors,
        window_rad=np.radians(window_deg),
        min_trials=min_trials,
    )
    ymax = max(
        [np.nanmax(ofr) if len(ofr) else 0.0 for _, _, ofr, _ in fits] + [0.0]
    )
    # The colour wheel is rotated with the anchors, so the first anchor is always red.
    # `(t0, t1, fr)` comes back too: the curve panel fits a finer anchor set over the same
    # trials, and reading the unit off disk twice to do it would be both slow and a way for
    # the two anchor sets to end up describing different trials.
    return (
        fits,
        (ymax_pad * ymax if ymax > 0 else 1.0),
        np.radians(delta_theta_deg),
        (t0, t1, fr),
        delta_theta_deg,
    )


def line_figure(
    fits, ymax, hue_offset, *, n_anchors=6, window_deg=30.0, figsize=(8.8, 1.8)
):
    """The line-plot figure: one panel per anchor, all in a row.

    Every panel shares the first's axes, so the anchors are directly comparable and only the
    leftmost carries y tick labels. The key is not drawn here — it means the same thing for
    every unit, so :func:`legend_figure` draws it once.
    """
    fig = plt.figure(figsize=figsize, layout="constrained")
    fig.get_layout_engine().set(wspace=0.1, hspace=0.05)
    gs = fig.add_gridspec(1, n_anchors)

    axes = []
    for i in range(n_anchors):
        ax = fig.add_subplot(
            gs[0, i],
            sharex=axes[0] if axes else None,
            sharey=axes[0] if axes else None,
        )
        axes.append(ax)
        anchor, other, other_fr, fit = fits[i]
        _draw_line_panel(
            ax,
            anchor,
            other,
            other_fr,
            fit,
            ymax,
            window_deg,
            hue_rgb(anchor, HSV_VALUE, hue_offset),
            point_size=POINT_SIZE,
            point_alpha=POINT_ALPHA,
            fit_linewidth=FIT_LINEWIDTH,
            band_alpha=BAND_ALPHA,
            anchor_dot_size=LINE_DOT_SIZE,
            anchor_dot_y_frac=LINE_DOT_Y,
            star_size=LINE_STAR_SIZE,
            star_y_frac=LINE_STAR_Y,
            xticks=(0, 180, 360),
        )
        ax.set_xlabel("Non-anchor (°)")
        if i == 0:
            ax.set_ylabel("Firing rate (Hz)")
        else:
            ax.tick_params(labelleft=False)
    return fig, axes


def ring_figure(
    fits,
    hue_offset,
    *,
    idealized=None,
    cross_size=None,
    figsize=(4.0, 1.5),
    verbose=True,
):
    """The ring figure: every anchor on one ring with an arrow to its conditional RF centre,
    and — with `idealized` — a second ring beside it.

    The idealized ring is built from this unit's **own** fits (:func:`idealized_fits`), so the
    comparison is against what this neuron would look like if it were perfectly
    `'retinotopic'` or perfectly `'relational'`, not against another unit's parameters.

    Each title sits just above its **circle** rather than above the axes. The two are not the
    same place: the panel is square and the circle fills only `1 / RING_VIEW_MAX` of it, so an
    axes title would float over a band of empty space.
    """
    fig = plt.figure(figsize=figsize, layout="constrained")
    fig.get_layout_engine().set(wspace=0.05)
    gs = fig.add_gridspec(1, 1 if idealized is None else 2)

    def _ring(col, ring_fits, title):
        ax = fig.add_subplot(gs[0, col])
        _draw_ring_panel(
            ax,
            ring_fits,
            value=HSV_VALUE,
            hue_offset_rad=hue_offset,
            ring_color=RING_COLOR,
            ring_linewidth=RING_LINEWIDTH,
            r_anchor=1.0,
            anchor_dot_size=RING_DOT_SIZE,
            star_size=RING_STAR_SIZE,
            arrow_linewidth=RING_ARROW_LW,
            arrow_head_scale=RING_ARROW_HEAD,
            arrow_shrink=RING_ARROW_SHRINK,
            view_max=RING_VIEW_MAX,
            cross_size=cross_size,
        )
        if title:
            top = (
                0.5 + 0.5 / RING_VIEW_MAX
            )  # the circle's top edge, in axes fractions
            ax.text(
                0.5,
                top + RING_TITLE_GAP,
                title,
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=plt.rcParams["axes.titlesize"],
            )
        return ax

    ring_axes = [_ring(0, fits, "Data" if idealized else None)]
    if idealized is not None:
        ideal, ideal_param = idealized_fits(fits, idealized)
        ring_axes.append(_ring(1, ideal, f"Ideal {idealized}"))
        if verbose:
            label = (
                "RF centre"
                if idealized == "retinotopic"
                else "mean axis chi (mirror line at chi/2)"
            )
            print(
                f"  idealized {idealized}: {label} = "
                f"{np.degrees(ideal_param) % 360:.1f}°"
            )

    # Blow each ring up about its own centre and slide it down a little. Constrained layout
    # is run once and then switched off, or it would recompute — and undo — these positions.
    fig.canvas.draw()
    fig.set_layout_engine("none")
    for ax_ring in ring_axes:
        pos = ax_ring.get_position()
        cx, cy = pos.x0 + 0.5 * pos.width, pos.y0 + 0.5 * pos.height
        w, h = RING_SCALE * pos.width, RING_SCALE * pos.height
        ax_ring.set_position([cx - 0.5 * w, cy - 0.5 * h + RING_Y_SHIFT, w, h])
    return fig, ring_axes


def legend_figure(figsize=(5.2, 0.45), dpi=300):
    """The key on its own: a figure containing nothing but the legend box.

    The symbols mean the same thing in every unit's figures, so repeating the box under each
    one is the same statement printed n times. Drawn once, it can be placed beside the block
    of figures in the layout instead, and the panels get that strip of height back.

    The figure has no axes at all; `bbox_inches='tight'` crops it to the box.
    """
    handles = legend_handles(
        line_anchor_dot_size=LINE_DOT_SIZE,
        line_star_size=LINE_STAR_SIZE,
        point_size=POINT_SIZE,
        fit_linewidth=FIT_LINEWIDTH,
    )
    fig = plt.figure(figsize=figsize, dpi=dpi)
    legend = fig.legend(
        [h for h, _ in handles],
        [l for _, l in handles],
        loc="center",
        ncol=len(handles),
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        edgecolor="0.4",
        borderpad=0.3,
        handlelength=1.4,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_boxstyle("round", pad=0.0, rounding_size=0.15)
    return fig


def rf_per_anchor_figures(
    subject,
    session,
    unit_id,
    *,
    n_anchors=6,
    delta_theta_deg=5.0,
    window_deg=30.0,
    min_trials=None,
    smooth_half_win_ms=50,
    ymax_pad=1.12,
    idealized=None,
    figsize_lines=(8.8, 1.8),
    figsize_rings=None,
    n_curve_points=None,
    figsize_curve=(5.4, 2.6),
    curve_point_size=None,
    curve_height_ratio=1.0,
    curve_gap_in=None,
    curve_y_in=0.0,
    ring_cross_size=None,
    curve_marker=None,
    curve_marker_filled=None,
    curve_edge_linewidth=None,
    curve_ticks=None,
    verbose=True,
):
    """Both figures for one unit, from one pass over its trials.

    Returns `(fig_lines, fig_rings, fig_curve, fits)`. They are separate figures rather than
    one because they are separate objects on a page — a row of tuning curves, a pair of rings,
    a ring beside a curve — and a single figure could satisfy only one of their aspect ratios.
    They still share everything that has to agree: the same trials, the same y limit, the same
    colour wheel, the same symbols.

    `figsize_rings` controls the second figure — the data ring beside an *idealized* one.
    `None` (the default) skips it entirely and returns `fig_rings = None`, the same convention
    `n_curve_points` uses for the third. It defaulted to a size while `figure_2.ai` placed that
    ring; the assembled figure now carries the lines and the curve instead, so building it is
    opt-in. `idealized` is only read when it is built.

    `n_curve_points` controls the third figure: the number of anchors the **curve** panel is
    computed at, which is independent of `n_anchors` because the two panels have different
    limits. The ring carries one arrow per anchor and is unreadable past a handful; the curve
    is a line and wants as many points as the data support. `None` (the default) skips that
    figure entirely and returns `fig_curve = None`, so a caller that does not want it pays
    nothing for it.

    `curve_point_size` is the area of a dot in that panel, in points squared (matplotlib's
    `s`), defaulting to :data:`CURVE_POINT_SIZE`. It goes with `n_curve_points`: the more
    points on the curve, the smaller they have to be to stay separable.

    `ring_cross_size` is the fixation cross at the centre of **every** ring panel, in points
    (matplotlib's `markersize`); its stroke scales with it, so one number keeps the
    proportions. `None` uses :data:`RING_CROSS_SIZE`.

    `curve_height_ratio` is the curve panel's height as a fraction of the ring's **circle
    diameter** (`1.0` makes them equal), `curve_y_in` slides it vertically in inches (positive
    up), `curve_gap_in` the space between the panels in
    inches, `curve_marker` / `curve_marker_filled` / `curve_edge_linewidth` that panel's
    marker, and `curve_ticks` both of its axes. See :func:`ring_curve_figure` for all of
    them.
    """
    # Resolved at call time, never as a default: a default argument freezes the value at
    # import and `config.reload()` would not reach it. See config.reload.
    min_trials = (
        config.MIN_TRIALS_PER_ANCHOR if min_trials is None else min_trials
    )
    fits, ymax, hue_offset, trials, delta_resolved = _unit_fits(
        subject,
        session,
        unit_id,
        n_anchors=n_anchors,
        delta_theta_deg=delta_theta_deg,
        window_deg=window_deg,
        min_trials=min_trials,
        smooth_half_win_ms=smooth_half_win_ms,
        ymax_pad=ymax_pad,
        verbose=verbose,
    )
    fig_lines, _ = line_figure(
        fits,
        ymax,
        hue_offset,
        n_anchors=n_anchors,
        window_deg=window_deg,
        figsize=figsize_lines,
    )
    fig_rings = None
    if figsize_rings is not None:
        fig_rings, _ = ring_figure(
            fits,
            hue_offset,
            idealized=idealized,
            cross_size=ring_cross_size,
            figsize=figsize_rings,
            verbose=verbose,
        )

    fig_curve = None
    if n_curve_points:
        # `delta_resolved` rather than `delta_theta_deg`: the latter may be None, in which
        # case `_unit_fits` chose the rotation. Passing the resolved value is what keeps the
        # curve's anchor 0 on the ring's anchor 0, and so the two panels' colours in register.
        curve = curve_fits(
            *trials,
            n_points=n_curve_points,
            delta_theta_deg=delta_resolved,
            window_deg=window_deg,
            min_trials=min_trials,
        )
        n_fit = sum(f is not None for _, _, _, f in curve)
        if verbose:
            print(
                f"  curve: {n_fit}/{n_curve_points} anchors fit "
                f"(>= {min_trials} trials each, +/-{window_deg:g} deg windows)"
            )
        fig_curve, _ = ring_curve_figure(
            fits,
            curve,
            hue_offset,
            figsize=figsize_curve,
            point_size=curve_point_size,
            curve_height_ratio=curve_height_ratio,
            curve_gap_in=curve_gap_in,
            curve_y_in=curve_y_in,
            cross_size=ring_cross_size,
            marker=curve_marker,
            filled=curve_marker_filled,
            edge_linewidth=curve_edge_linewidth,
            ticks=curve_ticks,
        )
    return fig_lines, fig_rings, fig_curve, fits

"""The receptive-field-per-anchor figures of `rf_per_anchor.ipynb`.

On a 2-object trial one object is treated as the **anchor** and the other as the variable.
Conditioning on the anchor being near each of `n_anchors` positions leaves, for each anchor,
a tuning curve over where the *other* object was — so the question the figures ask is whether
this unit's receptive field moves with the anchor.

Three figures come out of one pass over a unit's trials:

* **the line plots** — one panel per anchor, the raw trials and a cosine fit, with a dot at
  the anchor's angle and a star at the fitted receptive-field centre;
* **the ring and curve** — those same centres as arrows from anchor to centre on a ring,
  beside the anchor angle plotted against the preferred direction;
* **the key**, drawn once for all units rather than repeated under each figure.

The styling constants below are fixed here; the notebook sets only the curve panel's marker,
size, ticks and placement.
"""

import colorsys

import numpy as np
from common import FR_WINDOW
from common import circ_dist as _circ_dist
from common import load_behavior, load_unit, per_trial_mean_fr
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

# ── the look ──────────────────────────────────────────────────────────────────
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
LINE_XTICKS = (0, 180, 360)
YMAX_PAD = (
    1.12  # the line panels' shared y limit, as a multiple of the highest rate
)
RING_DOT_SIZE, RING_STAR_SIZE = 150, 180
# The fixation cross at the ring's centre. The stroke is quoted against a reference size, so a
# caller that changes only the size gets a cross with the same proportions rather than a
# spindly one -- see `_cross_lw`.
RING_CROSS_SIZE, RING_CROSS_LW = 12, 1.8
RING_COLOR, RING_LINEWIDTH = "0.8", 1.2
RING_ARROW_LW, RING_ARROW_HEAD, RING_ARROW_SHRINK = 1.7, 20, 7
RING_VIEW_MAX = 1.22
CURVE_LINEWIDTH = 1.8  # the anchor-vs-preferred-direction panel
CURVE_EDGE_LW = 0.4  # black outline of its markers
# The curve panel's axis labels.
CURVE_XLABEL = "Anchor angle (°)"
CURVE_YLABEL = "Preferred dir (°)"
LEGEND_COLOR = "0.5"  # the key's symbols are neutral grey; the real colours code the anchor


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
    return dict(
        mean=float(np.arctan2(c2, c1) % (2 * np.pi)),
        amplitude=float(np.hypot(c1, c2)),
        baseline=float(b),
    )


# ── Anchor colors ─────────────────────────────────────────────────────────────
def hue_rgb(angle, hue_offset_rad):
    """RGB for an anchor at ``angle`` (rad): full-saturation rainbow hue at HSV value
    :data:`HSV_VALUE`.

    ``hue_offset_rad`` rotates the color wheel with the anchors, so the first anchor is
    always red however far the anchor set is rotated (pass the anchors' rotation).
    """
    hue = ((angle - hue_offset_rad) % (2 * np.pi)) / (2 * np.pi)
    return colorsys.hsv_to_rgb(hue, 1.0, HSV_VALUE)


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


def _draw_line_panel(ax, anchor, oth, ofr, fit, ymax, window_deg, color):
    """One anchor's line plot: firing rate vs. the non-anchor object's angle.

    Raw trials (dots) + cosine fit (solid), with the anchor (big dot) and the fitted RF
    center (star) marked near the top of the panel at their angles — the same two symbols
    the ring panel uses.
    """
    adeg = np.degrees(anchor) % 360.0
    _axvspan_wrapped(
        ax, adeg, window_deg, color=color, alpha=BAND_ALPHA, zorder=0
    )

    if len(ofr):
        ax.scatter(
            np.degrees(oth),
            ofr,
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            color=color,
            linewidths=0,
            zorder=1,
        )
    if fit is not None:
        grid = np.linspace(0, 2 * np.pi, 361)
        ax.plot(
            np.degrees(grid),
            fit["baseline"] + fit["amplitude"] * np.cos(grid - fit["mean"]),
            lw=FIT_LINEWIDTH,
            color=color,
            zorder=4,
        )

    # Anchor: a big dot at the anchor's angle, `LINE_DOT_Y` up the y-range (drawn just inside
    # the axes so `bbox_inches='tight'` keeps it — artists outside their axes are dropped from
    # the tight bounding box). Wrapped, so a 0°/360° anchor shows on both edges.
    dot_y = LINE_DOT_Y * ymax
    for xa in (adeg - 360.0, adeg, adeg + 360.0):
        if -0.5 <= xa <= 360.5:
            ax.scatter(
                [xa],
                [dot_y],
                s=LINE_DOT_SIZE,
                color=color,
                edgecolors="k",
                linewidths=0.6,
                zorder=6,
            )

    # Conditional RF center: star at the cosine peak's angle, `LINE_STAR_Y` up the y-range
    # (just below the anchor dot). Unclipped, so a peak near 0°/360° still shows as a whole
    # star instead of being cut by the panel's edge.
    if fit is not None:
        ax.scatter(
            [np.degrees(fit["mean"])],
            [LINE_STAR_Y * ymax],
            marker="*",
            s=LINE_STAR_SIZE,
            color=color,
            edgecolors="k",
            linewidths=0.5,
            zorder=7,
            clip_on=False,
        )

    ax.set_xlim(0, 360)
    ax.set_ylim(0, ymax)
    ax.set_xticks(list(LINE_XTICKS))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _cross_lw(size):
    """Stroke width for a fixation cross of ``size``, keeping the reference proportions.

    ``markersize`` and ``markeredgewidth`` are independent in matplotlib, so scaling only the
    first gives a large cross drawn with a hairline. The width is scaled from the reference
    pair :data:`RING_CROSS_SIZE` / :data:`RING_CROSS_LW`.
    """
    return RING_CROSS_LW * float(size) / RING_CROSS_SIZE


def _draw_ring_panel(ax, fits, hue_offset_rad, cross_size):
    """All anchors on the ring, each with an arrow to its conditional RF center and a
    star at that center."""
    r_anchor = 1.0
    circ = np.linspace(0, 2 * np.pi, 400)
    ax.plot(
        r_anchor * np.cos(circ),
        r_anchor * np.sin(circ),
        color=RING_COLOR,
        lw=RING_LINEWIDTH,
        zorder=1,
    )
    for anchor, _oth, _ofr, fit in fits:
        color = hue_rgb(anchor, hue_offset_rad)
        if fit is not None:  # anchor → RF center
            m = fit["mean"]
            ax.annotate(
                "",
                xy=(r_anchor * np.cos(m), r_anchor * np.sin(m)),
                xytext=(r_anchor * np.cos(anchor), r_anchor * np.sin(anchor)),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=RING_ARROW_LW,
                    alpha=0.9,
                    shrinkA=RING_ARROW_SHRINK,
                    shrinkB=RING_ARROW_SHRINK,
                    mutation_scale=RING_ARROW_HEAD,
                ),
                zorder=3,
            )
            ax.scatter(
                r_anchor * np.cos(m),
                r_anchor * np.sin(m),
                marker="*",
                s=RING_STAR_SIZE,
                color=color,
                edgecolors="k",
                linewidths=0.5,
                zorder=5,
            )
        ax.scatter(
            r_anchor * np.cos(anchor),
            r_anchor * np.sin(anchor),
            s=RING_DOT_SIZE,
            color=color,
            edgecolors="k",
            linewidths=0.6,
            zorder=4,
        )
    ax.plot(
        0,
        0,
        marker="+",
        color="k",
        markersize=cross_size,
        markeredgewidth=_cross_lw(cross_size),
        zorder=6,
    )
    ax.set_xlim(-RING_VIEW_MAX, RING_VIEW_MAX)
    ax.set_ylim(-RING_VIEW_MAX, RING_VIEW_MAX)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.patch.set_alpha(0.0)  # transparent, so the ring panel never
    for spine in ax.spines.values():  #   hides a neighbor's tick labels
        spine.set_visible(False)  # schematic panel: no box


#: The nine translations that tile the panel. Everything is drawn once per offset and then
#: clipped to the axes, which is what makes the panel a torus: a marker or a line leaving one
#: edge appears at the opposite one because its neighbouring copy is standing there.
_TILE_OFFSETS = [
    (dx, dy) for dx in (-360.0, 0.0, 360.0) for dy in (-360.0, 0.0, 360.0)
]


def _draw_curve_panel(ax, fits, hue_offset_rad, point_size, marker, ticks):
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

    ``marker`` is any matplotlib marker code, drawn filled with a black outline. ``ticks``
    sets both axes' ticks (they are the same quantity, so one setting covers both).
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
    colors = [hue_rgb(a, hue_offset_rad) for a in a_rad]

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
        mid_color = hue_rgb(a_rad[i] + 0.5 * step, hue_offset_rad)
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
                    linewidths=CURVE_LINEWIDTH,
                    zorder=2,
                )
            )
        ax.scatter(
            x + ox,
            y + oy,
            s=point_size,
            color=colors,
            edgecolors="k",
            marker=marker,
            linewidths=CURVE_EDGE_LW,
            zorder=3,
        )

    # The limits are what make the tiling a torus rather than nine plots.
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 360)
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
    """Per-anchor fits on the anchor set of the curve panel.

    The ring carries one arrow per anchor and is unreadable past a handful; the curve is a
    line and can take as many points as the data support. Both are the same computation
    (:func:`anchor_fits`) on the same trials, differing only in how many anchors are asked
    for, so a point on the curve means exactly what an arrow on the ring means.

    ``delta_theta_deg`` is the same rotation the ring uses, so anchor 0 of the curve is anchor
    0 of the ring and the colour wheel lines up between the two panels.

    Note that the anchors **overlap** once there are more than ``360 / (2 * window_deg)`` of
    them: each conditions on an object within ``window_deg`` of it, so the windows then share
    trials. That is a sliding window rather than a partition — neighbouring points are not
    independent, and the curve should not be read as if they were.
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
    figsize,
    point_size,
    curve_height_ratio,
    curve_gap_in,
    curve_y_in,
    cross_size,
    marker,
    ticks,
):
    """The two-panel figure: the data ring on the left, anchor against preferred direction on
    the right.

    The left panel is every anchor on one ring with an arrow to its conditional RF centre; the
    right is :func:`_draw_curve_panel` over ``curve``. Sharing ``hue_offset`` is what ties them
    together — a colour means the same anchor angle in both panels, and in the line figure too.

    Both panels are square — the ring because it is a circle, the curve because its two axes
    are the same quantity in the same units — so the gridspec is 1:1 and the figure is sized
    to fit two squares side by side. Widening it only adds white space.

    ``point_size`` is the area of a curve dot in **points squared**, matplotlib's ``s``, so it
    scales as the square of the diameter: doubling it makes a dot about 1.4x wider, not 2x.

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
    inside this gap, to the left of the spine.

    ``curve_y_in`` slides the curve panel **vertically**, in inches, positive upwards. ``0``
    centres it on the ring, which is the alignment the two panels want when they are the same
    size — but the ring's arrows can crowd one side of it, so a curve panel smaller than the
    ring can look off-centre even when it is exactly centred. This is the correction for that,
    and it is deliberately a free number rather than something inferred: where the panel
    *looks* right depends on the unit being drawn.

    ``cross_size`` is the fixation cross at the ring's centre, in points; ``marker`` is the
    curve scatter's marker (any matplotlib code) and ``ticks`` both of the curve's axes.

    Returns ``(fig, (ax_ring, ax_curve))``.
    """
    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0])

    ax_ring = fig.add_subplot(gs[0, 0])
    _draw_ring_panel(ax_ring, fits, hue_offset, cross_size)

    ax_curve = fig.add_subplot(gs[0, 1])
    _draw_curve_panel(ax_curve, curve, hue_offset, point_size, marker, ticks)

    # Size the curve against the ring's circle. This has to happen after the layout engine
    # has run — only then do the axes have positions to measure — and the engine is then
    # switched off, or it would recompute at draw time and undo the placement.
    fig.canvas.draw()
    fig.set_layout_engine("none")
    ring_pos = ax_ring.get_position()
    w_in, h_in = fig.get_size_inches()
    # The ring axes spans [-RING_VIEW_MAX, RING_VIEW_MAX] and the circle has radius 1, so the
    # circle's diameter is this fraction of the axes height.
    circle_h = ring_pos.height / RING_VIEW_MAX
    target_h = curve_height_ratio * circle_h
    target_w = (
        target_h * h_in / w_in
    )  # the curve panel is square (equal aspect)
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
    cy = ring_pos.y0 + 0.5 * ring_pos.height + curve_y_in / h_in
    ax_curve.set_position([left, cy - 0.5 * target_h, target_w, target_h])
    return fig, (ax_ring, ax_curve)


def legend_handles():
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
                markerfacecolor=LEGEND_COLOR,
                markeredgecolor="none",
                markersize=pt(4 * POINT_SIZE),
            ),
            "2-object trials",
        ),
        (
            Line2D(
                [0], [0], linestyle="-", color=LEGEND_COLOR, lw=FIT_LINEWIDTH
            ),
            "Cosine fit",
        ),
        (
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=LEGEND_COLOR,
                markeredgecolor="k",
                markeredgewidth=0.6,
                markersize=pt(LINE_DOT_SIZE),
            ),
            "Anchor",
        ),
        (
            Line2D(
                [0],
                [0],
                marker="*",
                linestyle="none",
                markerfacecolor=LEGEND_COLOR,
                markeredgecolor="k",
                markeredgewidth=0.5,
                markersize=pt(LINE_STAR_SIZE),
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
):
    """Everything the figures are drawn from: the per-anchor fits, the shared y limit, the
    colour-wheel offset and the trial arrays. Computed once so the line, ring and curve
    panels cannot disagree about a unit."""
    behavior_2obj = load_behavior(subject, session, num_objects=2)
    spike_counts, trials, quality_label = load_unit(subject, session, unit_id)
    t0, t1, fr = two_obj_arrays(
        behavior_2obj,
        spike_counts,
        trials,
        smooth_half_win_ms=smooth_half_win_ms,
    )
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
    # `(t0, t1, fr)` comes back too: the curve panel fits its own anchor set over the same
    # trials, and reading the unit off disk twice to do it would be both slow and a way for
    # the two anchor sets to end up describing different trials.
    return (
        fits,
        (YMAX_PAD * ymax if ymax > 0 else 1.0),
        np.radians(delta_theta_deg),
        (t0, t1, fr),
    )


def line_figure(fits, ymax, hue_offset, *, n_anchors, window_deg, figsize):
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
            hue_rgb(anchor, hue_offset),
        )
        ax.set_xlabel("Non-anchor (°)")
        if i == 0:
            ax.set_ylabel("Firing rate (Hz)")
        else:
            ax.tick_params(labelleft=False)
    return fig, axes


def legend_figure(figsize):
    """The key on its own: a figure containing nothing but the legend box.

    The symbols mean the same thing in every unit's figures, so repeating the box under each
    one is the same statement printed n times. Drawn once, it can be placed beside the block
    of figures in the layout instead, and the panels get that strip of height back.

    The figure has no axes at all; `bbox_inches='tight'` crops it to the box.
    """
    handles = legend_handles()
    fig = plt.figure(figsize=figsize, dpi=300)
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
    n_anchors,
    delta_theta_deg,
    window_deg,
    min_trials,
    smooth_half_win_ms,
    figsize_lines,
    n_curve_points,
    figsize_curve,
    curve_point_size,
    curve_height_ratio,
    curve_gap_in,
    curve_y_in,
    ring_cross_size,
    curve_marker,
    curve_ticks,
):
    """Both figures for one unit, from one pass over its trials.

    Returns `(fig_lines, fig_curve)`. They are separate figures rather than one because they
    are separate objects on a page — a row of tuning curves, and a ring beside a curve — and a
    single figure could satisfy only one of their aspect ratios. They still share everything
    that has to agree: the same trials, the same colour wheel, the same symbols.

    `n_anchors` anchors, rotated by `delta_theta_deg`, give the line panels and the ring's
    arrows; anchors with fewer than `min_trials` trials within `window_deg` get no fit.

    `n_curve_points` is the number of anchors the **curve** panel is computed at, which is
    independent of `n_anchors` because the two panels have different limits. The ring carries
    one arrow per anchor and is unreadable past a handful; the curve is a line and can take as
    many points as the data support.

    `curve_point_size` is the area of a dot in that panel, in points squared (matplotlib's
    `s`). It goes with `n_curve_points`: the more points on the curve, the smaller they have to
    be to stay separable.

    `ring_cross_size` is the fixation cross at the centre of the ring, in points (matplotlib's
    `markersize`); its stroke scales with it, so one number keeps the proportions.

    `curve_height_ratio` is the curve panel's height as a fraction of the ring's **circle
    diameter** (`1.0` makes them equal), `curve_y_in` slides it vertically in inches (positive
    up), `curve_gap_in` the space between the panels in inches, `curve_marker` that panel's
    marker, and `curve_ticks` both of its axes. See :func:`ring_curve_figure` for all of them.
    """
    fits, ymax, hue_offset, trials = _unit_fits(
        subject,
        session,
        unit_id,
        n_anchors=n_anchors,
        delta_theta_deg=delta_theta_deg,
        window_deg=window_deg,
        min_trials=min_trials,
        smooth_half_win_ms=smooth_half_win_ms,
    )
    fig_lines, _ = line_figure(
        fits,
        ymax,
        hue_offset,
        n_anchors=n_anchors,
        window_deg=window_deg,
        figsize=figsize_lines,
    )

    # The same rotation as the ring's anchors, which keeps the curve's anchor 0 on the ring's
    # anchor 0, and so the two panels' colours in register.
    curve = curve_fits(
        *trials,
        n_points=n_curve_points,
        delta_theta_deg=delta_theta_deg,
        window_deg=window_deg,
        min_trials=min_trials,
    )
    n_fit = sum(f is not None for _, _, _, f in curve)
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
        ticks=curve_ticks,
    )
    return fig_lines, fig_curve

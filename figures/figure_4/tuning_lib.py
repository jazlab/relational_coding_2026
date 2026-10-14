"""The tuning column of ``figure_2/psth/psth.ipynb``, for several units side by side.

This module deliberately contains **no** tuning logic of its own: every firing rate,
prototype set, color scale and ring drawer comes from ``figure_2/psth/psth_lib.py`` (added
to ``sys.path`` below), so the panels here are the same objects as the right-hand column of
the single-unit PSTH figures, with the same parameters. All this file adds is the grid
layout: one **column per unit**, and the three rows of ``psth_lib.build_rows``

    row 0 : 1 object            — a continuous shaded annulus over object direction
    row 1 : ``sep1_deg`` apart  — a dense family of same-separation chords
    row 2 : 180° apart          — the antipodal family (spans half a turn)

plus an optional **bottom row of line plots** (:func:`draw_tuning_lineplot`), which shows the
same tuning curves as ordinary curves — firing rate against direction — with one line per
inter-object angle, so the same-separation families of several separations can be compared
directly (see :func:`unit_curves`).

Color is the mean firing rate over ``fr_window`` (stimulus + delay), cool-warm, and — as in
``psth.ipynb`` — each row is normalized to **its own** firing-rate range, which here means
*per panel*: colors are comparable within a panel, never across units. Every panel therefore
carries its own range label (``low–high Hz``), and the colorbar drawn at the right is a key
for the color→rate direction only, with no numbers on it.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# The PSTH toolkit of figure 2 — reused as-is (see the module docstring).
#
# It has to be imported with figure 2's directory FIRST on the path, because `psth_lib`
# does `import common` and figure 4 has a `common` of its own. Both are then put back:
# `psth_lib` keeps a direct reference to the module object it imported, so restoring
# `sys.modules['common']` afterwards leaves it working and leaves `import common` in this
# directory meaning what it should.
_PSTH_DIR = (Path(__file__).resolve().parent / "../figure_2").resolve()
_saved_common = sys.modules.pop("common", None)
sys.path.insert(0, str(_PSTH_DIR))
try:
    import psth_lib as pl  # noqa: E402  (path set up above)
finally:
    sys.path.remove(str(_PSTH_DIR))
    sys.modules.pop("common", None)
    if _saved_common is not None:
        sys.modules["common"] = _saved_common

COOLWARM = pl.COOLWARM
# ... and the brightened-red variant of it, so this directory's figures can share the red
# that figure_3/model_comparison/ideal_neurons.ipynb introduced.
boosted_cmap = pl.boosted_cmap

#: The colormap every panel is drawn in: coolwarm with its hot end lifted in HSV at constant
#: hue. Every panel's scale runs the whole map, so the extreme red is worth setting
#: deliberately — this brightens it without changing which red it is, where cutting the map
#: short would only fade it towards the pale centre.
FIRING_RATE_CMAP = pl.boosted_cmap(value=0.9, saturation=1.0, blend=0.5)


#: The binning every panel of this figure shares with ``figures/figure_2/psth.ipynb``.
#: Every value here is that notebook's, so the panels are directly comparable — except
#: ``SMOOTH_HALF_WIN_MS``, which is 250 there. That one is temporal, and a mean over a 2 s
#: window barely feels it, but the two figures are not identical until it is reconciled.
SEP1_DEG = 90  # inter-object angle of the middle ring row
SIMILARITY_WINDOW_DEG = (
    15.0  # a trial joins a prototype within this angular distance
)
SMOOTH_HALF_WIN_MS = (
    200.0  # temporal smoothing half-window of the firing rates
)
FR_WINDOW_MS = (0, 2000)  # stimulus + delay, stopping before the response
N_DENSE = 18  # 2-object prototypes per ring panel ...
N_DENSE_1OBJ = 180  # ... and the resolution of the 1-object annulus


def unit_rows(
    subject,
    session,
    unit_id,
    orientation_deg=0.0,
    similarity_window_deg=SIMILARITY_WINDOW_DEG,
):
    """``psth_lib.build_rows`` for one unit, keeping only what the tuning column needs.

    ``similarity_window_deg`` is the pooling window shared with :func:`unit_curves`: a trial
    joins a prototype when **both** of its objects are within it of their own slot. Pass the
    same value to both, or the ring rows and the line row stop measuring the same thing.
    """
    _, _, rows = pl.build_rows(
        subject,
        session,
        unit_id,
        orientation_deg=orientation_deg,
        sep1_deg=SEP1_DEG,
        similarity_window_deg=similarity_window_deg,
        smooth_half_win_ms=SMOOTH_HALF_WIN_MS,
        fr_window=FR_WINDOW_MS,
        n_dense=N_DENSE,
        n_dense_1obj=N_DENSE_1OBJ,
        share_2obj_color_scale=True,
    )
    return rows


def unit_curves(
    subject,
    session,
    unit_id,
    seps_deg,
    orientation_deg=0.0,
    n_dense=36,
    similarity_window_deg=SIMILARITY_WINDOW_DEG,
    fr_window=FR_WINDOW_MS,
    smooth_half_win_ms=SMOOTH_HALF_WIN_MS,
):
    """Dense 2-object tuning curves of one unit, one per inter-object angle.

    Returns ``{sep_deg: (phis_rad, mean_frs)}``, where ``phis`` are ``n_dense`` directions
    evenly spanning the full circle from ``orientation_deg`` and the firing rates come from
    the same similarity-window binning as every other panel of the figure
    (:func:`psth_lib.dense_2obj_tuning`).

    ``similarity_window_deg`` is what sets the curves' angular resolution, and it is a much
    bigger effect than the cosmetic Gaussian applied at draw time
    (``tuning_grid(line_smooth_deg_2obj=...)``). Both objects must be within the window of
    their own slot, which for a pair works out to ``|Δmean| + |Δseparation| / 2 <= window``:
    along the x axis of the line row it is a top-hat of half-width ``window``, so the drawn
    curve has a FWHM of about ``2 · window`` however fine ``n_dense`` is. At the default 15°
    — ``figure_2/psth.ipynb``'s value — neighbouring samples' windows still overlap threefold
    on the 10° grid, so adjacent points are not independent; shrinking it further sharpens the
    curves and costs trials per sample.

    ``phis`` is the direction of the **midpoint of the segment joining the two objects**:
    the pair of a prototype is ``(phi - sep/2, phi + sep/2)``, whose cartesian midpoint sits
    at radius ``R·cos(sep/2)`` in direction ``phi`` — i.e. the circular mean of the two
    object angles, taken on the side where the objects are.

    At ``sep = 180`` that midpoint is the origin and the direction is undefined: the two
    candidates ``phi`` and ``phi + 180`` are equidistant. Nothing special is needed for it
    here, because the prototype pairs ``(phi ± 90)`` and ``(phi + 180 ± 90)`` are *the same
    unordered pair*, so the two candidates are assigned exactly the same trials and the
    curve comes out 180°-periodic on its own — which is the mean over both candidates.
    """
    behavior = pl.load_behavior(subject, session, num_objects=2)
    spike_counts, trials, _ = pl.load_unit(subject, session, unit_id)
    offset = np.radians(orientation_deg)
    return {
        sep: pl.dense_2obj_tuning(
            behavior,
            spike_counts,
            trials,
            sep,
            n_dense,
            window_rad=np.radians(similarity_window_deg),
            offset_rad=offset,
            fr_window=fr_window,
            smooth_half_win_ms=smooth_half_win_ms,
        )
        for sep in seps_deg
    }


def sep_colors(seps_deg, cmap="viridis", lo=0.05, hi=0.9):
    """One colour per inter-object angle, sampled across ``cmap`` — the same mapping
    ``../tuning_plots/tuning_plots.ipynb`` uses, so the two figures' legends agree.
    """
    cm = plt.get_cmap(cmap)
    n = max(len(seps_deg) - 1, 1)
    return {
        float(s): cm(lo + (hi - lo) * i / n) for i, s in enumerate(seps_deg)
    }


class FlatCurve(ValueError):
    """Raised by :func:`rf_center` for a curve with no peak to centre on."""


def rf_center(phis, frs, method="peak", smooth_deg=0.0, flat_tol=1e-9):
    """Direction (rad) of the centre of a tuning curve, used to put 0° at the receptive field.

    ``'peak'``: the direction of the largest (smoothed) rate. ``'com'``: the circular centre of
    mass above the curve's minimum, less sensitive to one noisy sample but pulled off the peak by
    a secondary lobe — which several of these models have, so ``'peak'`` is the default.

    **Raises** :class:`FlatCurve` when the curve has no peak — every sample within ``flat_tol``
    (relative) of every other. That is not a pedantic guard. A *model's* predicted tuning can be
    exactly flat: `future_field` fitted with ``amplitude_retinotopic = 0`` predicts no 1-object
    tuning whatsoever, and `numpy.nanargmax` on a constant array returns index 0, so this used to
    return ``phis[0]`` — the first sample of the angle grid, i.e. wherever ``orientation_deg``
    happens to start it. The whole column was then drawn against a meaningless origin, and it
    looked like a phase bug in the model rather than a missing centre. Callers that have a
    sensible fallback (the model's own ``theta``) should catch this; see
    :func:`model_tuning_grid`.
    """
    y = np.asarray(frs, float)
    if smooth_deg > 0:
        y = pl._circ_smooth(y, smooth_deg)
    phis = np.asarray(phis, float)
    scale = max(float(np.nanmax(np.abs(y))), 1.0)
    if float(np.nanmax(y) - np.nanmin(y)) <= flat_tol * scale:
        raise FlatCurve(
            f"the curve is flat at {float(np.nanmin(y)):.4g}, so it has no peak and no centre. "
            f"Catch this and fall back to the model's theta rather than centring on an "
            f"arbitrary sample."
        )
    if method == "peak":
        return float(phis[np.nanargmax(y)])
    if method == "com":
        w = np.nan_to_num(y - np.nanmin(y))
        return float(
            np.arctan2((w * np.sin(phis)).sum(), (w * np.cos(phis)).sum())
        )
    raise ValueError(f"unknown rf_center method {method!r}")


def _unwrapped_deg(phis, frs, center_rad=None, smooth_deg=0.0):
    """``(x_deg, y, span)`` for one line: sorted, optionally circularly smoothed, and padded by
    one wrapped sample at each end so the drawn line runs continuously across the whole x range
    instead of stopping short of the axis limits."""
    x = np.degrees(np.asarray(phis, float))
    if center_rad is None:
        x, span = x % 360.0, (0.0, 360.0)
    else:
        x = (x - np.degrees(center_rad) + 180.0) % 360.0 - 180.0
        span = (-180.0, 180.0)
    y = np.asarray(frs, float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    if smooth_deg > 0:
        y = pl._circ_smooth(y, smooth_deg)
    x = np.concatenate([[x[-1] - 360.0], x, [x[0] + 360.0]])
    y = np.concatenate([[y[-1]], y, [y[0]]])
    return x, y, span


def draw_tuning_lineplot(
    ax,
    curves,
    colors,
    one_obj=None,
    center_rad=None,
    smooth_deg_2obj=0.0,
    smooth_deg_1obj=0.0,
    lw=1.4,
    one_obj_color="0.6",
    one_obj_lw=1.4,
    xticks=(-180, 0, 180),
    xticklabels=None,
):
    """Firing rate against direction: the 1-object tuning curve in gray, plus one line per
    inter-object angle.

    ``curves`` is ``{sep_deg: (phis_rad, frs)}`` (from :func:`unit_curves`), ``colors`` is
    ``{sep_deg: color}``, and ``one_obj`` is the optional ``(phis_rad, frs)`` of the
    1-object curve. With ``center_rad`` the x axis is the signed angle from that direction,
    spanning [-180, 180]; without it, the absolute direction over [0, 360].
    ``xticklabels`` overrides the tick text — pass ``''`` for a tick that should mark the
    axis extent without a (colliding) label. Returns ``{label: Line2D}`` for the legend.

    The two families are smoothed **independently** — ``smooth_deg_1obj`` for the gray
    1-object curve, ``smooth_deg_2obj`` for the per-separation lines — because they are not
    sampled alike: the 1-object curve carries ``n_dense_1obj`` samples around the ring, the
    2-object lines ``line_n_dense``, and the widest separations pool fewer trials per sample.
    """
    handles = {}
    span = (0.0, 360.0) if center_rad is None else (-180.0, 180.0)
    if one_obj is not None:
        x, y, span = _unwrapped_deg(
            *one_obj, center_rad=center_rad, smooth_deg=smooth_deg_1obj
        )
        (handles["1 obj"],) = ax.plot(
            x, y, color=one_obj_color, lw=one_obj_lw, zorder=2
        )
    for sep, (phis, frs) in curves.items():
        x, y, span = _unwrapped_deg(
            phis, frs, center_rad=center_rad, smooth_deg=smooth_deg_2obj
        )
        (handles[f"{sep:g}°"],) = ax.plot(
            x, y, color=colors[sep], lw=lw, zorder=3
        )
    ax.set_xlim(*span)
    ax.set_xticks(list(xticks))
    if xticklabels is not None:
        ax.set_xticklabels(list(xticklabels))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return handles


def add_legend(
    fig,
    handles,
    anchor_ax,
    title=None,
    ncol=None,
    loc="center left",
    x=1.06,
    y=0.5,
    frame=True,
    rounding_size=0.4,
    borderpad=0.6,
    labelspacing=0.5,
    handletextpad=0.6,
    handlelength=1.6,
    columnspacing=1.4,
    borderaxespad=0.0,
    edgecolor="0.3",
    framealpha=1.0,
    linewidth=0.8,
):
    """The line row's key, in a rounded box, **anchored to `anchor_ax`** rather than to the
    figure — ``(x, y)`` is the anchor point in that axes' own fractions (x = 1 its right
    edge, y = 0.5 its vertical middle) and ``loc`` says which part of the legend box lands
    there. An ``'outside ...'`` figure loc would center the box on the whole figure instead,
    which puts it well above a bottom row.

    Every spacing is in font-size units except ``rounding_size`` (inches). Shared by
    :func:`tuning_grid` and by the line-row-only figures of
    ``figure_4/neuron_models/tuning_style_lib.py``, so both look the same.
    """
    leg = fig.legend(
        handles.values(),
        handles.keys(),
        loc=loc,
        bbox_to_anchor=(x, y),
        bbox_transform=anchor_ax.transAxes,
        ncol=ncol or 1,
        title=title,
        title_fontsize=plt.rcParams["legend.fontsize"],
        frameon=frame,
        borderpad=borderpad,
        labelspacing=labelspacing,
        handletextpad=handletextpad,
        handlelength=handlelength,
        columnspacing=columnspacing,
        borderaxespad=borderaxespad,
        fancybox=True,
        framealpha=framealpha,
        edgecolor=edgecolor,
    )
    if frame:
        # `fancybox` rounds the corners; only the boxstyle exposes the radius. Its `pad` is
        # what `borderpad` already set, so it is carried over rather than re-specified.
        f = leg.get_frame()
        f.set_boxstyle(
            "round", pad=f.get_boxstyle().pad, rounding_size=rounding_size
        )
        f.set_linewidth(linewidth)
    return leg


def draw_tuning_panel(
    ax,
    row,
    radius=0.43,
    cmap=COOLWARM,
    annulus_width=0.1,
    annulus_smooth_deg=10.0,
    dense_linewidth=1.5,
    front_of="clockwise",
    rotate_deg=0.0,
):
    """One tuning ring: the 1-object row as a shaded annulus, the 2-object rows as a
    dense family of ``row['sep']``-apart chords (identical to ``psth_lib.plot_rows``'s
    right-hand column).

    ``rotate_deg`` turns the drawn ring counter-clockwise by that many degrees: every
    sample direction moves, and so both ends of every chord with it, which is a rigid
    rotation of the picture. Nothing else changes — the conditions, their firing rates and
    their color scale are what they were."""
    phis = np.asarray(row["phis"], float) + np.radians(rotate_deg)
    if row["kind"] == "chord":
        pl.draw_dense_2obj(
            ax,
            phis,
            row["sep"],
            row["frs"],
            row["norm"],
            radius=radius,
            cmap=cmap,
            front_of=front_of,
            lw=dense_linewidth,
        )
    else:
        pl.draw_1obj_annulus(
            ax,
            phis,
            row["frs"],
            row["norm"],
            radius=radius,
            width=annulus_width,
            cmap=cmap,
            smooth_deg=annulus_smooth_deg,
        )


def _range_text(row, fmt="{lo:.1f}–{hi:.1f} Hz"):
    return fmt.format(lo=row["norm"].vmin, hi=row["norm"].vmax)


def fit_ring_size(
    fig,
    ring_axes,
    target_in,
    n_ring,
    n_cols,
    max_iter=10,
    tol=0.004,
    min_scale=0.5,
):
    """Resize the figure until every ring panel is at least ``target_in`` inches square.

    Necessary because ``constrained_layout``'s ``height_ratios`` divide up the *decorated*
    height of each row: the line row's tick labels and x label, the column titles and the
    legend all come out of the same total, so simply adding inches to ``figsize`` for the
    line row shrinks the ring rows instead of leaving them alone.

    The ring panels are square (``aspect='equal'``), so their drawn size is the smaller side
    of their grid cell — which is what ``get_position(original=True)`` reports, before the
    aspect shrink. Each pass takes the *tightest* cell over ``ring_axes`` (the first column
    is narrower than the rest, since it carries the row labels), adds its width deficit to
    the figure width (times the number of columns) and its height deficit to the figure
    height (times the number of ring rows), then re-solves the layout; both are near-linear
    in the leftover space, so this settles in a few passes. ``min_scale`` floors the shrink
    so a too-small target cannot collapse the axes.
    """
    ring_axes = np.ravel(ring_axes)
    w0, h0 = fig.get_size_inches()
    for _ in range(max_iter):
        fig.canvas.draw()
        w, h = fig.get_size_inches()
        cells = [ax.get_position(original=True) for ax in ring_axes]
        dw = target_in - min(c.width for c in cells) * w
        dh = target_in - min(c.height for c in cells) * h
        if abs(dw) < tol and abs(dh) < tol:
            break
        fig.set_size_inches(
            max(w + dw * n_cols, min_scale * w0),
            max(h + dh * n_ring, min_scale * h0),
        )
    return fig.get_size_inches()


def tuning_grid(
    rows_per_unit,
    column_titles,
    figsize=(6.0, 3.5),
    dpi=300,
    cmap=FIRING_RATE_CMAP,
    radius=0.43,
    annulus_width=0.1,
    annulus_smooth_deg=10.0,
    dense_linewidth=1.5,
    front_of="clockwise",
    rotate_deg=None,
    row_labels=None,
    range_labels=False,
    range_fontsize=12,
    range_pad=0.02,
    colorbar=True,
    colorbar_label="Mean firing rate",
    colorbar_ticklabels=("low", "high"),
    colorbar_fraction=0.02,
    colorbar_pad=0.01,
    colorbar_shrink=0.4,
    wspace=0.02,
    hspace=0.02,
    ring_size_in=1.0,
    curves_per_unit=None,
    bottom_drawer=None,
    bottom_data=None,
    bottom_kw=None,
    line_height_in=1.4,
    line_colors=None,
    line_center="peak",
    line_smooth_deg_1obj=20.0,
    line_smooth_deg_2obj=10.0,
    line_linewidth=1.4,
    line_1obj=False,
    line_ylim=None,
    line_yticks=None,
    line_1obj_color="0.6",
    line_xlabel="Mean angle (°)",
    line_ylabel="Firing rate (Hz)",
    line_xticks=(-180, 0, 180),
    line_xticklabels=None,
    legend=True,
    legend_title=None,
    legend_ncol=None,
    legend_loc="center left",
    legend_x=1.1,
    legend_y=0.48,
    legend_frame=True,
    legend_rounding_size=0.3,
    legend_borderpad=0.3,
    legend_labelspacing=0.5,
    legend_handletextpad=0.4,
    legend_handlelength=1.0,
    legend_columnspacing=1.4,
    legend_borderaxespad=0.4,
    legend_edgecolor="0.3",
    legend_framealpha=1.0,
    legend_linewidth=0.8,
):
    """The grid: one column per unit (``rows_per_unit[i]`` = that unit's rows), one row
    per condition family. Returns ``(fig, axes)`` with ``axes[row, unit]``.

    ``range_labels`` writes each panel's own firing-rate range underneath it — necessary
    because every panel has its own color scale. ``colorbar`` adds a single unlabeled key
    for the blue→red direction.

    ``rotate_deg`` turns each unit's rings: a scalar rotates every column by the same
    angle, a sequence gives one angle per unit (``None`` = no rotation). All three ring
    rows of a column turn together, so they stay registered with one another, and only the
    drawing moves — which conditions are shown, their firing rates and their color scales
    are untouched. This is where a unit's ``orientation_deg`` lands, so that the number
    that rotates its prototype set turns its rings by the same angle; the line row, which
    measures x from the unit's own receptive field, is unaffected.

    ``curves_per_unit`` (a list of ``{sep_deg: (phis, frs)}``, one per unit, from
    :func:`unit_curves`) adds a **bottom row of line plots**: the unit's 1-object tuning
    curve in gray plus one line per inter-object angle, keyed by the legend drawn below the
    figure. The line row shares its y-axis with nothing — each panel is on its own
    firing-rate scale, as the rings are. It is ``line_height_in`` **inches** tall, added
    below ``figsize``.

    ``line_center`` (``'peak'`` / ``'com'``, see :func:`rf_center`) puts x = 0 of each line
    panel at the center of *that unit's* 1-object receptive field, so the x axis is the
    signed angle from the receptive field over [-180, 180] and the columns are aligned with
    each other. ``None`` keeps absolute directions over [0, 360]. The 1-object and
    2-object lines take their own angular smoothing (``line_smooth_deg_1obj`` /
    ``line_smooth_deg_2obj``); the receptive-field center is found on the 1-object curve, so
    it follows ``line_smooth_deg_1obj``.

    The legend sits **beside the line row**, anchored to the last line panel rather than to
    the figure, so it lines up with that row instead of with the whole column stack:
    ``(legend_x, legend_y)`` is the anchor point in that panel's own axes fractions — x = 1
    is its right edge, y = 0.5 its vertical middle — and ``legend_loc`` says which corner or
    edge of the legend box lands there (``'center left'`` = the box's left-middle, so it
    grows rightward and stays centered on the row).

    The box's spacing is fully exposed: ``legend_borderpad`` (entries to box),
    ``legend_labelspacing`` (between entries), ``legend_handletextpad`` (line to its label),
    ``legend_handlelength`` (length of the line samples), ``legend_columnspacing``,
    ``legend_borderaxespad`` (an extra pad on top of ``legend_x`` / ``legend_y``) and
    ``legend_rounding_size`` (corner radius), all in font-size units except the last.

    Because it is anchored by hand, the legend is outside ``constrained_layout``'s
    bookkeeping: it cannot squeeze the panels (nor :func:`fit_ring_size`'s target), and
    ``bbox_inches='tight'`` is what pulls it into the saved figure.

    ``ring_size_in`` makes every ring panel that many inches square, growing the figure past
    ``figsize`` as needed (:func:`fit_ring_size`) — without it, the decorations of the line
    row and the legend eat into the ring rows.

    ``line_ylim`` and ``line_yticks`` are **per column** — one entry per unit, ``None`` for a
    column that should keep matplotlib's autoscale and automatic ticks. A unit whose curves
    autoscale to an awkward range (an axis that runs 5.6-13.0 and ticks at 6, 8, 10, 12) can
    be given round numbers without touching the units beside it, and
    :func:`line_ylim_for_unit` reports whatever a column ends up with, so another figure can
    borrow it.
    """
    n_ring = len(rows_per_unit[0])
    n_cols = len(rows_per_unit)
    # A bottom row exists if EITHER the line plots or a caller-supplied drawer asks for one.
    # Deciding it from `curves_per_unit` alone would give a `bottom_drawer` no row of its own,
    # and `axes[-1]` would then be the last *ring* row — which the drawer would overwrite.
    has_bottom = curves_per_unit is not None or bottom_drawer is not None
    n_rows = n_ring + has_bottom
    heights = [1.0] * n_ring
    if has_bottom:
        heights.append(n_ring * line_height_in / figsize[1])
        figsize = (figsize[0], figsize[1] + line_height_in)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        dpi=dpi,
        squeeze=False,
        layout="constrained",
        height_ratios=heights,
    )
    fig.get_layout_engine().set(wspace=wspace, hspace=hspace)

    # One rotation per unit: a scalar applies to all of them, None to none of them.
    rotations = (
        np.zeros(n_cols)
        if rotate_deg is None
        else np.broadcast_to(np.asarray(rotate_deg, float), (n_cols,))
    )

    for c, rows in enumerate(rows_per_unit):
        for r, row in enumerate(rows):
            ax = axes[r, c]
            draw_tuning_panel(
                ax,
                row,
                radius=radius,
                cmap=cmap,
                annulus_width=annulus_width,
                annulus_smooth_deg=annulus_smooth_deg,
                dense_linewidth=dense_linewidth,
                front_of=front_of,
                rotate_deg=rotations[c],
            )
            if range_labels:
                ax.text(
                    0.5,
                    -range_pad,
                    _range_text(row),
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=range_fontsize,
                )
        axes[0, c].set_title(column_titles[c])

    labels = (
        row_labels
        if row_labels is not None
        else [r["label"] for r in rows_per_unit[0]]
    )
    for r, label in enumerate(labels):
        axes[r, 0].set_ylabel(label)

    handles = {}
    if bottom_drawer is not None:
        # A caller-supplied bottom row, replacing the line plots. The grid stays agnostic
        # about what goes there: it owns the geometry, the drawer owns the panel. This is how
        # `tuning_plots_anchor.ipynb` swaps in the anchor-vs-preferred-direction panel without
        # a second copy of this function.
        kw = dict(bottom_kw or {})
        for c, data in enumerate(bottom_data):
            # Only the leftmost column carries the y label and its tick labels — every panel
            # is on the same fixed 0-360 axes, so repeating them per column says nothing.
            bottom_drawer(
                axes[-1, c], data, **{"show_yticklabels": c == 0, **kw}
            )
            if c:
                axes[-1, c].set_ylabel(None)
    elif curves_per_unit is not None:
        seps = list(curves_per_unit[0])
        colors = line_colors if line_colors is not None else sep_colors(seps)
        for c, curves in enumerate(curves_per_unit):
            # The gray curve is the same 1-object tuning the annulus of row 0 shades, and
            # it is also what sets x = 0: the center of this unit's receptive field.
            one_obj = (rows_per_unit[c][0]["phis"], rows_per_unit[c][0]["frs"])
            center = (
                None
                if line_center is None
                else rf_center(
                    *one_obj,
                    method=line_center,
                    smooth_deg=line_smooth_deg_1obj,
                )
            )
            handles = draw_tuning_lineplot(
                axes[-1, c],
                curves,
                colors,
                one_obj=one_obj if line_1obj else None,
                center_rad=center,
                smooth_deg_1obj=line_smooth_deg_1obj,
                smooth_deg_2obj=line_smooth_deg_2obj,
                lw=line_linewidth,
                one_obj_color=line_1obj_color,
                one_obj_lw=line_linewidth,
                xticks=line_xticks,
                xticklabels=line_xticklabels,
            )
            if line_ylim is not None and line_ylim[c] is not None:
                axes[-1, c].set_ylim(*line_ylim[c])
            if line_yticks is not None and line_yticks[c] is not None:
                axes[-1, c].set_yticks(list(line_yticks[c]))
            axes[-1, c].set_xlabel(
                line_xlabel
            )  # every panel carries its own x label
        axes[-1, 0].set_ylabel(line_ylabel)

    if colorbar:
        cb = fig.colorbar(
            ScalarMappable(norm=Normalize(0, 1), cmap=cmap),
            ax=axes[:n_ring].ravel().tolist(),
            fraction=colorbar_fraction,
            pad=colorbar_pad,
            shrink=colorbar_shrink,
            ticks=[0, 1],
        )
        cb.ax.set_yticklabels(list(colorbar_ticklabels))
        cb.set_label(colorbar_label)
        cb.outline.set_visible(False)

    if legend and handles:
        add_legend(
            fig,
            handles,
            axes[-1, -1],
            title=legend_title,
            ncol=legend_ncol,
            loc=legend_loc,
            x=legend_x,
            y=legend_y,
            frame=legend_frame,
            rounding_size=legend_rounding_size,
            borderpad=legend_borderpad,
            labelspacing=legend_labelspacing,
            handletextpad=legend_handletextpad,
            handlelength=legend_handlelength,
            columnspacing=legend_columnspacing,
            borderaxespad=legend_borderaxespad,
            edgecolor=legend_edgecolor,
            framealpha=legend_framealpha,
            linewidth=legend_linewidth,
        )

    if ring_size_in:
        fit_ring_size(fig, axes[:n_ring], ring_size_in, n_ring, n_cols)
    return fig, axes

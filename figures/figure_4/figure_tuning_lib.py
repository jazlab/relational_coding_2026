r"""Model tuning curves in the style of ``tuning_plots.ipynb``.

The figure ``figure_tuning.ipynb`` draws: for one neuron, **one column per model**, with

* a **90°-apart chord ring** on top — the re-oriented ``90° apart`` row of
  ``tuning_plots.ipynb``, drawn from the model's prediction instead of the
  measured rate;
* the **line row** below it: firing rate against the pair's mean direction, one curve per inter-object angle,
  with the model's own 1-object curve in grey.

Self-contained per ``figures/FORMATTING.md``: the sampling, smoothing and x-unwrapping are
copied from ``tuning_lib.py`` here rather than imported across directories. The ring drawing and the model
evaluation come from ``ring_lib`` in this directory, so a ring here is the same object as a ring
in ``model_comparison.ipynb``.

Two choices worth knowing before reading a figure
-------------------------------------------------
**Where x = 0 is.** Each line panel is centred on *that model's own* predicted 1-object peak, not
on a shared angle. The models disagree about where the neuron's receptive field is, and centring
each on its own estimate is what makes the *shapes* comparable; it also means a horizontal offset
between two panels is not something you can read off.

**What the colour scale spans.** By default one scale covers every model column in a unit's ring
row, so a column that looks like another really is predicting the same rates. Set
``share='panel'`` to normalise each panel to its own range, which is what
``../tuning_plots/tuning_plots.ipynb`` does for the data — better for seeing the shape of a weak
model, worse for comparing across columns.
"""

import matplotlib.pyplot as plt
import models_lib as ml
import numpy as np
import ring_lib as rl
from matplotlib.colors import Normalize

# The centring and unwrapping helpers are `../tuning_plots`'s, imported rather than copied:
# they were duplicated here and the two copies had already drifted apart.
from tuning_lib import (  # noqa: F401
    FlatCurve,
    _unwrapped_deg,
    rf_center,
    sep_colors,
)


# ── Sampling and model evaluation ─────────────────────────────────────────────
def line_phis(n_dense=36, orientation_deg=0.0):
    """The ``n_dense`` sample directions of a curve, rotated by the unit's prototype
    orientation exactly as ``../tuning_plots``'s ring rows are.

    The rotation matters for the ring row and not for the line row: it decides which chords are
    drawn, and a family of 36 chords starting at a different phase looks different even when the
    underlying function is identical.
    """
    return np.radians(orientation_deg) + np.linspace(
        0, 2 * np.pi, n_dense, endpoint=False
    )


def predicted_curves(
    model, params, seps_deg, n_dense=36, orientation_deg=0.0, dctr=None
):
    """``{separation: (phis, frs)}`` — the model's 2-object prediction at each separation.

    No binning: unlike the measured curves of ``../tuning_plots``, which pool trials within an
    angular window, a model is evaluated at the exact prototype angles. So these curves are
    smooth by construction and the smoothing knobs below exist only to match the data figure's
    line weight, never to rescue a noisy estimate.

    ``dctr`` is the relative contrast of the drawn pair, which only
    :class:`models_lib.WeightedAverageModel` reads — see :func:`build_panels`.
    """
    phis = line_phis(n_dense, orientation_deg)
    return {
        float(s): (phis, rl.predicted_2obj(model, params, phis, s, dctr))
        for s in seps_deg
    }


def predicted_one_obj(model, params, n_dense=180, orientation_deg=0.0):
    """``(phis, frs)`` — the model's 1-object prediction, the grey reference curve.

    It is also what sets ``x = 0``: :func:`rf_center` takes its peak, or, when a model predicts
    no 1-object tuning at all, the model's own ``theta`` stands in. Worth knowing that this row
    is a *prediction*, never a fit — nothing in this directory fits 1-object trials — so a model
    whose grey curve misses the neuron's measured 1-object tuning is telling you something, and
    ``../tuning_plots/tuning_plots.ipynb`` is where to check it against.
    """
    phis = line_phis(n_dense, orientation_deg)
    return phis, rl.predicted_1obj(model, params, phis)


# ── Drawing ───────────────────────────────────────────────────────────────────
def draw_lineplot(
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
    """Rate against direction: the 1-object curve in grey plus one line per inter-object angle.

    ``curves`` is ``{sep_deg: (phis_rad, frs)}`` and ``colors`` is ``{sep_deg: colour}``. With
    ``center_rad`` the x axis is the signed angle from that direction over [-180, 180].

    The two families are smoothed **independently**. They are not sampled alike — the 1-object
    curve carries ``n_dense_1obj`` points and the 2-object lines ``n_dense`` — so one smoothing
    width in degrees means different things to them. Returns ``{label: Line2D}`` for the legend.
    """
    handles = {}
    span = (0.0, 360.0) if center_rad is None else (-180.0, 180.0)
    if one_obj is not None:
        x, y, span = _unwrapped_deg(
            *one_obj, center_rad=center_rad, smooth_deg=smooth_deg_1obj
        )
        (handles["1 object"],) = ax.plot(
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


def model_tuning_grid(
    panels,
    ring_sep_deg=90.0,
    rotate_deg=0.0,
    figsize=None,
    dpi=300,
    share="row",
    ring_rows=None,
    cmap=None,
    radius=None,
    chord_lw=1.6,
    front_of="clockwise",
    line_lw=1.4,
    smooth_deg_1obj=0.0,
    smooth_deg_2obj=0.0,
    center="peak",
    one_obj=True,
    one_obj_color="0.6",
    colors=None,
    xlabel="Mean angle",
    ylabel="Firing rate (Hz)",
    ylim="2obj",
    ylim_headroom=1.4,
    line_share_y=False,
    line_ylim=None,
    line_yticks=None,
    xticks=(-180, 0, 180),
    xticklabels=None,
    show_xticklabels=True,
    range_labels=True,
    range_fontsize=12,
    colorbar=False,
    colorbar_label="Firing rate",
    legend=True,
    legend_title="Inter-object angle",
    suptitle=None,
    ring_frac=0.62,
    wspace=0.30,
    hspace=0.30,
    bottom_drawer=None,
    bottom_data=None,
    bottom_kw=None,
    ring_size_in=None,
    line_height_in=1.4,
    col_gap_in=0.45,
    row_gap_in=None,
    line_gap_in=None,
    panel_width_in=None,
    margins_in=(0.70, 0.08, 0.32, 0.60),
):
    r"""**The figure.** One column per model: a ``ring_sep_deg``-apart chord ring on top, the
    line row below.

    ``ring_rows`` draws **several ring rows** instead of one, in the style of
    ``figure_2/psth.ipynb`` — pass the same list given to :func:`build_panels`, whose entries
    are ``'1obj'`` or a separation in degrees, top to bottom. Each row gets its **own** colour
    scale under ``share='row'``, because a 1-object rate and a 2-object rate are not on one
    scale and forcing them to be flattens both. Left at ``None`` the figure keeps its single
    ring row, so existing callers are unaffected.

    Each entry of ``panels`` is a dict with ``label``, ``ring`` (``(phis, frs)`` at
    ``ring_sep_deg``), ``curves`` (``{sep: (phis, frs)}``) and ``one_obj`` (``(phis, frs)``).

    ``rotate_deg`` turns every ring counter-clockwise by that many degrees — a rigid rotation
    of the drawn picture, exactly what ``../tuning_plots/tuning_plots.ipynb`` does with a
    unit's ``orientation_deg``, so a unit's rings here can be laid against its rings there.
    Only the drawing moves: the conditions, their predicted rates and the colour scale are
    untouched, and the line row (centred on each model's own predicted receptive field) is
    unaffected.

    ``share='row'`` puts every column's ring on one colour scale, so columns are comparable and
    the range is printed once. ``share='panel'`` normalises each to its own, which is what the
    data figure does — better for reading a weak model's shape, useless for comparing columns.
    The line row always keeps per-column y axes, because the models differ in predicted rate by
    more than a shared axis can show without flattening most of them.

    ``line_share_y`` does the same for the **line row**: off, each column is scaled to its own
    curves, so every model's shape is legible but their heights say nothing; on, one axis
    spans them all and the columns become comparable in rate, at the cost of flattening any
    model that predicts a much smaller modulation than its rivals. Only the leftmost panel
    keeps its tick labels when shared, since the others now repeat it.

    ``ylim='2obj'`` scales each line panel to its **2-object** curves. That is not cosmetic. The
    1-object row is an *extrapolation* — nothing in this directory fits 1-object trials — and for
    the normalization model in particular it can be an order of magnitude above every 2-object
    rate, because that model's amplitude is a numerator divided by ``f_A + f_B + alpha`` and one
    object leaves the denominator small. Scaling to it would flatten the curves the figure exists
    to show. When the grey curve then leaves the panel, its peak is **printed in the corner**
    rather than silently cropped. ``ylim='all'`` includes it and accepts the flattening. ``ylim_headroom`` is the slack in
    between: a 1-object peak within that multiple of the 2-object range stretches the axis to
    fit rather than being clipped, so only genuine blow-ups get the annotation.

    ``line_ylim=(lo, hi)`` overrides all of that and pins every line panel to those limits.
    It is what puts this figure on the **data figure's** rate axis:
    ``tuning_plots.line_ylim`` computes the limits ``../tuning_plots/tuning_plots.ipynb``
    gives the same unit, from that figure's own settings, and they come in here. The
    1-object peak annotation still applies — a grey curve that leaves a borrowed axis is
    exactly as worth saying as one that leaves a computed axis.

    ``line_yticks`` sets the line row's y ticks, for the same reason: an axis borrowed from
    another figure should carry that figure's ticks too, or the two read as different scales.

    ``show_xticklabels=False`` keeps the ticks and drops their numbers, for a panel that sits
    under another figure carrying the same axis. The x label is separate (pass
    ``xlabel=None``).

    Spacing
    -------
    Setting ``ring_size_in`` switches the figure to an **inch-absolute** layout, and then the
    six knobs below are the whole story — ``figsize``, ``wspace``, ``hspace`` and
    ``ring_frac`` are ignored, because they are fractions of a figure whose size is now an
    output rather than an input. Every distance is in inches, so a number means the same
    thing however many columns there are:

    ==================  ========================================================
    ``ring_size_in``    the rings' diameter; the ring axes are square at this size
    ``panel_width_in``  a column's width. Defaults to ``ring_size_in``, which is
                        why columns used to be exactly as wide as their rings —
                        the setting to raise when the column *titles* crowd each
                        other, since it widens the columns without inflating the
                        rings. The rings stay square and centred in the column;
                        the line panel spans its full width.
    ``col_gap_in``      the gap between columns. Column pitch is
                        ``panel_width_in + col_gap_in``.
    ``row_gap_in``      the gap between **ring rows**, when there is more than one
    ``line_gap_in``     the gap between the bottom ring row and the **line row**.
                        Defaults to ``row_gap_in``, which is what it used to be
                        fixed at; raise it alone to drop the line row without
                        also spreading the rings apart.
    ``margins_in``      ``(left, right, top, bottom)`` around the whole grid. The
                        left margin is what the y label and the row labels live
                        in; the top margin holds the column titles.
    ==================  ========================================================

    The figure's own size follows from them::

        width  = left + n·panel_width_in + (n−1)·col_gap_in + right
        height = top + n_ring·ring_size_in + (n_ring−1)·row_gap_in
                     + line_gap_in + line_height_in + bottom

    so a panel keeps its size when you add a column, which is the point of laying it out this
    way — see the note below.
    """
    cmap = cmap or rl.COOLWARM
    radius = rl.RING_RADIUS if radius is None else radius
    # `ring_rows` generalises the single ring row; without it, one row at `ring_sep_deg`
    # reading its data from `panel['ring']`, exactly as before.
    ring_rows = list(ring_rows) if ring_rows else [ring_sep_deg]
    n_ring = len(ring_rows)

    def ring_data(panel, key):
        if len(ring_rows) == 1 and "rings" not in panel:
            return panel["ring"]
        return panel["rings"][key]

    def ring_label(key):
        return "1 object" if key == "1obj" else f"{float(key):g}° apart"

    seps = list(panels[0]["curves"])
    colors = colors or sep_colors(seps)
    n = len(panels)
    figsize = figsize or (2.0 * n + 1.9, 4.4)

    if ring_size_in:
        # Inch-absolute layout, the contract ../tuning_plots/tuning_plots.ipynb uses: the panels
        # are the fixed quantity and the figure is sized to hold them, rather than a fixed figure
        # being divided by fractions. Sizing by fraction cannot hold a panel size steady, because
        # the decorations (column titles, tick labels, x label) are a fixed size in points while
        # the fractions scale -- change n or figsize and the rings silently resize.
        #
        # Laid out with explicit rectangles instead of a gridspec: `hspace`/`wspace` are fractions
        # of *average* axes height/width, which the two rows do not share once their heights are
        # set independently, so gaps in inches are the only way to say what is actually meant.
        gap_r = (
            0.30
            if (row_gap_in is None and range_labels)
            else (row_gap_in or 0.08)
        )
        # The ring-to-line gap is its own knob but defaults to the ring-to-ring one, so a
        # caller that never sets it gets the single gap this layout used to have.
        gap_l = gap_r if line_gap_in is None else line_gap_in
        # A column is as wide as its rings unless told otherwise. Widening the column is how
        # to give the titles room without making the rings bigger.
        panel_w = ring_size_in if panel_width_in is None else panel_width_in
        ml, mr, mt, mb = margins_in
        mr += 1.35 if legend else 0.0
        mr += 0.55 if colorbar else 0.0
        fig_w = ml + n * panel_w + (n - 1) * col_gap_in + mr
        fig_h = (
            mt
            + n_ring * ring_size_in
            + (n_ring - 1) * gap_r
            + gap_l
            + line_height_in
            + mb
        )
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

        def rect(c, row):
            is_line = row == n_ring
            x0 = ml + c * (panel_w + col_gap_in)
            # The line panel fills the column; the ring stays square and sits centred in it,
            # so the two rows share a column centre however wide the column is.
            x, w = (
                (x0, panel_w)
                if is_line
                else (x0 + (panel_w - ring_size_in) / 2, ring_size_in)
            )
            h = (line_height_in if is_line else ring_size_in) / fig_h
            # rows are numbered top-down; y is measured from the bottom
            y = (
                mb
                if is_line
                else mb
                + line_height_in
                + gap_l
                + (n_ring - 1 - row) * (ring_size_in + gap_r)
            ) / fig_h
            return [x / fig_w, y, w / fig_w, h]

        cell = lambda c, row: fig.add_axes(rect(c, row))
    else:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        # ONE gridspec for both rows, so ring column c and line column c occupy the same x extent.
        # Two gridspecs with different `wspace` put their column centres in different places, and a
        # colorbar attached to the ring axes only would shrink that row out of alignment again --
        # which is why `colorbar` defaults to off.
        right = 0.84 if legend else 0.97
        gs = fig.add_gridspec(
            n_ring + 1,
            n,
            left=0.09,
            right=right,
            top=0.90,
            bottom=0.14,
            wspace=wspace,
            hspace=hspace,
            height_ratios=[ring_frac / n_ring] * n_ring + [1.0 - ring_frac],
        )
        cell = lambda c, row: fig.add_subplot(gs[row, c])

    # One colour scale per ROW, not one for the whole grid: a 1-object rate and a 2-object
    # rate are not on the same scale, and forcing them to be flattens both.
    norm_row = {}
    if share == "row":
        for key in ring_rows:
            allv = np.concatenate(
                [np.asarray(ring_data(p, key)[1], float) for p in panels]
            )
            norm_row[key] = Normalize(
                float(np.nanmin(allv)), float(np.nanmax(allv))
            )

    ring_axes, handles = [], {}
    for r, key in enumerate(ring_rows):
        for c, p in enumerate(panels):
            phis, frs = ring_data(p, key)
            frs = np.asarray(frs, float)
            norm = (
                norm_row[key]
                if share == "row"
                else Normalize(float(np.nanmin(frs)), float(np.nanmax(frs)))
            )
            ax = cell(c, r)
            rot = np.asarray(phis, float) + np.radians(rotate_deg)
            if key == "1obj":
                rl.draw_1obj_annulus(
                    ax, rot, frs, norm, radius=radius, cmap=cmap
                )
            else:
                rl.draw_dense_2obj(
                    ax,
                    rot,
                    float(key),
                    frs,
                    norm,
                    radius=radius,
                    cmap=cmap,
                    lw=chord_lw,
                    front_of=front_of,
                )
            if r == 0:
                ax.set_title(
                    p["label"], fontsize=plt.rcParams["axes.titlesize"], pad=6
                )
            ring_axes.append(ax)
            if range_labels and (share == "panel" or c == 0):
                ax.text(
                    0.5,
                    -0.02,
                    f"{np.nanmin(frs):.1f}-{np.nanmax(frs):.1f} Hz",
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=range_fontsize,
                    color="0.35",
                )
            if c == 0:
                ax.text(
                    -0.10,
                    0.5,
                    ring_label(key),
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=range_fontsize,
                )

    # A caller-supplied bottom row replaces the line plots. The grid owns the geometry, the
    # drawer owns the panel — which is how `figure_tuning_anchor.ipynb` swaps in the
    # anchor-vs-preferred-direction panel without a second copy of this function. The line
    # loop below is skipped by iterating nothing, so its body stays exactly as it was.
    if bottom_drawer is not None:
        kw = dict(bottom_kw or {})
        for c, data in enumerate(bottom_data):
            ax = cell(c, n_ring)
            # Only the leftmost column carries the y label and its tick labels: every panel
            # is on the same fixed 0-360 axes, so repeating them per column says nothing and
            # runs the neighbouring columns' labels into each other.
            bottom_drawer(ax, data, **{"show_yticklabels": c == 0, **kw})
            if c:
                ax.set_ylabel(None)

    for c, p in [] if bottom_drawer is not None else list(enumerate(panels)):
        ref = p.get("one_obj")
        ax = cell(c, n_ring)
        center_rad = None
        if center is not None and ref is not None:
            try:
                center_rad = rf_center(
                    *ref, method=center, smooth_deg=smooth_deg_1obj
                )
            except FlatCurve:
                # A model that predicts no 1-object tuning has no receptive field to read off
                # its grey curve, but it still has a preferred direction. Centre on that, and
                # say so — silently centring on an arbitrary angle is what made this column
                # look 90 deg out of phase with its neighbours.
                center_rad = p.get("theta")
                print(
                    f"  {p.get('label', '?')}: 1-object prediction is flat, so there is no "
                    f"peak to centre on; "
                    + (
                        f"using the fitted theta ({np.degrees(center_rad) % 360:.1f} deg)"
                        if center_rad is not None
                        else "left uncentred"
                    )
                )
        handles = draw_lineplot(
            ax,
            p["curves"],
            colors,
            one_obj=ref if (one_obj and ref is not None) else None,
            center_rad=center_rad,
            smooth_deg_1obj=smooth_deg_1obj,
            smooth_deg_2obj=smooth_deg_2obj,
            lw=line_lw,
            one_obj_color=one_obj_color,
            one_obj_lw=line_lw,
            xticks=xticks,
            xticklabels=xticklabels,
        )
        ax.set_xlabel(xlabel, fontsize=range_fontsize)
        if c == 0:
            ax.set_ylabel(ylabel, fontsize=range_fontsize)
        if not show_xticklabels:
            ax.set_xticklabels([""] * len(list(xticks)))
        if line_ylim is not None or ylim == "2obj":
            # gated on `one_obj` as well as `ref`: with the curve not drawn there is nothing
            # to make room for and nothing to annotate as clipped.
            refs = (
                [q.get("one_obj") for q in panels] if line_share_y else [ref]
            )
            peak = (
                None
                if not one_obj or all(r is None for r in refs)
                else max(
                    float(np.nanmax(np.asarray(r[1], float)))
                    for r in refs
                    if r is not None
                )
            )
            if line_ylim is not None:
                bottom, top = float(line_ylim[0]), float(line_ylim[1])
            else:
                # shared: every column's curves set one range, so the panels are comparable
                src = panels if line_share_y else [p]
                vals = np.concatenate(
                    [
                        np.asarray(v, float)
                        for q in src
                        for _, v in q["curves"].values()
                    ]
                )
                lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
                pad = 0.10 * max(hi - lo, 1e-9)
                top = hi + pad
                # A 1-object curve that only just clears the 2-object range is worth showing:
                # stretch the axis up to `ylim_headroom` times the range to fit it, and only
                # clip (and say so) when it is further out than that.
                if peak is not None and top < peak <= lo + ylim_headroom * (
                    top - lo
                ):
                    top = peak + pad
                bottom = lo - pad
            ax.set_ylim(bottom, top)
            if line_yticks is not None:
                ax.set_yticks(list(line_yticks))
            if line_share_y and c > 0:
                ax.set_yticklabels([])  # the leftmost panel already says it
            if peak is not None:
                if peak > top:
                    # above the axes, not inside it — inside, this collides with whichever
                    # curve happens to peak there, which varies unit to unit.
                    # with a shared axis the clipped peak is the whole row's, so it is
                    # said once rather than over every column
                    if not line_share_y or c == 0:
                        ax.annotate(
                            f"1 obj peaks at {peak:.0f}",
                            xy=(0.5, 1.02),
                            xycoords="axes fraction",
                            ha="center",
                            va="bottom",
                            fontsize=range_fontsize,
                            color=one_obj_color,
                        )

    if colorbar:
        sm = plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap)
        cb = fig.colorbar(
            sm, ax=ring_axes, fraction=0.025, pad=0.02, ticks=[0, 1]
        )
        cb.ax.set_yticklabels(["low", "high"], fontsize=range_fontsize)
        cb.set_label(colorbar_label, fontsize=range_fontsize)
        cb.outline.set_visible(False)
    if legend and handles:
        fig.legend(
            list(handles.values()),
            list(handles),
            frameon=False,
            fontsize=range_fontsize,
            title=legend_title,
            loc="center left",
            bbox_to_anchor=(0.855, 0.24),
        )
    if suptitle:
        fig.suptitle(suptitle, fontsize=plt.rcParams["axes.titlesize"])
    return fig, fig.axes


def build_panels(
    models,
    fits,
    model_names,
    seps_deg,
    ring_sep_deg=90.0,
    n_dense=36,
    n_dense_1obj=180,
    orientation_deg=0.0,
    labels=None,
    ring_n_dense=18,
    ring_rows=None,
    dctr=None,
):
    """One panel per model, ready for :func:`model_tuning_grid`.

    ``models`` is ``{name: Model}`` from :func:`models_lib.build_models` and ``fits`` is
    ``{name: params}`` from :func:`ring_lib.unit_fits`, so every panel is that model's fit to
    *this* neuron — the columns differ only in the combination rule.

    ``ring_rows`` asks for **more than one ring row**, in the style of
    ``figure_2/psth.ipynb``, which shows a 1-object annulus above its two 2-object chord
    families. Pass a list whose entries are either ``'1obj'`` or a separation in degrees, top
    to bottom — e.g. ``['1obj', 90, 180]``. Left at ``None`` the panel carries the single
    ``ring_sep_deg`` row it always has, so existing callers are unaffected.

    ``dctr`` — the relative contrast of the pair being drawn. Every model here predicts a rate
    from geometry alone *except* :class:`models_lib.WeightedAverageModel`, whose rate also
    depends on which two object identities are on screen, so there is no single curve to draw
    for it: a contrast has to be chosen. Pass a scalar to use one value everywhere, or a
    ``{model: value}`` dict. Models that do not read a contrast ignore it.

    This cannot be dodged by averaging over identities. The task shows each unordered pair in
    both spatial orders about equally often, ``d`` is antisymmetric, and the rate is linear in
    ``d`` — so the identity-averaged prediction is the plain retinotopic sum for *any*
    ``gamma``. The weighting shows up trial by trial, never in the mean, which is why the drawn
    condition has to be named rather than marginalised away.
    """
    labels = labels or {}
    dctr = (
        dict(dctr)
        if isinstance(dctr, dict)
        else {m: dctr for m in model_names}
    )
    missing = [
        m
        for m in model_names
        if models[m].needs_contrast and dctr.get(m) is None
    ]
    if missing:
        raise ValueError(
            f'{", ".join(missing)} predicts a rate from the object identities as well as '
            f"their angles, so `dctr` (the pair's relative contrast) must be given for it. "
            f"See contrast_lib.relative_contrast."
        )
    out = []
    for m in model_names:
        mod, par, d = models[m], fits[m], dctr.get(m)
        curves = predicted_curves(
            mod, par, seps_deg, n_dense, orientation_deg, d
        )
        # The rings get their own, coarser sampling: `n_dense` chords drawn over each other
        # read as a solid disc, where the line row wants the resolution. 18 / 36 are the
        # values ../tuning_plots/tuning_plots.ipynb uses.
        ring_phis = line_phis(ring_n_dense, orientation_deg)
        panel = {
            "model": m,
            "label": labels.get(m, ml.MODEL_LABELS_SHORT.get(m, m)),
            "ring": (
                ring_phis,
                rl.predicted_2obj(mod, par, ring_phis, ring_sep_deg, d),
            ),
            "curves": curves,
            "one_obj": predicted_one_obj(
                mod, par, n_dense_1obj, orientation_deg
            ),
            # The preferred direction, kept for `model_tuning_grid` to centre on when the
            # 1-object prediction is flat. Not every model has one (`inhibition` does not).
            "theta": par.get("theta"),
        }
        if ring_rows:
            # An antipodal family spans only half a turn: past 180 deg the unordered pair
            # repeats, so drawing the full circle would draw every chord twice.
            rings = {}
            for key in ring_rows:
                if key == "1obj":
                    rings[key] = panel["one_obj"]
                else:
                    sep = float(key)
                    n_r = (
                        ring_n_dense
                        if abs(sep - 180.0) > 1e-6
                        else 2 * ring_n_dense
                    )
                    phis = np.radians(orientation_deg) + np.linspace(
                        0,
                        np.pi if abs(sep - 180.0) <= 1e-6 else 2 * np.pi,
                        n_r,
                        endpoint=False,
                    )
                    rings[key] = (
                        phis,
                        rl.predicted_2obj(mod, par, phis, sep, d),
                    )
            panel["rings"] = rings
        out.append(panel)
    return out

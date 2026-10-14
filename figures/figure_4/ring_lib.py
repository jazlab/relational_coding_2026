"""Measured and model-predicted tuning curves, drawn in the style of
``figure_4/tuning_plots/tuning_plots.ipynb``.

Three condition families per neuron, the same three that figure the PSTH and tuning-plot
panels:

    row 0   1 object          — a continuous shaded annulus over object direction
    row 1   SEP_DEG° apart    — a dense family of same-separation chords
    row 2   180° apart        — the antipodal family (spans half a turn)

**The 1-object row is a prediction, not a fit.** Nothing in this directory is fit to
1-object trials, so that row asks a question the fits never saw: does the receptive field
inferred from 2-object trials alone match the neuron's measured single-object tuning? A
model that gets the 2-object rows right and the 1-object row wrong is telling you
something. Drop the row with ``build_rows(..., include_1obj=False)``.

Two differences from ``figure_4/tuning_plots``, both deliberate:

* **The firing rates come from this analysis's own cache**, not from ``psth_lib`` — they
  are the exact per-trial rates the models were fit to (mean of the stimulus and delay
  windows, no temporal smoothing), so a fitted curve and its data are commensurable. The
  ring-drawing primitives are copied from ``figure_2/psth/psth_lib.py`` (per
  ``figures/FORMATTING.md`` this directory imports no other figure's code).
* **One color scale per row, shared by the data panel and every model panel**, so a model
  panel that looks like the data panel *is* predicting the data. In
  ``figure_4/tuning_plots`` each panel has its own scale, which is right when comparing
  units but wrong when comparing a fit to its data.

Measured rates bin trials by angular similarity (``window_deg``), exactly as
``psth_lib.window_trials_*`` does. Predicted rates need no binning: the model is evaluated
at the exact prototype angles.
"""

import matplotlib.pyplot as plt
import models_lib as ml
import numpy as np

COOLWARM = plt.cm.coolwarm  # blue = low FR, red = high FR
RING_RADIUS = 0.43
NAN_COLOR = (0.85, 0.85, 0.85, 1.0)


# ── Angular helpers ───────────────────────────────────────────────────────────
def _circ_dist(a, b):
    """Absolute circular distance (rad) in [0, π]; broadcasts."""
    return np.abs(
        (np.asarray(a, float) - np.asarray(b, float) + np.pi) % (2 * np.pi)
        - np.pi
    )


def _dist_2obj(t0, t1, p0, p1):
    """Symmetric average per-object angular distance between an unordered trial pair
    ``(t0, t1)`` and a prototype pair ``(p0, p1)`` — ``psth_lib._dist_2obj``, vectorised
    over trials."""
    return np.minimum(
        (_circ_dist(t0, p0) + _circ_dist(t1, p1)) / 2,
        (_circ_dist(t0, p1) + _circ_dist(t1, p0)) / 2,
    )


# ── Measured tuning ───────────────────────────────────────────────────────────
def boosted_cmap(
    cmap=None, value=1.0, saturation=1.0, blend=0.5, n=256, name="boosted"
):
    """A colormap whose hot end is lifted **in HSV, at constant hue**.

    ``coolwarm`` ends at a dark maroon: in HSV that colour is (h 0.97, s 0.98, v 0.71), and
    it is what every per-panel scale reaches, since each panel runs the whole map whatever
    its firing-rate range. Cutting the map short would brighten it only by moving toward the
    pale centre, which desaturates the red — a different colour, not a brighter one. This
    instead keeps each sample's **hue** and interpolates its saturation and value towards
    ``saturation`` and ``value``, so the extreme becomes the same red at full chroma and
    higher brightness (v 0.71 -> ``value``, s 0.98 -> ``saturation``).

    ``blend`` is the fraction of the map, measured from the hot end, over which that lift
    ramps in: at ``t = 1`` it is complete, at ``t = 1 - blend`` it is zero, so the midpoint
    and the cold half are untouched. ``value=None`` returns the colormap unchanged.

    Copied from ``figure_3/model_comparison/ideal_lib.py`` per ``figures/FORMATTING.md``,
    so the figures that share this red do not share an import.
    """
    from matplotlib.colors import (
        LinearSegmentedColormap,
        hsv_to_rgb,
        rgb_to_hsv,
    )

    cmap = (
        COOLWARM
        if cmap is None
        else (plt.get_cmap(cmap) if isinstance(cmap, str) else cmap)
    )
    if value is None or blend <= 0:
        return cmap
    t = np.linspace(0.0, 1.0, n)
    rgba = cmap(t)
    hsv = rgb_to_hsv(rgba[:, :3])
    w = np.clip((t - (1.0 - blend)) / blend, 0.0, 1.0)[
        :, None
    ]  # 0 -> 1 over the top
    target = np.stack(
        [hsv[:, 0], np.full(n, float(saturation)), np.full(n, float(value))],
        axis=1,
    )  # same hue, new s and v
    rgba[:, :3] = hsv_to_rgb((1.0 - w) * hsv + w * target)
    return LinearSegmentedColormap.from_list(f"{name}_{cmap.name}", rgba)


def measured_1obj(unit, phis, window_deg=30.0):
    """Mean 1-object rate (and s.e.m., and n) at each angle in ``phis``."""
    th, fr = unit["theta1"], unit["fr1"]
    w = np.radians(window_deg)
    out = np.full((3, len(phis)), np.nan)
    for i, phi in enumerate(phis):
        sel = _circ_dist(th, phi) <= w
        if sel.sum():
            v = fr[sel]
            out[:, i] = (
                v.mean(),
                v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0,
                len(v),
            )
    return out[0], out[1], out[2]


def measured_2obj(unit, phis, sep_deg, window_deg=30.0):
    """Mean 2-object rate (and s.e.m., and n) for ``sep_deg``-apart pairs centred on each
    angle in ``phis``."""
    t0, t1, fr = unit["t0"], unit["t1"], unit["fr2"]
    w = np.radians(window_deg)
    half = np.radians(sep_deg) / 2
    out = np.full((3, len(phis)), np.nan)
    for i, phi in enumerate(phis):
        sel = _dist_2obj(t0, t1, phi - half, phi + half) <= w
        if sel.sum():
            v = fr[sel]
            out[:, i] = (
                v.mean(),
                v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0,
                len(v),
            )
    return out[0], out[1], out[2]


def predicted_1obj(model, params, phis):
    return ml.rectify(model.rate_1obj(params, np.asarray(phis, float)))


def predicted_2obj(model, params, phis, sep_deg, dctr=None):
    """Predicted rate over the pair's midpoint direction at one separation.

    ``dctr`` is the relative contrast, which only the contrast-weighted model reads. Drawing
    that model's tuning curve means choosing which pair of objects to draw it for — there is
    no single curve — so it has to be supplied rather than defaulted."""
    half = np.radians(sep_deg) / 2
    phis = np.asarray(phis, float)
    if dctr is not None:
        dctr = np.broadcast_to(np.asarray(dctr, float), phis.shape)
    return ml.rectify(
        ml.rate_2obj(model, params, phis - half, phis + half, dctr)
    )


def build_rows(
    unit,
    sep_deg=90.0,
    n_dense=36,
    n_dense_1obj=180,
    window_deg=30.0,
    orientation_deg=0.0,
    include_1obj=True,
):
    """The condition families for one neuron, with their measured rates.

    Returns a list of dicts with keys ``label, kind, sep, phis, fr, sem, n``. ``kind`` is
    ``'annulus'`` for the 1-object row and ``'chord'`` for the two 2-object rows. The
    antipodal row spans only ``[0, π)`` because past 180° the unordered pair repeats.

    ``include_1obj`` keeps the 1-object row. It is on by default because that row is the
    one thing in this directory the fits have never seen — see the module docstring — but
    a neuron with few or no 1-object trials should drop it.

    ``orientation_deg`` rotates every sample angle, exactly as the *prototype orientation*
    does in ``figure_4/tuning_plots`` — it lines the sampled conditions up with the unit's
    receptive field, and it must be the same number that notebook uses for the unit or the
    panels will not be comparable.
    """
    offset = np.radians(orientation_deg)
    phis_1o = offset + np.linspace(0, 2 * np.pi, n_dense_1obj, endpoint=False)
    phis_s = offset + np.linspace(0, 2 * np.pi, n_dense, endpoint=False)
    phis_180 = offset + np.linspace(0, np.pi, n_dense, endpoint=False)

    rows = []
    if include_1obj and len(unit["fr1"]):
        fr, sem, n = measured_1obj(unit, phis_1o, window_deg)
        rows.append(
            dict(
                label="1 object\n(not fit)",
                kind="annulus",
                sep=None,
                phis=phis_1o,
                fr=fr,
                sem=sem,
                n=n,
            )
        )
    for label, sep, phis in (
        (f"{sep_deg:g}° apart", sep_deg, phis_s),
        ("180° apart", 180.0, phis_180),
    ):
        fr, sem, n = measured_2obj(unit, phis, sep, window_deg)
        rows.append(
            dict(
                label=label,
                kind="chord",
                sep=sep,
                phis=phis,
                fr=fr,
                sem=sem,
                n=n,
            )
        )
    return rows


def params_from_summary(summary_row):
    """``{parameter: value}`` from a summary/units row's ``p_*`` columns."""
    return {
        k[2:]: float(v)
        for k, v in summary_row.items()
        if k.startswith("p_") and np.isfinite(v)
    }


def unit_fits(
    subject, session, unit_id, cfg, brain_area=None, tag=None, verbose=True
):
    """Every model's **all-trial** parameter fit for one neuron, as
    ``{model: {parameter: value}}``.

    Reads the per-session cache when that session has been fit, and otherwise fits just
    this one neuron on the spot (a few seconds for all the models). The parameters are the
    ones fit to **all** of the neuron's 2-object trials — what ``fit_session`` stores in
    its ``units`` file — which is what a tuning curve should be drawn from; the held-out
    likelihoods come from the per-split fits.
    """
    import fit_lib as fl
    import pandas as pd

    path = (
        fl.cache_dir(cfg, brain_area, tag)
        / f"{subject}_{session}_units.csv.gz"
    )
    if path.exists():
        units = pd.read_csv(path)
        row = units[units["unit"] == unit_id]
        if len(row) and set(cfg.models) <= set(row["model"]):
            if verbose:
                print(f"  {subject}/{session} unit {unit_id}: from cache")
            return {
                r["model"]: params_from_summary(r) for _, r in row.iterrows()
            }

    if verbose:
        print(
            f"  {subject}/{session} unit {unit_id}: not in the cache — fitting now"
        )
    _, units_df = fl.fit_session(
        subject,
        session,
        cfg,
        brain_area=brain_area,
        unit_ids=[unit_id],
        n_jobs=1,
        verbose=False,
        cache=False,
    )
    return {r["model"]: params_from_summary(r) for _, r in units_df.iterrows()}


# ── Ring drawing (copied from figure_2/psth/psth_lib.py) ──────────────────────
def _prep_stim_ax(ax, radius=RING_RADIUS, ring=True):
    """Fixation cross (+ faint stimulus ring) on a unit square, no ticks or spines."""
    if ring:
        circ = np.linspace(0, 2 * np.pi, 200)
        ax.plot(
            0.5 + radius * np.cos(circ),
            0.5 + radius * np.sin(circ),
            "k--",
            lw=0.7,
            alpha=0.35,
            zorder=1,
        )
    ax.plot(
        0.5,
        0.5,
        marker="+",
        color="k",
        markersize=7,
        markeredgewidth=1.3,
        zorder=1,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _ring_xy(theta, radius=RING_RADIUS):
    return 0.5 + radius * np.cos(theta), 0.5 + radius * np.sin(theta)


def draw_chords(ax, chords, radius=RING_RADIUS, front_of="clockwise", lw=1.6):
    """A family of same-separation chords with rotationally consistent occlusion: each
    chord is split at its midpoint and the two halves go in different z-layers, which makes
    every chord pass in front of its clockwise neighbours and behind its counter-clockwise
    ones with no seam at the wrap-around (see ``psth_lib.draw_chords`` for the geometry).
    """
    n = max(len(chords), 1)
    for k, (t0, t1, color) in enumerate(chords):
        x0, y0 = _ring_xy(t0, radius)
        x1, y1 = _ring_xy(t1, radius)
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        cw, ccw = ([xm, x0], [ym, y0]), ([xm, x1], [ym, y1])
        fg, bg = (cw, ccw) if front_of == "clockwise" else (ccw, cw)
        ax.plot(
            *bg,
            color=color,
            lw=lw,
            zorder=2 + k / (n + 1),
            solid_capstyle="round",
        )
        ax.plot(
            *fg,
            color=color,
            lw=lw,
            zorder=3 + k / (n + 1),
            solid_capstyle="round",
        )


def _fill_nan_circular(v):
    v = np.asarray(v, float)
    good = np.isfinite(v)
    if good.all() or not good.any():
        return np.where(good, v, 0.0)
    idx = np.arange(len(v))
    xp, fp = idx[good], v[good]
    n = len(v)
    return np.interp(
        idx, np.concatenate([xp - n, xp, xp + n]), np.concatenate([fp, fp, fp])
    )


def _circ_smooth(vals, smooth_deg):
    vals = _fill_nan_circular(vals)
    n = len(vals)
    sigma = smooth_deg / 360.0 * n
    if sigma <= 0:
        return vals
    half = int(np.ceil(3 * sigma))
    x = np.arange(-half, half + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    ext = np.concatenate([vals[-half:], vals, vals[:half]])
    return np.convolve(ext, k, mode="same")[half : half + n]


def draw_1obj_annulus(
    ax,
    phis,
    frs,
    norm,
    radius=RING_RADIUS,
    width=0.1,
    cmap=COOLWARM,
    smooth_deg=10.0,
):
    """Gouraud-shaded annulus coloured by firing rate around the ring."""
    frs = _circ_smooth(frs, smooth_deg)
    th = np.append(phis, phis[0] + 2 * np.pi)
    c = np.append(frs, frs[0])
    TH, RR = np.meshgrid(
        th, np.array([radius - width / 2, radius + width / 2])
    )
    ax.pcolormesh(
        0.5 + RR * np.cos(TH),
        0.5 + RR * np.sin(TH),
        np.vstack([c, c]),
        cmap=cmap,
        norm=norm,
        shading="gouraud",
    )
    _prep_stim_ax(ax, radius, ring=False)


def draw_dense_2obj(
    ax,
    phis,
    sep_deg,
    frs,
    norm,
    radius=RING_RADIUS,
    cmap=COOLWARM,
    front_of="clockwise",
    lw=1.6,
):
    _prep_stim_ax(ax, radius)
    half = np.radians(sep_deg) / 2
    draw_chords(
        ax,
        [
            (
                p - half,
                p + half,
                cmap(norm(v)) if np.isfinite(v) else NAN_COLOR,
            )
            for p, v in zip(phis, frs)
        ],
        radius=radius,
        front_of=front_of,
        lw=lw,
    )


# ── The two figures ───────────────────────────────────────────────────────────

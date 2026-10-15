"""The PSTH figure of `psth.ipynb`: one unit, three rows, three columns.

Each row is a family of stimulus conditions — one object, two objects `SEP1_DEG` apart, and
two objects antipodal — and each column shows the same conditions three ways:

* **Stimuli** — the selected prototypes, as chords on the ring, coloured by mean firing rate.
* **PSTH** — the firing rate over the trial for each of those prototypes, in the same colours.
* **Tuning** — every orientation of that row's condition, not just the selected few.

A condition is a **prototype** — a nominal pair of object angles — and a trial joins it when
both of its objects are within `similarity_window_deg` of their slot. The stimulus set is
discrete and never places two objects closer than about 80 degrees, so the trials in a
prototype's bin are not centred on it; the panels are keyed to the nominal prototypes, which
is what the PSTHs and the colours refer to.

`build_rows` assembles the data for one unit and `plot_rows` draws it; `psth_figure` is the
two together, which is all the notebook calls.
"""

import common
import numpy as np
from common import BIN_MS, FR_WINDOW
from common import bin_times_ms as _bin_times_ms
from common import circ_dist as _circ_dist
from common import load_behavior, load_unit, per_trial_mean_fr, smooth
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import (
    LinearSegmentedColormap,
    Normalize,
    hsv_to_rgb,
    rgb_to_hsv,
)
from matplotlib.ticker import MaxNLocator

RING_RADIUS = 0.38  # ring radius of the stimulus schematics (axes span [0, 1])
STIM_TIME_WINDOW = (
    0,
    2500,
)  # ms after stimulus onset, the plotted PSTH extent
COOLWARM = plt.cm.coolwarm  # blue = low firing rate, red = high


def firing_rate_cmap():
    """The colormap every panel's firing rate is drawn in.

    `coolwarm` with two changes. Its hot end is lifted in HSV at constant hue
    (:func:`boosted_cmap`), because every panel's scale runs the whole map and coolwarm's own
    red is dull at full scale. Then the neutral colour is moved to 60% of the way up the
    range, so the lower 60% reads as blue and red is reserved for the top — the extremes are
    unchanged, only what counts as middling moves.

    The skew **reparameterises position, not colour**: the blue half of the lifted map is
    stretched over ``[0, 0.6]`` and the red half compressed into ``[0.6, 1]``. Every colour
    still appears, in the same order, and the two ends are unchanged. Doing it with a
    two-slope norm instead would have the same effect on the panels but would need every
    colorbar to know about it too; this way the map itself carries the skew.

    The order matters: the lift is defined in the map's own coordinate, so skewing first
    would lift a different amount of red.
    """
    cmap, skew = boosted_cmap(), 0.6
    t = np.linspace(0.0, 1.0, 256)
    src = np.where(
        t < skew, 0.5 * t / skew, 0.5 + 0.5 * (t - skew) / (1.0 - skew)
    )
    return LinearSegmentedColormap.from_list(f"skewed_{cmap.name}", cmap(src))


def boosted_cmap():
    """``coolwarm`` with its hot end lifted **in HSV, at constant hue**.

    ``coolwarm`` ends at a dark maroon: in HSV that colour is (h 0.97, s 0.98, v 0.71), and
    it is what every per-panel scale reaches, since each panel runs the whole map whatever
    its firing-rate range. Cutting the map short would brighten it only by moving toward the
    pale centre, which desaturates the red — a different colour, not a brighter one. This
    instead keeps each sample's **hue** and interpolates its saturation and value towards
    ``saturation`` and ``value``, so the extreme becomes the same red at full chroma and
    higher brightness (v 0.71 -> ``value``, s 0.98 -> ``saturation``).

    ``blend`` is the fraction of the map, measured from the hot end, over which that lift
    ramps in: at ``t = 1`` it is complete, at ``t = 1 - blend`` it is zero, so the midpoint
    and the cold half are untouched.

    Figure 4's ``tuning_lib`` builds its colormap from this function too, so the figures
    share this red.
    """
    value, saturation, blend = 0.9, 1.0, 0.5
    n = 256
    t = np.linspace(0.0, 1.0, n)
    rgba = COOLWARM(t)
    hsv = rgb_to_hsv(rgba[:, :3])
    w = np.clip((t - (1.0 - blend)) / blend, 0.0, 1.0)[
        :, None
    ]  # 0 -> 1 over the top
    target = np.stack(
        [hsv[:, 0], np.full(n, float(saturation)), np.full(n, float(value))],
        axis=1,
    )  # same hue, new s and v
    rgba[:, :3] = hsv_to_rgb((1.0 - w) * hsv + w * target)
    return LinearSegmentedColormap.from_list(f"boosted_{COOLWARM.name}", rgba)


def prototypes_1obj(n_orient, offset_rad):
    """``n_orient`` object angles evenly spaced around the ring, rotated by
    ``offset_rad``. Keyed by orientation index."""
    step = 2 * np.pi / n_orient
    return {i: offset_rad + i * step for i in range(n_orient)}


def prototypes_sep(n_orient, separation_deg, offset_rad):
    """``n_orient`` ``separation_deg``-apart (θ0, θ1) pairs, centered on evenly
    spaced orientations and rotated by ``offset_rad``.

    Each pair is symmetric about its center angle, so at ``offset_rad = 0`` the
    chords stay exactly horizontal/vertical whatever the separation.
    """
    half = np.radians(separation_deg) / 2
    mids = offset_rad + np.radians(
        np.linspace(0, 360, n_orient, endpoint=False)
    )
    return {i: (m - half, m + half) for i, m in enumerate(mids)}


def prototypes_180(n_orient, offset_rad):
    """``n_orient`` antipodal (180°-apart) pairs at evenly spaced orientations
    over ``[0, 180)`` (past 180° the unordered pair repeats), rotated by
    ``offset_rad``."""
    step = np.pi / n_orient
    return {
        i: (offset_rad + i * step, offset_rad + i * step + np.pi)
        for i in range(n_orient)
    }


def _dist_2obj(t0, t1, p0, p1):
    """Distance between a trial's object pair and a prototype pair, in radians.

    Symmetric in the two objects — both ways of matching trial objects to prototype slots
    are tried and the better kept — and the distance is the **worse** of the two per-object
    distances, so a trial joins a prototype's bin only when *both* of its objects are within
    the window of their own slot.
    """
    d00, d01 = _circ_dist(t0, p0), _circ_dist(t0, p1)
    d10, d11 = _circ_dist(t1, p0), _circ_dist(t1, p1)
    return min(max(d00, d11), max(d01, d10))


def window_trials_2obj(behavior, protos, window_rad):
    """`{key: the trials within window_rad of that prototype}`.

    A trial can land in several prototypes' bins (the windows overlap) or in none (a small
    window and a discrete stimulus set), so this is a binning, not a partition.
    """
    out = {key: [] for key in protos}
    for _, row in behavior.iterrows():
        t0, t1 = row["object_0_theta"], row["object_1_theta"]
        for key, (p0, p1) in protos.items():
            if _dist_2obj(t0, t1, p0, p1) <= window_rad:
                out[key].append(int(row["trial_num"]))
    return {k: np.array(v) for k, v in out.items()}


def window_trials_1obj(behavior_1obj, protos, window_rad):
    """``{key: 1-object trials within window_rad of the prototype angle}``."""
    out = {key: [] for key in protos}
    for _, row in behavior_1obj.iterrows():
        th = row["object_0_theta"]
        for key, p in protos.items():
            if _circ_dist(th, p) <= window_rad:
                out[key].append(int(row["trial_num"]))
    return {k: np.array(v) for k, v in out.items()}


def mean_fr_by_condition(
    spike_counts,
    trials,
    trials_per_condition,
    window=FR_WINDOW,
    smooth_half_win_ms=50,
):
    """``{condition: mean over trials of the per-trial mean firing rate}``."""
    out = {}
    for cond, cond_trials in trials_per_condition.items():
        frs = list(
            per_trial_mean_fr(
                spike_counts, trials, cond_trials, window, smooth_half_win_ms
            ).values()
        )
        out[cond] = float(np.mean(frs)) if frs else np.nan
    return out


def psth_curve(spike_counts, trials, trial_nums, smooth_half_win_ms=50):
    """Mean ± sem PSTH across ``trial_nums``. Returns (time_ms, mean, sem) over
    the plotted window."""
    bps = 1000 / BIN_MS
    trial_index = {int(t): i for i, t in enumerate(trials)}
    curves = None
    time_keep = None
    stack = []
    for t in trial_nums:
        idx = trial_index.get(int(t))
        if idx is None:
            continue
        sc = smooth(spike_counts[idx], smooth_half_win_ms) * bps
        time = _bin_times_ms(len(sc))
        keep = (time > STIM_TIME_WINDOW[0]) & (time < STIM_TIME_WINDOW[1])
        stack.append(sc[keep])
        time_keep = time[keep]
    if not stack:
        return np.array([]), np.array([]), np.array([])
    curves = np.vstack(stack)
    mean = curves.mean(axis=0)
    sem = curves.std(axis=0, ddof=0) / np.sqrt(len(curves))
    return time_keep, mean, sem


def dense_2obj_tuning(
    behavior,
    spike_counts,
    trials,
    separation_deg,
    n_dense,
    window_rad,
    orient_span=2 * np.pi,
    offset_rad=0.0,
    fr_window=FR_WINDOW,
    smooth_half_win_ms=50,
):
    """`n_dense` chords of `separation_deg` evenly spanning `[offset, offset + orient_span)`.

    Returns `(phis, mean_frs)` with `phis` the chords' centre angles. The firing rates come
    from the same similarity window as the selected prototypes, so the dense panel and the
    PSTHs are measuring the same thing at different resolutions.
    """
    half = np.radians(separation_deg) / 2
    phis = offset_rad + np.linspace(0, orient_span, n_dense, endpoint=False)
    protos = {k: (phi - half, phi + half) for k, phi in enumerate(phis)}
    tpc = window_trials_2obj(behavior, protos, window_rad)
    mfr = mean_fr_by_condition(
        spike_counts,
        trials,
        tpc,
        window=fr_window,
        smooth_half_win_ms=smooth_half_win_ms,
    )
    return phis, np.array([mfr[k] for k in range(n_dense)])


def dense_1obj_tuning(
    behavior_1obj,
    spike_counts,
    trials,
    n_dense,
    window_rad,
    offset_rad=0.0,
    fr_window=FR_WINDOW,
    smooth_half_win_ms=50,
):
    """``n_dense`` single-object prototypes around the ring; (phis, mean_frs)."""
    phis = offset_rad + np.linspace(0, 2 * np.pi, n_dense, endpoint=False)
    protos = {k: phi for k, phi in enumerate(phis)}
    tpc = window_trials_1obj(behavior_1obj, protos, window_rad)
    mfr = mean_fr_by_condition(
        spike_counts,
        trials,
        tpc,
        window=fr_window,
        smooth_half_win_ms=smooth_half_win_ms,
    )
    return phis, np.array([mfr[k] for k in range(n_dense)])


def _fill_nan_circular(v):
    """Fill NaNs by circular linear interpolation around the ring."""
    v = np.asarray(v, float)
    n = len(v)
    good = np.isfinite(v)
    if good.all() or not good.any():
        return np.where(np.isfinite(v), v, 0.0)
    idx = np.arange(n)
    xp, fp = idx[good], v[good]
    xp_ext = np.concatenate([xp - n, xp, xp + n])
    fp_ext = np.concatenate([fp, fp, fp])
    return np.interp(idx, xp_ext, fp_ext)


def _circ_smooth(vals, smooth_deg):
    """Circular Gaussian smoothing of ring-indexed values (``smooth_deg`` = kernel
    sigma in degrees). NaNs are interpolated first."""
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


def _fr_norm(values):
    """Normalize over the finite range of a dense tuning curve."""
    v = np.asarray(values, float)
    return Normalize(np.nanmin(v), np.nanmax(v))


def _fr_colors(norm, mean_frs, cmap):
    """``{condition: cmap color}`` from ``{condition: mean FR}`` (gray if NaN)."""
    return {
        k: (cmap(norm(v)) if np.isfinite(v) else (0.7, 0.7, 0.7, 1.0))
        for k, v in mean_frs.items()
    }


def _prep_stim_ax(ax, radius=RING_RADIUS, ring=True):
    """Fixation cross (+ faint stimulus ring of ``radius``) on a unit [0, 1]
    square, no ticks. ``radius`` is in axes fractions, so 0.5 exactly fills the
    panel."""
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


def draw_chords(
    ax,
    chords,
    radius=RING_RADIUS,
    lw=2.0,
):
    """Draw a family of same-separation chords with *rotationally consistent*
    occlusion.

    ``chords`` is an ordered list of ``(theta0, theta1, color)`` with
    ``theta1 = theta0 + separation`` (so ``theta1`` is the counter-clockwise end)
    and the list ordered by increasing center angle.

    Why splitting each chord at its midpoint works
    ----------------------------------------------
    Every chord of separation ``s`` on a ring of radius ``R`` is tangent, at its
    own midpoint, to the inner circle of radius ``R·cos(s/2)``. Two chords whose
    centers differ by ``Δ`` are two tangents of that inner circle, so they meet at
    angular direction ``(center₁ + center₂)/2`` — i.e. **between** their midpoints
    — at radius ``R·cos(s/2)/cos(Δ/2)``, which is inside the ring (so the crossing
    exists at all) exactly when ``Δ ≤ s``.

    Therefore *every* crossing of chord *i* with a counter-clockwise neighbor lies
    on the counter-clockwise half of chord *i*, and every crossing with a
    clockwise neighbor lies on its clockwise half. Drawing all clockwise halves in
    one z-layer and all counter-clockwise halves in another makes each chord pass
    in front of all its clockwise neighbors and behind all its counter-clockwise
    ones — with **no seam at the wrap-around**, which a single z-order per chord
    can never achieve (the "in front of" relation would have to be a cyclic total
    order).

    The clockwise halves go on top, so each line is in front of its clockwise
    neighbors.
    """
    z0 = 2.0
    n = max(len(chords), 1)
    for k, (t0, t1, color) in enumerate(chords):
        x0, y0 = _ring_xy(t0, radius)  # clockwise end
        x1, y1 = _ring_xy(t1, radius)  # counter-clockwise end
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2  # chord midpoint
        cw_half = ([xm, x0], [ym, y0])
        ccw_half = ([xm, x1], [ym, y1])
        fg, bg = cw_half, ccw_half
        # A tie-break within each layer keeps degenerate cases (e.g. the 180°
        # chords, which all cross at the center) from flickering.
        ax.plot(
            *bg,
            color=color,
            lw=lw,
            zorder=z0 + k / (n + 1),
            solid_capstyle="round",
        )
        ax.plot(
            *fg,
            color=color,
            lw=lw,
            zorder=z0 + 1 + k / (n + 1),
            solid_capstyle="round",
        )


def draw_2obj_prototypes(ax, protos, colors, radius=RING_RADIUS):
    """The selected prototype pairs: a chord per pair, coloured by `colors[key]`, with an
    **unfilled** circle at each object.

    The circle is open on purpose. It says where the objects are without hiding what runs
    underneath, so two prototypes that share an object still show both their chords through
    it — which a filled dot would cover.
    """
    _prep_stim_ax(ax, radius)
    draw_chords(
        ax,
        [(t0, t1, colors[key]) for key, (t0, t1) in sorted(protos.items())],
        radius=radius,
    )
    ends = [t for _, pair in sorted(protos.items()) for t in pair]
    xs, ys = zip(*(_ring_xy(t, radius) for t in ends))
    ax.scatter(
        xs,
        ys,
        s=40,
        facecolors="none",
        edgecolors="k",
        linewidths=0.5,
        zorder=6,
    )


def draw_1obj_prototypes(ax, protos, colors, radius):
    """Selected prototype object positions colored by ``colors[key]``."""
    _prep_stim_ax(ax, radius)
    for key, t in protos.items():
        x, y = _ring_xy(t, radius)
        ax.scatter(
            x,
            y,
            color=colors[key],
            s=44,
            edgecolors="k",
            linewidths=0.4,
            zorder=3,
        )


def draw_dense_2obj(
    ax,
    phis,
    separation_deg,
    mean_frs,
    norm,
    radius=RING_RADIUS,
    cmap=COOLWARM,
    lw=1.3,
):
    """The dense family of `separation_deg`-apart chords, centred at `phis` and coloured by
    mean firing rate, with the rotational occlusion of :func:`draw_chords`.

    A prototype no trial fell into is drawn grey rather than skipped, so a gap in the
    stimulus set reads as a gap rather than as a missing chord.
    """
    _prep_stim_ax(ax, radius)
    half = np.radians(separation_deg) / 2
    chords = [
        (
            phi - half,
            phi + half,
            cmap(norm(fr)) if np.isfinite(fr) else (0.85, 0.85, 0.85, 1.0),
        )
        for phi, fr in zip(phis, mean_frs)
    ]
    draw_chords(ax, chords, radius=radius, lw=lw)


def draw_1obj_annulus(
    ax,
    orientations,
    mean_frs,
    norm,
    radius=RING_RADIUS,
    width=0.1,
    cmap=COOLWARM,
    smooth_deg=0.0,
):
    """Continuous (Gouraud-shaded) annulus of radial thickness ``width``, centered
    on ``radius`` and colored by mean FR around the ring, circularly smoothed by
    ``smooth_deg`` (Gaussian sigma)."""
    frs = _circ_smooth(mean_frs, smooth_deg)
    th = np.append(orientations, orientations[0] + 2 * np.pi)  # close the ring
    c1 = np.append(frs, frs[0])
    TH, RR = np.meshgrid(
        th, np.array([radius - width / 2, radius + width / 2])
    )
    ax.pcolormesh(
        0.5 + RR * np.cos(TH),
        0.5 + RR * np.sin(TH),
        np.vstack([c1, c1]),
        cmap=cmap,
        norm=norm,
        shading="gouraud",
    )
    _prep_stim_ax(ax, radius, ring=False)


def plot_psth(
    ax,
    spike_counts,
    trials,
    trials_per_condition,
    colors,
    smooth_half_win_ms,
    lw,
):
    """One PSTH line per condition, colored by ``colors[condition]``, with a ±sem
    band at half the line's opacity. Returns max y."""
    y_max = 0.0
    for cond, cond_trials in trials_per_condition.items():
        time, mean, sem = psth_curve(
            spike_counts, trials, cond_trials, smooth_half_win_ms
        )
        if not len(time):
            continue
        c = colors[cond]
        ax.fill_between(
            time,
            mean - sem,
            mean + sem,
            color=(c[0], c[1], c[2], c[3] * 0.5),
            lw=0,
            zorder=2,
        )
        ax.plot(time, mean, color=c, lw=lw, zorder=3)
        y_max = max(y_max, float((mean + sem).max()))
    return y_max


#: The binning of :func:`build_rows`, shared by figure 2, figure 4's rings and supp figure 10.
SEP1_DEG = 90  # inter-object angle of the middle row
N_DENSE = 18  # orientations in the 2-object tuning panels
N_DENSE_1OBJ = 180  # resolution of the 1-object annulus


def build_rows(
    subject,
    session,
    unit_id,
    orientation_deg=0.0,
    similarity_window_deg=15.0,
    smooth_half_win_ms=250.0,
    n_proto=4,
    share_2obj_color_scale=True,
    cmap=COOLWARM,
):
    """Load a unit and prepare the figure's three rows.

    Row 1 is `n_proto` 1-object positions, row 2 is `n_proto` prototypes `SEP1_DEG` apart,
    row 3 is `n_proto` antipodal ones. All three sets — and the dense tuning of the right
    column — are rotated by `orientation_deg`, so the conditions can be lined up with this
    unit's receptive field.

    The colour of a condition, in all three columns, is its mean firing rate over
    `FR_WINDOW` (ms after stimulus onset), stimulus + delay, which excludes
    the response. The PSTH traces themselves always span `STIM_TIME_WINDOW`.

    With `share_2obj_color_scale` the two 2-object rows share one firing-rate range spanning
    both their dense tuning curves, so their colours can be compared directly; otherwise each
    row is normalized to its own range and only within-row comparisons mean anything. The
    1-object row is always on its own scale — its rates are not comparable to the 2-object
    ones in the first place.

    Returns `(spike_counts, trials, rows)`, each row a dict of everything
    :func:`plot_rows` needs to draw it.
    """
    behavior_2obj = load_behavior(subject, session, num_objects=2)
    behavior_1obj = load_behavior(subject, session, num_objects=1)
    assert (
        len(behavior_2obj) > 0
    ), f"no 2-object ring behavior for {subject} {session}"
    spike_counts, trials, quality_label = load_unit(subject, session, unit_id)
    print(
        f"{subject} {session}: {len(behavior_2obj)} 2-obj + {len(behavior_1obj)} "
        f"1-obj trials | unit {unit_id} @ {quality_label} "
        f"({len(trials)} trials with spikes) | prototypes at {orientation_deg:g}°"
    )

    window_rad = np.radians(similarity_window_deg)
    offset = np.radians(orientation_deg)
    sel_1obj = prototypes_1obj(n_proto, offset_rad=offset)
    sel_sep1 = prototypes_sep(n_proto, SEP1_DEG, offset_rad=offset)
    sel_180 = prototypes_180(n_proto, offset_rad=offset)

    dense = dict(
        window_rad=window_rad,
        offset_rad=offset,
        fr_window=FR_WINDOW,
        smooth_half_win_ms=smooth_half_win_ms,
    )
    phis_s1, fr_s1 = dense_2obj_tuning(
        behavior_2obj, spike_counts, trials, SEP1_DEG, N_DENSE, **dense
    )
    phis_180, fr_180 = dense_2obj_tuning(
        behavior_2obj,
        spike_counts,
        trials,
        180,
        N_DENSE,
        orient_span=np.pi,
        **dense,
    )
    phis_1obj, fr_1obj = dense_1obj_tuning(
        behavior_1obj, spike_counts, trials, N_DENSE_1OBJ, **dense
    )

    if share_2obj_color_scale:
        norm_s1 = norm_180 = _fr_norm(np.concatenate([fr_s1, fr_180]))
    else:
        norm_s1, norm_180 = _fr_norm(fr_s1), _fr_norm(fr_180)

    rows = []
    for label, protos, tpc, kind, sep, phis, frs, norm in [
        (
            "1 object",
            sel_1obj,
            window_trials_1obj(behavior_1obj, sel_1obj, window_rad),
            "dot",
            None,
            phis_1obj,
            fr_1obj,
            _fr_norm(fr_1obj),
        ),
        (
            f"{SEP1_DEG:g}° apart",
            sel_sep1,
            window_trials_2obj(behavior_2obj, sel_sep1, window_rad),
            "chord",
            SEP1_DEG,
            phis_s1,
            fr_s1,
            norm_s1,
        ),
        (
            "180° apart",
            sel_180,
            window_trials_2obj(behavior_2obj, sel_180, window_rad),
            "chord",
            180,
            phis_180,
            fr_180,
            norm_180,
        ),
    ]:
        mfr = mean_fr_by_condition(
            spike_counts,
            trials,
            tpc,
            window=FR_WINDOW,
            smooth_half_win_ms=smooth_half_win_ms,
        )
        rows.append(
            dict(
                label=label,
                protos=protos,
                trials_per_condition=tpc,
                colors=_fr_colors(norm, mfr, cmap),
                kind=kind,
                sep=sep,
                phis=phis,
                frs=frs,
                norm=norm,
            )
        )
    return spike_counts, trials, rows


def layout_columns(fig, axes, caxes):
    """Lay the columns out horizontally by hand, in inches.

    Each row is packed as::

        [margin][row label][stim square][stim_gap][y label+ticks][PSTH]
                [tuning_gap][tuning square][cbar_gap][colorbar][ticks][margin]

    The two stimulus panels are **squares whose side equals the height of that
    row's PSTH axes** (so their size is set by the vertical layout), the colorbar
    is ``cbar_width_in`` wide and spans the PSTH axes' height, and the PSTH axes
    absorb whatever width is left over. The ``*_gap_in`` values therefore set the
    white space around the stimulus/tuning panels directly, in inches.

    Call after the layout engine has run, since only then are the vertical
    positions and the label extents final.
    """
    psth_col = 1
    stim_gap_in, tuning_gap_in = 0.06, 0.10
    cbar_gap_in, cbar_width_in = 0.10, 0.10
    margin_in = 0.02
    n_rows = axes.shape[0]
    # Measure label/tick overhang with the column titles hidden — a title wider
    # than its axes would otherwise inflate the tight bbox horizontally.
    titles = [ax.get_title() for ax in axes[0]]
    for ax in axes[0]:
        ax.set_title("")
    fig.canvas.draw()
    w_in, h_in = fig.get_size_inches()

    def _tight(ax):
        return ax.get_tightbbox().transformed(fig.dpi_scale_trans.inverted())

    def _overhang_left(ax):
        return max(0.0, ax.get_position().x0 * w_in - _tight(ax).x0)

    def _overhang_right(ax):
        return max(0.0, _tight(ax).x1 - ax.get_position().x1 * w_in)

    label_stim = max(_overhang_left(axes[r, 0]) for r in range(n_rows))
    label_psth = max(_overhang_left(axes[r, psth_col]) for r in range(n_rows))
    label_cbar = max(_overhang_right(caxes[r]) for r in range(n_rows))

    for ax, t in zip(axes[0], titles):
        ax.set_title(t)

    side = (
        axes[0, psth_col].get_position().height * h_in
    )  # square side (inches)
    x_stim = margin_in + label_stim
    x_psth = x_stim + side + stim_gap_in + label_psth
    w_psth = (
        w_in
        - margin_in
        - label_cbar
        - cbar_width_in
        - cbar_gap_in
        - side
        - tuning_gap_in
        - x_psth
    )
    assert (
        w_psth > 0
    ), "No room left for the PSTH axes — shrink the gaps or widen FIGSIZE"
    x_tuning = x_psth + w_psth + tuning_gap_in
    x_cbar = x_tuning + side + cbar_gap_in

    for r in range(n_rows):
        p_psth = axes[r, psth_col].get_position()
        y_c = p_psth.y0 + p_psth.height / 2  # row center
        h_sq = side / h_in
        axes[r, psth_col].set_position(
            [x_psth / w_in, p_psth.y0, w_psth / w_in, p_psth.height]
        )
        axes[r, 0].set_position(
            [x_stim / w_in, y_c - h_sq / 2, side / w_in, h_sq]
        )
        axes[r, 2].set_position(
            [x_tuning / w_in, y_c - h_sq / 2, side / w_in, h_sq]
        )
        caxes[r].set_position(
            [x_cbar / w_in, p_psth.y0, cbar_width_in / w_in, p_psth.height]
        )

    # Pin every colorbar label to one x, clear of the *widest* row's tick labels. Left to
    # itself each label hugs its own ticks, so a row reading '5' sits further left than one
    # reading '15'. The caxes are all `cbar_width_in` wide, so a shared axes-fraction x is a
    # shared figure x. The right margin above already reserved room for the widest row.
    cbar_labels = [cax.get_ylabel() for cax in caxes]
    for cax in caxes:
        cax.set_ylabel("")
    fig.canvas.draw()
    ticks_in = max(_overhang_right(cax) for cax in caxes)
    for cax, text in zip(caxes, cbar_labels):
        cax.set_ylabel(text)
        pad_in = cax.yaxis.labelpad / 72
        cax.yaxis.set_label_coords(
            1 + (ticks_in + pad_in) / cbar_width_in, 0.5
        )


def plot_rows(
    spike_counts,
    trials,
    rows,
    *,
    figsize,
    cmap,
    smooth_half_win_ms,
    response_time_ms,
):
    """Draw the 3x3 figure, plus a colorbar column: stimuli | PSTHs | dense tuning.

    The dashed vertical lines mark stimulus, delay and cue onset; `response_time_ms` adds one
    more at the mean response time, drawn the same way — it is an average over trials rather
    than a task boundary, but it belongs to the same family of events.

    The two 2-object rows share a PSTH y-scale so their heights can be compared; the
    1-object row is on its own, as its rates are not comparable to theirs.
    """
    # Fixed sizes, in the units they are measured in. The rings are drawn in axes fractions
    # (0.5 reaches the panel edge); the column gaps are in inches, because the panels are
    # squares pinned to the PSTH height and the gaps set the space around them directly.
    STIM_RADIUS, TUNING_RADIUS = 0.4, 0.43
    DENSE_LW, PSTH_LW = 1.5, 1.2
    EVENT_TIMES_MS = (0, 1000, 2000)  # stimulus, delay and cue onset

    n_rows = len(rows)
    fig = plt.figure(figsize=figsize, dpi=300)
    gs = fig.add_gridspec(n_rows, 4, width_ratios=(1, 3, 1, 0.1))
    axes = np.array(
        [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(n_rows)]
    )
    caxes = [fig.add_subplot(gs[r, 3]) for r in range(n_rows)]

    row_ymax = []
    for r, row in enumerate(rows):
        # Col 0: the selected prototypes, coloured by mean firing rate.
        if row["kind"] == "chord":
            draw_2obj_prototypes(
                axes[r, 0],
                row["protos"],
                row["colors"],
                radius=STIM_RADIUS,
            )
        else:
            draw_1obj_prototypes(
                axes[r, 0],
                row["protos"],
                row["colors"],
                radius=STIM_RADIUS,
            )
        axes[r, 0].set_ylabel(row["label"])

        # Col 1: the PSTHs of those same conditions, in those same colours.
        row_ymax.append(
            plot_psth(
                axes[r, 1],
                spike_counts,
                trials,
                row["trials_per_condition"],
                row["colors"],
                smooth_half_win_ms=smooth_half_win_ms,
                lw=PSTH_LW,
            )
        )

        # Col 2: every orientation of the row's condition — chords for the 2-object rows,
        # a smoothed annulus for the 1-object one.
        if row["kind"] == "chord":
            draw_dense_2obj(
                axes[r, 2],
                row["phis"],
                row["sep"],
                row["frs"],
                row["norm"],
                radius=TUNING_RADIUS,
                cmap=cmap,
                lw=DENSE_LW,
            )
        else:
            draw_1obj_annulus(
                axes[r, 2],
                row["phis"],
                row["frs"],
                row["norm"],
                radius=TUNING_RADIUS,
                cmap=cmap,
                smooth_deg=10.0,
            )
        cb = fig.colorbar(
            ScalarMappable(norm=row["norm"], cmap=cmap), cax=caxes[r]
        )
        # Whole numbers only: the default locator's 2.5 step puts decimals on the narrow
        # colorbars, and a rate to the tenth of a Hz is more precision than these mean.
        cb.locator = MaxNLocator(
            nbins="auto", integer=True, steps=[1, 2, 5, 10]
        )
        cb.update_ticks()
        cb.ax.tick_params(labelsize=8)
        cb.set_label(
            "Mean FR (Hz)", fontsize=12
        )  # FORMATTING.md: nothing below 12 pt

    for col, text in enumerate(("Stimuli", "PSTH", "Tuning")):
        axes[0, col].set_title(text)

    y_2obj = max(
        [y for row, y in zip(rows, row_ymax) if row["kind"] == "chord"],
        default=0,
    )
    for r, row in enumerate(rows):
        y = y_2obj if row["kind"] == "chord" else row_ymax[r]
        ax = axes[r, 1]
        ax.vlines(
            EVENT_TIMES_MS,
            0,
            y,
            colors="k",
            linestyles="dashed",
            linewidth=0.7,
        )
        if np.isfinite(response_time_ms):
            ax.vlines(
                [response_time_ms],
                0,
                y,
                colors="k",
                linestyles="dashed",
                linewidth=0.7,
                zorder=4,
            )
        ax.set_xlim(*STIM_TIME_WINDOW)
        ax.set_ylim(0, y)
        ax.set_ylabel("Firing rate (Hz)")
        if r == n_rows - 1:
            ax.set_xlabel("Time after stimulus (ms)")
        else:
            ax.set_xticklabels([])

    fig.tight_layout()
    layout_columns(fig, axes, caxes)
    return fig, axes


def psth_figure(
    subject,
    session,
    unit_id,
    orientation_deg=0.0,
    similarity_window_deg=15.0,
    smooth_half_win_ms=250.0,
    n_proto=4,
    share_2obj_color_scale=True,
    figsize=(6.0, 4.5),
):
    """The whole figure for one unit: :func:`build_rows` then :func:`plot_rows`.

    The PSTHs also get a dashed line at the **mean response time of the trials this figure
    draws** — pooled over every condition of every row, so the line describes the panel
    rather than the session. Drawn in :func:`firing_rate_cmap`.
    """
    cmap = firing_rate_cmap()
    spike_counts, trials, rows = build_rows(
        subject,
        session,
        unit_id,
        orientation_deg=orientation_deg,
        similarity_window_deg=similarity_window_deg,
        smooth_half_win_ms=smooth_half_win_ms,
        n_proto=n_proto,
        share_2obj_color_scale=share_2obj_color_scale,
        cmap=cmap,
    )
    drawn = {
        t
        for row in rows
        for ts in row["trials_per_condition"].values()
        for t in ts
    }
    response_ms = common.mean_response_time_ms(
        common.load_behavior(subject, session), drawn
    )
    print(f"mean response time {response_ms:.0f} ms after stimulus onset")
    return plot_rows(
        spike_counts,
        trials,
        rows,
        cmap=cmap,
        figsize=figsize,
        smooth_half_win_ms=smooth_half_win_ms,
        response_time_ms=response_ms,
    )

"""The titration figure, and the one test under it.

Everything here is reachable from `titration.ipynb`; nothing here is reachable from anywhere
else. What the notebook asks is the whole scope of the module: **the curve**
(:func:`plot_figure`) and **whether any composition decodes better than the recorded one**
(:func:`unbiased_verdict`).

The sweep variable **x is the target weighted mean rank**. λ(x) inverts
``m(λ) = 1/(1 − e^−λ) − 1/λ``, the mean of a density ∝ e^{λr} on the unit interval; because
within-animal ranks are uniform on [0, 1] by construction, that inverse depends on no data at
all, so **one λ grid serves both animals** and **λ(0.5) = 0 exactly**. x = 0.5 is therefore the
unbiased population, which turns "is the recorded composition at the optimum?" into the direct
question the verdict cell asks: can any other composition do better?

Removed on 2026-09-13, and recoverable from git if ever wanted again: a shuffled-ρ null battery
(`null_summary`, `load_null`, `null_test`) and the 40 shuffle CSVs it read; a bootstrap
argmin-CI / curvature report (`report`, `cached_report`, `argmin_ci`, `argmin_ci_rank`,
`curvature_ci`); five alternative optimality tests (`optimality`, `baseline_test`,
`interior_test`, `three_point_contrast`, `matched_permutation_test`); and two alternative
figures (`plot_curve_axes`, `plot_sweep_insets`, `plot_baseline`). None was called by the
notebook or by any other module — together they were half this file.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

HERE = Path(__file__).resolve().parent
for _p in (
    str(HERE),
    str(HERE.parent),
    str(HERE.parent / "model_comparison"),
    str(HERE.parent.parent),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pool_lib as pool  # noqa: E402
import selection_lib as sel  # noqa: E402

DF = HERE / "dataframes"
SELECTION = dict(
    coordinate="angle", exclusion="config", min_fit=0.0, min_fit_mode="both"
)
MEAN_RANKS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
UNBIASED_X = 0.5


def units():
    return pool.selection_units(verbose=False, **SELECTION)


def population(u=None):
    return sel.population_rho(u if u is not None else units())


def load(name):
    """A results CSV by stem, or by arm key — ``'k40_b20'`` and ``'results_k40_b20'`` both work."""
    stem = str(name)
    if not stem.startswith("results_"):
        stem = f"results_{stem}"
    return load_results(stem, directory=DF)


def draw(stat, lam, k, rng):
    """``k`` distinct units without replacement with probability proportional to exp(lam*stat).

    The Gumbel top-k trick: adding i.i.d. Gumbel noise to ``lam*stat`` and taking the largest
    ``k`` keys is exactly successive sampling without replacement proportional to
    ``exp(lam*stat)``, in one vectorised pass rather than ``k`` renormalisations.

    This is the same routine `run_titration.tilt_positions` uses for the fits; it is duplicated
    here so the figures can be drawn without importing the sweep driver, which sets torch thread
    counts at import time.
    """
    z = lam * (np.asarray(stat, float) - float(np.mean(stat)))
    keys = z + rng.gumbel(size=len(z))
    k = min(int(k), len(z))
    return np.argpartition(-keys, k - 1)[:k]


def mean_rank_of_tilt(lam):
    """Mean of the density ∝ ``exp(lam*r)`` on [0, 1]; ``m(0) = 1/2`` exactly."""
    lam = float(lam)
    if abs(lam) < 1e-6:
        return 0.5 + lam / 12.0
    if lam > 60:
        return 1.0 - 1.0 / lam
    if lam < -60:
        return -1.0 / lam
    return 1.0 / (1.0 - np.exp(-lam)) - 1.0 / lam


def lam_for_mean_rank(x):
    """Invert :func:`mean_rank_of_tilt`. Data-independent, and exactly 0 at x = 0.5."""
    if abs(float(x) - 0.5) < 1e-12:
        return 0.0
    from scipy.optimize import brentq

    return float(
        brentq(
            lambda L: mean_rank_of_tilt(L) - float(x),
            -200.0,
            200.0,
            xtol=1e-10,
        )
    )


def lam_grid(xs=MEAN_RANKS):
    return pd.DataFrame(
        {"x": list(xs), "lam": [lam_for_mean_rank(v) for v in xs]}
    )


# ── realized vs target ───────────────────────────────────────────────────────


# ── figures ──────────────────────────────────────────────────────────────────


# ── Phase-1 decision figures ─────────────────────────────────────────────────


# ── The curve on all three x axes ────────────────────────────────────────────
def curve_axes_table(df, u, subject, n_rep=400, seed=0, ycol="error_deg"):
    """Per sweep point: the three x candidates and the error, with its s.e.m.

    ``target_rank`` is the sweep variable itself. ``empirical_rank`` is the mean rank of the
    units actually drawn — read from the run when `run_titration` recorded it, and estimated by
    re-simulating the draw otherwise (phase-1 runs predate the column). ``lam`` is the tilt.
    """
    g = fitted(df)
    g = g[g["subject"] == subject]
    have_rank = (
        "mean_rank_inputs" in g.columns and g["mean_rank_inputs"].notna().any()
    )
    if not have_rank:
        gu = sel.add_rank(u[u["subject"] == subject].reset_index(drop=True))
        rank_all = gu["rho_rank"].to_numpy(float)
        k = int(g["n_units"].max())
        rng = np.random.default_rng(seed)
    rows = []
    for x, gg in g.groupby("sweep_fraction"):
        lam = float(gg["lam"].iloc[0])
        if have_rank:
            emp = float(gg["mean_rank_inputs"].mean())
        else:
            emp = float(
                np.mean(
                    [
                        rank_all[draw(rank_all, lam, k, rng)].mean()
                        for _ in range(n_rep)
                    ]
                )
            )
        v = gg[ycol].to_numpy(float)
        rows.append(
            dict(
                target_rank=float(x),
                empirical_rank=emp,
                lam=lam,
                mean_rho=float(gg["mean_rho_inputs"].mean()),
                y=float(v.mean()),
                y_sem=float(v.std(ddof=1) / np.sqrt(len(v))),
                n=len(v),
            )
        )
    return pd.DataFrame(rows).sort_values("target_rank").reset_index(drop=True)


#: (column, axis label, where the unbiased population sits on that axis)
CURVE_AXES = [
    ("target_rank", "target mean rank  x", 0.5),
    ("empirical_rank", "realized mean rank of the draw", 0.5),
    ("lam", "tilt strength  λ", 0.0),
]


# ── Caching, so the notebook is cheap to re-run ──────────────────────────────
#: Derived quantities live here. The *fitting* is already cached — it is the CSVs in
#: `dataframes/` — but the notebook's own expensive steps are the neuron pool (~25 s to load and
#: re-apply the selection criteria) and the bootstrap CIs (a few thousand quadratic fits each).
#: Those are cached separately so that re-running the notebook with a **different inset seed**
#: redraws the figures and touches nothing else.
CACHE = HERE / "cache"
FINAL = "results_k20_b100"
#: K -- distinct units per sweep point -- is 160 (Monkey P) / 51 (Monkey E) at every point of
#: every run here, derived by `run_titration.matched_k` rather than chosen. :func:`plot_figure`
#: still reads it off the run it is plotting (the `n_units` column) rather than trusting this
#: note.


def _cache_path(name):
    CACHE.mkdir(exist_ok=True)
    return CACHE / name


def cached_units(force=False):
    """The pool, cached to parquet/csv. Rebuilding it re-reads the fits and re-applies the
    selection criteria, which is the single slowest thing the notebook does."""
    p = _cache_path("units.csv")
    if p.exists() and not force:
        return pd.read_csv(p)
    u = units()
    u.to_csv(p, index=False)
    return u


def cached_population(u=None, force=False):
    p = _cache_path("population_rho.csv")
    if p.exists() and not force:
        return pd.read_csv(p)
    r = population(u if u is not None else cached_units())
    r.to_csv(p, index=False)
    return r


def cached_axes_table(name=FINAL, u=None, force=False):
    """The three-x-axis table per animal, cached. Exact from phase 2 on — `run_titration` now
    records the realized mean rank, so nothing is simulated here.

    **The cache is invalidated when the sweep CSV it summarises is newer than it.** Without that
    check, re-running the sweep leaves a stale table behind and the figure silently plots the
    OLD sweep points while every statistic in the notebook — `unbiased_verdict` reads `load()`
    directly — comes from the new ones. That happened on 2026-09-13: a partially-failed sweep
    cached 12 of 18 points, and after the missing six were recomputed the curve still showed 12.
    A figure disagreeing with the test beneath it is the worst failure mode this module has.
    """
    p = _cache_path(f"axes_{name}.csv")
    src = DF / f"{name}.csv"
    fresh = p.exists() and (
        not src.exists() or p.stat().st_mtime >= src.stat().st_mtime
    )
    if fresh and not force:
        return pd.read_csv(p)
    u = cached_units() if u is None else u
    d = load(name)
    t = pd.concat(
        [
            curve_axes_table(d, u, s).assign(subject=sel.subject_title(s))
            for s in sel.SUBJECTS
        ],
        ignore_index=True,
    )
    t.to_csv(p, index=False)
    return t


def clear_cache():
    """Delete the derived cache. The fits in `dataframes/` are untouched."""
    n = 0
    if CACHE.exists():
        for f in CACHE.glob("*.csv"):
            f.unlink()
            n += 1
    print(f"cleared {n} cached files from {CACHE} (dataframes/ untouched)")


# ── The paper figure: insets over the curve ──────────────────────────────────
def log_tilt_density(r, lam):
    """log of the density ∝ ``exp(lam*r)`` on [0, 1], normalised to integrate to 1.

    This is the sampling weight the sweep actually applies, as a density on **rank**: ranks are
    uniform on [0, 1] by construction, so a value of 1 means "as likely as under no tilt at all",
    and ``lam = 0`` gives exactly 1 everywhere. Written in logs because ``exp(lam)`` overflows
    long before the densities themselves become extreme.
    """
    r = np.asarray(r, float)
    lam = float(lam)
    if abs(lam) < 1e-9:
        return np.zeros_like(r)
    a = abs(lam)
    log_z = np.log(a) - np.log1p(-np.exp(-a))
    return log_z + (lam * (r - 1.0) if lam > 0 else lam * r)


def grey_ramp(top=0.85, name="Greys"):
    """``Greys`` truncated at ``top``, so the darkest shading stays short of black."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "greys_cut", plt.get_cmap(name)(np.linspace(0.0, float(top), 256))
    )


#: How the two animals are told apart on the shared axis -- marker shape and line style, not
#: colour.
SUBJECT_MARKERS = {"Monkey P": "o", "Monkey E": "^"}
SUBJECT_LINESTYLES = {"Monkey P": "-", "Monkey E": "--"}


def _connection_patch_cls():
    from matplotlib.patches import ConnectionPatch

    class _UnderLabel(ConnectionPatch):
        """A connector whose top end sits ``pad_pt`` points below ``label_ax``'s x label.

        The label's position is only known once the layout engine has run, so the start point
        is recomputed at every draw, in the axes' x-axis coordinates.
        """

        label_ax = None
        pad_pt = 3.0

        def draw(self, renderer):
            ax = self.label_ax
            y_disp = (
                ax.xaxis.label.get_window_extent(renderer).y0
                - self.pad_pt * self.figure.dpi / 72.0
            )
            y_axes = ax.transAxes.inverted().transform((0.0, y_disp))[1]
            self.xy1 = (self.xy1[0], y_axes)
            super().draw(renderer)

    return _UnderLabel


_UnderLabelConnection = _connection_patch_cls()


def plot_figure(
    u,
    name=FINAL,
    inset_xs=(0.1, 0.5, 0.9),
    seed=0,
    figsize=(3.4, 2.6),
    ycol="error_deg",
    xlabel="Relational sampling bias",
    ylabel="Stimulus decoder error (°)",
    xticks=(0.2, 0.5, 0.8),
    unbiased_line=False,
    inset_subject="Perle",
    inset_row_height=0.45,
    inset_width=0.30,
    inset_labels=("Retino.", "Relat."),
    inset_label_fontsize=None,
    inset_anchor="N",
    inset_grid=240,
    inset_cmap="Greys",
    inset_shade_max=0.55,
    inset_density_range=10.0,
    inset_point_size=5,
    inset_point_color="tab:red",
    inset_point_alpha=0.5,
    qmax=99,
    legend=True,
    connectors=True,
    connector_color="0.85",
    line_color="k",
    marker_face="0.4",
    marker_size=8,
    band_alpha=0.25,
    legend_rounding=0.15,
    insets_below=False,
    connector_label_pad=3.0,
    row_gap=None,
    direction_arrow=False,
    arrow_y=1.06,
    arrow_label_y=1.10,
    arrow_x=(0.02, 0.98),
    arrow_labels=("Retinotopic", "Relational"),
    arrow_label_colors=("tab:blue", "tab:orange"),
    arrow_label_fontsize=None,
    arrow_length=None,
    arrow_head_size=10,
    arrow_label_weight="normal",
    arrow_label_stroke=0.0,
    connector_in_plot=True,
    rep=None,
):
    """The headline panel: **both animals on one axis**, with example draws above it.

        The x axis is the **realized mean rank of the draw** -- what the decoder actually received,
        not what the sweep asked for. The two differ at the ends because drawing K distinct units
        without replacement caps the realized mean rank; plotting the target would overstate the
        range by hiding that. The two animals realize almost the same ranks (0.14 / 0.86 at the
        ends), which is what makes one shared axis honest here. The shaded band is ±1 s.e.m. of
        the plotted mean over the 100 training repeats.

        The animals are told apart by **marker shape and line style**, not colour. The insets are
        square boxed panels in their own gridspec row, tied to their sweep point
        by a rule down the panel, with the axis labels on the left-most one only.

        Each inset shows the sweep point's **sampling distribution** as a grey shading, with one
        **single draw** from it in red -- what one training run receives -- so the red dots move with
        ``seed`` and the shading does not. One animal's pool is shown because the insets illustrate
        the sampling scheme, which is identical in both; showing both would double the row to say the
        same thing twice.

    The shading is the **tilt density itself**, and depends on the polar angle alone. A pixel at
        angle θ has ρ = (2/π)·θ; that ρ has a rank in this animal's pool; the tilt weights it by
        ``exp(λ·rank)``. Normalised as a density on rank (:func:`log_tilt_density`), so **1 means
        "as likely as under no tilt"** and the unbiased panel is *exactly* flat rather than
        accidentally so. The scale is logarithmic and **shared across panels**, spanning
        ``1/inset_density_range`` to ``inset_density_range``, so the three panels are directly
        comparable: white is a region the tilt has emptied, dark is one it has concentrated into.

        Note it shows the *weight field*, not where the neurons are — the pool's own density in the
        plane is visible as the red draw, which is the weight field times the pool.

        Deliberately absent: the fitted quadratic, the shaded CI of its vertex, and the red shading
        of the points that decode significantly worse than unbiased. The quadratic and its vertex
        are a summary of the curve, not a measurement, and inviting the eye to read optimality off
        them competes with the test that actually answers it; :func:`unbiased_verdict` is that test.
        The single grey dashed line marks the unbiased population at rank 0.5, which is exact by
        construction (λ(0.5) = 0).

        ``inset_shade_max`` is the one knob for how dark the shading gets: it truncates the grey
        ramp, so 1.0 takes the densest wedge to black and lower values lift the whole scale toward
        white without changing what any shade means relative to the others.

        ``insets_below`` moves the inset row under the curve, below its x axis and x label. The
        rules then run down out of the curve panel to the top of each inset, except that the
        middle one -- which would cut through the x label -- starts ``connector_label_pad`` points
        below the label instead of at the axis. ``row_gap`` is the layout engine's ``hspace``
        between the curve and the inset row (a fraction of the average row height); ``None`` keeps
        its default.

        ``direction_arrow`` draws a black double-headed arrow over the curve, spanning ``arrow_x``
        at height ``arrow_y``, with ``arrow_labels`` in ``arrow_label_colors`` above its left and
        right ends at ``arrow_label_y`` -- which way along the x axis the mix tilts. All positions
        are in the curve panel's axes fractions, so values above 1 sit above the plot.
        ``arrow_length`` (an axes fraction) shortens the arrow about the centre of ``arrow_x``
        while the labels stay at ``arrow_x``; ``None`` spans ``arrow_x`` exactly.
        ``arrow_head_size`` scales the arrowheads (matplotlib's ``mutation_scale``).
        ``arrow_label_stroke`` thickens the labels continuously, by outlining each glyph in its own
        colour with a stroke that many points wide -- Arial ships only regular and bold, so
        ``arrow_label_weight`` alone cannot give anything in between. 0 adds nothing.

        ``connector_in_plot=False`` keeps the grey rules out of the curve panel: they run only
        between it and the insets.

        ``rep`` is accepted and ignored, so the old call signature still works.
    """
    d = load(name)
    axes_t = cached_axes_table(name, u)
    rng = np.random.default_rng(seed)
    subjects = [sel.subject_title(s) for s in sel.SUBJECTS]

    heights = [inset_row_height, 1.0]
    fig, axs = plt.subplots(
        2,
        1,
        figsize=(figsize[0], figsize[1] * (1 + inset_row_height)),
        height_ratios=heights[::-1] if insets_below else heights,
        layout="constrained",
    )
    ax, inset_row = axs if insets_below else axs[::-1]
    if row_gap is not None:
        fig.get_layout_engine().set(hspace=row_gap)

    # ── the curve, both animals ──
    for title in subjects:
        t = axes_t[axes_t["subject"] == title].sort_values("empirical_rank")
        # A band rather than caps: with both animals on one axis and nine points each, eighteen
        # sets of caps read as clutter, and it is ±1 s.e.m. of the plotted mean either way.
        ax.fill_between(
            t["empirical_rank"],
            t["y"] - t["y_sem"],
            t["y"] + t["y_sem"],
            color=line_color,
            alpha=band_alpha,
            lw=0,
            zorder=2,
        )
        ax.plot(
            t["empirical_rank"],
            t["y"],
            marker=SUBJECT_MARKERS[title],
            ls=SUBJECT_LINESTYLES[title],
            color=line_color,
            ms=marker_size,
            mfc=marker_face,
            mec=line_color,
            mew=0.5,
            lw=1.0,
            label=title,
            zorder=3,
        )
    if unbiased_line:
        ax.axvline(UNBIASED_X, color="0.55", lw=1.0, ls="--", zorder=1)
    ax.set_xticks(list(xticks))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend:
        # Boxed with slightly rounded corners, as in the v1 notebook and figure_1/swap_errors/.
        # `borderpad` alone sets how tight the box is -- the box style's own pad is 0, or the
        # frame would grow outside the extent the legend reports and a tight bbox would clip it.
        lg = ax.legend(
            frameon=True,
            framealpha=1.0,
            edgecolor="0.4",
            handlelength=1.8,
            borderpad=0.3,
            labelspacing=0.25,
            handletextpad=0.5,
        )
        lg.get_frame().set_linewidth(0.8)
        lg.get_frame().set_boxstyle(
            "round", pad=0.0, rounding_size=legend_rounding
        )
    sel._despine(ax)
    if direction_arrow:
        if arrow_length is None:
            a0, a1 = arrow_x
        else:
            mid = 0.5 * (arrow_x[0] + arrow_x[1])
            a0, a1 = mid - arrow_length / 2, mid + arrow_length / 2
        ax.annotate(
            "",
            xy=(a1, arrow_y),
            xytext=(a0, arrow_y),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops=dict(
                arrowstyle="<|-|>",
                color="k",
                lw=1.2,
                shrinkA=0,
                shrinkB=0,
                mutation_scale=arrow_head_size,
            ),
            annotation_clip=False,
        )
        for x, ha, txt, c in zip(
            arrow_x, ("left", "right"), arrow_labels, arrow_label_colors
        ):
            tx = ax.text(
                x,
                arrow_label_y,
                txt,
                color=c,
                ha=ha,
                va="bottom",
                fontsize=arrow_label_fontsize,
                fontweight=arrow_label_weight,
                transform=ax.transAxes,
            )
            if arrow_label_stroke:
                from matplotlib import patheffects

                tx.set_path_effects(
                    [
                        patheffects.withStroke(
                            linewidth=arrow_label_stroke, foreground=c
                        )
                    ]
                )

    # ── the inset row ──
    # It shares the curve's x limits, so an inset placed at a sweep point sits exactly over it.
    # Drawn last, when the x limits have stopped moving.
    g = u[u["subject"] == inset_subject]
    rank = sel.add_rank(g.reset_index(drop=True))["rho_rank"].to_numpy(float)
    xr = (
        np.maximum(g["retinotopic_excess"].to_numpy(float), 0)
        * sel.ACCURACY_SCALE
    )
    yr = (
        np.maximum(g["relational_excess"].to_numpy(float), 0)
        * sel.ACCURACY_SCALE
    )
    lim = float(np.percentile(np.concatenate([xr, yr]), qmax)) * 1.05
    # K off the run itself, not a constant: the arms differ in exactly this number.
    k = int(d.loc[d["subject"] == inset_subject, "n_units"].iloc[0])
    # The pool's own rho, sorted: the empirical CDF that turns a pixel's angle into a rank.
    rho_sorted = np.sort(g["rho"].to_numpy(float))
    _gx = np.linspace(0, lim, inset_grid)
    X_px, Y_px = np.meshgrid(_gx, _gx)  # 'xy' indexing: row = y, column = x
    from matplotlib.colors import LogNorm

    t = axes_t[axes_t["subject"] == sel.subject_title(inset_subject)]

    inset_row.set_xlim(ax.get_xlim())
    inset_row.set_xticks([])
    inset_row.set_yticks([])
    for spine in inset_row.spines.values():
        spine.set_visible(False)
    x0, x1 = ax.get_xlim()
    for j, x in enumerate(inset_xs):
        hit = t.loc[np.isclose(t["target_rank"], x), "empirical_rank"]
        if not len(hit):
            # Without this the failure is an opaque `IndexError` from `.iloc[0]`, several
            # frames deep, for a run whose CSV looks perfectly fine.
            raise ValueError(
                f"no curve point at x = {x:g} for {sel.subject_title(inset_subject)} in run "
                f'{name!r}; available: {sorted(t["target_rank"].round(3))}. Either `inset_xs` '
                f"names a point this run does not have, or the point exists in the CSV but was "
                f"dropped by `fitted` (k_matched=False, only possible in a CSV predating the "
                f'always-matched rule), or {_cache_path(f"axes_{name}.csv")} is stale — delete '
                f"it, or pass force=True, to rebuild it from the CSV."
            )
        emp = float(hit.iloc[0])
        frac = (emp - x0) / (x1 - x0)
        ins = inset_row.inset_axes(
            [frac - inset_width / 2, 0.0, inset_width, 1.0]
        )
        ins.set_box_aspect(1.0)
        # The square is narrower than its slot is tall, and matplotlib centres the slack by
        # default -- which shows up as white space above the row, at the top of the figure.
        # Anchoring north puts all of it below the panels instead, where the x label and the
        # connectors already live.
        ins.set_anchor(inset_anchor)
        lam = lam_for_mean_rank(x)
        idx = draw(rank, lam, k, rng)
        # A function of the ANGLE only: each pixel's rho, that rho's rank in this animal's pool,
        # and the tilt's density at that rank. The rank comes from the pool's own empirical CDF,
        # so the shading is the weight the sweep really applies and not an idealisation of it.
        rho_px = (2.0 / np.pi) * np.arctan2(Y_px, X_px)
        rank_px = (
            np.searchsorted(rho_sorted, rho_px, side="right") - 0.5
        ) / len(rho_sorted)
        ins.imshow(
            np.exp(log_tilt_density(rank_px, lam)),
            origin="lower",
            extent=(0, lim, 0, lim),
            cmap=grey_ramp(inset_shade_max, inset_cmap),
            norm=LogNorm(
                vmin=1.0 / inset_density_range, vmax=inset_density_range
            ),
            aspect="auto",
            interpolation="bilinear",
            zorder=0,
        )
        ins.scatter(
            xr[idx],
            yr[idx],
            s=inset_point_size,
            color=inset_point_color,
            alpha=inset_point_alpha,
            lw=0,
            zorder=2,
        )
        ins.set_xlim(0, lim)
        ins.set_ylim(0, lim)
        ins.set_xticks([])
        ins.set_yticks([])
        for side in ("top", "right", "bottom", "left"):
            ins.spines[side].set_color("0.4")
            ins.spines[side].set_linewidth(0.8)
        if j == 0 and inset_labels:  # the three share their axes
            # None inherits rcParams' 12 pt, the floor in figures/FORMATTING.md.
            lkw = (
                {}
                if inset_label_fontsize is None
                else dict(fontsize=inset_label_fontsize)
            )
            ins.set_xlabel(inset_labels[0], labelpad=2, **lkw)
            ins.set_ylabel(inset_labels[1], labelpad=2, **lkw)
        if connectors:
            # A rule saying which sweep point this inset is of: down the curve panel behind the
            # data, continued across the gap. Axis coordinates on both ends, so the layout
            # engine can still move either row.
            from matplotlib.patches import ConnectionPatch

            if connector_in_plot:
                ax.axvline(emp, color=connector_color, lw=0.9, zorder=0.5)
            if not insets_below:
                fig.add_artist(
                    ConnectionPatch(
                        xyA=(emp, 0.0),
                        coordsA=inset_row.get_xaxis_transform(),
                        xyB=(emp, 1.0),
                        coordsB=ax.get_xaxis_transform(),
                        color=connector_color,
                        lw=0.9,
                        zorder=-1,
                    )
                )
                continue
            cls = (
                _UnderLabelConnection
                if j == len(inset_xs) // 2
                else ConnectionPatch
            )
            con = cls(
                xyA=(emp, 0.0),
                coordsA=ax.get_xaxis_transform(),
                xyB=(emp, 1.0),
                coordsB=inset_row.get_xaxis_transform(),
                color=connector_color,
                lw=0.9,
                zorder=-1,
            )
            if cls is _UnderLabelConnection:
                con.label_ax, con.pad_pt = ax, connector_label_pad
            fig.add_artist(con)
    return fig


# ── Is any composition actually better than the unbiased one? ────────────────
def compare_to_unbiased(
    df, subject, ref_x=UNBIASED_X, ycol="error_deg", alpha=0.05
):
    """Every sweep point tested against the unbiased point, Holm-corrected within animal.

    This asks the question the argmin CI only answers indirectly. The argmin is the vertex of a
    fitted quadratic, so it can sit slightly off 0.5 — and be *significantly* off — while no
    composition the sweep actually visited decodes measurably better than the unbiased one.
    Those are different claims, and this is the one with a direct behavioural reading: could the
    animal do better by shifting its mix?

    **The test.** Welch's t between the 100 training repeats at each point and the 100 at the
    reference. Unpaired, because repeat *r* at two different sweep points shares no seed — each
    point draws its own neurons and trains its own network — so there is nothing to pair on. The
    across-repeat spread therefore includes the "which K neurons were drawn" variance, which is
    exactly what belongs in the denominator: the claim is about a *population of that
    composition*, averaged over which neurons realize it.

    **Holm** across the eight comparisons within each animal, not Benjamini–Hochberg, because
    the headline is a claim about whether *any* point is better — a family-wise statement — and
    Holm controls that. It is also uniformly more conservative, which is the right side to err on
    for a negative result.

    Two-sided throughout, with the direction read off the sign afterwards; testing one-sided in
    both directions would spend the error budget twice.
    """
    from scipy import stats

    g = df[df["subject"] == subject]
    ref = g.loc[np.isclose(g["sweep_fraction"], ref_x), ycol].to_numpy(float)
    if not len(ref):
        raise ValueError(f"no reference point at x = {ref_x}")
    rows = []
    for x, gg in g.groupby("sweep_fraction"):
        v = gg[ycol].to_numpy(float)
        is_ref = bool(np.isclose(x, ref_x))
        t, p = (
            (np.nan, np.nan)
            if is_ref
            else stats.ttest_ind(v, ref, equal_var=False)
        )
        rows.append(
            dict(
                x=float(x),
                rank=float(gg["mean_rank_inputs"].mean()),
                mean=float(v.mean()),
                sem=float(v.std(ddof=1) / np.sqrt(len(v))),
                n=len(v),
                delta=float(v.mean() - ref.mean()),
                t=float(t),
                p_raw=float(p),
                is_ref=is_ref,
            )
        )
    t_ = pd.DataFrame(rows).sort_values("x").reset_index(drop=True)

    m = ~t_["is_ref"]
    ps = t_.loc[m, "p_raw"].to_numpy(float)
    adj = np.ones_like(ps)
    run = 0.0
    for k, i in enumerate(np.argsort(ps)):  # Holm step-down
        run = max(run, (len(ps) - k) * ps[i])
        adj[i] = min(run, 1.0)
    t_.loc[m, "p_holm"] = adj
    t_["verdict"] = np.where(
        t_["is_ref"],
        "reference",
        np.where(
            (t_["p_holm"] < alpha) & (t_["delta"] > 0),
            "worse",
            np.where(
                (t_["p_holm"] < alpha) & (t_["delta"] < 0), "better", "n.s."
            ),
        ),
    )
    return t_


def unbiased_verdict(
    df, u=None, ref_x=UNBIASED_X, ycol="error_deg", alpha=0.05
):
    """One row per animal: how many points are worse, how many better, and the best challenger.

    ``best_challenger_delta`` is the largest *improvement* any sweep point achieves over the
    unbiased one, significant or not. A negative number that is nowhere near significance is the
    useful summary: it bounds how much there was to gain.
    """
    rows = []
    for s in sel.SUBJECTS:
        t = compare_to_unbiased(df, s, ref_x=ref_x, ycol=ycol, alpha=alpha)
        nonref = t[~t["is_ref"]]
        best = nonref.loc[nonref["delta"].idxmin()]
        rows.append(
            dict(
                subject=sel.subject_title(s),
                n_worse=int((t["verdict"] == "worse").sum()),
                n_better=int((t["verdict"] == "better").sum()),
                n_ns=int((t["verdict"] == "n.s.").sum()),
                best_challenger_x=float(best["x"]),
                best_challenger_delta=float(best["delta"]),
                best_challenger_p_holm=float(best["p_holm"]),
                ref_mean=float(t.loc[t["is_ref"], "mean"].iloc[0]),
            )
        )
    return pd.DataFrame(rows)


# ── Constants inherited from the curve/statistics layer ──────────────────────
#: The three sweep points the shuffled-null contrast is taken at, and the repeat count the
#: shuffle arms were run with.
NULL_POINTS = (0.2, 0.5, 0.8)
NULL_REPEATS = 10
#: Default columns for the curve statistics: the realized composition and the reported error.
XCOL = "mean_rho_inputs"
YCOL = "error_deg"
_HERE = HERE


# ══════════════════════════════════════════════════════════════════════════════
# Curve statistics
# ══════════════════════════════════════════════════════════════════════════════


def load_results(name, directory=None):
    d = Path(directory or (_HERE / "dataframes"))
    p = d / (name if str(name).endswith(".csv") else f"{name}.csv")
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found — run `python run_titration.py` first."
        )
    return pd.read_csv(p)


def fitted(df):
    """Only the rows a fit may use — every row, now.

    ``k_matched`` is True throughout: a sweep point is drawn at K distinct units or skipped, so
    no point on a curve is measured with fewer neurons than another. The filter is kept because
    it costs nothing and older CSVs, written when an under-sized point was run at its own size
    and flagged, are still readable through it.
    """
    return df[df["k_matched"].astype(bool)] if "k_matched" in df else df

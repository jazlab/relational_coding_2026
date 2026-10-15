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
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import ConnectionPatch

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

#: The titration fits (`regenerate.py` writes them) and the notebook's derived quantities.
_REPO = (
    Path(__file__).resolve().parents[3]
)  # repository root, where paths.py lives
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
import paths  # noqa: E402

CACHE_ROOT = paths.FIGURES_CACHE / "figure_3/efficient_coding"
DF = CACHE_ROOT / "dataframes"
#: The fit the figure plots: `dataframes/results.csv`.
FINAL = "results"
SELECTION = dict(exclusion="config", min_fit=0.0)
MEAN_RANKS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
UNBIASED_X = 0.5


def units():
    return pool.selection_units(verbose=False, **SELECTION)


def load():
    """The titration fit, `dataframes/results.csv`."""
    p = DF / f"{FINAL}.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found — run `python regenerate.py` first."
        )
    return pd.read_csv(p)


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


def lam_grid():
    """λ at every sweep point of :data:`MEAN_RANKS`."""
    return pd.DataFrame(
        {
            "x": list(MEAN_RANKS),
            "lam": [lam_for_mean_rank(v) for v in MEAN_RANKS],
        }
    )


# ── The curve, on its realized x axis ────────────────────────────────────────
def curve_axes_table(df, subject):
    """Per sweep point: the sweep variable, the realized mean rank of the draw, the tilt, and
    the error with its s.e.m.

    ``target_rank`` is the sweep variable itself. ``empirical_rank`` is the mean rank of the
    units actually drawn, as `run_titration` recorded it. ``lam`` is the tilt.
    """
    g = df[df["subject"] == subject]
    rows = []
    for x, gg in g.groupby("sweep_fraction"):
        v = gg["error_deg"].to_numpy(float)
        rows.append(
            dict(
                target_rank=float(x),
                empirical_rank=float(gg["mean_rank_inputs"].mean()),
                lam=float(gg["lam"].iloc[0]),
                mean_rho=float(gg["mean_rho_inputs"].mean()),
                y=float(v.mean()),
                y_sem=float(v.std(ddof=1) / np.sqrt(len(v))),
                n=len(v),
            )
        )
    return pd.DataFrame(rows).sort_values("target_rank").reset_index(drop=True)


# ── Caching, so the notebook is cheap to re-run ──────────────────────────────
#: Derived quantities live here. The *fitting* is already cached — it is the CSV in
#: `dataframes/` — but the notebook's own expensive step is the neuron pool (~25 s to load and
#: re-apply the selection criteria). The pool, its mean-ρ bootstrap CI and the curve table are
#: cached separately so that re-running the notebook with a **different inset seed** redraws
#: the figures and touches nothing else.
CACHE = CACHE_ROOT / "derived"
#: K -- distinct units per sweep point -- is 186 (Monkey P) / 62 (Monkey E) at every point of
#: the fit, derived by `run_titration.matched_k` rather than chosen. :func:`plot_figure` reads it
#: off the fit (the `n_units` column) rather than trusting this note.


def _cache_path(name):
    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE / name


def cached_units():
    """The pool, cached to csv. Rebuilding it re-reads the fits and re-applies the selection
    criteria, which is the single slowest thing the notebook does."""
    p = _cache_path("units.csv")
    if p.exists():
        return pd.read_csv(p)
    u = units()
    u.to_csv(p, index=False)
    return u


def cached_population(u):
    """:func:`selection_lib.population_rho` of the pool ``u``, cached."""
    p = _cache_path("population_rho.csv")
    if p.exists():
        return pd.read_csv(p)
    r = sel.population_rho(u)
    r.to_csv(p, index=False)
    return r


def cached_axes_table():
    """The curve table (:func:`curve_axes_table`) per animal, cached.

    **The cache is invalidated when the sweep CSV it summarises is newer than it.** Without that
    check, re-running the sweep would leave a stale table behind and the figure would plot the
    old sweep points while every statistic in the notebook — `unbiased_verdict` reads `load()`
    directly — comes from the new ones.
    """
    p = _cache_path(f"axes_{FINAL}.csv")
    src = DF / f"{FINAL}.csv"
    fresh = p.exists() and (
        not src.exists() or p.stat().st_mtime >= src.stat().st_mtime
    )
    if fresh:
        return pd.read_csv(p)
    d = load()
    t = pd.concat(
        [
            curve_axes_table(d, s).assign(subject=sel.subject_title(s))
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


# ── The paper figure: insets below the curve ─────────────────────────────────
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


def grey_ramp(top=0.85):
    """``Greys`` truncated at ``top``, so the darkest shading stays short of black."""
    return LinearSegmentedColormap.from_list(
        "greys_cut", plt.get_cmap("Greys")(np.linspace(0.0, float(top), 256))
    )


#: How the two animals are told apart on the shared axis -- marker shape and line style, not
#: colour.
SUBJECT_MARKERS = {"Monkey P": "o", "Monkey E": "^"}
SUBJECT_LINESTYLES = {"Monkey P": "-", "Monkey E": "--"}


class _UnderLabelConnection(ConnectionPatch):
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


def plot_figure(
    u,
    *,
    inset_xs,
    seed,
    inset_subject,
    figsize,
    xlabel,
    ylabel,
    xticks,
    inset_shade_max,
    inset_row_height,
    row_gap,
    arrow_x,
    arrow_y,
    arrow_label_y,
    arrow_label_fontsize,
    arrow_label_weight,
    arrow_label_stroke,
    arrow_length,
    arrow_head_size,
    connector_label_pad,
    connector_color,
):
    """The headline panel: **both animals on one axis**, with example draws below it.

    The x axis is the **realized mean rank of the draw** -- what the decoder actually received,
    not what the sweep asked for. The two differ at the ends because drawing K distinct units
    without replacement caps the realized mean rank; plotting the target would overstate the
    range by hiding that. The two animals realize almost the same ranks (0.14 / 0.86 at the
    ends), which is what makes one shared axis honest here. The shaded band is ±1 s.e.m. of
    the plotted mean over the 100 training repeats.

    The animals are told apart by **marker shape and line style**, not colour. The insets
    (``inset_xs``) are square boxed panels in their own gridspec row under the curve's x axis
    and x label, tied to their sweep point by a grey rule (``connector_color``) running down out
    of the curve panel to the top of each inset, except that the middle one -- which would cut
    through the x label -- starts ``connector_label_pad`` points below the label instead.
    ``row_gap`` is the layout engine's ``hspace`` between the curve and the inset row (a
    fraction of the average row height). The axis labels are on the left-most inset only.

    Each inset shows the sweep point's **sampling distribution** as a grey shading, with one
    **single draw** from it in red -- what one training run receives -- so the red dots move
    with ``seed`` and the shading does not. One animal's pool (``inset_subject``) is shown
    because the insets illustrate the sampling scheme, which is identical in both; showing both
    would double the row to say the same thing twice.

    The shading is the **tilt density itself**, and depends on the polar angle alone. A pixel
    at angle θ has ρ = (2/π)·θ; that ρ has a rank in this animal's pool; the tilt weights it by
    ``exp(λ·rank)``. Normalised as a density on rank (:func:`log_tilt_density`), so **1 means
    "as likely as under no tilt"** and the unbiased panel is *exactly* flat rather than
    accidentally so. The scale is logarithmic and **shared across panels**, spanning 1/10 to
    10, so the three panels are directly comparable: white is a region the tilt has emptied,
    dark is one it has concentrated into. ``inset_shade_max`` is the one knob for how dark the
    shading gets: it truncates the grey ramp, so 1.0 takes the densest wedge to black and lower
    values lift the whole scale toward white without changing what any shade means relative to
    the others.

    Note it shows the *weight field*, not where the neurons are — the pool's own density in
    the plane is visible as the red draw, which is the weight field times the pool.

    Deliberately absent: a fitted quadratic and the shaded CI of its vertex. They are a
    summary of the curve, not a measurement, and inviting the eye to read optimality off them
    competes with the test that actually answers it; :func:`unbiased_verdict` is that test.

    A black double-headed arrow over the curve, ``arrow_length`` long (an axes fraction) and
    centred between the ``arrow_x`` ends, at height ``arrow_y``, carries the labels
    "Retinotopic" and "Relational" above its left and right ends (at ``arrow_x``,
    ``arrow_label_y``) -- which way along the x axis the mix tilts. All positions are in the
    curve panel's axes fractions, so values above 1 sit above the plot. ``arrow_head_size``
    scales the arrowheads (matplotlib's ``mutation_scale``). ``arrow_label_stroke`` thickens
    the labels continuously, by outlining each glyph in its own colour with a stroke that many
    points wide -- Arial ships only regular and bold, so ``arrow_label_weight`` alone cannot
    give anything in between.
    """
    d = load()
    axes_t = cached_axes_table()
    rng = np.random.default_rng(seed)
    subjects = [sel.subject_title(s) for s in sel.SUBJECTS]

    fig, (ax, inset_row) = plt.subplots(
        2,
        1,
        figsize=(figsize[0], figsize[1] * (1 + inset_row_height)),
        height_ratios=[1.0, inset_row_height],
        layout="constrained",
    )
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
            color="k",
            alpha=0.25,
            lw=0,
            zorder=2,
        )
        ax.plot(
            t["empirical_rank"],
            t["y"],
            marker=SUBJECT_MARKERS[title],
            ls=SUBJECT_LINESTYLES[title],
            color="k",
            ms=8,
            mfc="0.4",
            mec="k",
            mew=0.5,
            lw=1.0,
            label=title,
            zorder=3,
        )
    ax.set_xticks(list(xticks))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # Boxed with slightly rounded corners, as in figure_1/swap_errors.ipynb. `borderpad` alone sets
    # how tight the box is -- the box style's own pad is 0, or the frame would grow outside the
    # extent the legend reports and a tight bbox would clip it.
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
    lg.get_frame().set_boxstyle("round", pad=0.0, rounding_size=0.15)
    sel._despine(ax)

    # ── the direction arrow over the curve ──
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
        arrow_x,
        ("left", "right"),
        ("Retinotopic", "Relational"),
        ("tab:blue", "tab:orange"),
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
            tx.set_path_effects(
                [
                    patheffects.withStroke(
                        linewidth=arrow_label_stroke, foreground=c
                    )
                ]
            )

    # ── the inset row ──
    # It shares the curve's x limits, so an inset placed at a sweep point sits exactly under
    # it. Drawn last, when the x limits have stopped moving.
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
    # The insets' axis limit: the 99th percentile of either coordinate, plus 5%.
    lim = float(np.percentile(np.concatenate([xr, yr]), 99)) * 1.05
    # K off the fit itself, not a constant.
    k = int(d.loc[d["subject"] == inset_subject, "n_units"].iloc[0])
    # The pool's own rho, sorted: the empirical CDF that turns a pixel's angle into a rank.
    rho_sorted = np.sort(g["rho"].to_numpy(float))
    _gx = np.linspace(0, lim, 240)
    X_px, Y_px = np.meshgrid(_gx, _gx)  # 'xy' indexing: row = y, column = x

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
            # frames deep, for a fit whose CSV looks perfectly fine.
            raise ValueError(
                f"no curve point at x = {x:g} for {sel.subject_title(inset_subject)}; "
                f'available: {sorted(t["target_rank"].round(3))}. Either `inset_xs` names a '
                f"point the fit does not have, or "
                f'{_cache_path(f"axes_{FINAL}.csv")} is stale — delete it to rebuild it from '
                f"the CSV."
            )
        emp = float(hit.iloc[0])
        frac = (emp - x0) / (x1 - x0)
        ins = inset_row.inset_axes([frac - 0.30 / 2, 0.0, 0.30, 1.0])
        ins.set_box_aspect(1.0)
        # The square is narrower than its slot is tall, and matplotlib centres the slack by
        # default -- which shows up as white space between the rows. Anchoring north puts all
        # of it below the panels instead.
        ins.set_anchor("N")
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
            cmap=grey_ramp(inset_shade_max),
            norm=LogNorm(vmin=1.0 / 10.0, vmax=10.0),
            aspect="auto",
            interpolation="bilinear",
            zorder=0,
        )
        ins.scatter(
            xr[idx],
            yr[idx],
            s=5,
            color="tab:red",
            alpha=0.5,
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
        if j == 0:  # the three share their axes
            ins.set_xlabel("Retino.", labelpad=2)
            ins.set_ylabel("Relat.", labelpad=2)
        # A rule saying which sweep point this inset is of, from the curve panel's x axis to the
        # inset's top. Axis coordinates on both ends, so the layout engine can still move
        # either row.
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
def compare_to_unbiased(df, subject):
    """Every sweep point tested against the unbiased point (x = 0.5), Holm-corrected within
    animal at α = 0.05.

    The question with a direct behavioural reading: could the animal do better by shifting its
    mix? A summary of the curve's shape (a fitted minimum, say) can sit slightly off 0.5 while
    no composition the sweep actually visited decodes measurably better than the unbiased one;
    this tests the visited compositions directly.

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

    ref_x, alpha = UNBIASED_X, 0.05
    g = df[df["subject"] == subject]
    ref = g.loc[np.isclose(g["sweep_fraction"], ref_x), "error_deg"].to_numpy(
        float
    )
    if not len(ref):
        raise ValueError(f"no reference point at x = {ref_x}")
    rows = []
    for x, gg in g.groupby("sweep_fraction"):
        v = gg["error_deg"].to_numpy(float)
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


def unbiased_verdict(df):
    """One row per animal: how many points are worse, how many better, and the best challenger.

    ``best_challenger_delta`` is the error of the best sweep point minus the unbiased one's,
    significant or not: negative when that point beats the unbiased one, positive when every
    sweep point is worse. A negative number that is nowhere near significance is the useful
    summary: it bounds how much there was to gain.
    """
    rows = []
    for s in sel.SUBJECTS:
        t = compare_to_unbiased(df, s)
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

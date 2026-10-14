"""Model-comparison figures for the 2-object-only single-neuron fits.

Everything is plotted as **Δ log-likelihood against the constant-rate null**, in nats per
trial. Raw per-trial log-likelihoods are dominated by each neuron's firing rate (a neuron
firing at 30 Hz has a much more negative ``log Γ(y+1)`` than one firing at 3 Hz), so a plot
of raw LL is a plot of firing rate. The difference against a null fit to the *same trials*
removes that, and because every model saw the same train/test splits the difference is
taken **split by split** before averaging — the CIs are paired.

Sign convention: **higher is better** everywhere; ΔLL > 0 means the model beats a constant
rate.

Every likelihood in here is on **2-object** trials, because that is all this directory
fits — the column names carry no ``_2obj`` tag for that reason. ``n_2obj`` is the only
survivor, and it is a trial *count*, not a likelihood.
"""

import data_lib as dl
import matplotlib.pyplot as plt
import models_lib as ml
import numpy as np
import pandas as pd
from matplotlib.transforms import ScaledTranslation
from scipy.stats import gaussian_kde

REFERENCE = "constant"
UNIT_KEYS = ["subject", "session", "unit"]


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _models_in(summary, models=None, drop_reference=True):
    """The models to draw, in the order to draw them.

    ``models=None`` falls back to the canonical :data:`models_lib.MODEL_ORDER`. An explicit
    **sequence** is taken in the order given — so a caller that wants a particular x-order
    just lists the models in it — while an unordered ``set`` has no order to honour and is
    sorted canonically instead.
    """
    have = set(summary["model"])
    if models is None or isinstance(models, (set, frozenset)):
        out = [
            m
            for m in ml.MODEL_ORDER
            if m in have and (models is None or m in models)
        ]
    else:
        out = [m for m in models if m in have]  # the caller's order wins
    if drop_reference:
        out = [m for m in out if m != REFERENCE]
    return out


def _subjects_in(summary):
    present = list(dict.fromkeys(summary["subject"]))
    return [s for s in dl.SUBJECT_ORDER if s in present] + [
        s for s in present if s not in dl.SUBJECT_ORDER
    ]


# ── 1. Distribution of ΔLL per model ──────────────────────────────────────────


def levels(summary, col):
    """Ordered unique values of a paneling column (subjects in the canonical order,
    everything else alphabetical). Public because the notebooks loop over it.
    """
    present = list(dict.fromkeys(summary[col].dropna()))
    if col == "subject":
        return [s for s in dl.SUBJECT_ORDER if s in present] + [
            s for s in present if s not in dl.SUBJECT_ORDER
        ]
    return sorted(present)


# ── 4. Head-to-head, neuron by neuron ─────────────────────────────────────────


# ── 5b. Population log-likelihood ─────────────────────────────────────────────


def _population_matrix(summary, models, column, n_col):
    """``(D, w)`` — a neurons × models matrix of per-trial ΔLL and a per-neuron trial-count
    weight, with the rows **aligned across models** so bootstraps can be paired.
    """
    D = summary.pivot_table(index=UNIT_KEYS, columns="model", values=column)
    missing = [m for m in models if m not in D.columns]
    if missing:
        raise KeyError(f"models not present in this summary: {missing}")
    D = D[models]
    if n_col in summary.columns:
        w = summary.drop_duplicates(subset=UNIT_KEYS).set_index(UNIT_KEYS)[
            n_col
        ]
        w = w.reindex(D.index).to_numpy(float)
    else:
        w = np.ones(len(D))
    ok = np.isfinite(D.to_numpy(float)).all(axis=1) & np.isfinite(w)
    return D.to_numpy(float)[ok], w[ok]


def _density_weights(d):
    """Per-point relative violin width in [0, 1] — the kernel density at that point over
    its maximum. Scaling each point's x-jitter by this makes the dot cloud trace the
    violin's outline instead of filling a uniform band (the KDE uses the same 'scott'
    bandwidth as ``Axes.violinplot``, so the two agree). Copied from
    ``figure_3/model_comparison/stats_lib``."""
    d = np.asarray(d, float)
    if len(d) < 3 or np.ptp(d) < 1e-12:
        return np.ones(len(d))
    try:
        with np.errstate(all="ignore"):
            dens = gaussian_kde(d)(d)
    except Exception:  # singular covariance (e.g. all-equal values)
        return np.ones(len(d))
    # A near-degenerate sample (thousands of draws whose posterior is ~0 for one model)
    # makes the KDE's effective-n blow up and return non-finite densities. Fall back to a
    # uniform band rather than propagating nan into the point positions.
    if not np.all(np.isfinite(dens)):
        return np.ones(len(d))
    mx = float(dens.max())
    return dens / mx if mx > 0 else np.ones(len(d))


def _mean_ci(d, level=0.95, n_boot=2000, rng=None):
    """Percentile-bootstrap CI of the mean of ``d`` over **neurons**. No normality
    assumption, which matters because these distributions are skewed and often bimodal.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    d = np.asarray(d, float)
    if len(d) < 2:
        return np.nan, np.nan
    means = np.mean(d[rng.integers(0, len(d), size=(n_boot, len(d)))], axis=1)
    lo, hi = np.percentile(means, [50 * (1 - level), 100 - 50 * (1 - level)])
    return float(lo), float(hi)


def shift_xticklabels(ax, dx_in=0.0, dy_in=0.0):
    """Nudge an axis's tick labels by ``(dx_in, dy_in)`` **inches**, without moving the ticks.

    Matplotlib aligns a rotated tick label by its **bounding box**, so ``xtick_ha`` decides
    which end of the labels is evenly spaced and which end is ragged. With ``ha='right'`` the
    right ends line up under the ticks and the left ends stagger by however much the names
    differ in length; with ``ha='left'`` it is the other way round — the left ends are evenly
    spaced and the block sits to the right of the ticks, which is what this shift pulls back.
    Either way the labels are correctly placed and the shift is cosmetic.

    Inches rather than data or axes units on purpose: the offset should not change when the
    figure is resized or the x-limits move, because what it corrects is a property of the text,
    not of the data.
    """
    offset = ScaledTranslation(
        float(dx_in), float(dy_in), ax.figure.dpi_scale_trans
    )
    for label in ax.get_xticklabels():
        label.set_transform(label.get_transform() + offset)
    return ax


def _violin(
    ax,
    data,
    colors,
    point_size=4,
    point_alpha=0.35,
    violin_alpha=0.2,
    jitter=0.2,
    density_jitter=True,
    mean_halfwidth=0.3,
    mean_linewidth=2.0,
    ci_level=0.95,
    ci_n_boot=2000,
    ci_linewidth=1.6,
    ci_capsize=4,
    seed=0,
    ci_bounds=None,
    max_points=None,
):
    """Violins + per-neuron points + a black mean line with its bootstrap CI — the same
    construction as ``figure_3/model_comparison/stats_lib._violin``, copied here for
    self-containedness so the two figures look like the same figure.

    ``ci_bounds`` overrides the internally computed **marginal** CI with one ``(lo, hi)``
    per violin (``None`` for a violin that should get no bar). That is what lets a caller
    draw an interval it computed itself — see :func:`plot_dataset_model_probability`,
    whose bar is a percentile range over bootstrap replicates rather than a CI of a mean.

    ``max_points`` caps how many points are **drawn** per violin (a random subset). It
    affects ink only: the violin, the mean and the CI always use every value, so a capped
    figure and an uncapped one report the same numbers.
    """
    parts = ax.violinplot(data, showmeans=False, showextrema=False)
    for body, c in zip(parts["bodies"], colors):
        body.set_facecolor(c)
        body.set_alpha(violin_alpha)
    rng = np.random.default_rng(seed)
    for i, (d, c) in enumerate(zip(data, colors)):
        d = np.asarray(d, float)
        d = d[np.isfinite(d)]
        if not len(d):
            continue
        w = _density_weights(d) if density_jitter else np.ones(len(d))
        x = (i + 1) + w * rng.uniform(-jitter, jitter, size=len(d))
        shown = slice(None)
        if max_points is not None and len(d) > max_points:
            shown = rng.choice(len(d), size=max_points, replace=False)
        ax.scatter(
            x[shown],
            d[shown],
            s=point_size,
            color=c,
            linewidths=0,
            alpha=point_alpha,
            zorder=3,
        )
        mean = float(np.mean(d))
        bounds = None if ci_bounds is None else ci_bounds[i]
        if ci_bounds is not None:
            if bounds is not None:
                # A strongly skewed sample (e.g. a posterior with atoms at 0 and 1) can put
                # the mean outside its own percentile range; widen to include it rather
                # than let matplotlib reject a negative error length.
                lo, hi = min(bounds[0], mean), max(bounds[1], mean)
                ax.errorbar(
                    i + 1,
                    mean,
                    yerr=[[mean - lo], [hi - mean]],
                    fmt="none",
                    ecolor="k",
                    elinewidth=ci_linewidth,
                    capsize=ci_capsize,
                    capthick=ci_linewidth,
                    zorder=6,
                )
        elif ci_level:
            lo, hi = _mean_ci(d, ci_level, ci_n_boot, rng)
            ax.errorbar(
                i + 1,
                mean,
                yerr=[[mean - lo], [hi - mean]],
                fmt="none",
                ecolor="k",
                elinewidth=ci_linewidth,
                capsize=ci_capsize,
                capthick=ci_linewidth,
                zorder=6,
            )
        ax.hlines(
            mean,
            i + 1 - mean_halfwidth,
            i + 1 + mean_halfwidth,
            color="k",
            lw=mean_linewidth,
            zorder=5,
        )
    return parts


def _dataset_weights(w, trials_per_neuron):
    """Trials contributed by each neuron: its own count, or a fixed number for all."""
    if trials_per_neuron is None:
        return w
    return np.full(len(w), float(trials_per_neuron))


def dataset_model_probabilities(
    summary,
    models=None,
    n_boot=2000,
    n_neurons=None,
    trials_per_neuron=None,
    column="test_ll_mean",
    n_col="n_2obj",
    seed=0,
    group_cols=("subject",),
):
    r"""``P(model | data)`` for a whole dataset, with a neuron bootstrap.

    Each replicate is a resampled **dataset**:

    1. draw ``n_neurons`` neurons **with replacement** (default: as many as were recorded);
    2. add up the **total** held-out log-likelihood of that replicate under each model —
       each neuron's per-trial held-out LL times ``trials_per_neuron`` (default: that
       neuron's own trial count);
    3. softmax over models with equal priors → one posterior for that dataset.

    Repeat ``n_boot`` times. Because a replicate is a resampled dataset and the posterior is
    a statistic of it, **the spread across replicates is the bootstrap confidence interval**
    of ``P(model | data)`` — it is not reduced by running more replicates, which only
    resolve it more finely.

    ``n_neurons`` and ``trials_per_neuron`` set how much evidence the posterior conditions
    on, and they matter because the posterior **saturates** at full size: a monkey's dataset
    is hundreds of neurons times hundreds of trials, the total log-likelihood gap between
    two models runs to thousands of nats, and ``exp(-2000)`` is zero. Every replicate then
    returns ~1 for its winner and ~0 for the rest, and the distribution collapses to two
    atoms whose mean is the fraction of resampled datasets each model wins. Dialling either
    knob down puts the posterior back in a regime where it varies continuously and the
    violin has a shape — which is the point of being able to set them.

    Returns a tidy frame with ``group_cols``, ``boot``, and one ``p_{model}`` column each.
    """
    models = _models_in(summary, models)
    group_cols = list(group_cols)
    frames = []
    for keys, sub in summary.groupby(group_cols, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        D, w = _population_matrix(sub, models, column, n_col)
        n = len(D)
        if n < 2:
            continue
        k = n if n_neurons is None else int(n_neurons)
        C = _dataset_weights(w, trials_per_neuron)[:, None] * D
        rng = np.random.default_rng(seed)
        # A with-replacement sample of k neurons is exactly a multinomial over how many
        # times each neuron is drawn — far cheaper than materialising the index array.
        counts = rng.multinomial(k, np.full(n, 1.0 / n), size=n_boot).astype(
            float
        )
        with np.errstate(
            all="ignore"
        ):  # spurious BLAS warnings; values are finite
            LL = counts @ C  # (n_boot, n_models) total held-out LL
        LL = LL - LL.max(axis=1, keepdims=True)
        P = np.exp(LL)
        P = P / P.sum(axis=1, keepdims=True)
        f = pd.DataFrame(P, columns=[f"p_{m}" for m in models])
        f["boot"] = np.arange(n_boot)
        f["n_available"] = n
        f["n_neurons"] = k
        for c, v in zip(group_cols, keys):
            f[c] = v
        frames.append(f)
    if not frames:
        raise ValueError("no group had enough neurons")
    return pd.concat(frames, ignore_index=True)


def _probability_error_bounds(data, error="mc", ci_level=0.95):
    """Per-violin ``(lo, hi)`` for :func:`plot_dataset_model_probability`'s error bar.

    ``error`` picks *what the bar measures*, which is the only way to change its length —
    the bar is as short as it is because of the quantity it reports, not how it is drawn:

    * ``'mc'``   — Monte-Carlo precision, the bootstrap CI of the mean over replicates.
      Returns ``None`` so ``_violin`` computes it itself. This is ``~sqrt(p(1-p)/n_boot)``
      and shrinks as ``n_boot`` grows, with no extra data.
    * ``'sd'``   — mean +/- 1 SD **of the replicates**: how far ``P(model | data)`` moves
      when the neuron sample changes. Roughly ``sqrt(n_boot)`` times longer than ``'mc'``,
      and unlike it does not shrink with more replicates.
    * ``'none'`` — no bar at all; the violin and its points carry the spread on their own.

    A violin with fewer than two finite values gets ``None`` (no bar) under ``'sd'``.
    """
    mode = str(error).lower()
    if mode in ("mc", "ci"):
        return None  # `_violin` draws its own CI of the mean
    if mode in ("none", "off"):
        return [None] * len(data)
    if mode != "sd":
        raise ValueError(f"error must be 'mc', 'sd' or 'none', got {error!r}")
    bounds = []
    for d in data:
        d = np.asarray(d, float)
        d = d[np.isfinite(d)]
        if len(d) < 2:
            bounds.append(None)
            continue
        m, sd = float(d.mean()), float(d.std(ddof=1))
        bounds.append((m - sd, m + sd))
    return bounds


#: A light-to-dark grey ramp for the also-rans, then one saturated colour for the model the
#: figure is about. The break is the point: a model that is merely the darkest step of a ramp
#: reads as one more step, while a model in a different hue reads as the answer. Crimson
#: because it survives being printed small and separates from every grey in the ramp.
ACCENT_COLOR = "#c1272d"
GRAY_LIGHT, GRAY_DARK = 0.82, 0.2


def plot_dataset_model_probability(
    summary,
    models=None,
    n_boot=100,
    n_neurons=30,
    trials_per_neuron=1,
    figsize=(6.2, 3.2),
    dpi=300,
    ci_level=0.95,
    error="mc",
    seed=0,
    title=None,
    ylabel="P(model | data)",
    xlabel=None,
    ylim="fit",
    ylim_pad=0.05,
    yticks=None,
    n_position=(0.02, 0.98),
    n_fontsize=12,
    n_ha=None,
    show_n=False,
    xtick_rotation=35,
    xtick_ha=None,
    xtick_dx_pt=0.0,
    colors=None,
    labels=None,
    column="test_ll_mean",
    n_col="n_2obj",
    wspace=0.17,
    point_size=4,
    point_alpha=0.40,
    violin_alpha=0.5,
    jitter=0.2,
    density_jitter=True,
    mean_halfwidth=0.3,
    mean_linewidth=1.2,
    ci_linewidth=1.2,
    ci_capsize=4,
    **violin_kw,
):
    """``P(model | data)`` as violins over bootstrap-resampled datasets, one panel per
    subject.

    **One point is one resampled dataset** — ``n_neurons`` neurons drawn with replacement,
    each contributing ``trials_per_neuron`` trials. The black line is the mean over
    replicates, ``P(model | data)``, and ``error`` chooses what the bar around it measures:

    * ``'mc'``   (default) — the bootstrap CI of that mean, at ``ci_level``. The points are
      already bootstrap replicates, so the error on their average is **Monte-Carlo
      precision**: ``~sqrt(p(1-p)/n_boot)``, tightening as ``n_boot`` grows without any
      extra data. It says how well ``n_boot`` resamples have pinned down the mean, and it
      is short for that reason — tens of times shorter than the violin beside it.
    * ``'sd'``   — mean +/- 1 SD **of the replicates**: how far the answer moves when the
      *neuron sample* changes, which is the uncertainty a reader takes the bar to mean.
      Does not shrink with ``n_boot``.
    * ``'none'`` — no bar; the violin and its points carry the spread on their own.

    Under ``'mc'`` the *spread of the violin* remains the separate thing, read off the
    points rather than the bar; ``'sd'`` puts a summary of that spread on the bar itself.

    At full dataset size expect the violins to collapse onto 0 and 1 — see
    :func:`dataset_model_probabilities` for why, and turn ``n_neurons`` or
    ``trials_per_neuron`` down to see the posterior vary continuously.

    The sample-size annotation sits top-**left** by default, because at full size the
    winning model's mean is pinned at 1.0 and a top-right label lands on top of it.
    ``show_n=False`` drops it entirely — for a paper panel whose caption carries the same
    information, where the in-axes block is redundant ink.

    ``wspace`` sets the horizontal gap between the subject panels, as a fraction of their
    average width (0 = touching). It is applied **after** ``tight_layout``, which otherwise
    picks its own value from the rotated tick labels' extents and leaves a gap wider than a
    paper panel wants; ``None`` keeps that automatic choice.

    ``xtick_dx_pt`` slides the model names horizontally, in points, positive to the right,
    without moving the ticks they belong to. It is the companion to ``xtick_rotation``: a
    rotated label is anchored at one end of its bounding box, so changing the angle moves the
    text sideways as a side effect, and this is what puts it back under its tick.

    ``xtick_rotation`` is the angle of the model names, and ``xtick_ha`` their horizontal
    anchor — left ``None`` it follows the angle ('right' for a rotated label, 'center' for a
    horizontal one), so the angle can be changed on its own.
    """
    models = _models_in(summary, models)
    P = dataset_model_probabilities(
        summary,
        models=models,
        n_boot=n_boot,
        n_neurons=n_neurons,
        trials_per_neuron=trials_per_neuron,
        column=column,
        n_col=n_col,
        seed=seed,
    )
    subjects = _subjects_in(P)
    if colors is None:
        colors = [ml.MODEL_COLORS[m] for m in models]
    fig, axes = plt.subplots(
        1, len(subjects), figsize=figsize, dpi=dpi, squeeze=False, sharey=True
    )
    axes = axes[0]
    spans = []
    for ax, subj in zip(axes, subjects):
        sub = P[P["subject"] == subj]
        data = [sub[f"p_{m}"].to_numpy(float) for m in models]
        # The styling arguments are named on this function rather than tunnelled through
        # **violin_kw, so they are visible and documented; they still have to be handed on.
        kw = dict(
            point_size=point_size,
            point_alpha=point_alpha,
            violin_alpha=violin_alpha,
            jitter=jitter,
            density_jitter=density_jitter,
            mean_halfwidth=mean_halfwidth,
            mean_linewidth=mean_linewidth,
            ci_linewidth=ci_linewidth,
            ci_capsize=ci_capsize,
            **violin_kw,
        )
        kw.setdefault("max_points", 400)
        kw.setdefault("ci_level", ci_level)
        # None = let `_violin` draw its own CI of the mean ('mc'); a list of `None`s
        # ('none') reaches it as "this violin gets no bar".
        kw["ci_bounds"] = _probability_error_bounds(data, error, ci_level)
        _violin(ax, data, list(colors), **kw)
        ax.axhline(1.0 / len(models), color="k", ls="--", lw=1.0, alpha=0.7)
        spans.append((min(v.min() for v in data), max(v.max() for v in data)))
        if show_n:
            tpn = (
                f"{trials_per_neuron} trials/neuron"
                if trials_per_neuron
                else "all trials/neuron"
            )
            n_neu = int(sub["n_neurons"].iloc[0])
            ax.text(
                *n_position,
                f'{n_neu} neuron{"" if n_neu == 1 else "s"}\n{tpn}\n'
                f"{n_boot} resamples",
                transform=ax.transAxes,
                ha=n_ha or ("left" if n_position[0] < 0.5 else "right"),
                va="top",
                fontsize=n_fontsize,
            )
        names = {**ml.MODEL_LABELS_SHORT, **(labels or {})}
        ax.set_xticks(range(1, len(models) + 1))
        # A rotated label is anchored at the end nearest its tick, so it needs 'right';
        # a horizontal one wants to be centred under it. `xtick_ha=None` picks whichever
        # the rotation calls for, so changing the angle alone never leaves the labels
        # hanging off their ticks.
        ha = xtick_ha or ("right" if xtick_rotation % 180 else "center")
        ax.set_xticklabels(
            [names[m] for m in models], rotation=xtick_rotation, ha=ha
        )
        if xtick_dx_pt:
            # Slide the labels along x without moving the ticks. Applied as an offset on each
            # label's own transform, in POINTS, so it does not rescale with the figure and a
            # value that looks right stays right at any `figsize` -- which a shift in data or
            # axes coordinates would not.
            from matplotlib.transforms import ScaledTranslation

            off = ScaledTranslation(
                xtick_dx_pt / 72.0, 0.0, fig.dpi_scale_trans
            )
            for lbl in ax.get_xticklabels():
                lbl.set_transform(lbl.get_transform() + off)
        ax.set_xlabel(xlabel)
        ax.set_title(dl.subject_title(subj))
        _despine(ax)

    # The panels share a y axis, so the limits are set once, from what ALL of them hold.
    # 'fit' is the default because the posterior only reaches 1 when one model wins every
    # resampled dataset; with several live models it lives in the bottom half of [0, 1] and
    # a fixed 0-1 axis spends most of its height on probabilities nothing occupies.
    # Pass an explicit (lo, hi) to pin it, or None to leave matplotlib's own limits alone.
    if ylim == "fit":
        lo = min(a for a, _ in spans)
        hi = max(b for _, b in spans)
        chance = 1.0 / len(models)  # the dashed line has to stay visible
        lo, hi = min(lo, chance), max(hi, chance)
        pad = ylim_pad * max(hi - lo, 1e-9)
        # A probability cannot be negative, so the bottom is clamped at 0 rather than padded
        # below it — an axis starting at -0.03 reads as a scale error.
        axes[0].set_ylim(max(0.0, lo - pad), min(1.0, hi + pad))
    elif ylim is not None:
        axes[0].set_ylim(*ylim)
    if yticks is not None:
        axes[0].set_yticks(yticks)
    axes[0].set_ylabel(ylabel)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if wspace is not None:
        fig.subplots_adjust(
            wspace=wspace
        )  # after tight_layout, which sets its own
    return fig, axes


# ── 6. Fitted parameters ──────────────────────────────────────────────────────


# ── 6b. The relational model's two amplitudes ─────────────────────────────────
# The relational model is  b + a_ret·[f(φ_A)+f(φ_B)] + a_rem·[f(r_AB)+f(r_BA)] , so its two
# amplitudes are a direct read-out of how a neuron combines the two drives: a_rem = 0 is a
# purely retinotopic neuron (the model then *is* the retinotopic model), a_ret = 0 a purely
# relational one. `future_field_index` is the scale-free version of that.
#
# Both amplitudes are constrained non-negative by the fitter, so the index is always in
# [0, 1] — but it has **atoms at both ends**: a neuron whose optimum sits on a bound gets
# exactly 0 or exactly 1, and those are a large share of the population. Every function
# here reports the atoms separately rather than letting a histogram bin hide them.
A_RET, A_REM = "p_amplitude_retinotopic", "p_amplitude_remapped"


# ── 7. Tables ─────────────────────────────────────────────────────────────────


# ── 8. The relational-bias model's alpha ──────────────────────────────────────
# The bias model's counterpart of §6's two amplitudes. Where the relational model splits
# its drive between a retinotopic and a remapped term, the bias model has a single drive
# whose *reference point* moves, and `alpha` is how far it moved: 0 = fixation
# (retinotopic), 1 = the pair's midpoint (relational). See `models_lib.bias_angles`.
#: The two ends of any relationality axis, whichever measure is on it.
RELATIONALITY_LANDMARKS = [
    (0.0, "retinotopic", "tab:blue"),
    (1.0, "future_field", "tab:orange"),
]


def plot_relationality_distribution(
    biases,
    figsize=(5.9, 2.9),
    dpi=300,
    bins=20,
    by_subject=True,
    normalize="fraction",
    alpha_max=None,
    bar_color="0.45",
    bar_edge_color="white",
    bar_edge_lw=0.4,
    landmark_color="0.15",
    landmark_lw=1.2,
    landmark_ls="--",
    landmark_labels=False,
    landmark_fontsize=12,
    landmarks=("retinotopic",),
    xlabel=r"Relationality ($\rho$)",
    xticks=(0.0, 0.5, 1.0),
    mark_units=(),
    mark_color="tab:red",
    mark_size=40,
    mark_labels=True,
    mark_label_fmt="{unit}",
    mark_fontsize=12,
    mark_rotation=0,
    mark_stem_lw=1.0,
    mark_stem_alpha=0.55,
    mark_base=0.84,
    mark_level_gap=0.13,
    mark_levels=2,
    mark_min_sep=0.12,
    mark_label_gap=0.08,
    mark_headroom=0.22,
    mark_label_ha="auto",
    show_n=True,
    title=None,
    sharey=True,
):
    """:func:`plot_relationality_hist` as a **figure panel**: grayscale, and normalized to a
    distribution rather than left as counts.

    Same quantity, same landmarks — ``alpha = 0`` *is* the retinotopic model and
    ``alpha = 1`` *is* the relational one, and both ends carry an **atom** of neurons whose
    optimum sits on a bound (see :func:`plot_relationality_hist`). What changes is only the
    reading: with ``normalize='fraction'`` each bar is the fraction of that monkey's
    neurons in its bin (the bars of a panel sum to 1), so the two populations are
    comparable however many neurons each contributes; ``'density'`` divides by the bin
    width as well (the bars integrate to 1), and ``None`` leaves raw counts.

    The color knobs take any matplotlib color, so the grayscale default is a default and
    not a constraint. ``landmarks`` picks which of :data:`RELATIONALITY_LANDMARKS` are drawn at all
    (by name or by value) — ``None`` draws both; ``('retinotopic',)`` leaves the
    ``alpha = 1`` end unmarked, which is what a figure whose right-hand landmark sits on
    the axis edge wants.

    ``mark_units`` calls out individual neurons — ``(subject, session, unit)`` triples,
    e.g. the example neurons of ``figure_4/tuning_plots/tuning_plots.ipynb`` via
    :func:`data_lib.units_from_notebook` — as a dot at that neuron's own ``alpha``, on a
    stem down to the axis, labelled with ``mark_label_fmt``. Each goes in its own subject's
    panel. A neuron that is not in ``biases`` (the selection dropped it, or it was never
    fit) cannot be placed and is named in a printed note rather than passed over in
    silence.

    The marks live in the **top** ``mark_headroom`` of each panel, which is added above the
    bars rather than taken out of them, so nothing is drawn over the distribution.
    Labels are written along their stem (``mark_rotation=90``), which is what keeps two
    neighbouring labels apart; only marks closer than ``mark_min_sep`` in ``alpha`` — close
    enough for the dots themselves to touch — are stacked over ``mark_levels`` heights. A
    landmark label with a mark within ``mark_label_gap`` of it is dropped rather than
    overprinted: the two are rotated text in the same band, the mark is what the figure is
    for, and the landmark's dashed line stays either way.
    """
    if normalize not in ("fraction", "density", None):
        raise ValueError(
            f"normalize must be 'fraction', 'density' or None "
            f"(got {normalize!r})"
        )
    subjects = _subjects_in(biases) if by_subject else [None]
    hi = 1.0 if alpha_max is None else alpha_max
    edges = np.linspace(0.0, hi, bins + 1)

    # Each called-out neuron -> (subject, alpha, label). A neuron the selection dropped has
    # no alpha to mark and is reported instead of silently vanishing.
    marks, missing = [], []
    for subject, session, unit in mark_units or ():
        row = biases[
            (biases["subject"] == subject)
            & (biases["session"] == session)
            & (biases["unit"] == unit)
        ]
        if len(row):
            marks.append(
                (
                    subject,
                    float(row["bias"].iloc[0]),
                    mark_label_fmt.format(
                        subject=subject, session=session, unit=unit
                    ),
                )
            )
        else:
            missing.append(f"{subject} {session} unit {unit}")
    if missing:
        print(
            f"not marked ({len(missing)} of {len(mark_units)} have no alpha in this "
            f"selection): " + ", ".join(missing)
        )

    fig, axes = plt.subplots(
        1,
        len(subjects),
        figsize=figsize,
        dpi=dpi,
        squeeze=False,
        sharey=sharey,
    )
    axes = axes[0]
    for ax, subj in zip(axes, subjects):
        s = biases if subj is None else biases[biases["subject"] == subj]
        v = s["bias"].to_numpy(float)
        # `weights` rather than `density` for the fraction, so a bar is read directly as
        # "this share of the neurons", independent of how many bins the axis is cut into.
        kw = (
            {"weights": np.full(len(v), 1.0 / len(v))}
            if normalize == "fraction" and len(v)
            else {"density": True} if normalize == "density" else {}
        )
        ax.hist(
            v,
            bins=edges,
            color=bar_color,
            edgecolor=bar_edge_color,
            linewidth=bar_edge_lw,
            **kw,
        )
        mark_xs = [m[1] for m in marks if subj is None or m[0] == subj]
        for x, name, _ in [
            lm
            for lm in RELATIONALITY_LANDMARKS
            if landmarks is None or lm[1] in landmarks or lm[0] in landmarks
        ]:
            ax.axvline(x, color=landmark_color, ls=landmark_ls, lw=landmark_lw)
            # A called-out neuron sitting on a landmark takes that label's place: both are
            # rotated text in the same band, and the mark is the one the figure is for.
            # The dashed line still says where the landmark is.
            crowded = any(abs(mx - x) < mark_label_gap * hi for mx in mark_xs)
            if landmark_labels and not crowded:
                # Nudged inward by a whole bin, so the label clears the bar the atom itself
                # makes — which is at the bound, is the tallest bar in the panel, and is
                # exactly where a label placed on the line would land. The opaque box is
                # the backstop for whatever the neighbouring bins do.
                inward = (hi / bins) * (1 if x <= 0.5 * hi else -1)
                ax.text(
                    x + inward,
                    0.98,
                    name,
                    transform=ax.get_xaxis_transform(),
                    rotation=90,
                    va="top",
                    ha="left" if inward > 0 else "right",
                    color=landmark_color,
                    fontsize=landmark_fontsize,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.0),
                )
        # Called-out neurons, in the headroom above the bars. `get_xaxis_transform` is x
        # in data units, y in axes fractions, so a mark's height is independent of how
        # tall the distribution happens to be.
        here = [m for m in marks if subj is None or m[0] == subj]
        level, prev_x = 0, -np.inf
        for _, x, label in sorted(here, key=lambda m: m[1]):
            level = (
                (level + 1) % mark_levels
                if x - prev_x < mark_min_sep * hi
                else 0
            )
            prev_x = x
            y = mark_base + level * mark_level_gap
            tr = ax.get_xaxis_transform()
            ax.plot(
                [x, x],
                [0, y],
                transform=tr,
                color=mark_color,
                lw=mark_stem_lw,
                alpha=mark_stem_alpha,
                zorder=5,
                clip_on=False,
            )
            ax.plot(
                [x],
                [y],
                transform=tr,
                marker="o",
                ls="none",
                color=mark_color,
                markersize=np.sqrt(mark_size),
                zorder=6,
                clip_on=False,
            )
            if mark_labels:
                # Written along the stem (``mark_rotation=90``) rather than across it: a
                # horizontal label is wider than the gap between two neighbouring biases
                # and would collide with its neighbour's, whichever height it sat at.
                # Rotated: ``rotation_mode='anchor'`` with ha='left' starts the text at
                # the dot and runs it upward, and va places it *across* the stem.
                # Horizontal: the label sits on top of the dot instead. Either way a mark
                # at a bound is pinned inward, so its label does not hang off the panel.
                if mark_rotation:
                    va = (
                        "top"
                        if x < 0.03 * hi
                        else "bottom" if x > 0.97 * hi else "center"
                    )
                    kw = dict(
                        rotation=mark_rotation,
                        ha="left",
                        va=va,
                        rotation_mode="anchor",
                    )
                else:
                    # mark_label_ha='auto' centres the label but pins a mark within 3% of a
                    # bound inward so it stays inside the panel; 'center' centres every label
                    # over its line, letting one at a bound overhang the panel edge.
                    ha = (
                        "center"
                        if mark_label_ha == "center"
                        else (
                            "left"
                            if x < 0.03 * hi
                            else "right" if x > 0.97 * hi else "center"
                        )
                    )
                    kw = dict(ha=ha, va="bottom")
                ax.text(
                    x,
                    y + 0.03,
                    label,
                    transform=tr,
                    color=mark_color,
                    fontsize=mark_fontsize,
                    zorder=7,
                    **kw,
                )

        name = dl.subject_title(subj) if subj else "All"
        ax.set_title(f"{name}  (n = {len(v)})" if show_n else name)
        ax.set_xlabel(xlabel)
        ax.set_xlim(0, hi)
        if xticks is not None:
            # labelled to one decimal, so the ends read as 0.0 and 1.0 rather than as bare
            # integers next to a 0.5
            ax.set_xticks(list(xticks))
            ax.set_xticklabels([f"{v:.1f}" for v in xticks])
        _despine(ax)
    axes[0].set_ylabel(
        {
            "fraction": "Fraction of neurons",
            "density": "Density",
            None: "Neurons",
        }[normalize]
    )
    if marks:
        # Room for the marks, made by growing the axis rather than by squeezing the bars:
        # after this the tallest bar reaches (1 - mark_headroom) of its panel. Each axis'
        # own top is read BEFORE any of them is changed — with ``sharey`` a set_ylim moves
        # every panel, so growing them one at a time would compound the expansion.
        tops = [ax.get_ylim()[1] for ax in axes]
        for ax, top in zip(axes, tops):
            ax.set_ylim(0, top / (1.0 - mark_headroom))
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def relationality_table(biases, landmark_tol=0.15):
    """Per subject: how many neurons sit at each landmark, and the median ``alpha``.

    ``landmark_tol`` is the half-width of the window counted as "at" a landmark. It is a
    display choice, not a test — the point of the table is to make the shape of the
    distribution legible next to the histogram, and a neuron 0.14 from a landmark is not
    meaningfully different from one 0.16 away.
    """
    rows = []
    for subj in _subjects_in(biases) + [None]:
        s = biases if subj is None else biases[biases["subject"] == subj]
        v = s["bias"].to_numpy(float)
        if not len(v):
            continue
        row = {
            "subject": dl.subject_title(subj) if subj else "All",
            "n": len(v),
            "median": float(np.median(v)),
        }
        for x, name, _ in RELATIONALITY_LANDMARKS:
            row[f"frac_{name}"] = float(np.mean(np.abs(v - x) <= landmark_tol))
        row["frac_at_bound"] = float(np.mean((v <= 1e-9) | (v >= 1.0 - 1e-9)))
        rows.append(row)
    return pd.DataFrame(rows).set_index("subject")


# ── 10. Session-by-session consistency ────────────────────────────────────────

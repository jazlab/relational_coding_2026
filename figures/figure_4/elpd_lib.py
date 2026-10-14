r"""Figure 4f's evidence comparison, done on a scale that has no free parameters in it.

Figure 4f shows a posterior over models built by softmaxing the summed held-out log-likelihood
of a **resampled dataset of 30 neurons contributing one trial each**, 100 times. Two analyst
choices set what a reader sees there and neither is a property of the data: the error bar is
Monte-Carlo precision, so it shrinks as the replicate count rises; and the violin's width is set
by that 30 x 1 dataset size, chosen because at full size every replicate returns 1 for its winner
and 0 for the rest. The panel therefore answers "what would a posterior look like if you had 30
neurons and one trial each", which is not the experiment that was run.

:func:`elpd_table` replaces it with the standard summary: differences in expected log predictive
density against a reference model, in **nats per trial**, with **cluster-robust** standard errors
taken across *sessions*. Simultaneously recorded neurons share trials, a stimulus sequence and
correlated receptive fields, so the session is the unit that is plausibly independent; a
neuron-level interval understates the uncertainty and a trial-level one badly so. The naive
neuron-level error is returned alongside, and their ratio reported, so the size of that
understatement is visible rather than asserted.

What goes in
------------
Per-neuron, per-model held-out log-likelihood **above the constant-rate model**, summed over that
neuron's trials:

    L[n, k] = n_2obj[n] * dtest_ll_mean[n, k]

`dtest_ll_mean` is already the paired per-split difference against a constant fit to the same
trials, averaged over the 10 splits, in nats per trial; multiplying by the trial count puts it on
the whole-neuron scale. Using a cross-validated predictive density rather than a marginal
likelihood is an approximation, and the thing that keeps the comparison free of the priors over
each model's parameters. It should be stated as such.

Nothing here refits anything: every number comes from `fit_lib.load_summary`.
"""

import numpy as np
import pandas as pd
from scipy import stats

#: Ask for this as ``reference`` to plot each model's gain over a constant-rate model, which is
#: the baseline every cached likelihood is already measured against.
CONSTANT = "constant"


def neuron_matrix(summary, models, subject=None):
    """``(L, units)`` -- the ``n_neurons x n_models`` log-evidence matrix and its index.

    ``L[n, k]`` is neuron *n*'s held-out log-likelihood above the constant-rate model under
    model *k*, summed over its own trials. Neurons missing any model are dropped, so the matrix
    is complete and every model is scored on the same neurons.
    """
    s = summary if subject is None else summary[summary["subject"] == subject]
    wide = s.pivot_table(
        index=["subject", "session", "unit"],
        columns="model",
        values="dtest_ll_mean",
        aggfunc="first",
    )
    n = s.pivot_table(
        index=["subject", "session", "unit"],
        columns="model",
        values="n_2obj",
        aggfunc="first",
    )
    models = list(models)
    wide = wide.reindex(columns=models).dropna()
    n_trials = n.reindex(wide.index)[models[0]].to_numpy(float)
    L = wide.to_numpy(float) * n_trials[:, None]
    units = wide.index.to_frame(index=False)
    units["n_2obj"] = n_trials
    return L, units


# ── elpd differences with cluster-robust standard errors ─────────────────────
def elpd_table(
    summary,
    models,
    reference=None,
    subject=None,
    n_boot=2000,
    seed=0,
    cluster="session",
    per="trial",
    ci_level=0.95,
):
    """Delta-elpd against ``reference``, per trial, with cluster-robust and naive errors.

    ``reference`` may be a model name, ``None`` (whichever model has the highest total), or
    ``'constant'``. The last is not a fourth way of saying the same thing: every likelihood in
    the cache is *already* measured against a constant-rate model, so asking for it plots each
    model's absolute predictive gain over a neuron that ignores the stimulus, rather than the
    much smaller differences between the models. Both views are worth having and they answer
    different questions -- "how much of the response does this model explain" against "which
    model explains it best".

    The point estimate is the total held-out log-likelihood difference over all neurons and all
    trials, divided by ``per``:

    * ``per='trial'`` -- divide by the total trial count, giving **nats per trial**: how much
      better each trial is predicted under one model than under the other. Small numbers, and
      the scale on which they are additive.
    * ``per='neuron'`` -- divide by the neuron count, giving **nats per neuron**: the evidence a
      whole neuron's data carries. These are large (tens of nats), so exponentiated they are
      Bayes-factor-sized and want a log axis. This is the more striking statement and the more
      trial-count-dependent one, since a neuron with twice the trials contributes twice the
      evidence.

    Three uncertainties are returned, deliberately side by side:

    * ``se_cluster`` -- the standard error across **sessions**, computed by summing the
      per-neuron differences within a session and taking the spread of those sums. This is the
      one to quote.
    * ``se_neuron`` -- the same thing across neurons, which is what a flat bootstrap gives and
      what almost every figure of this kind reports. ``neuron_ci_lo``/``neuron_ci_hi`` put a
      ``ci_level`` (0.95) Student-*t* interval on it, treating the neurons as independent
      **regardless of which session they came from**. That is the interval to use when the
      question is about the mean over neurons rather than over sessions; it is narrower than the
      cluster-robust one by ``inflation``, and says nothing about session-to-session variation.
    * ``lo``/``hi`` -- a percentile bootstrap **resampling sessions with replacement**, which
      handles the unequal number of neurons per session that the analytic form glosses over.

    ``inflation`` is ``se_cluster / se_neuron``: how much the naive interval understates things.
    A value near 1 means the clustering does not matter here; well above 1 means it does.
    """
    L, units = neuron_matrix(summary, models, subject)
    models = list(models)
    totals = L.sum(axis=0)
    ref = models[int(np.argmax(totals))] if reference is None else reference
    # 'constant' is the origin of the scale rather than a column of L.
    r = None if ref == CONSTANT else models.index(ref)
    if per not in ("trial", "neuron"):
        raise ValueError(f"per must be 'trial' or 'neuron', not {per!r}")
    denom = (
        float(units["n_2obj"].sum()) if per == "trial" else float(len(units))
    )
    keys = (
        units[cluster].to_numpy()
        if cluster in units
        else units["session"].to_numpy()
    )
    clusters = np.unique(keys)
    rng = np.random.default_rng(seed)

    rows = []
    for k, m in enumerate(models):
        d = (
            L[:, k] if r is None else L[:, k] - L[:, r]
        )  # per neuron, whole-neuron nats
        per_cluster = np.array([d[keys == c].sum() for c in clusters])
        n_c = len(clusters)
        # Var of a sum over i.i.d. clusters is n_c * var(cluster totals).
        se_cluster = (
            float(np.sqrt(n_c) * per_cluster.std(ddof=1))
            if n_c > 1
            else np.nan
        )
        se_neuron = (
            float(np.sqrt(len(d)) * d.std(ddof=1)) if len(d) > 1 else np.nan
        )
        # A t interval on the neuron mean, ignoring sessions entirely. With ~1000 neurons t is
        # within a hair of 1.96, but it costs nothing and is right for a small subject too.
        t_mult = (
            float(stats.t.ppf(0.5 * (1 + ci_level), len(d) - 1))
            if len(d) > 1
            else np.nan
        )
        boot = (
            np.array(
                [
                    per_cluster[rng.integers(0, n_c, n_c)].sum()
                    for _ in range(n_boot)
                ]
            )
            if n_c > 1
            else np.array([np.nan])
        )
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append(
            dict(
                model=m,
                is_reference=(m == ref),
                delta_elpd=float(d.sum()),
                delta_per_unit=float(d.sum() / denom),
                se_cluster_per_unit=se_cluster / denom,
                se_neuron_per_unit=se_neuron / denom,
                neuron_ci_lo_per_unit=float(
                    (d.sum() - t_mult * se_neuron) / denom
                ),
                neuron_ci_hi_per_unit=float(
                    (d.sum() + t_mult * se_neuron) / denom
                ),
                lo_per_unit=float(lo / denom),
                hi_per_unit=float(hi / denom),
                inflation=se_cluster / se_neuron if se_neuron else np.nan,
                n_neurons=len(d),
                n_clusters=n_c,
            )
        )
    out = pd.DataFrame(rows)
    out.attrs["reference"] = ref
    out.attrs["per"] = per
    return out


def per_neuron_delta(summary, models, reference, subject=None):
    """``(D, units, constant_offset)`` for the violins.

    ``D[n, k]`` is neuron *n*'s held-out log-likelihood difference **per trial** between model
    *k* and the reference -- the same units as the population estimate, so violin and point
    live on one axis. ``constant_offset`` is where the constant-rate model sits on that axis:
    minus the reference's own trial-weighted advantage over a constant rate, since every
    ``dtest_ll_mean`` is already measured against that baseline.
    """
    L, units = neuron_matrix(summary, models, subject)
    n = units["n_2obj"].to_numpy(float)
    per_trial = L / n[:, None]  # nats per trial, vs constant
    if reference == CONSTANT:
        # The axis already has the constant model at 0, so there is no separate line to draw.
        return per_trial, units, 0.0
    r = list(models).index(reference)
    trial_weighted = L.sum(axis=0) / n.sum()
    return per_trial - per_trial[:, [r]], units, -float(trial_weighted[r])


# ── Figure ────────────────────────────────────────────────────────────────────
def plot_elpd(
    tables,
    colors,
    figsize=(5.8, 3.3),
    ylabel=r"$\Delta$elpd (nats per trial)",
    which="cluster",
    labels=None,
    wspace=0.17,
    xtick_rotation=-50,
    xtick_ha="left",
    xtick_dx_pt=0.0,
    shift_in=-0.2,
    ylim_pad=0.08,
    point_size=7,
    err_linewidth=1.2,
    err_capsize=4,
    n_by_subject=None,
    per_neuron=None,
    constants=None,
    violin_alpha=0.5,
    dot_size=4,
    dot_alpha=0.40,
    jitter=0.2,
    density_jitter=True,
    qlim=(1, 99),
    constant_label="constant rate",
    seed=0,
    summary_style="point",
    show_dots=False,
    bar_width=0.6,
    bar_lw=0.5,
    column_shift_in=0.0,
    yscale="ratio",
    logy=False,
):
    """Delta-elpd against the reference model, drawn in `figure_model_comparison`'s style.

    One panel per animal, models along the x axis in the figure's order and colours, names
    rotated and left-anchored exactly as there, so the two panels can be read side by side
    without discounting a difference in how they are drawn. The reference model sits at 0 by
    construction; the dashed line marks it.

    Unlike that figure the panels do **not** share a y axis: the two animals differ several-fold
    in the size of the differences, and a shared scale makes the smaller one unreadable to make
    a point the numbers already make.

    ``which`` selects the interval: ``'neuron_ci'`` for a 95% confidence interval of the mean
    **across neurons**, computed with the neurons treated as independent regardless of session;
    ``'cluster'`` for +/-1 cluster-robust standard error across sessions; ``'boot'`` for the
    session bootstrap percentile interval; or ``'neuron'`` for +/-1 naive neuron-level standard
    error. The last two neuron-level options ignore that simultaneously recorded neurons share
    trials, a stimulus sequence and correlated receptive fields, so they answer "how much do
    neurons differ" and not "would another session look like this"; `elpd_table` reports
    ``inflation``, the factor by which the cluster-robust interval is wider, so the size of that
    difference stays visible.

    ``per_neuron`` adds the **distribution across neurons** as a violin with one point per
    neuron, from :func:`per_neuron_delta`. Read the two layers as the different things they
    are: the violin is how much neurons differ from one another, the black point and its bar
    are the population estimate and its uncertainty. The bar is emphatically *not* the error on
    the violin's spread, and at ~1000 neurons it is far narrower than that spread -- which is
    the confusion the panel this replaces invited.

    ``constants`` draws a line at the **constant-rate model**, one value per panel from
    :func:`per_neuron_delta`. It is the origin of the whole scale: every likelihood here is
    already measured against a neuron that ignores the stimulus, so this line says how much
    stimulus-driven predictability there is in total, and therefore what fraction of it the
    models are arguing over. Without it a reader has no way to tell whether the gaps between
    models are large or small.

    ``summary_style`` draws the population estimate as a **bar** from the reference line
    (``'bar'``, the default) or as a marker (``'point'``). Bars read the difference as a length
    against the reference, which is what this axis is; a marker reads it as a position, which is
    the right choice only when the reference is somewhere other than zero.

    ``show_dots`` overlays one dot per neuron on the violins. Off by default, because with bars
    the panel already carries a filled shape per model and the dot cloud competes with it.

    ``qlim`` clips the violins to a percentile range so a handful of extreme neurons do not set
    the axis; the summary, its interval and the constant line are never clipped.

    ``logy`` puts the y axis itself on a log scale, which per-neuron likelihood ratios need:
    they span tens of decades, so a linear axis shows only the reference and a row of points on
    the floor.

    ``yscale='ratio'`` plots the **likelihood** rather than the log-likelihood: every value is
    exponentiated, so the reference sits at 1 and a model at 0.91 makes each held-out trial 9%
    less likely than the reference does. Two things are worth knowing about that transform.
    Exponentiating is monotone, so the interval endpoints map exactly -- it is the same interval
    on a multiplicative scale, and the asymmetry it acquires is real rather than an artefact.
    And ``exp`` of a mean log-likelihood is a **geometric** mean of per-trial likelihood ratios,
    which is the right summary here (the whole dataset's likelihood ratio is ``exp`` of the
    summed log ratio) but is not the arithmetic mean of the ratios, and should not be described
    as one.

    ``wspace`` is constrained layout's *minimum* gap between panels, and here it bottoms out
    well short of zero -- the right panel carries its own y tick labels, since the animals do
    not share a scale, and those set the floor. Measured: 0.17 -> 0 closes the gap only from
    0.213 to 0.139 of the figure width. ``column_shift_in`` slides column *c* left by
    ``c * column_shift_in`` INCHES after the layout has settled, which is the only way past
    that floor; it is the same knob, with the same name and meaning, as
    ``figure_3/model_comparison/comparison_lib.plot_model_comparison``.
    """
    import models_lib as ml
    import plot_lib as pl
    from matplotlib import pyplot as plt

    if yscale not in ("log", "ratio"):
        raise ValueError(f"yscale must be 'log' or 'ratio', not {yscale!r}")
    tf = (
        (lambda v: np.exp(np.asarray(v, float)))
        if yscale == "ratio"
        else (lambda v: v)
    )
    ref_level = 1.0 if yscale == "ratio" else 0.0
    names = {**getattr(ml, "MODEL_LABELS_SHORT", {}), **(labels or {})}
    fig, axes = plt.subplots(
        1,
        len(tables),
        figsize=figsize,
        squeeze=False,
        sharey=False,
        layout="constrained",
    )
    fig.get_layout_engine().set(wspace=wspace)
    for ax, (title, t) in zip(axes[0], tables.items()):
        models = list(t["model"])
        x = np.arange(1, len(models) + 1)
        y = t["delta_per_unit"].to_numpy()
        if which == "boot":
            lo, hi = t["lo_per_unit"].to_numpy(), t["hi_per_unit"].to_numpy()
        elif which == "neuron_ci":
            lo = t["neuron_ci_lo_per_unit"].to_numpy()
            hi = t["neuron_ci_hi_per_unit"].to_numpy()
        else:
            e = t[f"se_{which}_per_unit"].to_numpy()
            lo, hi = y - e, y + e
        y, lo, hi = (
            tf(y),
            tf(lo),
            tf(hi),
        )  # monotone, so the endpoints map exactly
        if per_neuron is not None and title in per_neuron:
            D = per_neuron[title]
            keep = [np.isfinite(D[:, k]) for k in range(len(models))]
            vals = [tf(D[keep[k], k]) for k in range(len(models))]
            parts = ax.violinplot(
                vals, positions=x, showmeans=False, showextrema=False
            )
            for body, m in zip(parts["bodies"], models):
                body.set_facecolor(colors[m])
                body.set_alpha(violin_alpha)
            rng = np.random.default_rng(seed)
            for xi, v, m in zip(x, vals, models if show_dots else []):
                if density_jitter and len(v) > 2:
                    try:
                        from scipy.stats import gaussian_kde

                        w = gaussian_kde(v)(v)
                        w = w / w.max()
                    except Exception:
                        w = np.ones(len(v))
                else:
                    w = np.ones(len(v))
                ax.scatter(
                    xi + w * rng.uniform(-jitter, jitter, len(v)),
                    v,
                    s=dot_size,
                    color=colors[m],
                    linewidths=0,
                    alpha=dot_alpha,
                    zorder=2,
                )

        ax.axhline(ref_level, color="k", ls="--", lw=1.0, alpha=0.7, zorder=1)
        c = None if constants is None else constants.get(title)
        if (
            c is not None and abs(c) > 1e-12
        ):  # at the reference it would coincide
            c = float(tf(c))
            ax.axhline(c, color="0.35", ls=":", lw=1.2, zorder=1)
            # x in AXES fraction, y in DATA units -- that is what get_yaxis_transform gives,
            # and passing a data-space x here is what pushed the label off the figure.
            ax.text(
                0.98,
                c,
                constant_label,
                va="bottom",
                ha="right",
                fontsize=9,
                color="0.35",
                transform=ax.get_yaxis_transform(),
            )
        if summary_style == "bar":
            # Bars grow from the reference, which is 1 on a likelihood axis, not 0.
            ax.bar(
                x,
                y - ref_level,
                bottom=ref_level,
                width=bar_width,
                color=[colors[m] for m in models],
                edgecolor="k",
                linewidth=bar_lw,
                zorder=4,
            )
        elif summary_style != "point":
            raise ValueError(
                f"summary_style must be 'bar' or 'point', not {summary_style!r}"
            )
        for xi, yi, l, h, m in zip(x, y, lo, hi, models):
            ax.errorbar(
                xi,
                yi,
                yerr=[[yi - l], [h - yi]],
                fmt="none",
                ecolor="k",
                elinewidth=err_linewidth,
                capsize=err_capsize,
                capthick=err_linewidth,
                zorder=5,
            )
            if summary_style == "point":
                ax.plot(
                    [xi],
                    [yi],
                    marker="o",
                    ms=point_size,
                    color=colors[m],
                    mec="k",
                    mew=0.8,
                    zorder=6,
                )
        if logy:
            ax.set_yscale("log")
        bot, top = min(lo.min(), ref_level), max(hi.max(), ref_level)
        if per_neuron is not None and title in per_neuron:
            D = tf(per_neuron[title])
            bot = min(bot, float(np.nanpercentile(D, qlim[0])))
            top = max(top, float(np.nanpercentile(D, qlim[1])))
        if c is not None:
            bot = min(bot, c)
        if logy:
            lo_pos = min(v for v in (bot, top) if v > 0)
            ax.set_ylim(
                lo_pos * 10 ** (-ylim_pad * 8), top * 10 ** (ylim_pad * 2)
            )
        else:
            span = max(top - bot, 1e-12)
            ax.set_ylim(bot - ylim_pad * span, top + ylim_pad * span)
        ax.set_xlim(0.4, len(models) + 0.6)
        ax.set_xticks(x)
        ha = xtick_ha or ("right" if xtick_rotation % 180 else "center")
        ax.set_xticklabels(
            [names.get(m, m) for m in models], rotation=xtick_rotation, ha=ha
        )
        if xtick_dx_pt:
            from matplotlib.transforms import ScaledTranslation

            off = ScaledTranslation(
                xtick_dx_pt / 72.0, 0.0, fig.dpi_scale_trans
            )
            for lbl in ax.get_xticklabels():
                lbl.set_transform(lbl.get_transform() + off)
        n = (n_by_subject or {}).get(title)
        ax.set_title(f"{title}  (n = {n})" if n is not None else title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if shift_in:
            pl.shift_xticklabels(ax, shift_in)
    axes[0][0].set_ylabel(ylabel)
    if column_shift_in:
        fig.canvas.draw()  # let constrained layout finish before overriding it
        fig.set_layout_engine("none")
        dx = float(column_shift_in) / fig.get_size_inches()[0]
        for ci, ax in enumerate(axes[0]):
            box = ax.get_position()
            ax.set_position([box.x0 - ci * dx, box.y0, box.width, box.height])
    return fig

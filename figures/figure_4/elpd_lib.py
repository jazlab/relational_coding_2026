r"""Figure 4's model comparison: held-out evidence on a scale with no free parameters in it.

:func:`elpd_table` is the standard summary: differences in expected log predictive
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

#: The model every other model is compared against.
REFERENCE = "future_field"


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
def elpd_table(summary, models, subject=None, n_boot=2000):
    """Delta-elpd against :data:`REFERENCE`, per trial, with cluster-robust and naive errors.

    The point estimate is the total held-out log-likelihood difference over all neurons and all
    trials, divided by the total trial count: **nats per trial**, how much better each trial
    is predicted under one model than under the other.

    Three uncertainties are returned, deliberately side by side:

    * ``se_cluster`` -- the standard error across **sessions**, computed by summing the
      per-neuron differences within a session and taking the spread of those sums. This is the
      one to quote.
    * ``se_neuron`` -- the same thing across neurons, which is what a flat bootstrap gives and
      what almost every figure of this kind reports. ``neuron_ci_lo``/``neuron_ci_hi`` put a
      95% Student-*t* interval on it, treating the neurons as independent
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
    ref = REFERENCE
    r = models.index(ref)
    denom = float(units["n_2obj"].sum())
    keys = units["session"].to_numpy()
    clusters = np.unique(keys)
    rng = np.random.default_rng(0)

    rows = []
    for k, m in enumerate(models):
        d = L[:, k] - L[:, r]  # per neuron, whole-neuron nats
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
            float(stats.t.ppf(0.5 * (1 + 0.95), len(d) - 1))
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
    return out


def per_neuron_delta(summary, models, reference, subject=None):
    """``(D, units, constant_offset)``.

    ``D[n, k]`` is neuron *n*'s held-out log-likelihood difference **per trial** between model
    *k* and the reference -- the same units as the population estimate. ``constant_offset`` is where the constant-rate model sits on that axis:
    minus the reference's own trial-weighted advantage over a constant rate, since every
    ``dtest_ll_mean`` is already measured against that baseline.
    """
    L, units = neuron_matrix(summary, models, subject)
    n = units["n_2obj"].to_numpy(float)
    per_trial = L / n[:, None]  # nats per trial, vs constant
    r = list(models).index(reference)
    trial_weighted = L.sum(axis=0) / n.sum()
    return per_trial - per_trial[:, [r]], units, -float(trial_weighted[r])


# ── Figure ────────────────────────────────────────────────────────────────────
def plot_elpd(
    tables,
    colors,
    figsize=(5.8, 3.3),
    ylabel=r"$\Delta$elpd (nats per trial)",
    labels=None,
    wspace=0.17,
    xtick_rotation=-50,
    xtick_ha="left",
    shift_in=-0.2,
    point_size=7,
    err_linewidth=1.2,
    err_capsize=4,
    n_by_subject=None,
    column_shift_in=0.0,
):
    """Delta-elpd against the reference model, drawn in `figure_model_comparison`'s style.

    One panel per animal, models along the x axis in the figure's order and colours, names
    rotated and left-anchored exactly as there, so the two panels can be read side by side
    without discounting a difference in how they are drawn. The reference model sits at 1 by
    construction; the dashed line marks it.

    Unlike that figure the panels do **not** share a y axis: the two animals differ several-fold
    in the size of the differences, and a shared scale makes the smaller one unreadable to make
    a point the numbers already make.

    The interval is a 95% confidence interval of the mean **across neurons**, computed with the
    neurons treated as independent regardless of session. That ignores that simultaneously
    recorded neurons share trials, a stimulus sequence and correlated receptive fields, so it
    answers "how much do neurons differ" and not "would another session look like this";
    `elpd_table` reports ``inflation``, the factor by which the cluster-robust interval is wider,
    so the size of that difference stays visible.

    The population estimate is drawn as a marker with its interval.

    The y axis is the **likelihood** rather than the log-likelihood: every value is
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
    that floor.
    """
    import models_lib as ml
    import plot_lib as pl
    from matplotlib import pyplot as plt

    ref_level = 1.0
    names = {**ml.MODEL_LABELS_SHORT, **(labels or {})}
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
        lo = t["neuron_ci_lo_per_unit"].to_numpy()
        hi = t["neuron_ci_hi_per_unit"].to_numpy()
        # exp is monotone, so the endpoints map exactly
        y, lo, hi = np.exp(y), np.exp(lo), np.exp(hi)

        ax.axhline(ref_level, color="k", ls="--", lw=1.0, alpha=0.7, zorder=1)
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
        bot, top = min(lo.min(), ref_level), max(hi.max(), ref_level)
        span = max(top - bot, 1e-12)
        ax.set_ylim(bot - 0.08 * span, top + 0.08 * span)
        ax.set_xlim(0.4, len(models) + 0.6)
        ax.set_xticks(x)
        ha = xtick_ha or ("right" if xtick_rotation % 180 else "center")
        ax.set_xticklabels(
            [names.get(m, m) for m in models], rotation=xtick_rotation, ha=ha
        )
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

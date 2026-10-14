r"""A schematic of the cued-response model drawn by `synthetic.ipynb`.

The figure is the pipeline of `task_lib.CueMatrixModel`, in the order signals actually flow:

    stimulus ─► fixed population ──────────────► neural rate ─┐
             └► cue features ─► cue MLP ─► Poisson ─► W ≥ 0 ──┴─(+)─► ReLU ─► Poisson ─► readout ─► bump

Two things the drawing is built to make visible, because they are the whole point of the
model and neither is legible from the code:

* **What is trained and what is not.** The population is fixed — it says *where* the two
  objects are and, being symmetric under swapping them, nothing about *which is first*. Only
  the cue MLP, `W` and the readout learn. Fixed stages are drawn in grey, trainable ones in
  `TRAIN_COLOR`, and the legend says so.
* **Where the noise enters.** There are two independent Poisson draws — one inside the cue
  channel, one on the combined population state — and the cue's is the bottleneck the whole
  experiment is about. They are drawn as a distinct shape rather than as one more box.

Nothing here reads the model object: the boxes are labelled from the numbers the notebook
passes in, so the schematic states the configuration that was actually run rather than a
generic picture.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIXED_COLOR = "0.88"  # stages that never learn
TRAIN_COLOR = "#cfe3f5"  # the three trainable stages
NOISE_COLOR = "#f6dcc8"  # the two Poisson draws
EDGE_COLOR = "0.35"


def _box(ax, x, y, w, h, text, face, fontsize, lw=0.9, radius=0.02):
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=face,
            edgecolor=EDGE_COLOR,
            lw=lw,
            zorder=2,
        )
    )
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        zorder=3,
        linespacing=1.35,
    )


def _arrow(
    ax, xy_from, xy_to, lw=0.9, color=EDGE_COLOR, connectionstyle="arc3,rad=0"
):
    ax.add_patch(
        FancyArrowPatch(
            xy_from,
            xy_to,
            arrowstyle="-|>",
            mutation_scale=8,
            lw=lw,
            color=color,
            shrinkA=1.5,
            shrinkB=1.5,
            connectionstyle=connectionstyle,
            zorder=1,
        )
    )


def model_schematic(
    n_retinotopic,
    n_relational,
    gain,
    cue_output_neurons,
    cue_max_rate,
    cue_influence,
    cue_influence_kind,
    n_angles,
    figsize=(12.0, 4.6),
    dpi=300,
    fontsize=12,
    label_fontsize=12,
    legend=True,
    box_h=0.17,
    lw=0.9,
):
    """Draw the model, labelled with the configuration it was run at.

    The arguments are the notebook's CONFIG values, so the boxes carry the real neuron
    counts, cue capacity and weight budget rather than symbols.

    The defaults follow `figures/FORMATTING.md`: no text below 12 pt, which is what sets
    the default `figsize` — the boxes are laid out in axes fractions, so the figure has to be
    wide enough for 12 pt to fit inside them rather than the text being shrunk to fit.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    y_pop, y_cue, y_mid = 0.76, 0.28, 0.52
    stat = {"total_weight": "Σ", "mean_weight": "mean", "max_weight": "max"}[
        cue_influence_kind
    ]

    # ── the stimulus, and the two channels it feeds ──────────────────────────
    _box(
        ax,
        0.075,
        y_mid,
        0.13,
        box_h,
        "stimulus\n" r"$\theta_0,\ \theta_1$",
        FIXED_COLOR,
        fontsize,
    )

    _box(
        ax,
        0.30,
        y_pop,
        0.24,
        box_h,
        f"fixed population\n{n_retinotopic} retinotopic + {n_relational} relational\n"
        f"(gain {gain:g})",
        FIXED_COLOR,
        fontsize,
    )
    _box(
        ax,
        0.27,
        y_cue,
        0.15,
        box_h,
        "cue features\n" r"$\cos\theta,\ \sin\theta$",
        FIXED_COLOR,
        fontsize,
    )
    _box(
        ax,
        0.45,
        y_cue,
        0.13,
        box_h,
        "cue MLP\n" r"$\rightarrow$ rate",
        TRAIN_COLOR,
        fontsize,
    )
    _box(
        ax,
        0.60,
        y_cue,
        0.12,
        box_h,
        f"Poisson\n{cue_output_neurons} units\n" f"rate ≤ {cue_max_rate:g}",
        NOISE_COLOR,
        fontsize,
    )
    _box(
        ax,
        0.745,
        y_cue,
        0.12,
        box_h,
        r"$W \geq 0$" + f"\n{stat} = {cue_influence:g}",
        TRAIN_COLOR,
        fontsize,
    )

    _arrow(ax, (0.14, y_mid + 0.04), (0.18, y_pop - 0.06))
    _arrow(ax, (0.14, y_mid - 0.04), (0.195, y_cue + 0.06))
    _arrow(ax, (0.345, y_cue), (0.385, y_cue))
    _arrow(ax, (0.515, y_cue), (0.54, y_cue))
    _arrow(ax, (0.66, y_cue), (0.685, y_cue))

    # ── the two streams meet ─────────────────────────────────────────────────
    x_sum = 0.845
    # a scatter marker rather than a Circle patch: the axes is not square, so a patch of
    # radius r in data units draws as an ellipse
    ax.scatter(
        [x_sum],
        [y_mid],
        s=110,
        marker="o",
        facecolors="white",
        edgecolors=EDGE_COLOR,
        linewidths=lw,
        zorder=3,
    )
    ax.text(
        x_sum,
        y_mid,
        "+",
        ha="center",
        va="center",
        fontsize=fontsize + 2,
        zorder=4,
    )
    _arrow(
        ax,
        (0.42, y_pop),
        (x_sum, y_mid + 0.03),
        connectionstyle="angle3,angleA=0,angleB=70",
    )
    _arrow(
        ax,
        (0.805, y_cue),
        (x_sum, y_mid - 0.03),
        connectionstyle="angle3,angleA=0,angleB=-70",
    )
    ax.text(
        0.62,
        y_pop + 0.015,
        "neural rate",
        ha="center",
        va="bottom",
        fontsize=label_fontsize,
        color="0.35",
    )
    ax.text(
        0.815,
        y_cue - 0.115,
        r"$\Delta = W\,\mathrm{spikes}$",
        ha="center",
        va="top",
        fontsize=label_fontsize,
        color="0.35",
    )

    # ── and the readout ──────────────────────────────────────────────────────
    _box(ax, 0.925, y_mid, 0.10, box_h, "ReLU\nPoisson", NOISE_COLOR, fontsize)
    _box(ax, 0.925, 0.115, 0.10, box_h, "readout\nMLP", TRAIN_COLOR, fontsize)
    _arrow(ax, (x_sum + 0.025, y_mid), (0.873, y_mid))
    _arrow(ax, (0.925, y_mid - box_h / 2), (0.925, 0.115 + box_h / 2))
    ax.text(
        0.925,
        0.115 - box_h / 2 - 0.03,
        f"bump over {n_angles} angles"
        + "\n"
        + r"$\rightarrow\ \hat{\theta}_0$",
        ha="center",
        va="top",
        fontsize=label_fontsize,
        color="0.35",
    )

    if legend:
        handles = [
            FancyBboxPatch(
                (0, 0),
                1,
                1,
                boxstyle="round,pad=0,rounding_size=0.1",
                facecolor=c,
                edgecolor=EDGE_COLOR,
                lw=lw,
            )
            for c in (FIXED_COLOR, TRAIN_COLOR, NOISE_COLOR)
        ]
        ax.legend(
            handles,
            ["fixed", "trained", "Poisson noise"],
            loc="upper left",
            bbox_to_anchor=(0.0, 1.02),
            frameon=False,
            fontsize=label_fontsize,
            handlelength=1.1,
            handleheight=0.9,
            labelspacing=0.35,
            borderpad=0.0,
        )
    fig.tight_layout(pad=0.2)
    return fig, ax

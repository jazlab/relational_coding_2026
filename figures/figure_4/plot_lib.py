"""Figure 4's relationality histogram and its table, plus small shared plotting helpers.

:func:`plot_relationality_distribution` draws the distribution of the relational-bias model's
fitted ``bias`` (figure 4's relationality, see ``relationality_lib``) per monkey, and
:func:`relationality_table` summarises it. :func:`levels` and :func:`shift_xticklabels` are
used by ``ks_lib`` and ``elpd_lib``.
"""

import data_lib as dl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import ScaledTranslation


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _subjects_in(summary):
    present = list(dict.fromkeys(summary["subject"]))
    return [s for s in dl.SUBJECT_ORDER if s in present] + [
        s for s in present if s not in dl.SUBJECT_ORDER
    ]


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


def shift_xticklabels(ax, dx_in=0.0):
    """Nudge an axis's tick labels by ``dx_in`` **inches**, without moving the ticks.

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
    offset = ScaledTranslation(float(dx_in), 0.0, ax.figure.dpi_scale_trans)
    for label in ax.get_xticklabels():
        label.set_transform(label.get_transform() + offset)
    return ax


# ── The relational-bias model's bias ──────────────────────────────────────────
# `bias` is how far the reference point the neuron measures direction from has moved: 0 =
# fixation (retinotopic), 1 = the pair's midpoint (relational). See `models_lib.bias_angles`.
#: The two ends of any relationality axis, whichever measure is on it.
RELATIONALITY_LANDMARKS = [
    (0.0, "retinotopic", "tab:blue"),
    (1.0, "future_field", "tab:orange"),
]


def plot_relationality_distribution(
    biases,
    figsize=(5.9, 2.9),
    bins=20,
    bar_color="0.45",
    bar_edge_color="white",
    bar_edge_lw=0.4,
    landmark_color="0.15",
    landmark_lw=1.2,
    landmark_ls="--",
    xlabel=r"Relationality ($\rho$)",
    mark_units=(),
    mark_color="tab:red",
    mark_size=40,
    mark_label_fmt="{unit}",
    mark_fontsize=12,
    mark_stem_lw=1.0,
    mark_stem_alpha=0.55,
    mark_base=0.84,
    mark_level_gap=0.13,
    mark_levels=2,
    mark_min_sep=0.12,
    mark_headroom=0.22,
    mark_label_ha="auto",
):
    """The distribution of the relational-bias ``bias`` as a **figure panel**: one panel per
    monkey, grayscale, each bar the fraction of that monkey's neurons in its bin (the bars of
    a panel sum to 1), so the two populations are comparable however many neurons each
    contributes. The panels share a y axis.

    ``bias = 0`` *is* the retinotopic model and ``bias = 1`` the relational one, and both ends
    carry an **atom** of neurons whose optimum sits on a bound. Only the ``bias = 0`` landmark
    of :data:`RELATIONALITY_LANDMARKS` is drawn, as a dashed line: the other end sits on the
    axis edge.

    ``mark_units`` calls out individual neurons — ``(subject, session, unit)`` triples,
    e.g. the example neurons of ``figure_example_neurons.ipynb`` via
    :func:`data_lib.units_from_notebook` — as a dot at that neuron's own ``bias``, on a
    stem down to the axis, labelled with ``mark_label_fmt``. Each goes in its own subject's
    panel. A neuron that is not in ``biases`` (the selection dropped it, or it was never
    fit) cannot be placed and is named in a printed note rather than passed over in
    silence.

    The marks live in the **top** ``mark_headroom`` of each panel, which is added above the
    bars rather than taken out of them, so nothing is drawn over the distribution. Marks
    closer than ``mark_min_sep`` in ``bias`` — close enough for the dots themselves to touch —
    are stacked over ``mark_levels`` heights.
    """
    subjects = _subjects_in(biases)
    hi = 1.0
    edges = np.linspace(0.0, hi, bins + 1)

    # Each called-out neuron -> (subject, bias, label). A neuron the selection dropped has
    # no bias to mark and is reported instead of silently vanishing.
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
        dpi=300,
        squeeze=False,
        sharey=True,
    )
    axes = axes[0]
    for ax, subj in zip(axes, subjects):
        s = biases[biases["subject"] == subj]
        v = s["bias"].to_numpy(float)
        # `weights` rather than `density`, so a bar is read directly as "this share of the
        # neurons", independent of how many bins the axis is cut into.
        kw = {"weights": np.full(len(v), 1.0 / len(v))} if len(v) else {}
        ax.hist(
            v,
            bins=edges,
            color=bar_color,
            edgecolor=bar_edge_color,
            linewidth=bar_edge_lw,
            **kw,
        )
        for x, name, _ in RELATIONALITY_LANDMARKS:
            if name == "retinotopic":
                ax.axvline(
                    x, color=landmark_color, ls=landmark_ls, lw=landmark_lw
                )
        # Called-out neurons, in the headroom above the bars. `get_xaxis_transform` is x
        # in data units, y in axes fractions, so a mark's height is independent of how
        # tall the distribution happens to be.
        here = [m for m in marks if m[0] == subj]
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
            # The label sits on top of the dot. mark_label_ha='auto' centres it but pins a
            # mark within 3% of a bound inward so it stays inside the panel; 'center' centres
            # every label over its line, letting one at a bound overhang the panel edge.
            ha = (
                "center"
                if mark_label_ha == "center"
                else (
                    "left"
                    if x < 0.03 * hi
                    else "right" if x > 0.97 * hi else "center"
                )
            )
            ax.text(
                x,
                y + 0.03,
                label,
                transform=tr,
                color=mark_color,
                fontsize=mark_fontsize,
                zorder=7,
                ha=ha,
                va="bottom",
            )

        name = dl.subject_title(subj) if subj else "All"
        ax.set_title(f"{name}  (n = {len(v)})")
        ax.set_xlabel(xlabel)
        ax.set_xlim(0, hi)
        # labelled to one decimal, so the ends read as 0.0 and 1.0 rather than as bare
        # integers next to a 0.5
        xticks = (0.0, 0.5, 1.0)
        ax.set_xticks(list(xticks))
        ax.set_xticklabels([f"{v:.1f}" for v in xticks])
        _despine(ax)
    axes[0].set_ylabel("Fraction of neurons")
    if marks:
        # Room for the marks, made by growing the axis rather than by squeezing the bars:
        # after this the tallest bar reaches (1 - mark_headroom) of its panel. Each axis'
        # own top is read BEFORE any of them is changed — with a shared y axis a set_ylim
        # moves every panel, so growing them one at a time would compound the expansion.
        tops = [ax.get_ylim()[1] for ax in axes]
        for ax, top in zip(axes, tops):
            ax.set_ylim(0, top / (1.0 - mark_headroom))
    fig.tight_layout()
    return fig, axes


def relationality_table(biases, landmark_tol=0.15):
    """Per subject: how many neurons sit at each landmark, and the median ``bias``.

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

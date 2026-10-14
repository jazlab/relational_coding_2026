"""Figure 2's 2-object stimulus panels, with the chords moved to where the trials actually were.

Figure 2's PSTH colours each condition by the mean firing rate over every trial whose **two
objects were both within 15° of their prototype slot** (`psth_lib.window_trials_2obj`, and
`psth_lib._dist_2obj` for the matching rule). The prototype is therefore a label for a small
cloud of real stimuli, not a stimulus that was ever shown on its own.

This module redraws the two 2-object stimulus panels -- the `sep1_deg`-apart row and the
180°-apart row -- **unchanged except that each chord's endpoints are the circular means of the
objects in the trials that chord averaged**. Same four chords, same firing-rate colours, same
colour scale, drawn by figure 2's own `draw_2obj_prototypes`; only the positions move. What the
figure shows is how far the labels sit from the stimuli they name.

Two details decide whether the means are right.

* **The pair is unordered.** A trial joins a prototype's bin under whichever of the two
  object-to-slot assignments fits better, so before averaging, each trial's objects must be put
  back into the slots they were matched through. Averaging `object_0` with `object_0` across
  trials would mix the two slots and manufacture a deviation that is not there.
* **Angles are circular**, so the mean is the angle of the summed unit vectors. An arithmetic
  mean would be wrong for any bin straddling 0°.

Nothing about the prototypes, the window, the matching or the colours is reimplemented here;
all of it is imported from `figure_2/psth_lib.py`, which is the only way this can be a
statement about what figure 2 did.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# `figure_2` on the path, resolved from THIS FILE. Note there is deliberately no `common.py`
# in this directory: `psth_lib` does `from common import ...`, and a local one would shadow
# figure 2's and break it. See README.md.
_F2 = Path(__file__).resolve().parent.parent / "figure_2"
if str(_F2) not in sys.path:
    sys.path.insert(0, str(_F2))

import psth_lib as pl  # noqa: E402  figure_2 — prototypes, matching, drawing, colours


def _circmean(a):
    """Circular mean of angles in radians."""
    return float(np.angle(np.exp(1j * np.asarray(a, float)).sum()))


def _assign(t0, t1, p0, p1):
    """A trial's two objects, ordered into the prototype slots they were matched through.

    `psth_lib._dist_2obj` keeps the better of the two assignments; this reproduces that choice
    so slot 0 always means "the object matched to prototype slot 0".
    """
    straight = max(pl._circ_dist(t0, p0), pl._circ_dist(t1, p1))
    swapped = max(pl._circ_dist(t0, p1), pl._circ_dist(t1, p0))
    return (t0, t1) if straight <= swapped else (t1, t0)


def empirical_protos(behavior_2obj, protos, trials_per_condition):
    """``protos``, with each pair replaced by the circular means of its matched trials.

    Same keys and same shape as the input, so it drops straight into
    `psth_lib.draw_2obj_prototypes` in place of the prototypes. A prototype with no matched
    trials keeps its nominal position -- there is nothing to average -- and is reported by
    :func:`deviation_table` with ``n = 0``.
    """
    t0 = dict(
        zip(
            behavior_2obj["trial_num"].astype(int),
            behavior_2obj["object_0_theta"].astype(float),
        )
    )
    t1 = dict(
        zip(
            behavior_2obj["trial_num"].astype(int),
            behavior_2obj["object_1_theta"].astype(float),
        )
    )
    out = {}
    for key, (p0, p1) in protos.items():
        pairs = [
            _assign(t0[t], t1[t], p0, p1)
            for t in trials_per_condition.get(key, [])
            if t in t0
        ]
        out[key] = (
            (p0, p1)
            if not pairs
            else (
                _circmean([p[0] for p in pairs]),
                _circmean([p[1] for p in pairs]),
            )
        )
    return out


def deviation_table(protos, empirical, trials_per_condition, row_label):
    """How far each chord moved, per slot and worst-of-the-two.

    The worst of the two slots is the honest summary, because it is the rule that admitted the
    trial in the first place: `_dist_2obj` takes the worse of the two per-object distances.
    """
    rows = []
    for key in sorted(protos):
        (p0, p1), (m0, m1) = protos[key], empirical[key]
        d0 = float(np.degrees(pl._circ_dist(m0, p0)))
        d1 = float(np.degrees(pl._circ_dist(m1, p1)))
        rows.append(
            dict(
                row=row_label,
                chord=key,
                n=len(trials_per_condition.get(key, [])),
                dev0_deg=d0,
                dev1_deg=d1,
                dev_deg=max(d0, d1),
            )
        )
    return pd.DataFrame(rows)


def empirical_stimulus_figure(
    subject,
    session,
    unit_id,
    orientation_deg=0.0,
    sep1_deg=90,
    similarity_window_deg=15.0,
    n_proto=4,
    share_2obj_color_scale=True,
    fr_window=(0, 2000),
    smooth_half_win_ms=250.0,
    n_dense=18,
    panel_in=1.25,
    margin=(0.30, 0.45),
    cbar=True,
    cbar_width_in=0.10,
    cbar_gap_in=0.12,
    show_prototypes=False,
    verbose=True,
):
    """The two 2-object stimulus panels of `figure_2/psth.ipynb`, with empirical endpoints.

    Every argument that also exists in `psth_lib.build_rows` is passed straight through, so
    setting them to `psth.ipynb`'s values reproduces that figure's panels exactly -- colours,
    colour scale and all -- and the only difference is where the chords sit.

    ``show_prototypes`` additionally ghosts the nominal positions underneath in grey, for
    checking; the figure is about the empirical ones and it is off by default.

    Returns ``(fig, axes, table)``.
    """
    b2 = pl.load_behavior(subject, session, num_objects=2)
    _sc, _tr, rows = pl.build_rows(
        subject,
        session,
        unit_id,
        orientation_deg=orientation_deg,
        sep1_deg=sep1_deg,
        similarity_window_deg=similarity_window_deg,
        fr_window=fr_window,
        smooth_half_win_ms=smooth_half_win_ms,
        n_dense=n_dense,
        n_proto=n_proto,
        share_2obj_color_scale=share_2obj_color_scale,
        verbose=verbose,
    )
    two = [r for r in rows if r["kind"] == "chord"]

    fig_w = (
        panel_in
        + margin[0]
        + (cbar_gap_in + cbar_width_in + 0.45 if cbar else 0.0)
    )
    fig_h = panel_in * len(two) + margin[1]
    fig, axes = plt.subplots(
        len(two),
        1,
        figsize=(fig_w, fig_h),
        squeeze=False,
        layout="constrained",
    )
    axes = axes[:, 0]

    tables = []
    for ax, row in zip(axes, two):
        emp = empirical_protos(b2, row["protos"], row["trials_per_condition"])
        if show_prototypes:
            ghost = {k: (0.75, 0.75, 0.75, 1.0) for k in row["protos"]}
            pl.draw_2obj_prototypes(ax, row["protos"], ghost)
        pl.draw_2obj_prototypes(ax, emp, row["colors"])
        ax.set_title(row["label"])
        tables.append(
            deviation_table(
                row["protos"], emp, row["trials_per_condition"], row["label"]
            )
        )

    if cbar and len(two):
        # One bar for both panels when they share a scale, which is what makes their colours
        # comparable; otherwise one per panel, since each is then its own range.
        import matplotlib as mpl

        if share_2obj_color_scale:
            sm = mpl.cm.ScalarMappable(norm=two[0]["norm"], cmap=pl.COOLWARM)
            fig.colorbar(
                sm,
                ax=list(axes),
                fraction=0.08,
                pad=0.02,
                label="Firing rate (Hz)",
            )
        else:
            for ax, row in zip(axes, two):
                sm = mpl.cm.ScalarMappable(norm=row["norm"], cmap=pl.COOLWARM)
                fig.colorbar(
                    sm,
                    ax=ax,
                    fraction=0.08,
                    pad=0.02,
                    label="Firing rate (Hz)",
                )
    return fig, axes, pd.concat(tables, ignore_index=True)

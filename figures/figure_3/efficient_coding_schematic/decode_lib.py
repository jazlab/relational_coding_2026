r"""Two strict sub-populations of real neurons, and whether a decoder can tell three stimuli apart.

This module is the whole of `decoding.ipynb`: the selection rule that builds the two
sub-populations, the stimuli they are shown, the decoder, and the figures. It was two files
(`decode_lib.py` and `strict_lib.py`) until 2026-09-13; they were merged because the notebook is
the only consumer and the split cut the pipeline in half at an arbitrary point.

1. THE SELECTION
----------------
Two sub-populations, **disjoint** and at **equal n**, drawn from **one ranking over both
animals** (:func:`select_wedges`). Equal n is not cosmetic: any read-out improves with neuron
count, so two samples of different size are not comparable at all. The larger side is subsampled
*without* replacement, so no neuron is counted twice and the two samples share nothing.

The animals were ranked separately and pooled afterwards until 2026-09-13, which forced an even
split between them. Pooling first means the split is an **outcome**, and it is very uneven: held-
out accuracy is systematically higher in Monkey P, so Monkey P supplies almost all of both sides.
:func:`wedge_report` and the notebook print the composition; read any change in the result
against it.

The two samples are the N neurons each model fits **best**, not the extremes of rho: rho is an
angle, and its ends collect FAILED fits — a neuron scoring 0.501 and 0.500 sits at rho = 0 and
would be called purely retinotopic on no evidence.

**What is not fixed.** The two ends still differ in everything else that covaries with rho --
firing rate, tuning depth, brain area. :func:`wedge_report` prints those so the confound is
visible; it is not removed, and a difference in decoding accuracy should not be read as caused
by relationality alone.

2. THE STIMULI
--------------
Three, each a pair of angles on the ring:

* **A** -- two objects antipodal, north and south;
* **B** -- the same pair **rotated** slightly clockwise (separation unchanged);
* **C** -- both objects nudged **east** (the midpoint moves and the separation shrinks).

The pairs are chosen so the two comparisons probe *orthogonal directions* of the local stimulus
space:

    A -> B   is (dtheta0, dtheta1) prop (1, 1)    -- the pair ROTATES        (:data:`U_SUM`)
    A -> C   is (dtheta0, dtheta1) prop (1, -1)   -- the SEPARATION changes  (:data:`U_DIFF`)

and that is the whole point. A relational neuron is tuned to `theta0 + theta1`, unchanged by any
displacement along `u_diff`: such a population is **blind to the A/C distinction** however
strongly it responds. A retinotopic neuron sees both.

3. THE READ-OUT
---------------
The animal's **recorded** trials near each stimulus, and a linear classifier cross-validated on
them, scored as held-out accuracy (:func:`decode_pair_real`). One real trial per unit makes a
pseudo-trial -- the units were recorded in different sessions, so there is no simultaneous
population to take instead -- and the variability is therefore whatever the neurons did,
including everything a written-down noise model leaves out. The cost is that the classes are a
blur of the eligibility radius rather than the stimuli themselves.

The classifier is linear on purpose: the claim is that the *population code* separates the
stimuli, and a non-linear decoder could separate clouds no downstream neuron could read with a
weighted sum.

Two things this replaced, both recoverable from git: a simulated observation model (Poisson
counts drawn from the response tables, removed 2026-09-13 with the `USE_REAL_TRIALS` switch),
and before that a Fisher-information reading, `d'^2 = Delta^T J Delta` (`schematic_lib.py`,
`fisher_lib.py`). Fisher information is **local**, exact only for small `Delta`, and A vs C moves
the objects far enough that the expansion was an assumption rather than a fact.

**What the number is not.** Accuracy depends on how many trials the decoder is given, so read it
as a comparison *between populations at matched settings*: what the figure claims is the pattern
across bars, never a bar's height.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_V2 = _HERE.parent  # figures/figure_3
for _p in (
    str(_V2),
    str(_V2.parent),
    str(_V2 / "model_comparison"),
    str(_V2 / "efficient_coding"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pool_lib as pool  # noqa: E402  ../efficient_coding — the pool and response tables
import selection_lib as sel  # noqa: E402  the relationality coordinate

# ══════════════════════════════════════════════════════════════════════════════
# 1. Selecting the two sub-populations
# ══════════════════════════════════════════════════════════════════════════════

#: The animals, pooled into one ranking by :func:`select_wedges`.
SUBJECTS = ("Perle", "Elgar")


def wedge_report(
    units, positions, cols=("rho", "mean_rate", "n_2obj", "brain_area")
):
    """What differs between the two wedges *besides* relationality.

    Printed rather than corrected. Equal n makes the comparison meaningful; it does not make
    the two samples alike, and anything here that differs is a live alternative explanation for
    a gap in decoding accuracy.
    """
    rows = []
    for side, pos in positions.items():
        sub = units.iloc[pos]
        row = {"side": side, "n": len(sub)}
        for c in cols:
            if c not in sub.columns:
                continue
            v = sub[c]
            row[c] = (
                v.mode().iat[0]
                if v.dtype == object
                else float(np.nanmedian(v.to_numpy(float)))
            )
        rows.append(row)
    import pandas as pd

    return pd.DataFrame(rows).set_index("side")


# ── Selecting first, then building — and caching the result ───────────────────
def select_wedges(
    subject,
    n_per_side,
    sample_sigma_deg=10.0,
    seed=0,
    verbose=True,
    min_trials=None,
    stimuli=None,
    pairs=None,
    tol_deg=None,
    exclusive=True,
):
    """One subject's two sub-populations: the N best-fit neurons of each model.

    Returns ``dict(pop_df, rho, sides, positions)``, where ``positions`` maps each side to row
    positions **within the returned frame**.

    The ranking is **within one animal**, so ``n_per_side`` is a number per animal and the
    notebook calls this once per subject: each animal contributes its own top N and the two
    are pooled afterwards, which keeps the animals' contributions even by construction. A
    single pooled ranking over both animals was tried on 2026-09-13 and reverted — held-out
    accuracy is systematically higher in Monkey P, so one ranking on fit strength took almost
    all of its neurons from that animal.

    ``sample_sigma_deg`` is still passed to ``pool.subject_pool``, which takes it as part of
    its cache identity even with ``with_tables=False``; it no longer affects anything here.
    """
    # The pool frame only — `with_tables=False` reads no per-trial data.
    d = pool.subject_pool(
        subject,
        sample_sigma_deg=sample_sigma_deg,
        with_tables=False,
        verbose=False,
    )

    # The two samples are the N best-fit neurons of each model (:func:`top_n_positions`).
    # Selecting on the EXTREMES OF RHO was the other option until 2026-09-13 and was removed
    # with `strict_positions`: rho is an angle, so its ends collect failed fits and a sample
    # can be pure in composition while barely driven at all.
    #
    # With `min_trials`, a neuron must also have that many recorded trials for every stimulus
    # or it is skipped and the next one down the ranking takes its place — see
    # `enough_trials_mask` for what that couples.
    eligible = None
    if min_trials:
        if stimuli is None or pairs is None or tol_deg is None:
            raise ValueError(
                "min_trials needs stimuli, pairs and tol_deg to test against"
            )
        eligible = lambda cand: enough_trials_mask(  # noqa: E731
            d["pop_df"].iloc[cand],
            stimuli,
            pairs,
            tol_deg,
            min_trials,
            exclusive,
        )
    pos = top_n_positions(
        d["pop_df"], n_per_side=n_per_side, eligible=eligible, verbose=verbose
    )

    keep = np.concatenate([pos["retinotopic"], pos["relational"]])
    sides = np.array(
        ["retinotopic"] * len(pos["retinotopic"])
        + ["relational"] * len(pos["relational"])
    )
    out_pos = {
        "retinotopic": np.arange(len(pos["retinotopic"])),
        "relational": np.arange(len(pos["retinotopic"]), len(keep)),
    }
    return dict(
        pop_df=d["pop_df"].iloc[keep].reset_index(drop=True),
        rho=d["rho"][keep],
        sides=sides,
        positions=out_pos,
    )


def enough_trials_mask(
    units_df, stimuli, pairs, tol_deg, min_trials, exclusive=True
):
    """Which units have at least ``min_trials`` recorded trials for **every** stimulus used.

    A unit with fewer than ``n_splits`` eligible trials for one stimulus cannot be
    cross-validated -- some fold would have nothing of that class to test on -- so
    :func:`decode_pair_real` raises rather than quietly dropping it and leaving the two sides at
    unequal n. This is the test that lets the *selection* skip such a unit instead.

    Note what it couples: the population now depends on the stimuli, on ``tol_deg`` and on the
    fold count. Move a stimulus and a different set of neurons is eligible. That is the price of
    not silently unbalancing the two sides, and it should be stated wherever the population is
    described.
    """
    trials = pool.load_pool_trials(units_df)
    ok = np.ones(len(units_df), bool)
    for a, b in pairs:
        idx_a, idx_b = trial_pools(
            trials,
            stimuli[a],
            stimuli[b],
            tol_deg=tol_deg,
            exclusive=exclusive,
        )
        ok &= np.array(
            [
                len(x) >= min_trials and len(y) >= min_trials
                for x, y in zip(idx_a, idx_b)
            ]
        )
    return ok


def top_n_positions(units, n_per_side, eligible=None, verbose=True):
    """The ``n`` **most retinotopically** and ``n`` **most relationally** modulated neurons.

    ρ is an *angle* — a ratio of the two models' fits — so its extremes are a
    dumping ground for failed fits: a neuron scoring 0.501 against 0.500 sits at ρ = 0 and is
    called "purely retinotopic" on no evidence at all. This ranks on the **radius** instead:
    how strongly each model actually fits.

    The strength is ``2 * (accuracy - 1/2)`` — how far above chance the model calls held-out
    orderings (``selection_lib.fit_strength``). It is an effect size, which is what makes it
    rankable. The per-model p-value for beating chance is available as ``{model}_beats_null_p``
    and is *not* used: p-values saturate against the bootstrap's resolution floor, so ranking on
    them ties hundreds of neurons at the top and cannot pick an N.

    **Why this is not simply top-N of each column.** Taking the best-fit-retinotopic N and the
    best-fit-relational N independently produces samples that *overlap* — 21% of each side at
    N = 100 and 42% at N = 300, because a strongly driven neuron tends to be fit somewhat by
    both. A neuron in both samples argues both sides of the contrast. So each neuron is first
    **assigned to whichever model fits it better**, and the top N is taken within each group.
    The two samples are then disjoint by construction and still "the most modulated" in the
    sense the ranking means.

    ``eligible`` is an optional callable taking an array of row positions and returning a
    boolean mask over them. When given, the ranking is walked from the top and units failing the
    test are **skipped**, so the result is the top ``n`` *eligible* neurons of each side rather
    than the top ``n`` outright. :func:`enough_trials_mask` is the test this exists for.

    Returns ``{'retinotopic': positions, 'relational': positions}``, equal length.
    """
    ret = sel.fit_strength(units, "retinotopic")
    rel = sel.fit_strength(units, "relational")
    better_rel = rel > ret  # each neuron's own winner
    groups = {
        "retinotopic": np.where(~better_rel & np.isfinite(ret))[0],
        "relational": np.where(better_rel & np.isfinite(rel))[0],
    }
    strength = {"retinotopic": ret, "relational": rel}

    avail = min(len(groups["retinotopic"]), len(groups["relational"]))
    n = int(n_per_side)
    if n > avail:
        raise ValueError(
            f"n_per_side={n} exceeds the smaller group ({avail}); the two sides "
            f"would not be equal-n"
        )

    out, skipped = {}, {}
    for side, idx in groups.items():
        order = idx[np.argsort(-strength[side][idx])]  # strongest first
        if eligible is None:
            out[side], skipped[side] = np.sort(order[:n]), 0
            continue
        # Walk down the ranking in batches, keeping the first `n` that pass. Batching so the
        # test -- which reads trials -- is paid for a few more units than are needed, not for
        # the whole pool.
        chosen, cursor, n_skip = [], 0, 0
        while len(chosen) < n and cursor < len(order):
            take = max(n - len(chosen), 8)
            cand = order[cursor : cursor + take]
            ok = np.asarray(eligible(cand), bool)
            for pos, good in zip(cand, ok):
                if len(chosen) == n:
                    break
                if good:
                    chosen.append(pos)
                else:
                    n_skip += 1
            cursor += len(cand)
        if len(chosen) < n:
            raise ValueError(
                f"only {len(chosen)} of {len(order)} {side} neurons have enough trials for "
                f"every stimulus; asked for {n}. Lower n_per_side, widen tol_deg, or lower "
                f"n_splits."
            )
        out[side], skipped[side] = np.sort(np.array(chosen)), n_skip
    if verbose:
        for side in out:
            v = strength[side][out[side]]
            extra = (
                f", {skipped[side]} skipped for too few trials"
                if eligible is not None and skipped[side]
                else ""
            )
            print(
                f"  {side:12s}: {len(out[side])} of {len(groups[side])} "
                f"(fit strength {np.min(v):.3f}–{np.max(v):.3f}, median {np.median(v):.3f}"
                f"{extra})"
            )
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 2. The stimuli, the decoder and the figures
# ══════════════════════════════════════════════════════════════════════════════
U_SUM = np.array([1.0, 1.0]) / np.sqrt(
    2.0
)  # both objects move together (rotation)
U_DIFF = np.array([1.0, -1.0]) / np.sqrt(
    2.0
)  # they move oppositely (separation)


# ── Angles ────────────────────────────────────────────────────────────────────
def circ_dist(a, b):
    """Circular arc distance in [0, π]; broadcasts."""
    d = np.abs(np.asarray(a, float) - np.asarray(b, float)) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Stimuli ───────────────────────────────────────────────────────────────────
def stimulus_array(stimuli_deg):
    """`[n_stim, 2]` of stimulus angles in **radians**, from a list of `(θ0, θ1)` in degrees.

    Every object lies on the ring by construction: a stimulus *is* a pair of ring angles,
    and the drawing puts each object at radius 1 (:func:`plot_stimuli`).
    """
    return np.radians(np.asarray(stimuli_deg, float))


def separation_deg(stim):
    """Unordered separation of one stimulus, in degrees ∈ [0, 180]."""
    return float(np.degrees(circ_dist(stim[0], stim[1])))


def displacement(stim_a, stim_b):
    """`Δ = (dθ0, dθ1)` in radians, each component wrapped to (−π, π].

    Wrapped per component, because the two objects are tracked individually: a stimulus is
    an ordered pair here, and the wrap is what keeps "10° clockwise" from coming out as 350°.
    """
    d = np.asarray(stim_b, float) - np.asarray(stim_a, float)
    return (d + np.pi) % (2 * np.pi) - np.pi


def describe_pair(stim_a, stim_b, labels=("A", "B")):
    """One line saying what a comparison *is*: how far apart the two stimuli are, and how
    much of that displacement is a rotation of the pair versus a change of its separation.
    """
    d = displacement(stim_a, stim_b)
    norm = float(np.linalg.norm(d))
    rot = float(d @ U_SUM)
    sep = float(d @ U_DIFF)
    return (
        f"{labels[0]} → {labels[1]}: |Δ| = {np.degrees(norm):.1f}°, "
        f"rotation {np.degrees(rot):+.1f}°, separation change {np.degrees(sep):+.1f}° "
        f"({separation_deg(stim_a):.0f}° → {separation_deg(stim_b):.0f}° apart)"
    )


def plot_stimuli(
    stimuli,
    titles=None,
    figsize=(6.0, 2.2),
    dpi=300,
    radius=1.0,
    disc_radius=0.13,
    disc_color="0.6",
    disc_edge="k",
    disc_edge_lw=0.8,
    reference=True,
    ref_index=0,
    ref_color="0.45",
    ref_lw=1.2,
    ref_ls=(0, (3, 2)),
    arrows=True,
    arrow_color="tab:green",
    arrow_radius=0.78,
    arrow_lw=1.6,
    arrow_head=11,
    arrow_min_deg=0.5,
    ring_color="0.75",
    ring_lw=1.0,
    ring_ls="--",
    cross_size=8,
    view=1.35,
    wspace=None,
    title_fontsize=None,
    title_y=None,
):
    """The stimuli, one ring each: two gray discs on the ring, one per object.

    The objects are drawn as plain discs with no chord between them, so each panel shows the
    stimulus as the animal sees it — two objects at two ring positions — rather than
    asserting the pair's relative vector, which is one code's read of the stimulus and not a
    property of it.

    Every panel after `ref_index` also carries **dashed open circles at the reference
    stimulus's object positions**. That is what makes the comparison legible: the
    displacement between two stimuli is a few degrees, invisible when the panels are read
    side by side, but obvious as the offset between each filled disc and the dashed circle
    it moved out of.

    Each moved object also gets an **arrow** showing the perturbation that carries it there
    from the reference stimulus. The arrows are drawn as arcs at `arrow_radius`, *inside* the
    ring rather than on it, so they do not run underneath the discs they connect — at these
    displacements an arrow drawn on the ring itself would be almost entirely hidden by the
    two circles at its ends. An object that does not move (below `arrow_min_deg`) gets no
    arrow, since a zero-length arrow would draw a meaningless head.

    `disc_radius`, `arrow_radius` and the reference circles are in **data units**, where the
    ring has radius `radius` — so everything is drawn at a true size relative to the ring and
    stays that way at any figure size or dpi.

    `wspace` sets the horizontal gap between panels, as a fraction of the average panel width
    (matplotlib's convention; `None` leaves it to `tight_layout`). Note that each panel holds
    a *square* ring inside an axes box that `figsize` may make wider, and that extra width
    shows up as empty margin within the panel rather than between panels — so if the rings
    still look far apart at `wspace=0`, narrow the figure or lower `view` (the axis limit, in
    ring radii) instead of pushing `wspace` negative.
    """
    stimuli = np.asarray(stimuli, float)
    fig, axes = plt.subplots(
        1, len(stimuli), figsize=figsize, dpi=dpi, squeeze=False
    )
    circ = np.linspace(0, 2 * np.pi, 256)
    ref_xy = np.stack(
        [
            radius * np.cos(stimuli[ref_index]),
            radius * np.sin(stimuli[ref_index]),
        ],
        axis=1,
    )
    for i, (ax, stim) in enumerate(zip(axes[0], stimuli)):
        ax.plot(
            radius * np.cos(circ),
            radius * np.sin(circ),
            color=ring_color,
            lw=ring_lw,
            ls=ring_ls,
            zorder=1,
        )
        ax.plot(
            0,
            0,
            marker="+",
            color="k",
            markersize=cross_size,
            markeredgewidth=1.3,
            zorder=2,
        )
        if reference and i != ref_index:
            for x, y in ref_xy:
                ax.add_patch(
                    plt.Circle(
                        (x, y),
                        disc_radius,
                        facecolor="none",
                        edgecolor=ref_color,
                        lw=ref_lw,
                        ls=ref_ls,
                        zorder=3,
                    )
                )
        if arrows and i != ref_index:
            for a, b in zip(stimuli[ref_index], stim):
                d = float(
                    (b - a + np.pi) % (2 * np.pi) - np.pi
                )  # signed shortest move
                if abs(np.degrees(d)) < arrow_min_deg:
                    continue
                arc = a + np.linspace(0.0, d, 32)
                ax.plot(
                    arrow_radius * np.cos(arc),
                    arrow_radius * np.sin(arc),
                    color=arrow_color,
                    lw=arrow_lw,
                    solid_capstyle="round",
                    zorder=5,
                )
                ax.annotate(
                    "",
                    xy=(
                        arrow_radius * np.cos(arc[-1]),
                        arrow_radius * np.sin(arc[-1]),
                    ),
                    xytext=(
                        arrow_radius * np.cos(arc[-2]),
                        arrow_radius * np.sin(arc[-2]),
                    ),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=arrow_color,
                        lw=arrow_lw,
                        mutation_scale=arrow_head,
                        shrinkA=0,
                        shrinkB=0,
                    ),
                    zorder=5,
                )
        for x, y in zip(radius * np.cos(stim), radius * np.sin(stim)):
            ax.add_patch(
                plt.Circle(
                    (x, y),
                    disc_radius,
                    facecolor=disc_color,
                    edgecolor=disc_edge,
                    lw=disc_edge_lw,
                    zorder=4,
                )
            )
        if titles is not None:
            # `title_y` is the title's baseline in AXES FRACTIONS: 1.0 is the top of the axes
            # box, and lower values move it down toward the ring. The box is `view` ring radii
            # tall, so the ring's top sits at roughly `(1 + 1/view) / 2` -- about 0.87 at the
            # default view -- and that is how far the title can drop before it touches.
            # None leaves matplotlib's default placement (the top of the box plus `titlepad`).
            ax.set_title(
                titles[i],
                fontsize=(title_fontsize or plt.rcParams["axes.titlesize"]),
                y=title_y,
            )
        ax.set_xlim(-view, view)
        ax.set_ylim(-view, view)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.tight_layout()
    if wspace is not None:
        fig.subplots_adjust(
            wspace=wspace
        )  # after tight_layout, which would override it
    return fig, axes[0]


def trial_distance_deg(t0, t1, stim):
    """Swap-invariant distance from each recorded trial to one stimulus, in degrees.

    The **larger** of the two objects' arc distances, under whichever of the two pairings makes
    that smaller. Thresholding it at `tol` therefore means *both* objects are within `tol` of the
    stimulus objects they are matched to — a per-object guarantee, which a sum cannot give: a sum
    of 30° admits a trial whose first object is spot on and whose second is 30° away, and that
    trial is not the stimulus in any useful sense.

    Note this is NOT the metric the response tables' kernel uses
    (`decoder_lib._bump_weights_2obj` sums the two distances), so "within 15°" here and a kernel
    weight there are different statements. That is deliberate: eligibility is a hard membership
    question and wants the stricter, per-object reading.
    """
    d0, d1 = circ_dist(t0, stim[0]), circ_dist(t1, stim[1])
    s0, s1 = circ_dist(t0, stim[1]), circ_dist(t1, stim[0])
    return np.degrees(np.minimum(np.maximum(d0, d1), np.maximum(s0, s1)))


def trial_pools(unit_trials, stim_a, stim_b, tol_deg=30.0, exclusive=True):
    """Which recorded trials count as stimulus a and which as stimulus b, per unit.

    ``unit_trials`` is ``[(t0, t1, fr), ...]``, one entry per unit, as
    ``pool_lib.load_pool_trials`` returns it.

    A trial is eligible for a stimulus when **both** objects are within ``tol_deg`` of the
    stimulus objects they match (:func:`trial_distance_deg`).

    ``exclusive=True`` assigns a trial to whichever of the two stimuli is **nearer**, so the two
    classes are disjoint. That matters more than it sounds: A and B are ~21° apart and a useful
    tolerance is a good fraction of that, so overlapping pools would put the same trial in both
    classes and drive the decoder to chance for a reason that has nothing to do with the code.
    With ``False`` a trial eligible for both enters both, which is only meaningful when the two
    stimuli are far apart relative to the tolerance.

    Returns ``(idx_a, idx_b)``, lists of index arrays in unit order.
    """
    idx_a, idx_b = [], []
    for t0, t1, _fr in unit_trials:
        da = trial_distance_deg(t0, t1, stim_a)
        db = trial_distance_deg(t0, t1, stim_b)
        ka, kb = da <= float(tol_deg), db <= float(tol_deg)
        if exclusive:
            ka, kb = ka & (da <= db), kb & (db < da)
        idx_a.append(np.flatnonzero(ka))
        idx_b.append(np.flatnonzero(kb))
    return idx_a, idx_b


def _pseudo_population(unit_trials, idx, n_trials, rng):
    """`[n_trials, n_units]` of RECORDED firing rates: one real trial drawn per unit per row.

    The units were recorded in different sessions, so there is no such thing as a simultaneous
    population response here and the rows are **pseudo-trials** — each unit contributes a real
    trial of its own, drawn with replacement from its matching pool. That is the same
    independence assumption the simulated path makes by drawing each unit's Poisson count
    separately; what it buys is the real trial-to-trial variability and the real stimulus
    scatter in place of a model of them.
    """
    X = np.empty((int(n_trials), len(unit_trials)), float)
    for j, ((_t0, _t1, fr), take) in enumerate(zip(unit_trials, idx)):
        if not len(take):
            raise ValueError(
                f"unit {j} has no recorded trial within tolerance of this "
                f"stimulus; widen tol_deg"
            )
        X[:, j] = np.asarray(fr, float)[
            rng.choice(take, int(n_trials), replace=True)
        ]
    return X


def _trial_folds(idx, n_splits, rng):
    """Split one unit's eligible trials into ``n_splits`` held-out groups.

    The split is over **recorded trials**, which is the only place it can honestly go: a
    pseudo-trial is a fresh combination of real trials, so splitting pseudo-trials would put the
    same recorded trial on both sides of the fold and the accuracy would count memorised trials
    as generalisation.
    """
    if len(idx) < n_splits:
        return None
    return np.array_split(rng.permutation(idx), n_splits)


def decode_pair_real(
    unit_trials,
    stim_a,
    stim_b,
    n_trials=200,
    n_test=100,
    tol_deg=30.0,
    exclusive=True,
    n_splits=5,
    seed=0,
    C=1.0,
):
    """:func:`decode_pair` on **recorded** trials: no Poisson model, no counting window.

    Every trial the animal actually saw within ``tol_deg`` of a stimulus is eligible for that
    stimulus's class, and a pseudo-trial takes one real trial per unit
    (:func:`_pseudo_population`). The classifier is :func:`decode_pair`'s, so the two paths
    differ in where the numbers come from and in nothing else.

    **The cross-validation is over recorded trials, not over pseudo-trials.** Each unit's
    eligible trials are split into ``n_splits`` groups; a fold builds its training pseudo-trials
    from that unit's other groups and its test pseudo-trials from the held-out one, so no
    recorded trial is ever on both sides. This is not a formality — a unit contributes tens of
    eligible trials and hundreds of pseudo-trials are drawn from them, so a pseudo-trial split
    would score the decoder largely on rows it had already seen.

    The knobs of the simulated path — ``duration_s``, ``firing_rate_scale``, ``min_rate``,
    ``noise_scale`` — have no meaning here and are deliberately absent rather than ignored: the
    variability is whatever the animal's neurons did.

    Returns :func:`decode_pair`'s keys plus, per class, the median and minimum eligible trials
    per unit (``n_pool_a``/``n_pool_b``, ``n_pool_min``) and how far the eligible trials sat
    from the nominal stimulus (``dist_a_deg``, ``dist_b_deg``) — the numbers that say whether
    the classes are the stimuli you asked for.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    idx_a, idx_b = trial_pools(
        unit_trials, stim_a, stim_b, tol_deg=tol_deg, exclusive=exclusive
    )
    pools = [len(i) for i in idx_a] + [len(i) for i in idx_b]
    folds_a = [_trial_folds(i, n_splits, rng) for i in idx_a]
    folds_b = [_trial_folds(i, n_splits, rng) for i in idx_b]
    thin = [
        j
        for j, (fa, fb) in enumerate(zip(folds_a, folds_b))
        if fa is None or fb is None
    ]
    if thin:
        raise ValueError(
            f"{len(thin)} unit(s) have fewer than n_splits={n_splits} recorded trials within "
            f"tol_deg={tol_deg:g} of one of the stimuli (thinnest pool {min(pools)}). Widen "
            f"tol_deg, or lower n_splits."
        )

    accs, aucs = [], []
    for k in range(n_splits):

        def part(folds, held_out):
            return [
                np.concatenate(
                    [f[j] for j in range(n_splits) if (j == k) == held_out]
                )
                for f in folds
            ]

        Xtr = np.vstack(
            [
                _pseudo_population(
                    unit_trials, part(folds_a, False), n_trials, rng
                ),
                _pseudo_population(
                    unit_trials, part(folds_b, False), n_trials, rng
                ),
            ]
        )
        ytr = np.r_[np.zeros(n_trials), np.ones(n_trials)]
        Xte = np.vstack(
            [
                _pseudo_population(
                    unit_trials, part(folds_a, True), n_test, rng
                ),
                _pseudo_population(
                    unit_trials, part(folds_b, True), n_test, rng
                ),
            ]
        )
        yte = np.r_[np.zeros(n_test), np.ones(n_test)]
        clf = make_pipeline(
            StandardScaler(), LogisticRegression(C=C, max_iter=2000)
        )
        clf.fit(Xtr, ytr)
        accs.append(float(clf.score(Xte, yte)))
        aucs.append(float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])))
    accs = np.asarray(accs)

    def _dist(idx, stim):
        d = [
            trial_distance_deg(t0[i], t1[i], stim)
            for (t0, t1, _f), i in zip(unit_trials, idx)
            if len(i)
        ]
        return float(np.mean(np.concatenate(d))) if d else float("nan")

    return dict(
        accuracy=float(accs.mean()),
        accuracy_sem=float(accs.std(ddof=1) / np.sqrt(len(accs))),
        auc=float(np.mean(aucs)),
        n_features=len(unit_trials),
        folds=accs.tolist(),
        n_pool_a=float(np.median([len(i) for i in idx_a])),
        n_pool_b=float(np.median([len(i) for i in idx_b])),
        n_pool_min=int(min(pools)),
        dist_a_deg=_dist(idx_a, stim_a),
        dist_b_deg=_dist(idx_b, stim_b),
    )


def decoding_table(stimuli, pairs, labels, trials, **kw):
    """One decoder per population x stimulus pair, as a tidy frame.

    ``trials`` is a dict of population name -> ``[(t0, t1, fr), ...]``, one entry per unit, and
    the decoding is done on the animal's **recorded** trials (:func:`decode_pair_real`).

    A simulated path existed here until 2026-09-13 — Poisson counts drawn from per-neuron
    response tables under a written-down observation model — and was removed once the recorded
    trials became the only thing reported. Recover it from git if the simulation is wanted.
    """
    rows = []
    for name, unit_trials in trials.items():
        for a, b in pairs:
            r = decode_pair_real(unit_trials, stimuli[a], stimuli[b], **kw)
            rows.append(
                dict(
                    population=name,
                    pair=f"{labels[a]} vs {labels[b]}",
                    n_units=len(unit_trials),
                    **r,
                )
            )
    return pd.DataFrame(rows)


#: Legend styling for :func:`plot_decoding`, overridable key by key via `legend_kw`.
#: Everything except `rounded`/`rounding_size`/`x` is passed straight to `ax.legend`. The
#: first two are applied to the frame afterwards (`legend` has no box-style argument); `x`
#: slides the legend horizontally, replacing the x of `bbox_to_anchor` in axes coordinates —
#: 0.5 centres it, 0.0 puts its `loc` corner at the left edge, negative values move it
#: further left still. `None` leaves `bbox_to_anchor` alone.
LEGEND_DEFAULTS = dict(
    ncol=2,  # side by side, so the legend is one shallow strip
    loc="lower center",  # anchored ABOVE the axes ...
    bbox_to_anchor=(0.5, 1.0),  # ... at the top edge, clear of the tallest bar
    frameon=True,
    borderpad=0.6,  # space between the frame and its contents
    labelspacing=0.4,  # vertical space between entries
    handlelength=1.4,  # length of the color patch
    handletextpad=0.6,  # patch-to-text gap
    columnspacing=1.6,  # gap between the two columns
    fontsize=12,
    edgecolor="0.6",
    facecolor="white",
    framealpha=1.0,
    rounded=True,
    rounding_size=0.25,
    x=None,  # horizontal position; see above
)


def plot_decoding(
    table,
    pair_order=None,
    labels=None,
    colors=("tab:blue", "tab:orange"),
    figsize=(6.4, 3.0),
    dpi=300,
    ylabel=None,
    bar_width=0.36,
    bar_lw=0.6,
    error="sem",
    error_lw=1.4,
    error_capsize=3.0,
    error_color="k",
    chance=0.5,
    chance_color="k",
    chance_lw=1.0,
    chance_ls=(0, (4, 3)),
    ylim=(0.4, 1.02),
    legend_kw=None,
):
    """Grouped bars of held-out accuracy: one group per stimulus pair, one bar per population.

    The styling is the bar figure this directory used for `d'^2` — black-edged bars, the
    legend in a rounded box above the axes — so the two are readable as the same panel with
    the measure swapped.

    `error` chooses what the whiskers mean: `'sem'` (default) is the standard error of the
    plotted mean **over cross-validation folds**, which is a statement about how stable the
    fit is and *not* about how uncertain the accuracy is over neurons or trials; `'sd'` is the
    spread across folds; `'none'` draws none. The honest check on a close pair of bars is to
    re-run at another `seed`, since both bars are read off one draw of simulated trials.

    The chance line is at `chance` (0.5) and is not a fitted reference — the classes are
    balanced by construction, so it is where a decoder that has learnt nothing lands. It is
    drawn dashed and **in front of** the bars, so a bar that reaches it can be read against it
    rather than hiding it. `ylim` sets the y limits (`None` lets matplotlib choose), and
    `legend_kw` overrides :data:`LEGEND_DEFAULTS` key by key.
    """
    pops = list(dict.fromkeys(table["population"]))
    pairs = pair_order or list(dict.fromkeys(table["pair"]))
    if labels is None:
        legend = list(pops)
    elif isinstance(labels, dict):
        legend = [labels.get(n, n) for n in pops]
    else:
        legend = list(labels)
    x = np.arange(len(pairs), dtype=float)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    for i, (pop, color, text) in enumerate(zip(pops, colors, legend)):
        sub = table[table["population"] == pop].set_index("pair")
        off = (i - (len(pops) - 1) / 2) * bar_width
        heights = [float(sub.loc[p, "accuracy"]) for p in pairs]
        if error == "sem":
            errs = [float(sub.loc[p, "accuracy_sem"]) for p in pairs]
        elif error == "sd":
            errs = [float(np.std(sub.loc[p, "folds"], ddof=1)) for p in pairs]
        elif error in (None, "none"):
            errs = None
        else:
            raise ValueError(
                f"error must be 'sem', 'sd' or 'none' (got {error!r})"
            )
        bar_kw = (
            {}
            if errs is None
            else dict(
                yerr=errs,
                capsize=error_capsize,
                error_kw=dict(
                    elinewidth=error_lw,
                    capthick=error_lw,
                    ecolor=error_color,
                    zorder=5,
                ),
            )
        )
        ax.bar(
            x + off,
            heights,
            width=bar_width,
            color=color,
            edgecolor="k",
            linewidth=bar_lw,
            label=text,
            zorder=3,
            **bar_kw,
        )

    if chance is not None:
        ax.axhline(
            chance, color=chance_color, lw=chance_lw, ls=chance_ls, zorder=7
        )
    ax.set_xticks(x)
    ax.set_xticklabels(pairs)
    ax.set_ylabel(ylabel or "Decoding accuracy")
    if ylim is not None:
        ax.set_ylim(*ylim)
    kw = {**LEGEND_DEFAULTS, **(legend_kw or {})}
    rounded, rounding_size = kw.pop("rounded"), kw.pop("rounding_size")
    legend_x = kw.pop("x")
    if legend_x is not None:
        anchor = kw.get("bbox_to_anchor") or (0.5, 1.0)
        kw["bbox_to_anchor"] = (legend_x, anchor[1])
    leg = ax.legend(**kw)
    if rounded and leg.get_frame_on():
        # pad=0 and let `borderpad` do the spacing: a box-style pad grows the frame OUTSIDE
        # the extent the legend reports, which a tight bounding box then clips off.
        leg.get_frame().set_boxstyle(
            "round", pad=0.0, rounding_size=rounding_size
        )
    despine(ax)
    fig.tight_layout()
    return fig, ax

r"""Two strict sub-populations of real neurons, and whether a decoder can tell three stimuli apart.

This module is the whole of `decoding.ipynb`: the selection rule that builds the two
sub-populations, the stimuli they are shown, the decoder, and the figures.

1. THE SELECTION
----------------
Two sub-populations, **disjoint** and at **equal n**, ranked **within each animal**
(:func:`select_wedges`, called once per animal, the two animals pooled afterwards so they
contribute evenly). Equal n is not cosmetic: any read-out improves with neuron count, so two
samples of different size are not comparable at all.

The two samples are the N neurons each model fits **best** (:func:`top_n_positions`), not the
extremes of rho: rho is an angle, and its ends collect FAILED fits — a neuron scoring 0.501 and
0.500 sits at rho = 0 and would be called purely retinotopic on no evidence. Each neuron is
first assigned to whichever model fits it better, so the two samples share no neuron.

**What is not fixed.** The two ends still differ in everything else that covaries with rho --
firing rate, tuning depth, brain area. A difference in decoding accuracy should not be read as
caused by relationality alone.

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

A held-out accuracy needs no local approximation: a Fisher-information reading,
`d'^2 = Delta^T J Delta`, is exact only for small `Delta`, and A vs C moves the objects far
enough that the expansion would be an assumption rather than a fact.

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
_FIGURE_3 = _HERE.parent  # figures/figure_3
for _p in (
    str(_FIGURE_3),
    str(_FIGURE_3.parent),
    str(_FIGURE_3 / "model_comparison"),
    str(_FIGURE_3 / "efficient_coding"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pool_lib as pool  # noqa: E402  ../efficient_coding — the pool and response tables
import selection_lib as sel  # noqa: E402  the relationality coordinate

# ══════════════════════════════════════════════════════════════════════════════
# 1. Selecting the two sub-populations
# ══════════════════════════════════════════════════════════════════════════════

#: The animals. Each is ranked separately by :func:`select_wedges` and the two are pooled.
SUBJECTS = ("Perle", "Elgar")


def wedge_report(units, positions):
    """``n`` and the median number of recorded 2-object trials (``n_2obj``) per side."""
    rows = []
    for side, pos in positions.items():
        sub = units.iloc[pos]
        rows.append(
            dict(
                side=side,
                n=len(sub),
                n_2obj=float(np.nanmedian(sub["n_2obj"].to_numpy(float))),
            )
        )
    return pd.DataFrame(rows).set_index("side")


def select_wedges(subject, n_per_side, min_trials, stimuli, pairs, tol_deg):
    """One subject's two sub-populations: the N best-fit neurons of each model.

    Returns ``dict(pop_df, rho, sides, positions)``, where ``positions`` maps each side to row
    positions **within the returned frame**.

    The ranking is **within one animal**, so ``n_per_side`` is a number per animal and the
    notebook calls this once per subject: each animal contributes its own top N and the two
    are pooled afterwards, which keeps the animals' contributions even by construction. (One
    ranking over both animals would take almost all of its neurons from Monkey P, whose held-out
    accuracy is systematically higher.)

    A neuron must also have ``min_trials`` recorded trials within ``tol_deg`` of every stimulus
    of every pair in ``pairs``, or it is skipped and the next one down the ranking takes its
    place — see :func:`enough_trials_mask` for what that couples.
    """
    # The pool frame only — `with_tables=False` reads no per-trial data.
    d = pool.subject_pool(subject, with_tables=False, verbose=False)

    def eligible(cand):
        return enough_trials_mask(
            d["pop_df"].iloc[cand], stimuli, pairs, tol_deg, min_trials
        )

    pos = top_n_positions(
        d["pop_df"], n_per_side=n_per_side, eligible=eligible
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


def enough_trials_mask(units_df, stimuli, pairs, tol_deg, min_trials):
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
        idx_a, idx_b = trial_pools(trials, stimuli[a], stimuli[b], tol_deg)
        ok &= np.array(
            [
                len(x) >= min_trials and len(y) >= min_trials
                for x, y in zip(idx_a, idx_b)
            ]
        )
    return ok


def top_n_positions(units, n_per_side, eligible=None):
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

    Prints, per side, how many neurons were taken and their fit strengths. Returns
    ``{'retinotopic': positions, 'relational': positions}``, equal length.
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
    titles,
    figsize,
    disc_radius,
    disc_color,
    wspace,
    view,
    title_fontsize,
    title_y,
):
    """The stimuli, one ring each: two gray discs on the ring, one per object.

    The objects are drawn as plain discs with no chord between them, so each panel shows the
    stimulus as the animal sees it — two objects at two ring positions — rather than
    asserting the pair's relative vector, which is one code's read of the stimulus and not a
    property of it.

    Every panel after the first also carries **dashed open circles at the first (reference)
    stimulus's object positions**. That is what makes the comparison legible: the
    displacement between two stimuli is a few degrees, invisible when the panels are read
    side by side, but obvious as the offset between each filled disc and the dashed circle
    it moved out of.

    `disc_radius` and the reference circles are in **data units**, where the ring has radius
    1 — so everything is drawn at a true size relative to the ring and stays that way at any
    figure size or dpi.

    `wspace` sets the horizontal gap between panels, as a fraction of the average panel width
    (matplotlib's convention). Note that each panel holds a *square* ring inside an axes box
    that `figsize` may make wider, and that extra width shows up as empty margin within the
    panel rather than between panels — so if the rings still look far apart at `wspace=0`,
    narrow the figure or lower `view` (the axis limit, in ring radii) instead of pushing
    `wspace` negative. `title_y` is each title's baseline in AXES FRACTIONS: 1.0 is the top of
    the axes box, and lower values move it down toward the ring. The box is `view` ring radii
    tall, so the ring's top sits at roughly `(1 + 1/view) / 2` -- about 0.87 at view 1.35 --
    and that is how far the title can drop before it touches.
    """
    stimuli = np.asarray(stimuli, float)
    fig, axes = plt.subplots(
        1, len(stimuli), figsize=figsize, dpi=300, squeeze=False
    )
    circ = np.linspace(0, 2 * np.pi, 256)
    ref_xy = np.stack([np.cos(stimuli[0]), np.sin(stimuli[0])], axis=1)
    for i, (ax, stim) in enumerate(zip(axes[0], stimuli)):
        ax.plot(
            np.cos(circ),
            np.sin(circ),
            color="0.75",
            lw=1.0,
            ls="--",
            zorder=1,
        )
        ax.plot(
            0,
            0,
            marker="+",
            color="k",
            markersize=8,
            markeredgewidth=1.3,
            zorder=2,
        )
        if i != 0:
            for x, y in ref_xy:
                ax.add_patch(
                    plt.Circle(
                        (x, y),
                        disc_radius,
                        facecolor="none",
                        edgecolor="0.45",
                        lw=1.2,
                        ls=(0, (3, 2)),
                        zorder=3,
                    )
                )
        for x, y in zip(np.cos(stim), np.sin(stim)):
            ax.add_patch(
                plt.Circle(
                    (x, y),
                    disc_radius,
                    facecolor=disc_color,
                    edgecolor="k",
                    lw=0.8,
                    zorder=4,
                )
            )
        ax.set_title(titles[i], fontsize=title_fontsize, y=title_y)
        ax.set_xlim(-view, view)
        ax.set_ylim(-view, view)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.tight_layout()
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


def trial_pools(unit_trials, stim_a, stim_b, tol_deg):
    """Which recorded trials count as stimulus a and which as stimulus b, per unit.

    ``unit_trials`` is ``[(t0, t1, fr), ...]``, one entry per unit, as
    ``pool_lib.load_pool_trials`` returns it.

    A trial is eligible for a stimulus when **both** objects are within ``tol_deg`` of the
    stimulus objects they match (:func:`trial_distance_deg`).

    A trial eligible for both is assigned to whichever of the two stimuli is **nearer**, so the
    two classes are disjoint. That matters more than it sounds: A and B are ~21° apart and a
    useful tolerance is a good fraction of that, so overlapping pools would put the same trial in
    both classes and drive the decoder to chance for a reason that has nothing to do with the
    code.

    Returns ``(idx_a, idx_b)``, lists of index arrays in unit order.
    """
    idx_a, idx_b = [], []
    for t0, t1, _fr in unit_trials:
        da = trial_distance_deg(t0, t1, stim_a)
        db = trial_distance_deg(t0, t1, stim_b)
        ka, kb = da <= float(tol_deg), db <= float(tol_deg)
        ka, kb = ka & (da <= db), kb & (db < da)
        idx_a.append(np.flatnonzero(ka))
        idx_b.append(np.flatnonzero(kb))
    return idx_a, idx_b


def _pseudo_population(unit_trials, idx, n_trials, rng):
    """`[n_trials, n_units]` of RECORDED firing rates: one real trial drawn per unit per row.

    The units were recorded in different sessions, so there is no such thing as a simultaneous
    population response here and the rows are **pseudo-trials** — each unit contributes a real
    trial of its own, drawn with replacement from its matching pool. That assumes the units
    independent given the stimulus; what it buys is the real trial-to-trial variability and the
    real stimulus scatter in place of a model of them.
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
    unit_trials, stim_a, stim_b, n_trials, n_test, tol_deg, n_splits, seed, C
):
    """Held-out accuracy of a linear classifier telling ``stim_a`` from ``stim_b`` on
    **recorded** trials: no Poisson model, no counting window.

    Every trial the animal actually saw within ``tol_deg`` of a stimulus is eligible for that
    stimulus's class (:func:`trial_pools`), and a pseudo-trial takes one real trial per unit
    (:func:`_pseudo_population`): ``n_trials`` training and ``n_test`` held-out pseudo-trials
    per class per fold. The classifier is a standardised, L2-regularised (``C``) logistic
    regression.

    **The cross-validation is over recorded trials, not over pseudo-trials.** Each unit's
    eligible trials are split into ``n_splits`` groups; a fold builds its training pseudo-trials
    from that unit's other groups and its test pseudo-trials from the held-out one, so no
    recorded trial is ever on both sides. This is not a formality — a unit contributes tens of
    eligible trials and hundreds of pseudo-trials are drawn from them, so a pseudo-trial split
    would score the decoder largely on rows it had already seen.

    Returns ``accuracy`` (mean over folds), ``accuracy_sem``, ``auc``, ``n_features``,
    ``folds`` (per-fold accuracies) and, per class, the median and minimum eligible trials
    per unit (``n_pool_a``/``n_pool_b``, ``n_pool_min``) and how far the eligible trials sat
    from the nominal stimulus (``dist_a_deg``, ``dist_b_deg``) — the numbers that say whether
    the classes are the stimuli you asked for.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    idx_a, idx_b = trial_pools(unit_trials, stim_a, stim_b, tol_deg)
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
    the decoding is done on the animal's **recorded** trials (:func:`decode_pair_real`, which
    ``kw`` is passed to).
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
#: Everything except `rounding_size`/`x` is passed straight to `ax.legend`. `rounding_size` is
#: applied to the frame afterwards (`legend` has no box-style argument); `x` slides the legend
#: horizontally, replacing the x of `bbox_to_anchor` in axes coordinates — 0.5 centres it, 0.0
#: puts its `loc` corner at the left edge, negative values move it further left still. `None`
#: leaves `bbox_to_anchor` alone.
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
    rounding_size=0.25,
    x=None,  # horizontal position; see above
)


def plot_decoding(
    table,
    labels,
    colors,
    figsize,
    ylim,
    legend_kw,
    bar_width,
    error_lw,
    error_capsize,
):
    """Grouped bars of held-out accuracy: one group per stimulus pair, one bar per population.

    Black-edged bars, the legend in a rounded box above the axes. ``labels`` maps each
    population to its legend text.

    The whiskers are the standard error of the plotted mean **over cross-validation folds**,
    which is a statement about how stable the fit is and *not* about how uncertain the accuracy
    is over neurons or trials. The honest check on a close pair of bars is to re-run at another
    `seed`, since both bars are read off one draw of pseudo-trials.

    The chance line is at 0.5 and is not a fitted reference — the classes are balanced by
    construction, so it is where a decoder that has learnt nothing lands. It is drawn dashed
    and **in front of** the bars, so a bar that reaches it can be read against it rather than
    hiding it. `ylim` sets the y limits, and `legend_kw` overrides :data:`LEGEND_DEFAULTS` key
    by key.
    """
    pops = list(dict.fromkeys(table["population"]))
    pairs = list(dict.fromkeys(table["pair"]))
    legend = [labels.get(n, n) for n in pops]
    x = np.arange(len(pairs), dtype=float)
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    for i, (pop, color, text) in enumerate(zip(pops, colors, legend)):
        sub = table[table["population"] == pop].set_index("pair")
        off = (i - (len(pops) - 1) / 2) * bar_width
        heights = [float(sub.loc[p, "accuracy"]) for p in pairs]
        errs = [float(sub.loc[p, "accuracy_sem"]) for p in pairs]
        ax.bar(
            x + off,
            heights,
            width=bar_width,
            color=color,
            edgecolor="k",
            linewidth=0.6,
            label=text,
            zorder=3,
            yerr=errs,
            capsize=error_capsize,
            error_kw=dict(
                elinewidth=error_lw,
                capthick=error_lw,
                ecolor="k",
                zorder=5,
            ),
        )

    ax.axhline(0.5, color="k", lw=1.0, ls=(0, (4, 3)), zorder=7)
    ax.set_xticks(x)
    ax.set_xticklabels(pairs)
    ax.set_ylabel("Decoding accuracy")
    ax.set_ylim(*ylim)
    kw = {**LEGEND_DEFAULTS, **legend_kw}
    rounding_size = kw.pop("rounding_size")
    legend_x = kw.pop("x")
    if legend_x is not None:
        kw["bbox_to_anchor"] = (legend_x, kw["bbox_to_anchor"][1])
    leg = ax.legend(**kw)
    # pad=0 and let `borderpad` do the spacing: a box-style pad grows the frame OUTSIDE
    # the extent the legend reports, which a tight bounding box then clips off.
    leg.get_frame().set_boxstyle("round", pad=0.0, rounding_size=rounding_size)
    despine(ax)
    fig.tight_layout()
    return fig, ax

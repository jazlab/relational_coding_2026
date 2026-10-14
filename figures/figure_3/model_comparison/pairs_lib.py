r"""Trial-pair comparisons, and the tuning-vector fit on them — no tuning curve anywhere.

The idea, in one paragraph
--------------------------
Take two trials that share an object. Trial *i* shows objects at (a, φ_i), trial *j* at
(a, φ_j) — call the shared object the **anchor**. A hypothesis about where this neuron's
conditional preferred direction sits is enough to predict **which trial fired more**:
whichever of φ_i, φ_j is closer to it. Count how often each hypothesis calls that right, and
the comparison never touches baseline rate, modulation depth, receptive-field width or the
noise model.

The two hypotheses are two slopes of the conditional preferred direction μ(a) = k·a + c:

    k =  0   retinotopic — the receptive field does not move with the anchor
    k = -1   relational  — it moves the opposite way, so the pair's axis is fixed

(Proximity, k = +1, is absent *in this signed form*: it was the only one of the three that the
four anchor assignments made self-contradictory. It returns tuned to a preferred separation
rather than to a slope — see :func:`build_separation_comparisons`.)

What reads this module
----------------------
Two things, and they use different halves of it.

* `comparison_lib.py` — **figure 3's model comparison**. It takes the comparisons built here
  and scores each model on the sign of ``Δs`` alone: no likelihood, no fitted sharpness, one
  parameter per model. That is the published comparison.
* `run_phase_cache.py` — the **phase fit** behind panel 3e. It needs the fitted tuning
  *vector* ``γ`` and a bootstrap distribution for its angle, which the sign-only fit does not
  estimate, so it uses the Bradley-Terry fit described below.

A graded model comparison scored by held-out per-comparison dLL was figure 3b until
2026-09-13. It is gone; what survives of it here is the fit that supplies the phase.

The Bradley-Terry fit
---------------------
With ``Δs`` the model's score difference between the two trials,

    P(r_i > r_j) = σ(β · Δs)        σ = logistic

— the Bradley-Terry model of paired comparisons. ``β`` is the one extra parameter: how sharply
a score difference turns into a reliable ordering, the ordinal stand-in for signal-to-noise.
``β = 0`` is exactly the null (every comparison a coin flip, log ½ per comparison), so the
"does this neuron have any conditional structure at all" test comes free — and that test is
what `stats_lib.phase_eligible` uses to decide whether a neuron's phase is determined at all.

Why there is no phase sweep
---------------------------
Writing ``A = (u_i + u_j)/2`` and ``B = (u_i - u_j)/2`` with ``u = φ - k·ā``, the sum-to-product
identity gives

    Δs = cos(u_i - c) - cos(u_j - c) = -2 sin(A - c) sin(B) = p·cos c + q·sin c

with ``p = -2 sin A sin B`` and ``q = 2 cos A sin B`` **free of c**. So

    β·Δs = γ₁·p + γ₂·q,        γ = β·(cos c, sin c)

and the whole model is a **weighted logistic regression on two precomputed predictors, with no
intercept** — convex, solved by IRLS in a handful of steps, with the phase read back as
``c = atan2(γ₂, γ₁)`` and the sharpness as ``β = |γ|``. No grid, no multi-start, no local
optima. ``test_pairs.py`` checks the identity and checks IRLS against a brute-force sweep.

The absent intercept is a symmetry statement, not an omission: swapping *i* and *j* flips both
the outcome and ``(p, q)``, so the fit must be odd in them. An intercept would let a model
claim that trial *i* fires more irrespective of the stimulus.

What the likelihood is not
--------------------------
It is a **composite** (pairwise) likelihood. Every trial appears in thousands of comparisons,
so the number of terms overstates the information by roughly the trial count. Consequences,
all handled here rather than assumed away:

* its curvature is **not** an error bar. Every uncertainty in this module comes from
  :func:`bootstrap_gamma`, which resamples **trials**;
* its units are nats per *comparison*, not per trial, so these numbers must never be plotted
  on the same axis as a per-trial log-likelihood.

:func:`cv_delta_ll` and the ``*_dll`` fields of :func:`fit_neuron` scored the retired graded
comparison. They are still computed — the phase cache has not been rebuilt for the sake of
deleting columns — but nothing reads them, and `stats_lib.load_units` drops them on the way in.
"""

import numpy as np

#: Comparisons whose anchors are far apart contribute essentially nothing but dominate the
#: array. Dropping them at construction is a memory optimisation, not an analysis choice: at
#: the default sigma this cut is 3.03 sigma and removes ~2/3 of the pairs while changing every
#: statistic in the fifth decimal place. `test_pairs.py` checks that.
WEIGHT_FLOOR = 1e-2

#: Anchor smoothing. Two trials' anchors count as "the same object" to the extent this Gaussian
#: says so. 20 deg matches the +/-30 deg anchor window of `figure_2/rf_per_anchor.ipynb`. It is
#: a bias/variance trade: a wider sigma pools more trial pairs per anchor and so lowers the
#: variance of the statistic, at the cost of blurring together anchors whose conditional
#: preferred directions genuinely differ, which biases every slope towards zero.
SIGMA_DEG = 20.0

#: Above this many comparisons a neuron is uniformly subsampled. The statistic's precision is
#: set by the number of *trials*, not of comparisons, so this costs almost nothing and bounds
#: both memory and runtime. Sampling is uniform and seeded, never weight-proportional, which
#: would bias the weighted mean the estimator is built on.
MAX_COMPARISONS = 400_000

#: Trial resamples behind every uncertainty in this module. It is the knob that sets the
#: runtime: the fits themselves are exact and cost almost nothing, while the bootstrap is
#: linear in this and dominates everything else (~0.033 s per resample per neuron at the
#: default comparison cap, against ~0.5 s for the rest of a neuron put together).
#:
#: **Why 50 rather than 20.** The phase *histograms* pool bootstraps across hundreds of
#: neurons, so 20 apiece would be ample for them. The two per-neuron statistics built on the
#: bootstrap are what set the floor: `wald_p` estimates a 2x2 covariance from this many points
#: and `circ_sd_deg` a spread, and both are read *per neuron* by the exclusion criteria. At 20
#: resamples a circular s.d. carries ~16% relative error and the Wald p-values are noisiest in
#: exactly the tail Benjamini-Hochberg then reads across ~1300 neurons; at 50 that error is
#: ~10% and the covariance is estimated from enough points to be usable. It applies to the
#: 1-object receptive field as well as the two 2-object models — `fit_neuron` passes this one
#: value to all three, so the reference direction and the phase measured against it are never
#: estimated to different precisions.
#:
#: If the exclusion counts in `rf_orientation.ipynb` still move between seeds, raise it; that
#: notebook prints them for this reason.
N_BOOT = 50

MODELS = (0, -1)  # k, the two slope models
MODEL_NAME = {0: "retinotopic", -1: "relational"}
MODEL_KEY = {"retinotopic": 0, "relational": -1}

#: Every model the comparison scores, in the order the figures list them. Proximity is not a
#: slope, so it has no ``k``: see :func:`build_separation_comparisons`.
ALL_MODELS = ("retinotopic", "relational", "proximity")

#: Candidate preferred separations for the proximity profile, in degrees. The task shows
#: nothing closer than 80 deg, so values below that are all the same hypothesis — "prefers
#: them as close as the task allows" — and the profile is flat there by construction.
SEP_GRID_DEG = np.arange(10.0, 180.0, 15.0)

#: Step of the second, local profile run around the coarse grid's winner. Two stages of 12 and
#: 7 fits beat one stage of 35 by roughly half, and agree with it to the refine step.
SEP_REFINE_DEG = 2.5

#: Comparisons kept for the proximity profile. Lower than :data:`MAX_COMPARISONS` because the
#: profile is re-run inside every bootstrap resample and the statistic's precision is set by
#: the number of *trials*, not comparisons — the same argument that justifies the cap at all.
SEP_MAX_COMPARISONS = 100_000
NULL_LL = float(np.log(0.5))  # per comparison, the beta = 0 likelihood


# ── Angles ────────────────────────────────────────────────────────────────────
def circ_dist(a, b):
    """Circular arc distance in [0, pi]; broadcasts."""
    d = np.abs(np.asarray(a, float) - np.asarray(b, float)) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


def wrap_deg(x):
    """Wrap degrees to (-180, 180]."""
    return (np.asarray(x, float) + 180.0) % 360.0 - 180.0


def circ_sd_deg(x, period=360.0):
    """Circular standard deviation in degrees, on a circle of the given period.

    This is the number the receptive-field-orientation figure's precision criterion is built
    on: it says how well the phase is *determined*, in the units the histogram is drawn in.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    k = 2 * np.pi / period
    R = np.abs(np.mean(np.exp(1j * k * x)))
    if R >= 1.0:
        return 0.0
    if R <= 0.0:
        return np.inf
    return float(np.degrees(np.sqrt(-2.0 * np.log(R))) / (k * 180 / np.pi))


# ── Building the comparisons ──────────────────────────────────────────────────
def _predictors(u_i, u_j):
    """``(p, q)`` such that ``cos(u_i - c) - cos(u_j - c) = p cos c + q sin c`` for every c."""
    A, B = 0.5 * (u_i + u_j), 0.5 * (u_i - u_j)
    return -2.0 * np.sin(A) * np.sin(B), 2.0 * np.cos(A) * np.sin(B)


def build_comparisons(
    t0,
    t1,
    fr,
    k,
    sigma_deg=SIGMA_DEG,
    weight_floor=WEIGHT_FLOOR,
    max_comparisons=MAX_COMPARISONS,
    seed=0,
    block=256,
):
    """Every shared-anchor comparison for one neuron under model ``k``.

    For each unordered pair of 2-object trials and each of the **four** ways of choosing which
    object in each trial is the anchor: the anchor-agreement weight, the observed ordering, the
    two phase-free predictors, and the two trial indices.

    Taking all four assignments is what keeps the statistic symmetric in an ordering the task
    does not define. Both surviving models are invariant to it (this is exactly what proximity
    was not), so the four agree rather than cancelling.

    Returns a dict of equal-length arrays ``w, y, p, q, i, j`` plus ``n_trials``,
    ``n_raw`` (comparisons before the weight floor) and ``frac_tied``.

    Built in blocks over the first trial index so the full ``n²`` triangle is never
    materialised; at 1000 trials that would be 2 million rows before the floor.
    """
    t0, t1, fr = (
        np.asarray(t0, float),
        np.asarray(t1, float),
        np.asarray(fr, float),
    )
    n = len(fr)
    if n < 4:
        return dict(
            w=np.zeros(0),
            y=np.zeros(0),
            p=np.zeros(0),
            q=np.zeros(0),
            i=np.zeros(0, int),
            j=np.zeros(0, int),
            n_trials=n,
            n_raw=0,
            frac_tied=np.nan,
        )
    sig = np.radians(sigma_deg)
    assignments = (
        (t0, t1, t0, t1),
        (t0, t1, t1, t0),
        (t1, t0, t0, t1),
        (t1, t0, t1, t0),
    )
    W, Y, Pp, Qq, I, J = [], [], [], [], [], []
    n_raw = n_tied = 0
    for lo in range(0, n, block):
        hi = min(lo + block, n)
        ii, jj = np.meshgrid(np.arange(lo, hi), np.arange(n), indexing="ij")
        keep = ii < jj  # each unordered pair exactly once
        ii, jj = ii[keep], jj[keep]
        if not len(ii):
            continue
        dr = fr[ii] - fr[jj]
        untied = dr != 0.0
        n_raw += 4 * len(ii)
        n_tied += 4 * int((~untied).sum())
        for a_i, o_i, a_j, o_j in assignments:
            ai, bi, aj, bj = a_i[ii], o_i[ii], a_j[jj], o_j[jj]
            w = np.exp(-circ_dist(ai, aj) ** 2 / (2 * sig**2))
            m = untied & (w >= weight_floor)
            if not m.any():
                continue
            ai, bi, aj, bj, w = ai[m], bi[m], aj[m], bj[m], w[m]
            abar = np.angle(np.exp(1j * ai) + np.exp(1j * aj))
            p, q = _predictors(bi - k * abar, bj - k * abar)
            W.append(w.astype(np.float32))
            Y.append((dr[m] > 0).astype(np.float32))
            Pp.append(p.astype(np.float32))
            Qq.append(q.astype(np.float32))
            I.append(ii[m].astype(np.int32))
            J.append(jj[m].astype(np.int32))
    if not W:
        return dict(
            w=np.zeros(0),
            y=np.zeros(0),
            p=np.zeros(0),
            q=np.zeros(0),
            i=np.zeros(0, int),
            j=np.zeros(0, int),
            n_trials=n,
            n_raw=n_raw,
            frac_tied=n_tied / max(n_raw, 1),
        )
    out = dict(
        w=np.concatenate(W),
        y=np.concatenate(Y),
        p=np.concatenate(Pp),
        q=np.concatenate(Qq),
        i=np.concatenate(I),
        j=np.concatenate(J),
        n_trials=n,
        n_raw=n_raw,
        frac_tied=n_tied / max(n_raw, 1),
    )
    return _subsample(out, max_comparisons, seed)


# ── Proximity: the same construction on the unsigned separation ───────────────
def build_separation_comparisons(
    t0,
    t1,
    fr,
    sigma_deg=SIGMA_DEG,
    weight_floor=WEIGHT_FLOOR,
    max_comparisons=MAX_COMPARISONS,
    seed=0,
    block=256,
):
    """Shared-anchor comparisons carrying each trial's **separation** rather than a direction.

    The proximity hypothesis is that the rate is a function of how far apart the two objects
    are, peaking at some preferred separation ``d*``. Of two trials, the one whose separation
    is nearer ``d*`` fired more — an ordinal prediction like the other two models', and the
    form proximity has to take here, the *signed* slope-+1 version having turned out to be
    self-contradictory under the four anchor assignments.

    Two things differ from :func:`build_comparisons`. A separation does not care which object
    is called the anchor, so the four assignments give **identical** predictors and differ
    only in their anchor-agreement weight: the four are summed into one row per trial pair
    rather than kept apart, which is both faster and exactly what the symmetry says. And the
    score is not linear in the parameter, so the rows carry the raw separations ``di``, ``dj``
    and the predictor is built per candidate ``d*`` by :func:`separation_predictor`.

    The comparison *set* is the one the other models see — same pairs, same Gaussian anchor
    weight — so held-out likelihoods stay commensurable. Proximity could in principle use
    every pair of trials, anchor-sharing or not; it is held to the same set deliberately.
    """
    t0, t1, fr = (
        np.asarray(t0, float),
        np.asarray(t1, float),
        np.asarray(fr, float),
    )
    n = len(fr)
    empty = dict(
        w=np.zeros(0),
        y=np.zeros(0),
        di=np.zeros(0),
        dj=np.zeros(0),
        i=np.zeros(0, int),
        j=np.zeros(0, int),
        n_trials=n,
        n_raw=0,
        frac_tied=np.nan,
    )
    if n < 4:
        return empty
    sig = np.radians(sigma_deg)
    d = circ_dist(t0, t1)  # each trial's own separation
    W, Y, Di, Dj, I, J = [], [], [], [], [], []
    n_raw = n_tied = 0
    for lo in range(0, n, block):
        hi = min(lo + block, n)
        ii, jj = np.meshgrid(np.arange(lo, hi), np.arange(n), indexing="ij")
        keep = ii < jj
        ii, jj = ii[keep], jj[keep]
        if not len(ii):
            continue
        dr = fr[ii] - fr[jj]
        untied = dr != 0.0
        n_raw += len(ii)
        n_tied += int((~untied).sum())
        w = np.zeros(len(ii))
        for a_i in (t0, t1):  # the four anchor assignments,
            for a_j in (t0, t1):  # which share one predictor
                w += np.exp(-circ_dist(a_i[ii], a_j[jj]) ** 2 / (2 * sig**2))
        m = untied & (w >= weight_floor)
        if not m.any():
            continue
        W.append(w[m].astype(np.float32))
        Y.append((dr[m] > 0).astype(np.float32))
        Di.append(d[ii[m]].astype(np.float32))
        Dj.append(d[jj[m]].astype(np.float32))
        I.append(ii[m].astype(np.int32))
        J.append(jj[m].astype(np.int32))
    if not W:
        return dict(empty, n_raw=n_raw, frac_tied=n_tied / max(n_raw, 1))
    out = dict(
        w=np.concatenate(W),
        y=np.concatenate(Y),
        di=np.concatenate(Di),
        dj=np.concatenate(Dj),
        i=np.concatenate(I),
        j=np.concatenate(J),
        n_trials=n,
        n_raw=n_raw,
        frac_tied=n_tied / max(n_raw, 1),
    )
    return _subsample(out, max_comparisons, seed)


def separation_predictor(c, star_deg, mask=None):
    """The score difference ``-|d_i - d*| + |d_j - d*|`` for one candidate separation.

    Divided by 90 deg so that a unit of it means what a unit of the cosine predictors means,
    which keeps the fitted sharpness on a comparable scale across the three models.
    """
    star = np.radians(float(star_deg))
    di, dj = (
        (c["di"], c["dj"]) if mask is None else (c["di"][mask], c["dj"][mask])
    )
    return ((np.abs(dj - star) - np.abs(di - star)) / (np.pi / 2)).astype(
        np.float64
    )


def _profile(c, grid, w, y, zeros, mask, positive_only):
    """Best ``(ll, star_deg, beta)`` over one grid of candidate separations."""
    best = (-np.inf, np.nan, np.nan)
    for star in grid:
        x = separation_predictor(c, star, mask)
        g = fit_irls(w, y, x, zeros)
        if not np.isfinite(g[0]) or (positive_only and g[0] <= 0):
            continue
        ll = mean_loglik(w, y, x, zeros, g)
        if ll > best[0]:
            best = (ll, float(star), float(g[0]))
    return best


def fit_separation(
    c, grid=SEP_GRID_DEG, mask=None, refine=SEP_REFINE_DEG, positive_only=True
):
    """Profile the preferred separation over ``grid``. Returns ``(star_deg, beta, ll)``.

    One free shape parameter and one sharpness — the same two the other models spend on a
    phase and a sharpness, so held-out likelihoods are compared at equal degrees of freedom.

    ``positive_only`` keeps only fits with a positive sharpness, so that "proximity" means the
    neuron **prefers** a separation. A negative sharpness is a real hypothesis — the neuron
    *avoids* that separation — but it is a different claim, and over a support that stops at
    80 deg the two readings become interchangeable near the ends of the range: in simulation a
    neuron built to prefer 90 deg fitted marginally better as one avoiding 162 deg. Admitting
    both would put that ambiguity inside the model comparison.

    ``refine`` runs a second, finer profile around the winner, so a coarse ``grid`` costs a
    handful of extra fits instead of a fine grid's worth. The bootstrap re-profiles inside
    every resample, so this is the difference between a two-hour cache build and a half-hour
    one. It was checked against a fine grid when it was written: the refined coarse estimate
    agreed to within the grid's own step.
    """
    w = c["w"] if mask is None else c["w"][mask]
    y = c["y"] if mask is None else c["y"][mask]
    if len(w) < 40:
        return np.nan, np.nan, np.nan
    zeros = np.zeros(len(w))
    best = _profile(c, grid, w, y, zeros, mask, positive_only)
    if refine and np.isfinite(best[1]):
        step = (
            float(np.min(np.diff(np.sort(np.asarray(grid, float)))))
            if len(grid) > 1
            else 0
        )
        fine = np.arange(best[1] - step / 2, best[1] + step / 2 + 1e-9, refine)
        fine = fine[(fine > 0.0) & (fine < 180.0)]
        if len(fine):
            alt = _profile(c, fine, w, y, zeros, mask, positive_only)
            if alt[0] > best[0]:
                best = alt
    return best[1], best[2], best[0]


def cv_per_comparison_separation(c, grid=SEP_GRID_DEG, n_folds=5, seed=0):
    """Held-out log-likelihood per comparison, with ``d*`` chosen **inside** each fold.

    Selecting the separation on the training folds is what keeps the comparison with the other
    two models fair. Choosing it on all the data and then scoring held-out comparisons would
    let proximity see the test set through its one shape parameter, and it wins neurons it
    should not.
    """
    if len(c["w"]) < 40:
        return (np.zeros(0),) * 4
    LL, W, I, J = [], [], [], []
    for tr, te in fold_masks(c, n_folds, seed):
        if tr.sum() < 20 or te.sum() < 10:
            continue
        star, beta, _ = fit_separation(c, grid, mask=tr)
        if not np.isfinite(star):
            continue
        with np.errstate(all="ignore"):
            mu = _sigmoid(separation_predictor(c, star, te) * beta)
        mu = np.clip(mu, 1e-12, 1 - 1e-12)
        yt = c["y"][te]
        LL.append(yt * np.log(mu) + (1 - yt) * np.log(1 - mu))
        W.append(c["w"][te])
        I.append(c["i"][te])
        J.append(c["j"][te])
    if not LL:
        return (np.zeros(0),) * 4
    return (
        np.concatenate(LL),
        np.concatenate(W),
        np.concatenate(I),
        np.concatenate(J),
    )


def cv_delta_ll_separation(c, grid=SEP_GRID_DEG, n_folds=5, seed=0):
    """Held-out nats per comparison above chance, on the same folds as the other models."""
    ll, w, _, _ = cv_per_comparison_separation(c, grid, n_folds, seed)
    if not len(ll):
        return np.nan
    return float((w * ll).sum() / w.sum()) - NULL_LL


def bootstrap_separation(c, grid=SEP_GRID_DEG, n_boot=N_BOOT, seed=0):
    """``(n_boot, 2)`` of ``(star_deg, beta)`` from resampling **trials** with replacement.

    The separation is re-profiled inside every resample, because it is a fitted parameter and
    an interval that held it fixed would describe a different estimator. That is what makes
    proximity's bootstrap cost a multiple of the other models'.
    """
    n = c["n_trials"]
    out = np.full((n_boot, 2), np.nan)
    if len(c["w"]) < 40:
        return out
    rng = np.random.default_rng(seed)
    for b in range(n_boot):
        m = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float32)
        ww = c["w"] * m[c["i"]] * m[c["j"]]
        k = ww > 0
        if k.sum() < 20:
            continue
        cb = dict(c, w=ww)
        star, beta, _ = fit_separation(cb, grid, mask=k)
        out[b] = (star, beta)
    return out


def build_oneobj_comparisons(
    theta, fr, max_comparisons=MAX_COMPARISONS, seed=0
):
    """The same construction for **1-object** trials, where there is no anchor.

    With one object there is nothing to condition on, so every pair of trials is comparable and
    every weight is 1. The model is the plain preferred direction: the trial whose object sat
    nearer it should have fired more. This is what gives the receptive-field-orientation figure
    a reference direction measured by the *same* estimator as the 2-object phase it is
    subtracted from — which the parametric version, fitting a cosine here and a GLM there, does
    not have.
    """
    theta, fr = np.asarray(theta, float), np.asarray(fr, float)
    n = len(fr)
    if n < 4:
        return dict(
            w=np.zeros(0),
            y=np.zeros(0),
            p=np.zeros(0),
            q=np.zeros(0),
            i=np.zeros(0, int),
            j=np.zeros(0, int),
            n_trials=n,
            n_raw=0,
            frac_tied=np.nan,
        )
    i, j = np.triu_indices(n, 1)
    dr = fr[i] - fr[j]
    m = dr != 0.0
    p, q = _predictors(theta[i][m], theta[j][m])
    out = dict(
        w=np.ones(int(m.sum()), np.float32),
        y=(dr[m] > 0).astype(np.float32),
        p=p.astype(np.float32),
        q=q.astype(np.float32),
        i=i[m].astype(np.int32),
        j=j[m].astype(np.int32),
        n_trials=n,
        n_raw=len(i),
        frac_tied=float((~m).mean()),
    )
    return _subsample(out, max_comparisons, seed)


def _subsample(c, max_comparisons, seed):
    """Uniform, seeded subsample when a neuron has more comparisons than the cap."""
    n = len(c["w"])
    if not max_comparisons or n <= max_comparisons:
        c["n_used"] = n
        c["subsampled"] = False
        return c
    idx = np.random.default_rng(seed).choice(n, max_comparisons, replace=False)
    out = {
        k: (v[idx] if isinstance(v, np.ndarray) and v.shape[:1] == (n,) else v)
        for k, v in c.items()
    }
    out["n_used"] = int(max_comparisons)
    out["subsampled"] = True
    return out


# ── Fitting ───────────────────────────────────────────────────────────────────
def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * np.clip(z, -40.0, 40.0)))


def fit_irls(w, y, p, q, gamma0=None, max_iter=25, tol=1e-8, ridge=1e-9):
    """Weighted logistic regression of ``y`` on ``(p, q)`` with no intercept. Returns ``gamma``.

    The objective is convex, so this converges from anywhere; ``gamma0`` is a warm start, used
    by the bootstrap, where the previous full-data fit is a very good guess and cuts the
    iteration count by roughly four.
    """
    if len(w) < 8:
        return np.full(2, np.nan)
    X = np.column_stack([p, q]).astype(np.float64)
    y = np.asarray(y, np.float64)
    w = np.asarray(w, np.float64)
    g = np.zeros(2) if gamma0 is None else np.array(gamma0, float)
    if not np.all(np.isfinite(g)):
        g = np.zeros(2)
    if not (
        np.all(np.isfinite(X))
        and np.all(np.isfinite(w))
        and np.all(np.isfinite(y))
    ):
        return np.full(2, np.nan)
    for _ in range(max_iter):
        with np.errstate(
            all="ignore"
        ):  # macOS BLAS raises spurious matmul FP flags
            mu = _sigmoid(X @ g)
            s = np.maximum(mu * (1.0 - mu), 1e-6)
            ws = w * s
            H = X.T @ (ws[:, None] * X) + ridge * np.eye(2)
            grad = X.T @ (w * (y - mu))
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            return np.full(2, np.nan)
        if not np.all(np.isfinite(step)):
            return np.full(2, np.nan)
        g = g + step
        if np.max(np.abs(step)) < tol:
            break
    return g


def mean_loglik(w, y, p, q, gamma):
    """Weighted mean log-likelihood per comparison. ``gamma = 0`` gives ``NULL_LL``."""
    if len(w) == 0 or not np.all(np.isfinite(gamma)):
        return np.nan
    X = np.column_stack([p, q]).astype(np.float64)
    if not np.all(np.isfinite(X)):
        return np.nan
    with np.errstate(
        all="ignore"
    ):  # macOS BLAS raises spurious matmul FP flags
        mu = _sigmoid(X @ np.asarray(gamma, float))
    mu = np.clip(mu, 1e-12, 1 - 1e-12)
    w = np.asarray(w, np.float64)
    return float(
        (w * (y * np.log(mu) + (1 - y) * np.log(1 - mu))).sum() / w.sum()
    )


def gamma_to_phase(gamma, period=360.0):
    """``(phase_deg, beta)``. ``period`` 360 reports a direction, 180 an orientation."""
    if not np.all(np.isfinite(gamma)):
        return np.nan, np.nan
    return float(np.degrees(np.arctan2(gamma[1], gamma[0])) % period), float(
        np.hypot(*gamma)
    )


# ── Cross-validation ──────────────────────────────────────────────────────────
def fold_masks(c, n_folds, seed=0):
    """Train/test masks over **comparisons**, from folds taken over **trials**.

    A trial appears in thousands of comparisons, so folding comparisons would put the same
    trial on both sides and leak. Comparisons that straddle the boundary — one trial in the
    fold, one out — are in neither mask: they contain a trial the fit saw, and are the leak.
    With five folds this discards about a third of them and still tests every comparison whose
    two trials land in the same fold, once each.
    """
    rng = np.random.default_rng(seed)
    fold = rng.permutation(c["n_trials"]) % n_folds
    for f in range(n_folds):
        te = fold == f
        yield (~te[c["i"]]) & (~te[c["j"]]), te[c["i"]] & te[c["j"]]


def cv_per_comparison(c, n_folds=5, seed=0):
    """Per-comparison held-out log-likelihood, and which comparisons were scored.

    Returns ``(ll, w, i, j)`` over the test comparisons of every fold, each scored by the fit
    from the folds that excluded both of its trials. Keeping the per-comparison values, rather
    than only their mean, is what lets two models be compared **paired** and bootstrapped
    without refitting — see :func:`paired_bootstrap`.

    Both models must be given the same ``seed`` so they share folds, or the pairing is a
    fiction.
    """
    if len(c["w"]) < 40:
        return (np.zeros(0),) * 4
    LL, W, I, J = [], [], [], []
    for tr, te in fold_masks(c, n_folds, seed):
        if tr.sum() < 20 or te.sum() < 10:
            continue
        g = fit_irls(c["w"][tr], c["y"][tr], c["p"][tr], c["q"][tr])
        if not np.all(np.isfinite(g)):
            continue
        with np.errstate(all="ignore"):
            mu = _sigmoid(
                np.column_stack([c["p"][te], c["q"][te]]).astype(np.float64)
                @ g
            )
        mu = np.clip(mu, 1e-12, 1 - 1e-12)
        yt = c["y"][te]
        LL.append(yt * np.log(mu) + (1 - yt) * np.log(1 - mu))
        W.append(c["w"][te])
        I.append(c["i"][te])
        J.append(c["j"][te])
    if not LL:
        return (np.zeros(0),) * 4
    return (
        np.concatenate(LL),
        np.concatenate(W),
        np.concatenate(I),
        np.concatenate(J),
    )


def cv_delta_ll(c, n_folds=5, seed=0):
    """Held-out mean log-likelihood per comparison, **above the null**.

    Positive means the model predicts orderings of held-out trials better than a coin.
    """
    ll, w, _, _ = cv_per_comparison(c, n_folds, seed)
    if not len(w) or w.sum() <= 0:
        return np.nan
    return float((w * ll).sum() / w.sum()) - NULL_LL


def paired_bootstrap(ll_a, ll_b, w, i, j, n_trials, n_boot=N_BOOT, seed=0):
    """Bootstrap the paired difference of two models' held-out losses. ``(D, p, lo, hi)``.

    ``D`` is the weighted mean of ``ll_a - ll_b`` over the comparisons both models were scored
    on — the quantity whose sign names the winner. Resampling is over **trials**, the same
    ``w * m_i * m_j`` reweighting :func:`bootstrap_gamma` uses, because a trial appears in
    thousands of comparisons and resampling comparisons would badly understate the spread.

    ``p`` is two-sided, from the bootstrap **standard deviation** under a normal
    approximation, not from the tail fraction. The tail fraction has a hard resolution floor of
    ``1/(n_boot+1)`` — about 0.02 at the default 50 — and Benjamini-Hochberg across ~1300
    neurons needs p-values far below that, so a tail-based p would make the correction reject
    either nothing or everything depending on how many neurons piled onto the floor. The
    approximation is safe here and was checked rather than assumed: on real neurons the
    bootstrap distribution of ``D`` has |skew| < 0.1 and |excess kurtosis| < 0.2, and the two
    p-values agree to about 0.01 where both can be computed. It is a weighted mean over ~10^5
    comparisons resampled at the trial level, so the central limit theorem is doing exactly
    what it should. The percentile interval is returned alongside, and is what the figure
    draws.

    **What this interval is.** The fold fits are held fixed and only the evaluation set is
    resampled, so it measures the variability of the *evaluation*, not of the *fitting*. That
    is the standard paired held-out-loss comparison, and it is mildly **anticonservative** —
    narrower than a bootstrap that refit inside each resample (~6x the cost). ``validate_paired
    _bootstrap.py`` measures the gap on a sample of real neurons; the caption should say which
    version the figure used.
    """
    if not len(w) or w.sum() <= 0:
        return np.nan, np.nan, np.nan, np.nan
    d = np.asarray(ll_a, np.float64) - np.asarray(ll_b, np.float64)
    w = np.asarray(w, np.float64)
    D = float((w * d).sum() / w.sum())
    if n_boot < 2:
        return D, np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    out = np.full(n_boot, np.nan)
    for b in range(n_boot):
        m = np.bincount(
            rng.integers(0, n_trials, n_trials), minlength=n_trials
        ).astype(np.float64)
        ww = w * m[i] * m[j]
        tot = ww.sum()
        if tot > 0:
            out[b] = (ww * d).sum() / tot
    out = out[np.isfinite(out)]
    if len(out) < 2:
        return D, np.nan, np.nan, np.nan
    sd = float(np.std(out, ddof=1))
    p = float(2.0 * _norm_sf(abs(D) / sd)) if sd > 0 else np.nan
    lo, hi = (float(v) for v in np.percentile(out, [2.5, 97.5]))
    return D, p, lo, hi


def paired_bootstrap_unaligned(a, b, n_trials, n_boot=N_BOOT, seed=0):
    """The paired difference of two models scored on **different** comparison sets.

    :func:`paired_bootstrap` differences two models row for row, which they can only be when
    they were built from the same rows. Proximity is not: a separation does not care which
    object is the anchor, so its rows are one per trial pair where the slope models' are four.
    The pairing that matters is still there — the same trials, the same folds — so each
    resample draws one set of trial multiplicities and applies it to **both** models' own rows,
    differencing the two weighted means rather than the rows.

    ``a`` and ``b`` are each ``(ll, w, i, j)`` as returned by :func:`cv_per_comparison`.
    Returns ``(D, p, lo, hi)`` with the same meaning, and the same evaluation-only caveat.
    """
    (lla, wa, ia, ja), (llb, wb, ib, jb) = a, b
    if not len(wa) or not len(wb) or wa.sum() <= 0 or wb.sum() <= 0:
        return np.nan, np.nan, np.nan, np.nan
    lla, wa = np.asarray(lla, np.float64), np.asarray(wa, np.float64)
    llb, wb = np.asarray(llb, np.float64), np.asarray(wb, np.float64)
    D = float((wa * lla).sum() / wa.sum() - (wb * llb).sum() / wb.sum())
    if n_boot < 2:
        return D, np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    out = np.full(n_boot, np.nan)
    for k in range(n_boot):
        m = np.bincount(
            rng.integers(0, n_trials, n_trials), minlength=n_trials
        ).astype(np.float64)
        wwa, wwb = wa * m[ia] * m[ja], wb * m[ib] * m[jb]
        ta, tb = wwa.sum(), wwb.sum()
        if ta > 0 and tb > 0:
            out[k] = (wwa * lla).sum() / ta - (wwb * llb).sum() / tb
    out = out[np.isfinite(out)]
    if len(out) < 2:
        return D, np.nan, np.nan, np.nan
    sd = float(np.std(out, ddof=1))
    p = float(2.0 * _norm_sf(abs(D) / sd)) if sd > 0 else np.nan
    lo, hi = (float(v) for v in np.percentile(out, [2.5, 97.5]))
    return D, p, lo, hi


def _norm_sf(z):
    """Upper tail of the standard normal, via erfc — no scipy import on the hot path."""
    from math import erfc, sqrt

    return 0.5 * erfc(z / sqrt(2.0))


# ── Bootstrap and permutation ─────────────────────────────────────────────────
def bootstrap_gamma(c, n_boot=N_BOOT, seed=0, gamma0=None):
    """``(n_boot, 2)`` fits from resampling **trials** with replacement.

    A resample with trial multiplicities ``m`` turns each comparison's weight into
    ``w · m_i · m_j``, so the predictors never have to be rebuilt — the bootstrap is a
    reweighting of an array computed once, which is what makes a resample cost a fraction of
    the original fit. Self-pairs carry no ordering and drop out automatically, since a
    comparison needs two distinct trials.

    See :data:`N_BOOT` for what the default buys and what it costs.
    """
    n = c["n_trials"]
    out = np.full((n_boot, 2), np.nan)
    if len(c["w"]) < 40:
        return out
    rng = np.random.default_rng(seed)
    for b in range(n_boot):
        m = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float32)
        ww = c["w"] * m[c["i"]] * m[c["j"]]
        k = ww > 0
        if k.sum() < 20:
            continue
        out[b] = fit_irls(
            ww[k], c["y"][k], c["p"][k], c["q"][k], gamma0=gamma0
        )
    return out


def wald_p(gamma, boots):
    """Two-sided p for ``gamma != 0``, from the bootstrap covariance of ``gamma``.

    ``gamma`` is two-dimensional, so the natural test is a Wald statistic
    ``gamma' S^-1 gamma`` against chi-squared with two degrees of freedom, with ``S`` the
    bootstrap covariance — **not** the Hessian, which is meaningless for a composite
    likelihood (see the module docstring).

    This tests whether the neuron has a *determinable phase* under this model. That is the
    right question for the receptive-field-orientation figure, whose plotted quantity is an
    angle: a neuron with no determinable phase contributes a random angle at full weight. It is
    deliberately not a test of which model wins.
    """
    from scipy.stats import chi2

    g = np.asarray(gamma, float)
    b = np.asarray(boots, float)
    b = b[np.isfinite(b).all(axis=1)]
    if not np.all(np.isfinite(g)) or len(b) < 10:
        return np.nan
    S = np.cov(b.T)
    if not np.all(np.isfinite(S)):
        return np.nan
    try:
        stat = float(g @ np.linalg.solve(S + 1e-12 * np.eye(2), g))
    except np.linalg.LinAlgError:
        return np.nan
    return float(chi2.sf(max(stat, 0.0), df=2))


# ── One neuron, end to end ────────────────────────────────────────────────────
def fit_neuron(
    t0,
    t1,
    fr2,
    theta1=None,
    fr1=None,
    sigma_deg=SIGMA_DEG,
    n_folds=5,
    n_boot=N_BOOT,
    seed=0,
    max_comparisons=MAX_COMPARISONS,
    with_proximity=True,
    sep_grid=SEP_GRID_DEG,
    sep_max_comparisons=SEP_MAX_COMPARISONS,
):
    """Everything the figures need for one neuron, as a flat dict.

    Per 2-object model: the all-data fit (phase, beta), the held-out delta log-likelihood, the
    bootstrap phase mean and circular s.d., and the Wald p for a determinable phase. Plus the
    same for the 1-object receptive field when ``theta1``/``fr1`` are given, and the paired
    ``dll_relational_retinotopic`` whose sign is the winner.

    The phase is reported in the coordinate the figures plot. The relational model's ``c`` is
    the phase of a *sum*, so its receptive-field axis is ``(c + 180)/2`` and is defined modulo
    180 — that conversion happens here, once, rather than in each notebook.
    """
    out = {"sigma_deg": sigma_deg, "n_folds": n_folds, "n_boot": n_boot}
    per_comp = {}
    for k in MODELS:
        name = MODEL_NAME[k]
        c = build_comparisons(
            t0,
            t1,
            fr2,
            k,
            sigma_deg,
            seed=seed,
            max_comparisons=max_comparisons,
        )
        out[f"{name}_n_comparisons"] = len(c["w"])
        out[f"{name}_frac_tied"] = c["frac_tied"]
        out[f"{name}_subsampled"] = bool(c.get("subsampled", False))
        g = fit_irls(c["w"], c["y"], c["p"], c["q"])
        phase, beta = gamma_to_phase(g)
        out[f"{name}_phase_deg"] = phase
        out[f"{name}_beta"] = beta
        ll_c, w_c, i_c, j_c = cv_per_comparison(c, n_folds, seed)
        per_comp[name] = (ll_c, w_c, i_c, j_c)
        out[f"{name}_dll"] = (
            float((w_c * ll_c).sum() / w_c.sum()) - NULL_LL
            if len(w_c) and w_c.sum() > 0
            else np.nan
        )
        boots = bootstrap_gamma(c, n_boot, seed=seed + 1, gamma0=g)
        out[f"{name}_p"] = wald_p(g, boots)
        # The plotted coordinate: a direction for the retinotopic model, an axis (period 180,
        # after halving the sum-phase) for the relational one.
        if k == 0:
            ph = np.array([gamma_to_phase(b)[0] for b in boots])
            out[f"{name}_axis_deg"] = phase
            out[f"{name}_axis_sd_deg"] = circ_sd_deg(ph, 360.0)
            out[f"{name}_period"] = 360.0
        else:
            ph = np.array(
                [(gamma_to_phase(b)[0] + 180.0) / 2.0 % 180.0 for b in boots]
            )
            out[f"{name}_axis_deg"] = (phase + 180.0) / 2.0 % 180.0
            out[f"{name}_axis_sd_deg"] = circ_sd_deg(ph, 180.0)
            out[f"{name}_period"] = 180.0
        out[f"{name}_boot_phase"] = ph
    out["dll_relational_retinotopic"] = (
        out["relational_dll"] - out["retinotopic_dll"]
    )

    # ── Proximity: preferred separation, on the same trials and the same folds ──
    if with_proximity:
        cs = build_separation_comparisons(
            t0,
            t1,
            fr2,
            sigma_deg,
            seed=seed,
            max_comparisons=sep_max_comparisons,
        )
        star, beta, _ = fit_separation(cs, sep_grid)
        out["proximity_n_comparisons"] = len(cs["w"])
        out["proximity_frac_tied"] = cs["frac_tied"]
        out["proximity_subsampled"] = bool(cs.get("subsampled", False))
        out["proximity_sep_deg"] = star
        out["proximity_beta"] = beta
        ll_c, w_c, i_c, j_c = cv_per_comparison_separation(
            cs, sep_grid, n_folds, seed
        )
        per_comp["proximity"] = (ll_c, w_c, i_c, j_c)
        out["proximity_dll"] = (
            float((w_c * ll_c).sum() / w_c.sum()) - NULL_LL
            if len(w_c) and w_c.sum() > 0
            else np.nan
        )
        b_sep = bootstrap_separation(cs, sep_grid, n_boot, seed=seed + 1)
        stars = b_sep[:, 0][np.isfinite(b_sep[:, 0])]
        # No Wald test here: the separation was *selected*, so the 2-df chi-squared the other
        # models use does not apply. The bootstrap answers the same question directly — how
        # often the resamples fail to find a preferred separation at all.
        out["proximity_sep_sd_deg"] = (
            float(np.std(stars)) if len(stars) > 1 else np.nan
        )
        out["proximity_p"] = (
            float(1.0 - len(stars) / max(n_boot, 1)) if n_boot else np.nan
        )
        out["proximity_boot_sep"] = b_sep[:, 0]
    else:
        for key in (
            "n_comparisons",
            "frac_tied",
            "sep_deg",
            "beta",
            "dll",
            "sep_sd_deg",
            "p",
        ):
            out[f"proximity_{key}"] = np.nan
        out["proximity_subsampled"] = False
        out["proximity_boot_sep"] = np.full(n_boot, np.nan)

    # ── Who wins, both ways: the argmax, and the margin over the runner-up ──
    scored = {
        m: out[f"{m}_dll"]
        for m in ALL_MODELS
        if m in per_comp and np.isfinite(out.get(f"{m}_dll", np.nan))
    }
    order = sorted(scored, key=scored.get, reverse=True)
    out["best_model"] = order[0] if order else "none"
    out["runner_up_model"] = order[1] if len(order) > 1 else "none"
    if len(order) > 1:
        D, pv, lo, hi = paired_bootstrap_unaligned(
            per_comp[order[0]],
            per_comp[order[1]],
            len(fr2),
            n_boot=n_boot,
            seed=seed + 4,
        )
        out["margin"] = D
        out["margin_p"] = pv
        out["margin_lo"], out["margin_hi"] = lo, hi
    else:
        out["margin"] = out["margin_p"] = out["margin_lo"] = out[
            "margin_hi"
        ] = np.nan

    # The paired test. Both models were scored on the same folds, so their per-comparison
    # losses line up row for row and the difference can be bootstrapped directly.
    rel, ret = per_comp["relational"], per_comp["retinotopic"]
    if len(rel[0]) and len(rel[0]) == len(ret[0]):
        D, p, lo, hi = paired_bootstrap(
            rel[0],
            ret[0],
            rel[1],
            rel[2],
            rel[3],
            len(fr2),
            n_boot=n_boot,
            seed=seed + 3,
        )
        out["dll_paired"] = D
        out["dll_p"] = p
        out["dll_lo"], out["dll_hi"] = lo, hi
    else:
        out["dll_paired"] = out["dll_p"] = out["dll_lo"] = out["dll_hi"] = (
            np.nan
        )
    # The guard: a neuron cannot be decisive on the difference between two failures.
    out["winner_beats_null"] = bool(
        np.isfinite(out[f'{out["best_model"]}_dll'])
        and out[f'{out["best_model"]}_dll'] > 0
    )

    if theta1 is not None and fr1 is not None and len(fr1) >= 8:
        c1 = build_comparisons_oneobj = build_oneobj_comparisons(
            theta1, fr1, max_comparisons=max_comparisons, seed=seed
        )
        g1 = fit_irls(c1["w"], c1["y"], c1["p"], c1["q"])
        ph1, beta1 = gamma_to_phase(g1)
        b1 = bootstrap_gamma(c1, n_boot, seed=seed + 2, gamma0=g1)
        out["oneobj_phase_deg"] = ph1
        out["oneobj_beta"] = beta1
        out["oneobj_dll"] = cv_delta_ll(c1, n_folds, seed)
        out["oneobj_p"] = wald_p(g1, b1)
        out["oneobj_sd_deg"] = circ_sd_deg(
            np.array([gamma_to_phase(b)[0] for b in b1]), 360.0
        )
        out["oneobj_n_comparisons"] = len(c1["w"])
        out["oneobj_frac_tied"] = c1["frac_tied"]
        out["n_1obj"] = len(fr1)
    else:
        for key in (
            "phase_deg",
            "beta",
            "dll",
            "p",
            "sd_deg",
            "n_comparisons",
            "frac_tied",
        ):
            out[f"oneobj_{key}"] = np.nan
        out["n_1obj"] = 0 if fr1 is None else len(fr1)
    out["n_2obj"] = len(fr2)
    return out

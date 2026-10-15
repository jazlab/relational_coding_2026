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

(Proximity is not a slope: its signed form, k = +1, is self-contradictory under the four
anchor assignments. It is tuned to a preferred separation instead — see
:func:`build_separation_comparisons`.)

What reads this module
----------------------
Two things, and they use different halves of it.

* `comparison_lib.py` — **figure 3's model comparison**. It takes the comparisons built here
  and scores each model on the sign of ``Δs`` alone: no likelihood, no fitted sharpness, one
  parameter per model. That is the published comparison.
* `run_phase_cache.py` — the **phase fit** behind panel 3e. It needs the fitted tuning
  *vector* ``γ`` and a bootstrap distribution for its angle, which the sign-only fit does not
  estimate, so it uses the Bradley-Terry fit described below.

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
#: bootstrap are what set the floor: `wald_p` estimates a 2x2 covariance from this many points,
#: and is read *per neuron* by the exclusion criteria, and `circ_sd_deg` a spread. At 20
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

#: Every model the comparison scores, in the order the figures list them. Proximity is not a
#: slope, so it has no ``k``: see :func:`build_separation_comparisons`.
ALL_MODELS = ("retinotopic", "relational", "proximity")


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

    It says how well the phase is *determined*, in the units the orientation histogram is drawn
    in.
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
    does not define. Both slope models are invariant to it, so the four agree rather than
    cancelling.

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
    form proximity has to take here, since the *signed* slope-+1 version is self-contradictory
    under the four anchor assignments.

    Two things differ from :func:`build_comparisons`. A separation does not care which object
    is called the anchor, so the four assignments give **identical** predictors and differ
    only in their anchor-agreement weight: the four are summed into one row per trial pair
    rather than kept apart, which is both faster and exactly what the symmetry says. And the
    score is not linear in the parameter, so the rows carry the raw separations ``di``, ``dj``
    and the prediction is made per candidate ``d*`` (`comparison_lib.fit_separation_sign`).

    The comparison *set* is the one the other models see — same pairs, same Gaussian anchor
    weight — so held-out scores stay commensurable. Proximity could in principle use every
    pair of trials, anchor-sharing or not; it is held to the same set deliberately.
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


def build_oneobj_comparisons(
    theta, fr, max_comparisons=MAX_COMPARISONS, seed=0
):
    """The same construction for **1-object** trials, where there is no anchor.

    With one object there is nothing to condition on, so every pair of trials is comparable and
    every weight is 1. The model is the plain preferred direction: the trial whose object sat
    nearer it should have fired more. This is what gives the receptive-field-orientation figure
    a reference direction measured by the *same* estimator as the 2-object phase it is
    subtracted from.
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
    """Weighted mean log-likelihood per comparison. ``gamma = 0`` gives ``log(1/2)``."""
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
    n_boot=N_BOOT,
    seed=0,
    max_comparisons=MAX_COMPARISONS,
):
    """Everything the phase figure needs for one neuron, as a flat dict.

    Per 2-object slope model: the all-data phase, its bootstrap distribution, and the Wald p
    for a determinable phase. Plus the same for the 1-object receptive field when
    ``theta1``/``fr1`` are given.

    The phase is reported in the coordinate the figures plot. The relational model's ``c`` is
    the phase of a *sum*, so its receptive-field axis is ``(c + 180)/2`` and is defined modulo
    180 — that conversion happens here, once, rather than in each notebook.

    Every random draw has its own generator seeded from ``seed`` (``seed`` for the comparison
    subsample, ``seed + 1`` for the 2-object bootstraps, ``seed + 2`` for the 1-object one), so
    no quantity's draws depend on what else is computed.
    """
    out = {"sigma_deg": sigma_deg, "n_boot": n_boot}
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
        out[f"{name}_subsampled"] = bool(c.get("subsampled", False))
        g = fit_irls(c["w"], c["y"], c["p"], c["q"])
        phase, _ = gamma_to_phase(g)
        out[f"{name}_phase_deg"] = phase
        boots = bootstrap_gamma(c, n_boot, seed=seed + 1, gamma0=g)
        out[f"{name}_p"] = wald_p(g, boots)
        # The plotted coordinate: a direction for the retinotopic model, an axis (period 180,
        # after halving the sum-phase) for the relational one.
        if k == 0:
            ph = np.array([gamma_to_phase(b)[0] for b in boots])
            out[f"{name}_axis_deg"] = phase
            out[f"{name}_period"] = 360.0
        else:
            ph = np.array(
                [(gamma_to_phase(b)[0] + 180.0) / 2.0 % 180.0 for b in boots]
            )
            out[f"{name}_axis_deg"] = (phase + 180.0) / 2.0 % 180.0
            out[f"{name}_period"] = 180.0
        out[f"{name}_boot_phase"] = ph

    if theta1 is not None and fr1 is not None and len(fr1) >= 8:
        c1 = build_oneobj_comparisons(
            theta1, fr1, max_comparisons=max_comparisons, seed=seed
        )
        g1 = fit_irls(c1["w"], c1["y"], c1["p"], c1["q"])
        ph1, _ = gamma_to_phase(g1)
        b1 = bootstrap_gamma(c1, n_boot, seed=seed + 2, gamma0=g1)
        out["oneobj_phase_deg"] = ph1
        out["oneobj_p"] = wald_p(g1, b1)
        out["oneobj_sd_deg"] = circ_sd_deg(
            np.array([gamma_to_phase(b)[0] for b in b1]), 360.0
        )
        out["oneobj_n_comparisons"] = len(c1["w"])
        out["n_1obj"] = len(fr1)
    else:
        for key in ("phase_deg", "p", "sd_deg", "n_comparisons"):
            out[f"oneobj_{key}"] = np.nan
        out["n_1obj"] = 0 if fr1 is None else len(fr1)
    out["n_2obj"] = len(fr2)
    return out

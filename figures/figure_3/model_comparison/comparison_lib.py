r"""Figure 3's model comparison: each model scored on the orderings it predicts, and on
nothing else.

What it does
------------
:mod:`pairs_lib` turns a session into **comparisons** -- pairs of 2-object trials that share an
anchor, so the two differ in one object's position and are otherwise matched. Each carries
``y = 1{r_i > r_j}``, the sign of the firing-rate difference, with the size of that difference
never entering. This module scores a model by how many of those orderings it gets right:

    predict  sign(Delta_s),   score = weighted fraction of held-out comparisons called correctly

Chance is 1/2, and 1/2 is what a constant-rate neuron scores, so the null is built into the
scale. Nothing about the shape of a tuning curve survives this -- only the *ranking* it induces
-- so the comparison is invariant to any monotone reshaping of the tuning function, and the
three models are commensurable however differently shaped their predictions are.

Only the sign of ``Delta_s`` is used because its magnitude would be a **tuning-shape
assumption**: ``cos(u_i - c) - cos(u_j - c)`` asserts not just which trial fires more but by how
much, graded like a cosine, and that shape differs between proximity (a tent in separation) and
the two slope models (cosines in an angle).

The anchor weights stay analog (see :func:`pairs_lib.build_comparisons`). They say which pairs
of trials are comparable, not what a model predicts, so they carry no tuning assumption.

The fit is exact, and there is no grid
--------------------------------------
Write ``Delta_s = p cos c + q sin c = R cos(theta - c)`` with ``R = hypot(p, q)`` and
``theta = atan2(q, p)``. Then

    sign(Delta_s) = sign(cos(theta - c))

so a comparison is called **correctly** exactly when ``c`` lies within a quarter turn of
``theta`` (if ``y = 1``) or of ``theta + pi`` (if ``y = 0``). Every comparison therefore
contributes its weight on a **half-circle arc** of ``c``, and weighted accuracy is a sum of
weighted arc indicators: a step function whose exact maximiser is found by sorting the ``2N`` arc
endpoints and sweeping. No grid, no optimiser, no local optima, and ``O(N log N)``.

Proximity is the same algorithm on a line. ``|d_i - d*| < |d_j - d*|`` flips exactly when ``d*``
crosses the midpoint ``m = (d_i + d_j)/2``, so each comparison is correct on a **half-line** of
``d*``; sort the midpoints and sweep. Its "prefers ``d*``" and "avoids ``d*``" branches are
complements of each other on the decided set, so the polarity comes free: there is no
positive-only restriction, and all three models search their full hypothesis space.

Each model spends exactly **one** parameter (``c`` or ``d*`` plus a polarity that is free for the
slope models, being ``c -> c + pi``). There is no ``beta``.

What is lost
------------
Power. A comparison whose two stimuli sit nearly equidistant from ``c`` is a coin flip, and an
accuracy counts it at full weight as a 50/50 win or loss where a likelihood would correctly
discount it. Accuracy is therefore diluted by uninformative comparisons, and effect sizes are
smaller than a graded score's would be. The pairwise disagreement test
(:func:`disagreement_pieces`) is the answer where the question is *which model is right*: it
scores only the comparisons on which two models predict **different** orderings, which is
where all the discriminating information is.
"""

import numpy as np
import pairs_lib as pairs

MODELS = pairs.ALL_MODELS  # retinotopic, relational, proximity
CHANCE = 0.5

#: Trial resamples behind every uncertainty here. Higher than `pairs_lib.N_BOOT` because
#: nothing is refitted inside a resample -- the held-out correctness of each comparison is
#: already known, and a resample only reweights it -- so this costs a few array operations
#: rather than a fit. It sets the resolution of the p-values: 1/(N_BOOT + 1). The published
#: cache uses 1000.
N_BOOT = 1000


# ── The slope models: arcs on the circle ──────────────────────────────────────
def _slope_arc_centres(c, mask=None):
    """``(centre, weight, decided)``: the half-circle of ``c`` on which each comparison is
    called correctly, centred on ``theta`` when the first trial fired more and on
    ``theta + pi`` when it did not."""
    sel = slice(None) if mask is None else mask
    p = np.asarray(c["p"][sel], float)
    q = np.asarray(c["q"][sel], float)
    y = np.asarray(c["y"][sel], float)
    w = np.asarray(c["w"][sel], float)
    decided = np.hypot(p, q) > 0.0
    theta = np.arctan2(q, p)
    centre = np.where(y > 0.5, theta, theta + np.pi) % (2.0 * np.pi)
    return centre, w, decided


def fit_slope_sign(c, mask=None):
    """The ``c`` maximising weighted accuracy, exactly. Returns ``(c_hat, acc_train)``.

    A sweep over the ``2N`` arc endpoints. The maximum is attained on an interval; its midpoint
    is returned, which is the most robust point of the plateau rather than an edge of it.
    """
    centre, w, decided = _slope_arc_centres(c, mask)
    centre, w = centre[decided], w[decided]
    total = float(w.sum())
    if len(w) < 8 or total <= 0:
        return np.nan, np.nan
    half = np.pi / 2.0
    start = (centre - half) % (2.0 * np.pi)
    end = (centre + half) % (2.0 * np.pi)
    # Arcs that wrap past 0 already cover c = 0; everything else starts later.
    base = float(w[start > end].sum())
    pos = np.concatenate([start, end])
    delta = np.concatenate([w, -w])
    order = np.argsort(pos, kind="stable")
    pos, delta = pos[order], delta[order]
    cover = base + np.cumsum(delta)
    k = int(np.argmax(cover))
    # The plateau runs from this event to the next; take its middle.
    lo = pos[k]
    hi = pos[k + 1] if k + 1 < len(pos) else 2.0 * np.pi
    return float(((lo + hi) / 2.0) % (2.0 * np.pi)), float(cover[k] / total)


def slope_predict(c, c_hat, mask=None):
    """``(pred, decided)`` for the fitted phase: ``True`` where trial *i* is predicted to have
    fired more."""
    sel = slice(None) if mask is None else mask
    p = np.asarray(c["p"][sel], float)
    q = np.asarray(c["q"][sel], float)
    decided = np.hypot(p, q) > 0.0
    return (p * np.cos(c_hat) + q * np.sin(c_hat)) > 0.0, decided


# ── Proximity: half-lines on the separation axis ──────────────────────────────
def fit_separation_sign(c, mask=None):
    """``(d_star, prefer, acc_train)`` -- the exact maximiser over preferred separation *and*
    polarity.

    ``prefer=False`` is the "avoids this separation" branch, which is the complement of the
    "prefers" branch on the decided set and so costs one subtraction rather than a second fit.
    """
    sel = slice(None) if mask is None else mask
    di = np.asarray(c["di"][sel], float)
    dj = np.asarray(c["dj"][sel], float)
    y = np.asarray(c["y"][sel], float) > 0.5
    w = np.asarray(c["w"][sel], float)
    decided = di != dj
    di, dj, y, w = di[decided], dj[decided], y[decided], w[decided]
    total = float(w.sum())
    if len(w) < 8 or total <= 0:
        return np.nan, True, np.nan
    mid = 0.5 * (di + dj)
    # Correct on d* < mid when (i is the nearer separation) agrees with (i fired more).
    left = (di < dj) == y
    base = float(w[left].sum())  # coverage at d* = 0
    delta = np.where(left, -w, w)  # crossing `mid` flips each comparison
    order = np.argsort(mid, kind="stable")
    cover = base + np.cumsum(delta[order])
    pos = mid[order]
    k_hi = int(np.argmax(cover))
    k_lo = int(np.argmin(cover))
    acc_hi = cover[k_hi] / total  # best "prefers d*"
    acc_lo = 1.0 - cover[k_lo] / total  # best "avoids d*"
    if acc_hi >= acc_lo:
        k, acc, prefer = k_hi, acc_hi, True
    else:
        k, acc, prefer = k_lo, acc_lo, False
    lo = pos[k]
    hi = pos[k + 1] if k + 1 < len(pos) else np.pi
    return float((lo + hi) / 2.0), bool(prefer), float(acc)


def separation_predict(c, d_star, prefer=True, mask=None):
    """``(pred, decided)`` for a fitted preferred (or avoided) separation."""
    sel = slice(None) if mask is None else mask
    di = np.asarray(c["di"][sel], float)
    dj = np.asarray(c["dj"][sel], float)
    decided = di != dj
    nearer_i = np.abs(di - d_star) < np.abs(dj - d_star)
    return (nearer_i if prefer else ~nearer_i), decided


# ── Cross-validation ──────────────────────────────────────────────────────────
def cv_sign(c, kind, n_folds=5, seed=0):
    """Held-out correctness per comparison, folded over **trials**.

    Returns ``(correct, pred, w, i, j)`` over the test comparisons of every fold, each scored by
    the parameter fitted on the folds that excluded both of its trials. Uses
    :func:`pairs_lib.fold_masks` with the same ``seed`` for every model, so the three models
    share folds and every comparison is paired.
    """
    if len(c["w"]) < 40:
        return (np.zeros(0),) * 3 + (np.zeros(0, int),) * 2
    C, P, W, I, J = [], [], [], [], []
    for tr, te in pairs.fold_masks(c, n_folds, seed):
        if tr.sum() < 20 or te.sum() < 10:
            continue
        if kind == "separation":
            d_star, prefer, _ = fit_separation_sign(c, mask=tr)
            if not np.isfinite(d_star):
                continue
            pred, decided = separation_predict(c, d_star, prefer, mask=te)
        else:
            c_hat, _ = fit_slope_sign(c, mask=tr)
            if not np.isfinite(c_hat):
                continue
            pred, decided = slope_predict(c, c_hat, mask=te)
        y = np.asarray(c["y"][te], float) > 0.5
        C.append((pred == y)[decided].astype(np.float32))
        P.append(pred[decided])
        W.append(np.asarray(c["w"][te], float)[decided])
        I.append(np.asarray(c["i"][te])[decided])
        J.append(np.asarray(c["j"][te])[decided])
    if not C:
        return (np.zeros(0),) * 3 + (np.zeros(0, int),) * 2
    return (
        np.concatenate(C),
        np.concatenate(P),
        np.concatenate(W),
        np.concatenate(I),
        np.concatenate(J),
    )


def weighted_accuracy(correct, w):
    """The score: weighted fraction of held-out comparisons called correctly."""
    tot = float(np.asarray(w, float).sum())
    if not len(w) or tot <= 0:
        return np.nan
    return float(
        (np.asarray(w, float) * np.asarray(correct, float)).sum() / tot
    )


# ── Uncertainty: resample TRIALS, never comparisons ───────────────────────────
#
# Every statistic here is bootstrapped by **one shared loop over replicates**, and replicate `b`
# is seeded from ``(seed, b)`` alone, so within a replicate all three models and all three
# pairwise contrasts see the **same** trial resample, and differences between them are paired
# rather than carrying independent noise. The draws themselves are written to disk per session
# (see `run_comparison_cache.py`) and the p-values are computed from them by
# :func:`p_from_draws`.
#
# Nothing is refitted inside a replicate: the held-out correctness of every comparison is already
# known and a resample only reweights it. That is what makes the loop cheap (~0.9 ms per
# replicate per neuron).

#: The per-replicate quantities written to disk, in the order `bootstrap_draws` returns them.
BOOT_STATS = tuple(f"acc_{m}" for m in MODELS) + tuple(
    f"prop_{a}_vs_{b}"
    for a, b in (
        ("retinotopic", "relational"),
        ("retinotopic", "proximity"),
        ("relational", "proximity"),
    )
)


def bootstrap_draws(pieces, n_trials, n_boot, seed=0):
    """Weighted accuracies for replicates ``0..n_boot-1``, one row per statistic.

    ``pieces`` maps a statistic name to ``(correct, w, i, j)``. Returns
    ``{name: array of length n_boot}``.
    """
    names = list(pieces)
    out = {k: np.full(n_boot, np.nan, np.float32) for k in names}
    usable = {k: v for k, v in pieces.items() if v is not None and len(v[1])}
    if not usable:
        return out
    for b in range(n_boot):
        rng = np.random.default_rng([int(seed), int(b)])
        m = np.bincount(
            rng.integers(0, n_trials, n_trials), minlength=n_trials
        ).astype(np.float64)
        for k, (corr, w, i, j) in usable.items():
            ww = w * m[i] * m[j]
            tot = ww.sum()
            if tot > 0:
                out[k][b] = (ww * corr).sum() / tot
    return out


def _one_sided_p(draws):
    """p for "the statistic exceeds chance", from the draws. ``(k + 1)/(B + 1)``, so it is
    never zero and its floor states the resolution the replicate count bought.
    """
    d = np.asarray(draws, float)
    d = d[np.isfinite(d)]
    if not len(d):
        return np.nan
    k = np.sum(d <= CHANCE)
    return float((k + 1.0) / (len(d) + 1.0))


def _two_sided_p(draws):
    d = np.asarray(draws, float)
    d = d[np.isfinite(d)]
    if not len(d):
        return np.nan
    frac = float(np.mean(d >= CHANCE))
    return float(min(1.0, 2.0 * min(frac, 1.0 - frac) + 1.0 / (len(d) + 1.0)))


def win_fractions(draws):
    """``P(model wins | resampling the trials)``.

    For each bootstrap replicate the three models are scored on the **same** trial resample (see
    :func:`bootstrap_draws`), so "which of them is highest in this replicate" is a meaningful
    question. The fraction of replicates each model wins is therefore a well-defined quantity
    on [0, 1] whose three values sum to 1, with 1/3 the uninformative value.

    **It is not a posterior and it is not a likelihood.** No likelihood exists here: a sign-only
    score has none. What this measures is the *stability of the ranking under resampling the
    trials* -- how often this model would come out on top had the animal run a different draw of
    the same trials. It needs no evidence scale, only the replicates already computed.

    Ties share the win equally. Replicates where any model is missing are dropped.
    """
    A = np.vstack([np.asarray(draws[f"acc_{m}"], float) for m in MODELS])
    ok = np.all(np.isfinite(A), axis=0)
    out = {f"{m}_win_frac": np.nan for m in MODELS}
    out["n_win_replicates"] = int(ok.sum())
    if not ok.any():
        return out
    A = A[:, ok]
    top = A.max(axis=0)
    hit = (A == top[None, :]).astype(float)
    hit /= hit.sum(axis=0, keepdims=True)  # split ties
    for m, v in zip(MODELS, hit.mean(axis=1)):
        out[f"{m}_win_frac"] = float(v)
    return out


def p_from_draws(draws):
    """Every bootstrap p-value, from the per-replicate draws.

    ``draws`` maps the names in :data:`BOOT_STATS` to 1-D arrays.
    """
    out = {
        "n_boot_used": int(
            max(
                (
                    np.isfinite(np.asarray(v, float)).sum()
                    for v in draws.values()
                ),
                default=0,
            )
        )
    }
    for m in MODELS:
        out[f"{m}_beats_null_p"] = _one_sided_p(draws.get(f"acc_{m}", []))
    for a, b in _PAIR_NAMES:
        out[f"{a}_vs_{b}_p"] = _two_sided_p(draws.get(f"prop_{a}_vs_{b}", []))
    if all(f"acc_{m}" in draws for m in MODELS):
        out.update(win_fractions(draws))
    return out


_PAIR_NAMES = (
    ("retinotopic", "relational"),
    ("retinotopic", "proximity"),
    ("relational", "proximity"),
)


def disagreement_pieces(correct_a, pred_a, pred_b, w, i, j):
    """``(correct, w, i, j)`` restricted to the comparisons where two models predict different
    orderings, plus the point summary.

    On such a comparison exactly one model is right, so the weighted fraction *a* gets right is
    a complete summary and 1/2 is the null of "equally good". This is the sharpest form of the
    question "which model is right": it discards precisely the comparisons that carry no
    discriminating information, and it cannot be moved by the two models having differently
    shaped score differences, because the sizes of those differences never enter.
    """
    from scipy.stats import binomtest

    m = np.asarray(pred_a, bool) != np.asarray(pred_b, bool)
    n = int(m.sum())
    summary = dict(n=n, w_frac=np.nan, prop_a=np.nan, p_binom=np.nan)
    if n < 8:
        return None, summary
    ca = np.asarray(correct_a, float)[m]
    ww = np.asarray(w, float)[m]
    summary["w_frac"] = float(
        ww.sum() / max(float(np.asarray(w, float).sum()), 1e-12)
    )
    summary["prop_a"] = weighted_accuracy(ca, ww)
    # The exact sign test over comparisons, reported only as a foil: comparisons within a
    # neuron are massively dependent, so this is wildly anti-conservative and the trial-level
    # bootstrap is the test. The gap between them is worth seeing.
    summary["p_binom"] = float(
        binomtest(int(np.round(ca.sum())), n, 0.5).pvalue
    )
    return (ca, ww, np.asarray(i)[m], np.asarray(j)[m]), summary


# ── One neuron ────────────────────────────────────────────────────────────────
def fit_neuron_sign(
    t0,
    t1,
    fr2,
    sigma_deg=pairs.SIGMA_DEG,
    n_folds=5,
    seed=0,
    n_boot=N_BOOT,
):
    """Every sign-only quantity for one neuron: ``(row, draws)``.

    ``row`` is a flat dict — the fitted parameter, the held-out accuracy and its excess over
    chance per model, the three pairwise disagreement summaries, and the bootstrap p-values.
    ``draws`` holds the per-replicate values themselves, keyed by :data:`BOOT_STATS`.

    Every comparison is used: no cap is applied (``max_comparisons=0``), since the fit is
    O(N log N) in rows and the precision is set by the number of trials.
    """
    out = {"sigma_deg": sigma_deg, "n_folds": n_folds}
    sets, cvs = {}, {}

    for k in pairs.MODELS:  # 0 = retinotopic, -1 = relational
        name = pairs.MODEL_NAME[k]
        c = pairs.build_comparisons(
            t0,
            t1,
            fr2,
            k,
            sigma_deg,
            seed=seed,
            max_comparisons=0,
        )
        sets[name] = c
        c_hat, acc_tr = fit_slope_sign(c)
        out[f"{name}_n_comparisons"] = len(c["w"])
        out[f"{name}_subsampled"] = bool(c.get("subsampled", False))
        out[f"{name}_phase_deg"] = (
            np.degrees(c_hat) % 360.0 if np.isfinite(c_hat) else np.nan
        )
        out[f"{name}_acc_train"] = acc_tr
        if np.isfinite(c_hat):
            ph = np.degrees(c_hat) % 360.0
            out[f"{name}_axis_deg"] = (
                ph if k == 0 else (ph + 180.0) / 2.0 % 180.0
            )
        else:
            out[f"{name}_axis_deg"] = np.nan
        out[f"{name}_period"] = 360.0 if k == 0 else 180.0
        cvs[name] = cv_sign(c, "slope", n_folds, seed)

    c = pairs.build_separation_comparisons(
        t0, t1, fr2, sigma_deg, seed=seed, max_comparisons=0
    )
    sets["proximity"] = c
    d_star, prefer, acc_tr = fit_separation_sign(c)
    out["proximity_n_comparisons"] = len(c["w"])
    out["proximity_subsampled"] = bool(c.get("subsampled", False))
    out["proximity_sep_deg"] = (
        np.degrees(d_star) if np.isfinite(d_star) else np.nan
    )
    out["proximity_prefer"] = bool(prefer)
    out["proximity_acc_train"] = acc_tr
    cvs["proximity"] = cv_sign(c, "separation", n_folds, seed)

    n_trials = sets["retinotopic"]["n_trials"]
    pieces = {}
    for name in MODELS:
        corr, pred, w, i, j = cvs[name]
        acc = weighted_accuracy(corr, w) if len(w) else np.nan
        out[f"{name}_acc"] = acc
        out[f"{name}_excess"] = acc - CHANCE if np.isfinite(acc) else np.nan
        out[f"{name}_n_scored"] = int(len(w))
        pieces[f"acc_{name}"] = (
            (np.asarray(corr, float), np.asarray(w, float), i, j)
            if len(w)
            else None
        )

    for a, b in _PAIR_NAMES:
        piece, summary = _pair_pieces(cvs, a, b, n_trials)
        pieces[f"prop_{a}_vs_{b}"] = piece
        for key, v in summary.items():
            out[f"{a}_vs_{b}_{key}"] = v

    draws = bootstrap_draws(pieces, n_trials, n_boot, seed=seed + 7)
    out.update(p_from_draws(draws))
    out["boot_lo"], out["boot_hi"] = 0, int(n_boot)

    accs = {m: out[f"{m}_acc"] for m in MODELS}
    best = max(
        accs, key=lambda m: (-np.inf if not np.isfinite(accs[m]) else accs[m])
    )
    out["best_model"] = best if np.isfinite(accs[best]) else None
    out["best_acc"] = accs[best]
    out["beats_null_p"] = out.get(f"{best}_beats_null_p", np.nan)
    return out, draws


def _pair_pieces(cvs, a, b, n_trials):
    """Run :func:`disagreement_pieces` on the comparisons both models scored.

    The two slope models share a comparison set row for row. Proximity's set is one row per
    trial pair rather than four, so the two are joined on ``(i, j)`` -- the trial pair -- and a
    slope row is matched to the proximity row for the same pair. Rows without a partner are
    dropped, and the count that survives is reported as ``n``.
    """
    ca, pa, wa, ia, ja = cvs[a]
    cb, pb, wb, ib, jb = cvs[b]
    empty = dict(n=0, w_frac=np.nan, prop_a=np.nan, p_binom=np.nan)
    if not len(wa) or not len(wb):
        return None, empty
    if (
        len(wa) == len(wb)
        and np.array_equal(ia, ib)
        and np.array_equal(ja, jb)
    ):
        return disagreement_pieces(ca, pa, pb, wa, ia, ja)
    key_a = ia.astype(np.int64) * (n_trials + 1) + ja
    key_b = ib.astype(np.int64) * (n_trials + 1) + jb
    # First occurrence of each pair on the b side; a slope row matches the proximity row for
    # its own trial pair.
    order = np.argsort(key_b, kind="stable")
    kb, pos = key_b[order], order
    idx = np.searchsorted(kb, key_a)
    ok = (idx < len(kb)) & (kb[np.minimum(idx, len(kb) - 1)] == key_a)
    if ok.sum() < 8:
        return None, empty
    sel = pos[idx[ok]]
    return disagreement_pieces(ca[ok], pa[ok], pb[sel], wa[ok], ia[ok], ja[ok])


# ── Reading the fitted table ──────────────────────────────────────────────────
import pandas as pd  # noqa: E402
import stats_lib as sl  # noqa: E402

CACHE_DIR = sl.CACHE_DIR
UNITS_PATH = CACHE_DIR / "comparison_units.csv"

#: Threshold for a pairwise disagreement test.
DECISIVE_Q = 0.05


def load_units(path=UNITS_PATH):
    return pd.read_csv(sl._require(path, "`python run_comparison_cache.py`"))


def _holm(pvals, q):
    """Holm-Bonferroni over a handful of p-values. Returns the reject mask."""
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    order = np.argsort(np.where(ok, p, np.inf))
    m = int(ok.sum())
    rej = np.zeros(len(p), bool)
    for rank, idx in enumerate(order[:m]):
        if p[idx] <= q / (m - rank):
            rej[idx] = True
        else:
            break
    return rej


def add_decisive(units, q=DECISIVE_Q):
    """Add ``decisive`` and ``call``: which neurons one model distinguishably wins.

    The winner must beat **both** of the other two models on the comparisons where it and
    they predict different orderings (:func:`disagreement_pieces`), with the two tests
    Holm-corrected at ``q`` within the neuron. Everything else is ``undecided``, and the size of
    that category is itself a result about how separable these hypotheses are.
    """
    u = units.copy()
    win = u["best_model"].to_numpy(object)
    ok = pd.notna(u["best_model"]).to_numpy()
    decisive = np.zeros(len(u), bool)
    for r in range(len(u)):
        if not ok[r]:
            continue
        ps, wins = [], []
        for a, b in _PAIR_NAMES:
            if win[r] not in (a, b):
                continue
            p = u[f"{a}_vs_{b}_p"].iat[r]
            prop = u[f"{a}_vs_{b}_prop_a"].iat[r]
            if not (np.isfinite(p) and np.isfinite(prop)):
                ps, wins = [np.nan], [False]
                break
            ps.append(p)
            # `prop_a` is the share the FIRST model of the pair gets right.
            wins.append(prop > CHANCE if win[r] == a else prop < CHANCE)
        if len(ps) == 2 and all(np.isfinite(ps)):
            decisive[r] = bool(np.all(_holm(ps, q)) and all(wins))

    u["decisive"] = decisive
    u["call"] = np.where(decisive, u["best_model"], "undecided")
    return u


def win_table(units):
    """Three-way counts per subject, with the undecided share beside them."""
    rows = []
    for s in sl.ordered_subjects(units):
        g = units[units["subject"] == s]
        dec = g[g["decisive"]]
        row = {
            "subject": sl.subject_title(s),
            "n": len(g),
            "decisive": len(dec),
        }
        for m in MODELS:
            row[m] = int((dec["call"] == m).sum())
        row["undecided"] = int(len(g) - len(dec))
        row["undecided_pct"] = 100.0 * row["undecided"] / max(len(g), 1)
        rows.append(row)
    return pd.DataFrame(rows)


# ── The figure ────────────────────────────────────────────────────────────────
def plot_tail_with_count_insets(
    units,
    figsize=(6.4, 3.0),
    tail_kw=None,
    inset_bounds=(0.06, 0.48, 0.30, 0.36),
    inset_headroom=1.35,
    inset_title="# Units",
    inset_title_pad=2.0,
    n_xy=(0.66, 0.05),
    title_pad=30,
    legend_dy_in=0.03,
    legend_lw=None,
):
    """One row, one column per subject: :func:`plot_tail_fractions`, with the decisive-neuron
    counts (``call`` among the ``decisive`` neurons, from :func:`add_decisive`) drawn as an
    inset in each panel.

    ``inset_bounds`` is the inset's ``[x0, y0, width, height]`` in the parent's AXES FRACTIONS;
    the curves all start at 1/3 and only the relational one climbs, so the top left is empty.
    The inset has no y axis -- each bar carries its count -- and ``inset_title`` labels it,
    ``inset_title_pad`` points above the inset's top edge (larger raises it, negative lowers it
    into the inset).
    ``n_xy`` is where ``N = ...`` sits (bottom centre of the text), in AXES FRACTIONS; it counts
    every neuron, the inset only the decisive ones. A **list** of such positions is tried in
    order, per panel, and the first where the text touches no curve, band or inset is used
    (:func:`_place_clear`); if none is clear, the one touching least. ``legend_lw`` sets the
    line width of the legend's markers (points); ``None`` keeps the curves' own width.
    """
    from matplotlib import pyplot as plt

    subjects = sl.ordered_subjects(units)
    models = list(MODELS)
    tail_kw = dict(tail_kw or {})
    fig, axes = plt.subplots(
        1,
        len(subjects),
        figsize=figsize,
        squeeze=False,
        sharey=True,
        layout="constrained",
    )
    n_texts = []
    for ci, s in enumerate(subjects):
        g = units[units["subject"] == s]
        ax = axes[0][ci]
        plot_tail_fractions(ax, g, **tail_kw)
        cands = (
            [tuple(n_xy)] if np.ndim(n_xy) == 1 else [tuple(c) for c in n_xy]
        )
        n_texts.append(
            (
                ax,
                ax.text(
                    *cands[0],
                    f"N = {len(g)}",
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=12,
                ),
                cands,
            )
        )
        ax.set_title(sl.subject_title(s), fontsize=14, pad=title_pad)
        if ci:
            ax.set_ylabel("")

        ins = ax.inset_axes(list(inset_bounds))
        ins.patch.set_alpha(
            0
        )  # see-through: never hides a curve passing behind it
        counts = (
            g[g["decisive"]]["call"].value_counts().reindex(models).fillna(0)
        )
        ins.bar(
            range(len(models)),
            counts.values,
            color=[sl.MODEL_COLORS[m] for m in models],
            edgecolor="k",
            linewidth=0.5,
        )
        for i, m in enumerate(models):
            ins.text(
                i,
                counts[m],
                f"{int(counts[m])}",
                ha="center",
                va="bottom",
                fontsize=12,
            )
        ins.set_ylim(0, max(counts.max(), 1) * inset_headroom)
        ins.set_xticks([])
        ins.set_yticks([])
        for side in ("top", "right", "left"):
            ins.spines[side].set_visible(False)
        # `y=1.0` pins the title to the inset's top edge. Left unset, matplotlib auto-places a
        # title above everything in the axes, which silently overrides a negative pad.
        ins.set_title(inset_title, fontsize=12, pad=inset_title_pad, y=1.0)

    fig.canvas.draw()  # legend and N go in once the layout has settled
    for ax, txt, cands in n_texts:
        if len(cands) > 1:
            _place_clear(fig, ax, txt, cands)
    left, right = axes[0][0].get_position().x0, axes[0][-1].get_position().x1
    y = axes[0][0].get_position().y1 + legend_dy_in / fig.get_figheight()
    handles, labels = axes[0][0].get_legend_handles_labels()
    leg = fig.legend(
        handles,
        labels,
        ncol=len(models),
        loc="lower center",
        bbox_to_anchor=(0.5 * (left + right), y),
        frameon=True,
        fancybox=True,
        fontsize=12,
        handlelength=1.0,
        borderpad=0.3,
        handletextpad=0.5,
        columnspacing=1.2,
    )
    if legend_lw is not None:
        for h in leg.legend_handles:
            h.set_linewidth(legend_lw)
    return fig


def _place_clear(fig, ax, txt, candidates):
    """Move ``txt`` (in ``ax``'s AXES FRACTIONS) to the first of ``candidates`` where its box,
    grown by 3 pixels, crosses no line, filled band or inset of ``ax``; failing that, to the
    candidate crossing fewest. Run after the layout has settled, since it works in pixels.
    """
    from matplotlib.path import Path

    r = fig.canvas.get_renderer()
    obstacles = []  # (display-space path, filled?)
    for ln in ax.lines:
        xy = ln.get_transform().transform(np.asarray(ln.get_xydata(), float))
        ok = np.isfinite(xy).all(1)
        # one path per run of finite points, so a NaN gap is not bridged by a straight line
        for run in np.split(
            np.arange(len(xy)), np.flatnonzero(np.diff(ok.astype(int))) + 1
        ):
            if ok[run[0]] and len(run) > 1:
                obstacles.append((Path(xy[run]), False))
    for coll in ax.collections:
        tr = coll.get_transform()
        for pth in coll.get_paths():
            obstacles.append((pth.transformed(tr), True))
    insets = [c.get_tightbbox(r) for c in ax.child_axes]

    def hits(pos):
        txt.set_position(pos)
        bb = txt.get_window_extent(r).expanded(1.0, 1.0).padded(3.0)
        n = sum(p.intersects_bbox(bb, filled=f) for p, f in obstacles)
        return n + 10 * sum(bb.overlaps(ib) for ib in insets)

    scores = [hits(c) for c in candidates]
    txt.set_position(
        candidates[int(np.argmin(scores))]
    )  # argmin: the first zero, if any


def tail_fractions(acc, grid, min_tail=10):
    """``acc``: (n_units, n_models) held-out accuracies. Returns (n_models, len(grid)): of all
    scores >= x, the fraction belonging to each model; NaN where fewer than ``min_tail`` remain.
    """
    above = (acc[:, :, None] >= grid[None, None, :]).sum(
        axis=0
    )  # (n_models, n_grid)
    n_tail = above.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = above / n_tail
    frac[:, n_tail < min_tail] = np.nan
    return frac


def plot_tail_fractions(
    ax,
    g,
    min_tail=10,
    n_boot=1000,
    n_grid=200,
    seed=0,
    sem_alpha=0.25,
):
    """Of all (neuron, model) scores at or above each threshold, the fraction per model.

    Every neuron contributes one held-out accuracy per model, so the curves sum to 1 at every x;
    they stop where fewer than ``min_tail`` scores remain. The band is a bootstrap SEM: resample
    the units with replacement (each neuron keeps its scores together), recompute on the same
    grid, and take the SD across resamples, dropping a resample wherever it is under ``min_tail``.
    """
    import warnings

    acc = g[[f"{m}_acc" for m in MODELS]].to_numpy(float)
    grid = np.linspace(acc.min(), acc.max(), n_grid)
    frac = tail_fractions(acc, grid, min_tail)
    rng = np.random.default_rng(seed)
    boot = np.stack(
        [
            tail_fractions(
                acc[rng.integers(0, len(acc), len(acc))], grid, min_tail
            )
            for _ in range(n_boot)
        ]
    )
    with warnings.catch_warnings():  # thresholds where <2 resamples still have a curve
        warnings.simplefilter("ignore", RuntimeWarning)
        sem = np.nanstd(boot, axis=0, ddof=1)
    for i, m in enumerate(MODELS):
        ax.fill_between(
            grid,
            frac[i] - sem[i],
            frac[i] + sem[i],
            color=sl.MODEL_COLORS[m],
            alpha=sem_alpha,
            lw=0,
        )
        ax.plot(grid, frac[i], color=sl.MODEL_COLORS[m], lw=1.2, label=m)
    ax.set_xlabel("Held-out accuracy ≥ x")
    ax.set_ylabel("Fraction of units")
    ax.set_ylim(0, 1)
    sl._despine(ax)
    return ax

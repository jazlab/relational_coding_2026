"""The model-agnostic 2-object modulation statistic behind ``modulation.csv``.

For each neuron, from its completed 2-object trials only (east-west pair dropped):

* ``dev`` — held-out Poisson deviance explained by an exchange-symmetric Gaussian kernel
  smoother on the ``(θ0, θ1)`` torus (bandwidth ``config.BANDWIDTH_DEG``), from
  ``config.N_FOLDS``-fold cross-validation repeated ``config.N_CV_REPEATS`` times;
* ``p`` — a permutation p-value for ``dev`` against ``config.N_SHUFFLES`` within-neuron
  shuffles of the firing rates;
* ``cv_r2``, ``const_cv_r2``, ``cw_cv_r2``, ``ioa_cv_r2`` — held-out variance explained by
  the smoother and by the retinotopic (``const``), relational (``cw``) and proximity
  (``ioa``) Poisson GLMs on the same folds.

``figures/neuron_selection.py`` turns ``dev`` and ``p`` into the population every model
figure is drawn over; see its docstring for the full definition. ``run_modulation_cache.py``
builds the table.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import xlogy

_FIGURES = Path(__file__).resolve().parents[2]
if str(_FIGURES) not in sys.path:
    sys.path.insert(0, str(_FIGURES))
import config  # noqa: E402

DURATION_S = 2.0  # the analysis window is stimulus + delay = 2 s
RECTIFY_SCALE = 0.05  # Hz — softness of the positive-part rectifier
RATE_FLOOR = 1e-6  # Hz — keeps log(rate) finite
_IOA_N_GRID = 91  # candidate preferred distances for the proximity model


def mod_params():
    """The statistic's parameters, read from ``config`` when called."""
    return dict(
        bandwidth_deg=config.BANDWIDTH_DEG,
        n_folds=config.N_FOLDS,
        n_cv_repeats=config.N_CV_REPEATS,
        n_shuffles=config.N_SHUFFLES,
        min_2obj_trials=config.MIN_2OBJ_TRIALS,
        seed=config.SEED,
    )


# ── Model helpers ─────────────────────────────────────────────────────────────
def circ_dist(a, b):
    """Circular arc distance in [0, π]; broadcasts."""
    d = np.abs(np.asarray(a, float) - np.asarray(b, float)) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


def rectify(rate):
    """Smooth positive part of a predicted rate (Hz)."""
    return (
        RECTIFY_SCALE
        * np.logaddexp(0.0, np.asarray(rate, float) / RECTIFY_SCALE)
        + RATE_FLOOR
    )


def d01(t0, t1):
    """Chord length between the two objects, in ring-radius units (0 to 2)."""
    return 2.0 * np.sin(circ_dist(t0, t1) / 2.0)


def design(model, t0, t1):
    """Design matrix ``[1, cos·, sin·]`` of a linear 2-object model."""
    ones = np.ones_like(t0)
    if model == "const":  # retinotopic: RF drive from both objects
        return np.column_stack(
            [ones, np.cos(t0) + np.cos(t1), np.sin(t0) + np.sin(t1)]
        )
    if model == "cw":  # relational: tuned to θ0 + θ1
        s = t0 + t1
        return np.column_stack([ones, np.cos(s), np.sin(s)])
    raise ValueError(f"{model!r} is not a linear model (expected const or cw)")


def poisson_loglik(X, beta, y):
    """Poisson log-likelihood (dropping ``log Γ(y+1)``) of counts ``y`` over the
    ``DURATION_S`` window, at the rate ``rectify(X @ beta)``."""
    with np.errstate(
        all="ignore"
    ):  # macOS BLAS raises spurious matmul FP flags
        mu = DURATION_S * rectify(X @ beta)
        return float(np.sum(y * np.log(np.maximum(mu, 1e-300)) - mu))


def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * np.clip(z, -60.0, 60.0)))


def _poisson_nll_grad(beta, X, y):
    """Negative log-likelihood and its gradient, through the rectifier."""
    with np.errstate(all="ignore"):
        r = X @ beta
        rate = rectify(r)
        mu = DURATION_S * rate
        nll = float(np.sum(mu - y * np.log(np.maximum(mu, 1e-300))))
        if not np.isfinite(nll):
            return 1e18, np.zeros_like(beta)
        dmu = DURATION_S * _sigmoid(r / RECTIFY_SCALE)  # d mu / d eta
        grad = X.T @ ((1.0 - y / np.maximum(mu, 1e-300)) * dmu)
    return nll, grad


def poisson_irls(X, y, max_iter=50, tol=1e-9, ridge=1e-9):
    """Identity-link Poisson GLM by safeguarded IRLS, polished with L-BFGS-B on the exact
    rectified likelihood. ``y`` are counts over the ``DURATION_S`` window; column 0 of ``X``
    must be the intercept."""
    exposure = DURATION_S
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    beta = np.zeros(X.shape[1])
    beta[0] = max(y.mean(), 1e-6) / exposure  # intercept-only start (Hz)
    with np.errstate(all="ignore"):
        cur = poisson_loglik(X, beta, y)
        for _ in range(max_iter):
            rate = rectify(X @ beta)
            mu = exposure * rate
            w = 1.0 / np.maximum(
                rate, RECTIFY_SCALE
            )  # identity-link IRLS weight
            z = X @ beta + (y - mu) / np.maximum(exposure, 1e-12)
            XtW = X.T * w
            try:
                new = np.linalg.solve(
                    XtW @ X + ridge * np.eye(X.shape[1]), XtW @ z
                )
            except np.linalg.LinAlgError:
                break
            if not np.isfinite(new).all():
                break
            accepted = False
            step = 0.0
            for shrink in (1.0, 0.5, 0.25, 0.125):
                cand = beta + shrink * (new - beta)
                cand_ll = poisson_loglik(X, cand, y)
                if np.isfinite(cand_ll) and cand_ll >= cur:
                    step = float(np.max(np.abs(cand - beta)))
                    beta, cur, accepted = cand, cand_ll, True
                    break
            if not accepted or step < tol:
                break
    res = minimize(
        _poisson_nll_grad,
        beta,
        args=(X, y),
        jac=True,
        method="L-BFGS-B",
    )
    if res.success and np.isfinite(res.fun) and -res.fun >= cur - 1e-9:
        beta = res.x
    return beta


def ioa_feature_grid(t0, t1, n_grid=_IOA_N_GRID):
    """``(T, grid)``: the proximity feature ``|d01 − δ|`` at every candidate δ, spanning the
    observed range of ``d01``."""
    d = d01(t0, t1)
    lo, hi = float(np.min(d)), float(np.max(d))
    grid = np.linspace(lo, hi, n_grid) if hi > lo + 1e-9 else np.array([lo])
    return np.abs(d[:, None] - grid[None, :]), grid


def ioa_fit_from_grid(T, y, max_iter=25, tol=1e-9):
    """Fit ``rate = β + α·T[:, k]`` at every candidate δ at once (vectorised 2×2 IRLS) and
    return the best ``(k, β, α)`` by log-likelihood, the winner polished. ``y`` are counts
    over the ``DURATION_S`` window."""
    exposure = DURATION_S
    y = np.asarray(y, float)
    T = np.asarray(T, float)
    n, K = T.shape

    def loglik(b, a):
        with np.errstate(all="ignore"):
            mu = exposure * rectify(b[None, :] + a[None, :] * T)
            return (y[:, None] * np.log(np.maximum(mu, 1e-300)) - mu).sum(
                axis=0
            )

    b = np.full(K, max(y.mean(), 1e-6) / exposure)  # intercept-only start (Hz)
    a = np.zeros(K)
    best_ll = loglik(b, a)
    best_b, best_a = b.copy(), a.copy()
    with np.errstate(all="ignore"):
        for _ in range(max_iter):
            eta = b[None, :] + a[None, :] * T
            rate = rectify(eta)
            w = 1.0 / np.maximum(rate, RECTIFY_SCALE)
            z = eta + (y[:, None] - exposure * rate) / exposure
            s0 = w.sum(axis=0)
            s1 = (w * T).sum(axis=0)
            s2 = (w * T * T).sum(axis=0)
            r0 = (w * z).sum(axis=0)
            r1 = (w * T * z).sum(axis=0)
            det = s0 * s2 - s1 * s1
            ok = np.abs(det) > 1e-12
            nb, na = b.copy(), a.copy()
            nb[ok] = (s2[ok] * r0[ok] - s1[ok] * r1[ok]) / det[ok]
            na[ok] = (s0[ok] * r1[ok] - s1[ok] * r0[ok]) / det[ok]
            finite = np.isfinite(nb) & np.isfinite(na)
            b = np.where(finite, nb, b)
            a = np.where(finite, na, a)
            ll = loglik(b, a)
            better = np.isfinite(ll) & (ll > best_ll)
            best_ll = np.where(better, ll, best_ll)
            best_b = np.where(better, b, best_b)
            best_a = np.where(better, a, best_a)
            if np.max(np.abs(np.where(finite, nb - best_b, 0.0))) < tol:
                break

    k = int(np.nanargmax(best_ll))
    bk, ak = float(best_b[k]), float(best_a[k])
    X = np.column_stack([np.ones(n), T[:, k]])
    beta = poisson_irls(X, y, max_iter=max_iter, tol=tol)
    if poisson_loglik(X, beta, y) >= best_ll[k] - 1e-9:
        bk, ak = float(beta[0]), float(beta[1])
    return k, bk, ak


# ── The statistic ─────────────────────────────────────────────────────────────
def sym_weights(q0, q1, t0, t1, sigma):
    """Kernel weights ``[len(q), len(t)]`` on the (θ0, θ1) torus, symmetric under object
    swap (distance = min over the two pairings)."""
    q0, q1 = np.asarray(q0)[:, None], np.asarray(q1)[:, None]
    t0, t1 = np.asarray(t0)[None, :], np.asarray(t1)[None, :]
    d_direct = circ_dist(q0, t0) ** 2 + circ_dist(q1, t1) ** 2
    d_swap = circ_dist(q0, t1) ** 2 + circ_dist(q1, t0) ** 2
    return np.exp(-np.minimum(d_direct, d_swap) / (2.0 * sigma**2))


def poisson_deviance(y, mu):
    """Total Poisson deviance ``2 Σ [y·log(y/µ) − (y−µ)]`` (0·log0 = 0)."""
    mu = np.clip(mu, 1e-9, None)
    return float(2.0 * np.sum(xlogy(y, y / mu) - (y - mu)))


def _folds(n, k, rng):
    idx = rng.permutation(n)
    return [idx[i::k] for i in range(k)]  # k interleaved test folds


def _kernel_predict(W, test, train, y):
    """Kernel-weighted mean of the training counts at the test trials (fallback = the
    training mean where no neighbor carries weight)."""
    w = W[np.ix_(test, train)]
    sw = w.sum(1)
    return np.where(
        sw > 1e-9,
        (w @ y[train]) / np.where(sw > 1e-9, sw, 1.0),
        y[train].mean(),
    )


def _cv_dev_kernel(y, W, p, rng):
    """Held-out Poisson deviance explained of the kernel smoother alone (the shuffle null)."""
    n = len(y)
    dev_m = dev_0 = 0.0
    for test in _folds(n, p["n_folds"], rng):
        train = np.setdiff1d(np.arange(n), test)
        dev_m += poisson_deviance(y[test], _kernel_predict(W, test, train, y))
        dev_0 += poisson_deviance(y[test], np.full(len(test), y[train].mean()))
    return 1.0 - dev_m / dev_0 if dev_0 > 1e-9 else np.nan


def _cv_metrics(t0, t1, y, W, p, rng):
    """One shared k-fold pass: the kernel's held-out ``dev`` and ``cv_r2``, plus each model's
    held-out variance explained on the same folds."""
    n = len(y)
    Xc, Xw = design("const", t0, t1), design("cw", t0, t1)
    T_ioa, _ = ioa_feature_grid(t0, t1)
    dev_m = dev_0 = ss_tot = 0.0
    ss_res = {"kernel": 0.0, "const": 0.0, "cw": 0.0, "ioa": 0.0}
    for test in _folds(n, p["n_folds"], rng):
        train = np.setdiff1d(np.arange(n), test)
        ybar = y[train].mean()
        muk = _kernel_predict(W, test, train, y)
        dev_m += poisson_deviance(y[test], muk)
        dev_0 += poisson_deviance(y[test], np.full(len(test), ybar))
        ss_tot += np.sum((y[test] - ybar) ** 2)
        ss_res["kernel"] += np.sum((y[test] - muk) ** 2)
        for key, X in (("const", Xc), ("cw", Xw)):
            beta = poisson_irls(X[train], y[train])
            with np.errstate(all="ignore"):
                mu = DURATION_S * rectify(X[test] @ beta)
            ss_res[key] += np.sum((y[test] - mu) ** 2)
        k, b, a = ioa_fit_from_grid(T_ioa[train], y[train])
        mu = DURATION_S * rectify(b + a * T_ioa[test][:, k])
        ss_res["ioa"] += np.sum((y[test] - mu) ** 2)

    def r2(key):
        return (1.0 - ss_res[key] / ss_tot) if ss_tot > 1e-9 else np.nan

    return dict(
        dev=1.0 - dev_m / dev_0 if dev_0 > 1e-9 else np.nan,
        cv_r2=r2("kernel"),
        const_cv_r2=r2("const"),
        cw_cv_r2=r2("cw"),
        ioa_cv_r2=r2("ioa"),
    )


def neuron_modulation(t0, t1, fr, p, seed=0):
    """Every modulation metric for one neuron's 2-object trials."""
    y = np.asarray(fr, float) * DURATION_S  # spike counts over the 2-s window
    n = len(y)
    keys = [
        "dev",
        "cv_r2",
        "const_cv_r2",
        "cw_cv_r2",
        "ioa_cv_r2",
        "null_mean",
        "null_std",
        "z",
        "p",
    ]
    base = dict(n_2obj=n, mean_count=float(y.mean()) if n else np.nan)
    if n < p["min_2obj_trials"] or y.sum() <= 0:
        return {**base, **{k: np.nan for k in keys}}
    rng = np.random.default_rng(seed)
    sigma = np.radians(p["bandwidth_deg"])
    with np.errstate(all="ignore"):  # benign 0/0, masked by np.where guards
        W = sym_weights(t0, t1, t0, t1, sigma)
        cms = [
            _cv_metrics(
                t0, t1, y, W, p, np.random.default_rng(seed + 1000 + r)
            )
            for r in range(max(1, p["n_cv_repeats"]))
        ]
        cm = {
            k: float(np.nanmean([c[k] for c in cms]))
            for k in ("dev", "cv_r2", "const_cv_r2", "cw_cv_r2", "ioa_cv_r2")
        }
        null = np.array(
            [
                _cv_dev_kernel(
                    rng.permutation(y), W, p, np.random.default_rng(seed)
                )
                for _ in range(p["n_shuffles"])
            ]
        )
        null_mean, null_std = float(np.nanmean(null)), float(np.nanstd(null))
        z = (cm["dev"] - null_mean) / null_std if null_std > 1e-9 else np.nan
        pval = float((1 + np.sum(null >= cm["dev"])) / (1 + len(null)))
    return {
        **base,
        **cm,
        "null_mean": null_mean,
        "null_std": null_std,
        "z": z,
        "p": pval,
    }

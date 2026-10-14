"""Does the response error depend on the inter-object angle? — the test behind panel (d).

The figure shows a bias that swings from negative to positive as the non-target crosses from
one side of the target to the other. Two different claims can be made about it, and they want
different tests, so both are here.

1. **Omnibus** — does mean error differ across the inter-object-angle bins at all? Makes no
   assumption about the shape of the dependence, and has correspondingly little power.
2. **Directed** — is the dependence specifically *repulsion*: the response pushed **away**
   from the non-target? One degree of freedom, so far more powerful, and it is the claim the
   panel actually makes.

The regressor
-------------
Let ``d`` be the non-target's position relative to the cued object, signed, and ``e`` the
signed response error. Repulsion means the response is pushed away from the non-target: when
the non-target sits counter-clockwise of the target (``d`` in (0°, 180°)) the error should go
clockwise (negative), and the other way round for ``d`` in (180°, 360°). The natural smooth
odd function of ``d`` with that sign is ``x = -sin(d)``, so each session is fit

    e = a + b * x + noise

and **b is the repulsion amplitude in degrees** — the bias at a 90° separation, positive when
the response is repelled. Fitting an intercept ``a`` is not optional: an animal with a
constant clockwise bias would otherwise have that offset absorbed into ``b`` and be scored as
repulsed. (Monkey E has exactly such an offset; both its half-means are negative.)

The unit of replication is the **session**, not the trial
---------------------------------------------------------
Trials within a session share an animal, a day, a satiety level and a calibration, so they
are nowhere near independent. Testing over ~9,000 trials as if they were would produce a
p-value of no meaning — the usual pseudo-replication failure. So ``b`` is fit **once per
session** and the test is over those 11-12 numbers: a one-sample t-test against zero, with a
Wilcoxon signed-rank as a distribution-free check. That is the standard summary-statistic
approach, and its degrees of freedom are honest.

The omnibus test uses the same logic differently: the bin labels are permuted **within
session**, which destroys the trial-level association while leaving every session's own error
distribution and bin composition exactly as they are.

Everything runs on precisely the trials the panel plots — ``|error| <= config.MAX_ERROR_DEG``,
beyond which the response reports the wrong object and is a swap rather than a bias.
"""

import numpy as np
import pandas as pd
from scipy import stats


#: Repulsion regressor: `-sin(d)`, positive where the non-target lies clockwise of the target.
def repulsion_regressor(rel_rad):
    return -np.sin(np.asarray(rel_rad, float))


def session_slopes(trials):
    """`b` (and `a`) per session: one ordinary least-squares fit of `e = a + b*x` each.

    Returns a frame with `session`, `n`, `intercept_deg` and `slope_deg`, the latter being the
    repulsion amplitude — degrees of bias at a 90° separation, positive = repelled.
    """
    rows = []
    for session, g in trials.groupby("session", sort=True):
        x = repulsion_regressor(g["rel_theta"].to_numpy(float))
        e = g["error_deg"].to_numpy(float)
        if len(g) < 3 or np.ptp(x) == 0:
            continue
        b, a = np.polyfit(x, e, 1)
        rows.append(
            dict(session=session, n=len(g), intercept_deg=a, slope_deg=b)
        )
    return pd.DataFrame(rows)


def directed_test(trials):
    """The repulsion test: are the per-session slopes positive?

    Two-sided, because a *negative* slope would be attraction and is an equally real result
    that the test should be able to see.
    """
    per = session_slopes(trials)
    b = per["slope_deg"].to_numpy(float)
    t, p = stats.ttest_1samp(b, 0.0)
    try:
        _, p_w = stats.wilcoxon(b)
    except ValueError:  # too few sessions for the signed-rank test
        p_w = np.nan
    sem = b.std(ddof=1) / np.sqrt(len(b))
    return dict(
        n_sessions=len(b),
        slope_deg=b.mean(),
        sem_deg=sem,
        ci95=(b.mean() - 1.96 * sem, b.mean() + 1.96 * sem),
        t=t,
        df=len(b) - 1,
        p=p,
        p_wilcoxon=p_w,
        n_positive=int((b > 0).sum()),
        per_session=per,
    )


def eta_squared(errors, bins):
    """Fraction of the error variance lying between bin means — the figure's own statistic."""
    grand = errors.mean()
    between = sum(
        (bins == k).sum() * (errors[bins == k].mean() - grand) ** 2
        for k in np.unique(bins)
    )
    return between / ((errors - grand) ** 2).sum()


def omnibus_test(trials, n_perm=10000, seed=0):
    """Does mean error differ across bins? Bin labels permuted **within session**.

    Shuffling within session leaves each session's own errors and its own mix of inter-object
    angles untouched, so a session that happens to have both a large mean error and an unusual
    bin composition cannot manufacture the effect.
    """
    rng = np.random.default_rng(seed)
    e = trials["error_deg"].to_numpy(float)
    b = trials["bin"].to_numpy(int)
    session = trials["session"].to_numpy()
    obs = eta_squared(e, b)

    groups = [np.flatnonzero(session == s) for s in np.unique(session)]
    null = np.empty(n_perm)
    shuffled = b.copy()
    for i in range(n_perm):
        for idx in groups:
            shuffled[idx] = rng.permutation(b[idx])
        null[i] = eta_squared(e, shuffled)
    p = (1 + int((null >= obs).sum())) / (1 + n_perm)
    return dict(
        eta2=obs,
        p=p,
        n_perm=n_perm,
        null_mean=null.mean(),
        null_p95=float(np.quantile(null, 0.95)),
    )


def report(trials, label, n_perm=10000, seed=0):
    """Both tests for one subject, as printable lines."""
    d = directed_test(trials)
    o = omnibus_test(trials, n_perm=n_perm, seed=seed)
    lines = [
        f'{label}: {len(trials):,} trials, {d["n_sessions"]} sessions',
        f'   repulsion   b = {d["slope_deg"]:+.3f}° +/- {d["sem_deg"]:.3f} (SEM over sessions), '
        f'95% CI [{d["ci95"][0]:+.3f}, {d["ci95"][1]:+.3f}]',
        f'               t({d["df"]}) = {d["t"]:.2f}, p = {_fmt_p(d["p"])}'
        f'   (Wilcoxon p = {_fmt_p(d["p_wilcoxon"])}; '
        f'{d["n_positive"]}/{d["n_sessions"]} sessions positive)',
        f'   omnibus     eta^2 = {100 * o["eta2"]:.2f}%, permutation p = {_fmt_p(o["p"])} '
        f'({o["n_perm"]:,} within-session shuffles; null 95th pct {100 * o["null_p95"]:.2f}%)',
    ]
    return "\n".join(lines), d, o


def _fmt_p(p):
    if not np.isfinite(p):
        return "n/a"
    return f"{p:.3g}" if p >= 1e-4 else f"{p:.1e}"

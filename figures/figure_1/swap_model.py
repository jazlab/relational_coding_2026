"""The correct / swap / lapse mixture model behind `swap_errors.ipynb`.

On a 2-object trial the response comes from one of three kinds of trial, which cannot be
labelled individually but can be inferred as a mixture:

1. **correct** — aimed at the cued object, with some angular noise;
2. **swap** — aimed at the *other* object;
3. **lapse** — a guess, unrelated to either.

So the response angle is modelled as a von Mises mixture with one shared precision and a
uniform lapse term,

    p(theta) = p_correct * VM(theta; target, kappa)
             + p_swap    * VM(theta; nontarget, kappa)
             + p_lapse   * 1 / 2pi,

fit per session by EM. The lapse term is what keeps the other two honest: guesses land
uniformly around the ring, so without it the ones near the non-target would be counted as
swaps and the rest would inflate the spatial noise.

Reported per session: `spatial_error_deg`, the circular SD of the target component, and
`swap_error`, the fitted `p_swap`.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_FIGURES = Path(__file__).resolve().parent.parent
if str(_FIGURES) not in sys.path:
    sys.path.insert(0, str(_FIGURES))
import config  # noqa: E402  the one place selection criteria are set
from scipy.special import i0e, i1e


def _vm_pdf(error, kappa):
    """von Mises density at an angular error, via the scaled I0 so large kappa is stable."""
    return np.exp(kappa * (np.cos(error) - 1.0)) / (2.0 * np.pi * i0e(kappa))


def _mean_resultant(kappa):
    """A(kappa) = I1/I0, the mean resultant length of a von Mises."""
    return i1e(kappa) / i0e(kappa)


def _inv_mean_resultant(r):
    """The kappa whose mean resultant length is `r` (Best & Fisher's approximation)."""
    r = float(np.clip(r, 1e-8, 1 - 1e-8))
    if r < 0.53:
        return 2 * r + r**3 + 5 * r**5 / 6
    if r < 0.85:
        return -0.4 + 1.39 * r + 0.43 / (1 - r)
    return 1.0 / (r**3 - 4 * r**2 + 3 * r)


def fit_session(target, nontarget, response, n_iter=1000, tol=1e-9):
    """Fit the mixture to one session by EM. Returns `(spatial_error_deg, p_swap)`.

    E-step: each trial's responsibility for {correct, swap, lapse}. M-step: the weights are
    the mean responsibilities, and kappa is refit from the responsibility-weighted mean
    resultant length, each trial measured to the component's own mean.
    """
    from_target = (
        np.asarray(response, float) - np.asarray(target, float) + np.pi
    ) % (2 * np.pi) - np.pi
    from_other = (
        np.asarray(response, float) - np.asarray(nontarget, float) + np.pi
    ) % (2 * np.pi) - np.pi

    p_correct, p_swap, p_lapse = 0.8, 0.1, 0.1
    r0 = np.clip(np.abs(np.mean(np.exp(1j * from_target))), 0.05, 0.95)
    kappa = float(np.clip(_inv_mean_resultant(r0), 1e-3, 1e4))
    previous = -np.inf
    for _ in range(n_iter):
        a = p_correct * _vm_pdf(from_target, kappa)
        b = p_swap * _vm_pdf(from_other, kappa)
        c = p_lapse / (2.0 * np.pi)
        total = a + b + c
        loglik = float(np.sum(np.log(total)))
        resp_correct, resp_swap = a / total, b / total
        p_correct = float(np.mean(resp_correct))
        p_swap = float(np.mean(resp_swap))
        p_lapse = float(np.mean(c / total))
        weight = resp_correct + resp_swap
        cosines = resp_correct * np.cos(from_target) + resp_swap * np.cos(
            from_other
        )
        kappa = float(
            np.clip(
                _inv_mean_resultant(
                    np.clip(
                        cosines.sum() / max(weight.sum(), 1e-12),
                        1e-6,
                        1 - 1e-6,
                    )
                ),
                1e-6,
                1e4,
            )
        )
        if abs(loglik - previous) < tol:
            break
        previous = loglik

    r = max(_mean_resultant(kappa), 1e-12)
    return float(np.degrees(np.sqrt(-2.0 * np.log(r)))), p_swap


def fit_sessions(df_2obj, min_trials=None):
    """Fit every `(subject, session)` with at least `min_trials` trials.

    One row per session: `subject`, `session`, `n_trials`, `spatial_error_deg`,
    `swap_error` — the x and y of the figure.
    """
    # Resolved at call time, never as a default: a default argument freezes the value at
    # import and `config.reload()` would not reach it. See config.reload.
    min_trials = (
        config.MIN_TRIALS_PER_SESSION if min_trials is None else min_trials
    )
    rows = []
    for (subject, session), g in df_2obj.groupby(
        ["subject", "session"], sort=True
    ):
        if len(g) < min_trials:
            continue
        spatial_error_deg, p_swap = fit_session(
            g["target_theta"].to_numpy(),
            g["nontarget_theta"].to_numpy(),
            g["response_theta"].to_numpy(),
        )
        rows.append(
            dict(
                subject=subject,
                session=session,
                n_trials=len(g),
                spatial_error_deg=spatial_error_deg,
                swap_error=p_swap,
            )
        )
    return pd.DataFrame(rows)

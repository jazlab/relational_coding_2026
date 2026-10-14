r"""The maximum-response chords of `figure_example_neurons.ipynb`: real neurons and idealized ones.

For a neuron and an inter-object angle ``s``, the chords drawn are the ``s``-apart object pairs
that drive the neuron most: the local maxima of its firing rate as a function of the pair's
midpoint direction ``m`` (objects at ``m − s/2`` and ``m + s/2``).

Real neurons — the rate is measured
-----------------------------------
:func:`curve` is `figure_2/psth_lib.dense_2obj_tuning`'s binning exactly — the trials, rates and
window `tuning_plots.ipynb`'s line row uses — vectorised so it can be sampled every couple of
degrees:

* **Bin membership.** A trial joins chord ``k`` when *both* of its objects are within
  ``window`` of their own end of the chord, matching objects to ends whichever way fits better
  (`psth_lib._dist_2obj`). Windows overlap, so a trial can sit in several bins.
* **Rate.** Each trial's mean rate over ``tuning_lib.FR_WINDOW_MS`` (stimulus + delay) after
  ``tuning_lib.SMOOTH_HALF_WIN_MS`` temporal smoothing (`psth_lib.per_trial_mean_fr`), averaged
  over the bin's trials. The east-west antipodal pair is dropped (`psth_lib.load_behavior`).

:func:`check_against_tuning_lib` confirms the curve equals `tuning_lib.unit_curves` on its grid.
The notebook smooths it (`tuning_plots`' line-row smoothing) before :func:`local_maxima` picks the
peaks.

**The task never shows objects closer than about 80° apart**, so a narrow chord's bin can only
recruit *wider* pairs and measures a larger separation than its label; :func:`curve` returns each
bin's mean separation so that can be reported.

Idealized neurons — the rate is the model
-----------------------------------------
:func:`neuron_params` builds a relational-bias neuron (`models_lib.RelationalBiasModel`, the model
whose ``bias`` is figure 4's relationality) and :func:`extreme_pairs` finds its maxima from the
noise-free model rate.

Copied from `figures/tmp/` (`empirical_lib.py`, `frames_lib.py`), where the figure was
developed, so this directory does not import from a scratch one.
"""

import models_lib as ml
import numpy as np
import tuning_lib  # .pl is figure 2's psth_lib
from scipy.signal import find_peaks

pl = tuning_lib.pl
MODEL = ml.build_models(["relational_bias"])["relational_bias"]


def _cd(a, b):
    """Unsigned circular distance, radians."""
    return np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)


# ── Real neurons ──────────────────────────────────────────────────────────────
def unit_trials(subject, session, unit_id):
    """``(t0, t1, fr)`` over the unit's 2-object trials: object angles (rad) and per-trial rate
    (Hz), computed exactly as `tuning_lib` does."""
    beh = pl.load_behavior(subject, session, num_objects=2)
    spike_counts, trials, _ = pl.load_unit(subject, session, unit_id)
    rates = pl.per_trial_mean_fr(
        spike_counts,
        trials,
        beh["trial_num"].to_numpy(),
        window=tuning_lib.FR_WINDOW_MS,
        smooth_half_win_ms=tuning_lib.SMOOTH_HALF_WIN_MS,
    )
    beh = beh[beh["trial_num"].astype(int).isin(rates)]
    fr = np.array([rates[int(t)] for t in beh["trial_num"]])
    return (
        beh["object_0_theta"].to_numpy(float),
        beh["object_1_theta"].to_numpy(float),
        fr,
    )


def curve(trials, sep_deg, n_dense, window_deg, offset_rad=0.0):
    """``(mid, rate, n_trials, mean_sep_deg)`` over ``n_dense`` chord midpoints evenly spanning
    the circle from ``offset_rad``. NaN rate where a bin is empty."""
    t0, t1, fr = trials
    mid = offset_rad + np.linspace(0, 2 * np.pi, n_dense, endpoint=False)
    half = np.radians(sep_deg) / 2
    p0, p1 = (mid - half)[:, None], (mid + half)[:, None]
    d = np.minimum(
        np.maximum(_cd(t0, p0), _cd(t1, p1)),
        np.maximum(_cd(t0, p1), _cd(t1, p0)),
    )
    member = d <= np.radians(window_deg)
    n = member.sum(1)
    with np.errstate(invalid="ignore"):
        rate = (member * fr).sum(1) / n
        sep = (member * np.degrees(_cd(t0, t1))).sum(1) / n
    return mid, rate, n, sep


def local_maxima(mid, rate, n_max, min_prominence, period=2 * np.pi):
    """Indices of the ``n_max`` highest circular local maxima of ``rate`` whose prominence is at
    least ``min_prominence`` (Hz), highest first. ``period = π`` treats the curve as repeating
    every half turn (the 180°-apart curve), so each stimulus is counted once.
    """
    keep = mid - mid[0] < period - 1e-9
    idx_all = np.where(keep)[0]
    y = rate[keep]
    m = len(y)
    pk, _ = find_peaks(np.concatenate([y, y, y]), prominence=min_prominence)
    pk = np.unique(pk[(pk >= m) & (pk < 2 * m)] - m)
    if len(pk) == 0:
        return np.array([], int)
    pk = pk[np.argsort(y[pk])[::-1]][:n_max]
    return idx_all[pk]


def check_against_tuning_lib(
    subject, session, unit_id, sep_deg=90, n_dense=36, window_deg=15.0
):
    """Largest |difference| (Hz) between :func:`curve` and `tuning_lib.unit_curves` on the same
    grid — the evidence this is the same binning."""
    ref = tuning_lib.unit_curves(
        subject,
        session,
        unit_id,
        [sep_deg],
        n_dense=n_dense,
        similarity_window_deg=window_deg,
    )[sep_deg]
    _, r, _, _ = curve(
        unit_trials(subject, session, unit_id), sep_deg, n_dense, window_deg
    )
    return float(np.nanmax(np.abs(np.asarray(ref[1]) - r)))


# ── Idealized neurons ─────────────────────────────────────────────────────────
def neuron_params(
    bias, theta_deg=90.0, hwhm_deg=40.0, baseline=2.0, amplitude=10.0
):
    """One idealized neuron's parameters: the relational-bias model's five, with ``bias`` set."""
    return dict(
        baseline=float(baseline),
        amplitude=float(amplitude),
        theta=float(np.radians(theta_deg)),
        width=ml.width_from_hwhm(hwhm_deg),
        bias=float(bias),
    )


def rates(params, phi_a, phi_b):
    """The model's noise-free rate for each pair."""
    return ml.rectify(MODEL.rate_2obj(params, phi_a, phi_b))


def model_extreme_pairs(
    model, params, sep_deg, n, kind="max", dctr=None, step_deg=0.25
):
    """The ``n`` most extreme ``sep_deg``-apart pairs of **any** figure-4 model with parameters
    ``params``, as ``[(phi_a, phi_b, rate), ...]``, most extreme first: the highest local maxima
    (``kind='max'``) or lowest local minima (``'min'``) of the predicted rate against the pair's
    midpoint direction. The 180° search runs over half the circle, since ``(φ, φ + 180)`` repeats
    after half a turn.

    ``dctr`` is the pair's relative contrast, read only by the contrast-weighted model
    (`models_lib.rate_2obj` dispatches on it); every other model ignores it.
    """
    span = np.pi if np.isclose(sep_deg, 180.0) else 2 * np.pi
    mid = np.arange(0.0, span, np.radians(step_deg))
    half = np.radians(sep_deg) / 2
    d = None if dctr is None else np.full(len(mid), float(dctr))
    r = ml.rectify(ml.rate_2obj(model, params, mid - half, mid + half, d))
    y = r if kind == "max" else -r
    m = len(y)
    idx, _ = find_peaks(np.concatenate([y, y, y]))
    idx = np.unique(idx[(idx >= m) & (idx < 2 * m)] - m)
    if len(idx) == 0:
        return []
    idx = idx[np.argsort(y[idx])[::-1]][:n]
    return [(mid[i] - half, mid[i] + half, float(r[i])) for i in idx]


def extreme_pairs(params, sep_deg, n, kind="max", step_deg=0.25):
    """:func:`model_extreme_pairs` for an idealized relational-bias neuron (:func:`neuron_params`)."""
    return model_extreme_pairs(
        MODEL, params, sep_deg, n, kind, step_deg=step_deg
    )


def drawn_model_chords(
    model, params, separations, tie_tol, show_all=False, dctr=None
):
    """The maximum chords a model draws: per separation ``sep`` (``{sep: n}``), its highest
    ``n`` local maxima, keeping a second one only if it ties the first within ``tie_tol`` of the
    model's response range (the lowest to highest single extreme over ``separations``).
    ``tie_tol=None`` keeps every maximum found.
    Returns ``(chords, report)``: ``[(phi_a, phi_b, rate, sep), ...]`` and one line per maximum
    found, saying whether it was drawn."""
    allr = [
        r
        for sep in separations
        for k in ("max", "min")
        for *_, r in model_extreme_pairs(model, params, sep, 1, k, dctr)
    ]
    rng = max(allr) - min(allr)
    chords, report = [], []
    for sep, n in separations.items():
        found = model_extreme_pairs(model, params, sep, n, "max", dctr)
        for i, (a, c, r) in enumerate(found):
            drawn = (
                tie_tol is None
                or i == 0
                or abs(r - found[0][2]) <= tie_tol * rng
                or show_all
            )
            report.append(
                f"{sep:3d}° max #{i + 1}: objects at {np.degrees(a) % 360:5.1f}° and "
                f"{np.degrees(c) % 360:5.1f}°, {r:5.2f} Hz"
                f'{"" if drawn else "  (not drawn: does not tie the first)"}'
            )
            if drawn:
                chords.append((a, c, r, sep))
    return chords, report


def empirical_chords(
    subject,
    session,
    unit_id,
    separations,
    n_dense,
    window_deg,
    smooth_deg,
    min_prominence_frac,
):
    """A real neuron's maximum chords from its measured responses — `figure_example_neurons`'
    procedure: the binned curve (:func:`curve`) at each separation, smoothed by a circular
    Gaussian of ``smooth_deg``, then its highest ``separations[sep]`` local maxima that stand at
    least ``min_prominence_frac`` of the neuron's response range above their surroundings.
    Returns ``(chords, report)`` as :func:`drawn_model_chords` does."""
    trials = unit_trials(subject, session, unit_id)
    curves = {}
    for sep in separations:
        mid, rate, n, msep = curve(trials, sep, n_dense, window_deg)
        curves[sep] = (mid, pl._circ_smooth(rate, smooth_deg), n, msep)
    allr = np.concatenate([c[1] for c in curves.values()])
    rng = float(allr.max() - allr.min())
    chords, report = [], []
    for sep, n_max in separations.items():
        mid, sm, n, msep = curves[sep]
        idx = local_maxima(
            mid,
            sm,
            n_max,
            min_prominence_frac * rng,
            period=np.pi if sep == 180 else 2 * np.pi,
        )
        if len(idx) < n_max:
            report.append(
                f"{sep:3d}°: only {len(idx)} peak(s) clear the prominence threshold"
            )
        for k, i in enumerate(idx):
            a, c = mid[i] - np.radians(sep) / 2, mid[i] + np.radians(sep) / 2
            report.append(
                f"{sep:3d}° max #{k + 1}: objects at {np.degrees(a) % 360:5.1f}° and "
                f"{np.degrees(c) % 360:5.1f}°, {sm[i]:5.2f} Hz (smoothed), {n[i]} "
                f"trials, their mean separation {msep[i]:.0f}°"
            )
            chords.append((a, c, sm[i], sep))
    return chords, report

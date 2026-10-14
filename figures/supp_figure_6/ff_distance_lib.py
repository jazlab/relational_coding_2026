r"""Figure 4's future-field model with a **spatial (distance-dependent) receptive field** and
**convergent remapping**.

Figure 4's future-field model (`figure_4/models_lib.FutureFieldModel`)
----------------------------------------------------------------------
The receptive field is a von Mises in **direction only**, ``f(φ) = vM(φ; θ, width)``, and for a
planned saccade to object B the future-field drive from object A is ``f`` of the *direction* of A
as seen from B (`models_lib.remap_angle`). Distance never enters, and remapping is perfect::

    rate = b + a_ret·[f(φ_A) + f(φ_B)] + a_ff·[f(dir(A − B)) + f(dir(B − A))]

The model here
--------------
Everything lives in the plane, in units of the ring radius ``R = 1``. Object X sits at
``p_X = (cos φ_X, sin φ_X)``.

* **Receptive field.** A circular 2-D Gaussian bump centred at ``c = ρ·(cos θ, sin θ)``, with
  ``ρ = rf_ecc`` (1.2 by default: just outside the ring) **held fixed**, and spatial s.d.
  ``σ = width`` (in ring radii) **fit**::

      g(x) = exp(−|x − c|² / 2σ²)

  On the ring itself, ``|p_X − c|² = 1 + ρ² − 2ρ cos(φ_X − θ)``, so the retinotopic drive is
  *exactly* a von Mises in φ_X with concentration ``ρ/σ²``. The retinotopic half of the model is
  therefore the same family as figure 4's; only the future field changes.

* **Future field, perfect remapping (κ = 0).** For a planned saccade to target T, the receptive
  field is carried along with the eye: it sits at ``p_T + c`` on the screen, i.e. at ``c``
  relative to the target. Object X drives it by ``g(p_X − p_T)``. Unlike figure 4's model this
  depends on how **far** X is from T, not only on the direction.

* **Convergent remapping (Zirnsak et al., 2014).** The remapped field is pulled toward the saccade
  target: its offset from T shrinks from ``c`` to ``(1 − κ)·c``, so its eccentricity *relative to
  the target* falls by the factor ``1 − κ``. Size (σ) is unchanged. ``κ = 0`` is perfect remapping,
  ``κ = 1`` puts every future field on the target itself::

      h_T(x) = exp(−|x − p_T − (1 − κ)·c|² / 2σ²)

* **Rate.** Each object is a potential saccade target, with equal weight, as in figure 4::

      rate = b + a_ret·[g(p_A) + g(p_B)] + a_ff·[h_B(p_A) + h_A(p_B)]

  ``h_B(p_A)`` is A's drive on the future field for a saccade to B. By default the target's own
  drive on its future field, ``h_T(p_T) = exp(−(1 − κ)²ρ² / 2σ²)``, is **left out**, as figure 4
  leaves it out (the target is at the new fovea, where ``remap_angle`` is undefined). It is the
  same on every trial, so including it (``include_target=True``) only adds
  ``2·a_ff·h_T(p_T)`` to the baseline — but with the baseline bounded at 0 that can matter.

With ``κ = 1`` the future-field term is ``2·exp(−|p_A − p_B|² / 2σ²)``: a function of the
objects' separation alone, blind to where the pair is. Whatever relational structure the fits
show must then come from the receptive-field term.

Fitting
-------
Five free parameters — ``baseline, theta, width (σ), amplitude_retinotopic,
amplitude_remapped`` — with ``rf_ecc`` and ``convergence`` fixed per fit. Fit by figure 4's own
multi-start maximum-likelihood routine (`fit_lib._fit`: Poisson likelihood of the 2-s spike count
on every 2-object trial, coarse L-BFGS-B over all starts then the best polished), on the same
trials figure 4's fits use (`data_lib.session_unit_trials`). Starts: ``n_theta_starts`` preferred
directions × three σ × ``a_ff ∈ {0, a/2}``.

Fits are cached one row per (neuron, ρ, κ, include_target) in ``cache/fits.csv`` in this
directory, so changing the κ grid fits only the new values.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
FIGURE_4 = _HERE.parent / "figure_4"
if str(FIGURE_4) not in sys.path:
    sys.path.insert(1, str(FIGURE_4))

import data_lib as dl  # noqa: E402  figure 4
import fit_lib as fl  # noqa: E402
import models_lib as ml  # noqa: E402

EXAMPLES_NOTEBOOK = FIGURE_4 / "figure_example_neurons.ipynb"
CACHE_PATH = _HERE / "cache" / "fits.csv"
SIGMA_STARTS = (0.25, 0.5, 1.0)  # σ starting values, ring radii
PARAMS = (
    "baseline",
    "theta",
    "width",
    "amplitude_retinotopic",
    "amplitude_remapped",
)


def example_units():
    """``[(subject, session, unit, rotation_deg), ...]``: `figure_example_neurons.ipynb`'s ``UNITS``
    (uncommented entries), read from that notebook so the two figures show the same neurons.
    """
    return [
        tuple(u) for u in dl.notebook_literal(str(EXAMPLES_NOTEBOOK), "UNITS")
    ]


# ── Geometry ──────────────────────────────────────────────────────────────────
def ring_xy(phi):
    """Position on the unit ring, ``(x, y)``."""
    phi = np.asarray(phi, float)
    return np.cos(phi), np.sin(phi)


def gauss(dx, dy, cx, cy, sigma):
    """Circular Gaussian bump peaking at 1, centred at ``(cx, cy)``."""
    return np.exp(
        -((dx - cx) ** 2 + (dy - cy) ** 2) / (2.0 * max(sigma, 1e-6) ** 2)
    )


# ── The model ─────────────────────────────────────────────────────────────────
class DistanceFutureFieldModel(ml.Model):
    """See the module docstring. ``width`` is σ, the Gaussian's s.d. in ring radii."""

    name = "future_field_distance"
    label = "Future Field (distance)"
    params = PARAMS
    _one_object_amplitude = "amplitude_retinotopic"

    def __init__(self, rf_ecc=1.2, convergence=0.0, include_target=False):
        super().__init__()
        self.rf_ecc = float(rf_ecc)
        self.convergence = float(convergence)
        self.include_target = bool(include_target)

    def center(self, p):
        """``c``, the receptive field's centre relative to fixation."""
        return self.rf_ecc * np.cos(p["theta"]), self.rf_ecc * np.sin(
            p["theta"]
        )

    def rf(self, x, y, p):
        """The receptive field ``g`` at screen position ``(x, y)`` (fixation at the origin)."""
        cx, cy = self.center(p)
        return gauss(x, y, cx, cy, p["width"])

    def future_field(self, x, y, tx, ty, p):
        """``h_T`` at ``(x, y)`` for a planned saccade to ``(tx, ty)``."""
        cx, cy = self.center(p)
        k = 1.0 - self.convergence
        return gauss(x - tx, y - ty, k * cx, k * cy, p["width"])

    def drives(self, p, phi_a, phi_b):
        """``(retinotopic, remapped)`` drives, before their amplitudes."""
        xa, ya = ring_xy(phi_a)
        xb, yb = ring_xy(phi_b)
        ret = self.rf(xa, ya, p) + self.rf(xb, yb, p)
        rem = self.future_field(xa, ya, xb, yb, p) + self.future_field(
            xb, yb, xa, ya, p
        )
        if self.include_target:
            rem = rem + 2.0 * self.future_field(0.0, 0.0, 0.0, 0.0, p)
        return ret, rem

    def rate_2obj(self, p, phi_a, phi_b):
        ret, rem = self.drives(p, phi_a, phi_b)
        return (
            p["baseline"]
            + p["amplitude_retinotopic"] * ret
            + p["amplitude_remapped"] * rem
        )

    def rate_1obj(self, p, phi):
        x, y = ring_xy(phi)
        return p["baseline"] + p["amplitude_retinotopic"] * self.rf(x, y, p)

    def init_params(self, s):
        return dict(
            baseline=s["baseline"],
            theta=s["theta"],
            width=SIGMA_STARTS[1],
            amplitude_retinotopic=s["amplitude"],
            amplitude_remapped=0.5 * s["amplitude"],
        )


# ── Fitting ───────────────────────────────────────────────────────────────────
def unit_trials(subject, session, unit_id):
    """The neuron's 2-object trials as figure 4's fits see them."""
    return dl.session_unit_trials(subject, session, verbose=False)[unit_id]


def _d2(unit):
    return (unit["t0"], unit["t1"], unit["fr2"], np.zeros(len(unit["fr2"])))


def _starts(model, p0, cfg):
    out = []
    for k in range(cfg.n_theta_starts):
        th = p0["theta"] + 2 * np.pi * k / cfg.n_theta_starts
        for sig in SIGMA_STARTS:
            for a_rem in (0.0, p0["amplitude_remapped"]):
                out.append(
                    {
                        **p0,
                        "theta": th,
                        "width": sig,
                        "amplitude_remapped": a_rem,
                    }
                )
    return out


def constant_ll(unit):
    fr = unit["fr2"]
    return float(
        np.sum(ml.poisson_ll_per_trial(fr, np.full(len(fr), fr.mean())))
    )


def log_lik(model, params, unit):
    """Total all-trial log-likelihood of ``params``."""
    return -fl._neg_ll([], model, (), params, _d2(unit))


def fit(model, unit, cfg=None):
    """``(params, ll)``: the all-trial maximum-likelihood fit with ``rf_ecc`` and ``convergence``
    held at the model's values."""
    cfg = cfg or fl.FitConfig()
    p0 = model.init_params(ml.unit_init_stats(unit))
    return fl._fit(model, PARAMS, p0, _d2(unit), _starts(model, p0, cfg), cfg)


def _key(subject, session, unit_id, model):
    return (
        subject,
        str(session),
        int(unit_id),
        model.rf_ecc,
        model.convergence,
        model.include_target,
    )


_KEY_COLS = [
    "subject",
    "session",
    "unit",
    "rf_ecc",
    "convergence",
    "include_target",
]


def cached_fit(subject, session, unit_id, model, cfg=None, verbose=True):
    """:func:`fit`, cached one row per (neuron, ρ, κ, include_target) in ``CACHE_PATH``.
    Returns ``(params, dll)`` with ``dll`` the all-trial log-likelihood per trial above a
    constant-rate model (nats/trial)."""
    cfg = cfg or fl.FitConfig()
    key = _key(subject, session, unit_id, model)
    df = (
        pd.read_csv(CACHE_PATH)
        if CACHE_PATH.exists()
        else pd.DataFrame(columns=_KEY_COLS)
    )
    if len(df):
        df["session"] = df["session"].astype(str)
        hit = df[
            (df["cfg_tag"] == cfg.tag)
            & (df["subject"] == subject)
            & (df["session"] == str(session))
            & (df["unit"] == int(unit_id))
            & np.isclose(df["rf_ecc"], model.rf_ecc)
            & np.isclose(df["convergence"], model.convergence)
            & (df["include_target"].astype(bool) == model.include_target)
        ]
        if len(hit):
            r = hit.iloc[0]
            return {k: float(r[k]) for k in PARAMS}, float(r["dll"])
    unit = unit_trials(subject, session, unit_id)
    p, ll = fit(model, unit, cfg)
    dll = (ll - constant_ll(unit)) / len(unit["fr2"])
    if verbose:
        print(
            f"  fit {subject}/{session} unit {unit_id}  ρ={model.rf_ecc:g} "
            f"κ={model.convergence:.2f}: ΔLL {dll:.3f} nats/trial"
        )
    row = pd.DataFrame(
        [
            dict(
                zip(_KEY_COLS, key),
                **p,
                ll=ll,
                dll=dll,
                n_2obj=len(unit["fr2"]),
                cfg_tag=cfg.tag,
            )
        ]
    )
    df = pd.concat([df, row], ignore_index=True) if len(df) else row
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    os.replace(
        tmp, CACHE_PATH
    )  # one row at a time: an interrupt loses one fit
    return p, dll


def figure4_fit(subject, session, unit_id, cfg=None):
    """Figure 4's cached all-trial future-field fit, and its ΔLL (nats/trial)."""
    import ring_lib

    cfg = cfg or fl.FitConfig()
    p = ring_lib.unit_fits(subject, session, unit_id, cfg, verbose=False)[
        "future_field"
    ]
    model = ml.build_models(["future_field"])["future_field"]
    unit = unit_trials(subject, session, unit_id)
    dll = (log_lik(model, p, unit) - constant_ll(unit)) / len(unit["fr2"])
    return model, p, dll

r"""The single-neuron firing-rate models, their parameters, and the Poisson likelihood.

**Everything here is about 2-object trials.** Each model predicts a **firing rate in Hz**
as an explicit additive function of the two object angles (an *identity* link with a
non-negative baseline — not the log link of ``figure_3/model_comparison``), and is fit by
maximising the Poisson log-likelihood of the observed spike counts on 2-object trials
alone. See ``README.md`` for how this differs from ``../neuron_models``.

Shared structure
----------------
Every model is built out of one tuning function ``f(φ)`` with a preferred direction ``θ``
and (for the von Mises) a ``width``, plus a non-negative ``baseline`` and one or more
non-negative amplitudes. They differ only in how the drives from the two objects are
combined:

======================= ============================================================
model                    2-object firing rate
======================= ============================================================
Relational               ``b + a_ret·[f(φ_A)+f(φ_B)] + a_rem·[f(r_AB) + f(r_BA)]``
Relational bias          ``b + a·[f(φ̃_A(bias)) + f(φ̃_B(bias))]``
Softmax                  ``b + a·(f_A e^{β f_A} + f_B e^{β f_B})/(e^{β f_A}+e^{β f_B})``
Normalization            ``b + a·(f_A + f_B)/(f_A + f_B + α)``
Inhibition (tied)        ``b + a·[(f_{E,A}+f_{E,B}) − (f_{I,A}+f_{I,B})]``
Basis function           ``b + a·(f_A + f_B)·(1 + β·cos 2(ψ − θ))``, β ≥ 0
Priority map             ``b + a·[s·f_A + (1−s)·f_B]``, ``s = σ(λ·[cos(φ_A−θ_p) − cos(φ_B−θ_p)])``
Weighted average         ``b + a·[(1+γ·d_A)·f(φ_A) + (1+γ·d_B)·f(φ_B)]``
Constant                 ``b`` — the reference every likelihood is measured against
======================= ============================================================

``r_AB = remap(φ_A, φ_B)`` is the direction of A as seen from B (:func:`remap_angle`), and
``φ̃_A(α)`` is the direction of A as seen from a point ``α`` of the way from fixation to
the pair's midpoint (:func:`bias_angles`). Both are defined below.

The tuning function
-------------------
One shape, used by every model::

    f(φ) = exp[ (cos(φ − θ) − 1) / width² ]

This is a **von Mises bump, normalised to peak at 1** rather than to integrate to 1: it is
exactly proportional to a von Mises density with concentration ``κ = 1/width²``, differing
only by the constant ``2π·I₀(κ)·e^{−κ}``. Since every model multiplies ``f`` by a free
amplitude, that constant is absorbed and the family is identical to the density's — the
peak-normalised form just makes ``amplitude`` readable as "Hz at the preferred direction".

``width`` is an angular scale in radians. For narrow bumps it is close to the standard
deviation of a wrapped Gaussian (the two agree to 0.006 at ``width = 0.3``, diverging to 0.3
by ``width = 1.5``). The half-width at half maximum is ``arccos(1 − width²·ln2)`` — about
``1.177·width`` when narrow; use :func:`hwhm_deg` / :func:`width_from_hwhm` to convert.

.. note::
   A plain cosine ``f(φ) = cos(φ − θ)`` used to be available as a fixed-shape alternative and
   has been removed. It **cannot test the relational hypothesis**: ``remap(A, B)`` and
   ``remap(B, A)`` differ by exactly π and ``cos(x) + cos(x + π) = 0``, so the remapped drive
   vanishes identically and the relational model, the relational-bias model at ``α = 1`` and
   the inhibition model all collapse to the retinotopic one. Expressing relational tuning
   needs an even harmonic in ``f``, which the von Mises has and the pure cosine does not.

Non-negative rates
------------------
Amplitudes, baselines and widths are constrained non-negative. One thing can still push a
predicted rate below zero: the inhibition model's ``−a·f_I`` term. Rather than clipping — which creates a flat region the optimiser cannot escape —
the rate is passed through a **smooth softplus rectifier** before it enters the
likelihood::

    rate_used = 0.05 · log(1 + exp(rate / 0.05)) + 1e-6      (Hz)

With ``RECTIFY_SCALE = 0.05`` Hz this is indistinguishable from the identity above ~0.3 Hz
(the error at 1 Hz is < 1e-8 Hz) and smoothly folds negative predictions down towards zero
where the Poisson likelihood punishes them. It is applied to **every** model, so no model
is advantaged by it.
"""

import data_lib
import numpy as np
from scipy.special import gammaln

# ── Rectification and the likelihood ──────────────────────────────────────────
RECTIFY_SCALE = 0.05  # Hz — softness of the positive-part rectifier
RATE_FLOOR = 1e-6  # Hz — keeps log(rate) finite


def rectify(rate):
    """Smooth positive part of a predicted rate (see the module docstring)."""
    return (
        RECTIFY_SCALE
        * np.logaddexp(0.0, np.asarray(rate, float) / RECTIFY_SCALE)
        + RATE_FLOOR
    )


def poisson_ll_per_trial(obs_fr, pred_rate, duration=None):
    """Per-trial Poisson log-likelihood (nats) of the observed spike counts
    ``obs_fr · duration`` under the predicted mean count ``rectify(pred_rate) · duration``.
    The ``log Γ(y+1)`` term is **kept**, so the numbers are true log-likelihoods and are
    comparable across trials with different spike counts.

    ``duration`` defaults to ``data_lib.DURATION_S`` read **at call time**, not at import, so
    a caller that changes the analysis window (`supp_figure_2`) changes the likelihood too.
    """
    duration = data_lib.DURATION_S if duration is None else duration
    lam = rectify(pred_rate) * duration
    count = np.asarray(obs_fr, float) * duration
    return count * np.log(lam) - lam - gammaln(count + 1.0)


# ── Tuning functions ──────────────────────────────────────────────────────────
def von_mises(phi, theta, width):
    """Von Mises bump normalised to peak at 1.0 (see the module docstring)."""
    return np.exp(
        (np.cos(np.asarray(phi, float) - theta) - 1.0) / max(width, 1e-6) ** 2
    )


def chord_axis(phi_a, phi_b):
    r"""``(cos 2ψ, sin 2ψ)`` for the **orientation of the chord** joining two objects.

    ψ is the analogue, for a pair of objects, of the bar orientation Deneve & Pouget's
    basis units are tuned to: the direction of the line through the two of them, defined
    modulo π because the pair is unordered. It is returned as a doubled-angle unit vector
    rather than as ψ itself — that keeps it smooth (no ``atan2`` branch cut), makes the
    π-periodicity automatic, and lets a tuning for it be written as a dot product.

    Computed from the chord vector ``p_B − p_A`` rather than from the two directions'
    circular mean, which is a different quantity: the mean is undefined for an antipodal
    pair, whereas the chord through an antipodal pair is a perfectly good diameter. The
    only degenerate case left is two objects at the same place, which the task never shows;
    it returns ``(0, 0)``, i.e. no modulation.
    """
    phi_a, phi_b = np.asarray(phi_a, float), np.asarray(phi_b, float)
    dx = np.cos(phi_b) - np.cos(phi_a)
    dy = np.sin(phi_b) - np.sin(phi_a)
    n = np.hypot(dx, dy)
    safe = np.where(n > 1e-12, n, 1.0)
    ux, uy = dx / safe, dy / safe
    keep = n > 1e-12
    return np.where(keep, ux * ux - uy * uy, 0.0), np.where(
        keep, 2 * ux * uy, 0.0
    )


def width_from_hwhm(hwhm_degrees):
    """Inverse of :func:`hwhm_deg` — the ``width`` whose HWHM is ``hwhm_degrees``."""
    return float(
        np.sqrt((1.0 - np.cos(np.radians(hwhm_degrees))) / np.log(2.0))
    )


# ── The relational coordinate transform ───────────────────────────────────────
def remap_angle(phi_a, phi_b):
    """The direction of object **A** as seen from object **B**.

    Both objects sit on a ring of the same (constant) eccentricity ``R``, at angles
    ``phi_a`` and ``phi_b``. Their positions are ``R·(cos φ, sin φ)``, so the vector from B
    to A is ``R·(cos φ_A − cos φ_B, sin φ_A − sin φ_B)`` and its direction is the
    ``arctan2`` below. ``R`` cancels, which is why this transform needs no eccentricity
    argument.

    Three identities worth knowing:

    * ``remap(A, B)`` and ``remap(B, A)`` differ by **exactly π** — B sees A in the
      opposite direction from the one in which A sees B. The relational model always uses
      the *sum* of the two remapped drives, so it is insensitive to which object is which.
    * ``remap(A, B) = (φ_A + φ_B)/2 ± π/2``. The remapped direction is therefore
      **perpendicular to the chord's bisector** and depends only on the pair's *mean*
      angle, not on their separation.
    * **For antipodal pairs the remap is the identity**: when ``φ_B = φ_A + π`` the vector
      from B to A passes through fixation, so ``remap(A, B) = φ_A``. The relational model
      then reduces *exactly* to the retinotopic one, and 180°-apart trials carry **no**
      information about which of the two is right. This is a property of the stimulus set,
      not of the fit: it costs effective sample size (about a fifth of the 2-object trials
      are more than 160° apart) but it cannot bias the comparison, since both models make
      the same prediction there.

    Undefined when the two objects coincide; the stimulus set never separates them by less
    than ~79°, and ``arctan2(0, 0) = 0`` keeps it finite regardless.
    """
    phi_a = np.asarray(phi_a, float)
    phi_b = np.asarray(phi_b, float)
    return np.arctan2(
        np.sin(phi_a) - np.sin(phi_b), np.cos(phi_a) - np.cos(phi_b)
    )


# ── The relational-bias coordinate transform ──────────────────────────────────
# ── The relational-bias coordinate transform ──────────────────────────────────


def bias_angles(phi_a, phi_b, bias):
    r"""The two objects' directions **as seen from** ``pos_bias`` — the relational-bias
    model's coordinate transform. Returns ``(phi_a_tilde, phi_b_tilde)`` in radians.

    With both objects on a ring of eccentricity ``R``, writing ``u_X = (cos φ_X, sin φ_X)``,

    .. math::

        p_A - \mathrm{pos} = R\Big[(1 - \tfrac{b}{2})\,u_A - \tfrac{b}{2}\,u_B\Big],

    so ``R`` cancels — **the transform is scale-free** and needs no eccentricity argument.

    The two endpoints:

    * ``bias = 0`` → the angles are the ordinary **retinotopic** ones.
    * ``bias = 1`` → ``p_A − m = (p_A − p_B)/2``, a positive multiple of the vector from B to
      A, so the direction is that of A **as seen from B**: the fully **relational** frame.

    In between the reference point slides along the segment from fixation to the midpoint and
    the angles interpolate continuously.
    """
    phi_a, phi_b = np.asarray(phi_a, float), np.asarray(phi_b, float)
    h = 0.5 * np.asarray(bias, float)
    ca, sa, cb, sb = np.cos(phi_a), np.sin(phi_a), np.cos(phi_b), np.sin(phi_b)
    tilde_a = np.arctan2((1.0 - h) * sa - h * sb, (1.0 - h) * ca - h * cb)
    tilde_b = np.arctan2((1.0 - h) * sb - h * sa, (1.0 - h) * cb - h * ca)
    return tilde_a, tilde_b


# ── Parameter bounds ──────────────────────────────────────────────────────────
# Amplitudes, baselines and widths are non-negative, as specified. `theta` is periodic, so
# its "bounds" are only there to keep L-BFGS-B in a finite box — multi-start over the
# circle, not the box, is what handles its multimodality. Two parameter names mean
# different things in different models (`alpha`, `beta`); those models override the entry
# via their class-level `bounds` (see `Model.param_bounds`).
BOUNDS = {
    "bias": (0.0, 1.0),  # the relational-bias reference point
    "gamma": (-3.0, 3.0),  # contrast weighting; see the model
    "baseline": (0.0, 500.0),
    "theta": (-4 * np.pi, 4 * np.pi),
    "theta_E": (-4 * np.pi, 4 * np.pi),
    "theta_I": (-4 * np.pi, 4 * np.pi),
    "width": (0.05, 4.0),
    "amplitude": (0.0, 500.0),
    "amplitude_retinotopic": (0.0, 500.0),
    "amplitude_remapped": (0.0, 500.0),
    "alpha": (1e-6, 1e3),  # normalization's semi-saturation, > 0
    "beta": (-50.0, 50.0),  # softmax; exp(beta·f) must not overflow
    "theta_p": (
        -4 * np.pi,
        4 * np.pi,
    ),  # the priority map's preferred direction
}

WIDTH_PARAMS = ("width",)
ANGLE_PARAMS = ("theta", "theta_E", "theta_I", "theta_p")

# The upper bound on `bias`, and it is 1.0 for a reason that is not "because bias is a
# weight". Allowing it past 1 does not extrapolate to a *new* hypothesis, it re-describes an
# old one:
#
#     drive(theta, 2 - bias)  ==  drive(theta + pi, bias)      exactly, for every bias
#
# because replacing `bias` by `2 - bias` maps p_A - pos to -(p_B - pos) (see `bias_angles`),
# so the two transformed angles swap and both flip by pi. `bias` in [0, 2] with a free
# `theta` therefore covers every model in the family exactly *twice*, and the likelihood has
# a mirror-image optimum for every real one. Left unbounded, roughly one synthetic neuron in
# six lands in the mirror and reports a preferred direction 180 deg from the truth alongside
# a bias of 2 - bias_true — which would then be read as "more relational than relational"
# when it is the *same* neuron.
#
# So the fitted range is the identifiable one, [0, 1]. Note that bias = 1 is the symmetry's
# own fixed point: the fully relational model genuinely cannot tell theta from theta + pi
# (its two transformed angles differ by exactly pi), so a fitted theta from a neuron at
# bias = 1 is one of two equally good answers. Nothing else in [0, 1) is ambiguous.
BIAS_MAX = 1.0


# ── Models ────────────────────────────────────────────────────────────────────
class Model:
    """Base class. Subclasses define ``name``, ``label``, ``params``, ``rate_2obj`` and
    ``init_params``; ``rate_1obj`` has a shared default (see below).

    ``self.f`` is the shared tuning function; ``self.tuning`` its name.

    **``rate_1obj`` is never fit and never scored** in this directory — nothing here reads
    1-object trials. It is defined because the tuning-curve panels draw the 1-object row as
    a *check*: the receptive field these models infer from 2-object trials alone either
    does or does not match the neuron's measured single-object tuning, and that is worth
    seeing. Read those panels as a prediction, not as a fit.
    """

    name = ""
    label = ""
    #: How the model is named in a figure, where space is tight. Defaults to `label`.
    short_label = None
    params = ()  # every parameter, in a canonical order
    bounds = {}  # per-model overrides of the global BOUNDS
    #: True for a model whose rate needs a per-trial covariate beyond the two angles. Only
    #: the contrast-weighted model sets it; :func:`rate_2obj` is what dispatches on it, so
    #: nothing else has to know which models are which.
    needs_contrast = False

    def __init__(self):
        self.f = von_mises

    # -- to be overridden -----------------------------------------------------
    def rate_2obj(self, p, phi_a, phi_b):
        raise NotImplementedError

    def init_params(self, stats):
        """Starting values from cheap summary statistics of the neuron (see
        :func:`unit_init_stats`)."""
        raise NotImplementedError

    def rate_1obj(self, p, phi):
        """The single-bump prediction ``b + a·f(φ)`` shared by every model but the
        inhibition one. Extrapolation, not a fit — see the class docstring."""
        return p["baseline"] + p[self._one_object_amplitude] * self.f(
            phi, p["theta"], p.get("width")
        )

    _one_object_amplitude = "amplitude"

    # -- shared ---------------------------------------------------------------
    def param_bounds(self, name):
        """Optimiser bounds for one parameter, with this model's overrides applied."""
        return self.bounds.get(name, BOUNDS[name])

    def free_params(self, fixed_width=False):
        """The parameters actually optimised — everything except a width that is pinned
        by the configuration or absent from the tuning function."""
        return tuple(
            n for n in self.params if not self._frozen(n, fixed_width)
        )

    def _frozen(self, name, fixed_width):
        if name in WIDTH_PARAMS:
            return fixed_width
        return False

    def __repr__(self):
        return f"{type(self).__name__}()"


class FutureFieldModel(Model):
    """A retinotopic drive **plus** a drive from each object's position in a coordinate
    frame centred on the other object (see :func:`remap_angle`).

        2 object : baseline + amplitude_retinotopic · (f(φ_A) + f(φ_B))
                            + amplitude_remapped    · (f(r(φ_A, φ_B)) + f(r(φ_B, φ_A)))

    ``amplitude_remapped = 0`` recovers the retinotopic model exactly, so the relational
    model **nests** it and the fitted ``amplitude_remapped`` is a direct, continuous
    measure of how relational the neuron is. The two amplitudes are the natural read-out;
    The two amplitudes are the natural read-out; their ratio is scale-free.
    """

    name = "future_field"
    label = "Future Field"
    params = (
        "baseline",
        "theta",
        "width",
        "amplitude_retinotopic",
        "amplitude_remapped",
    )
    _one_object_amplitude = "amplitude_retinotopic"

    def rate_2obj(self, p, phi_a, phi_b):
        w = p.get("width")
        ret = self.f(phi_a, p["theta"], w) + self.f(phi_b, p["theta"], w)
        rem = self.f(remap_angle(phi_a, phi_b), p["theta"], w) + self.f(
            remap_angle(phi_b, phi_a), p["theta"], w
        )
        return (
            p["baseline"]
            + p["amplitude_retinotopic"] * ret
            + p["amplitude_remapped"] * rem
        )

    def init_params(self, s):
        return dict(
            baseline=s["baseline"],
            theta=s["theta"],
            width=s["width"],
            amplitude_retinotopic=s["amplitude"],
            amplitude_remapped=0.5 * s["amplitude"],
        )


class RelationalBiasModel(Model):
    r"""**One continuous axis from retinotopic to relational**, obtained by moving the point
    the neuron measures object direction *from*.

        2 object : baseline + amplitude · (f(φ̃_A) + f(φ̃_B))

    where ``(φ̃_A, φ̃_B) = bias_angles(φ_A, φ_B, bias)`` are the two objects' directions seen
    from

        pos = (1 − bias) · fixation  +  bias · midpoint(A, B)      (in Cartesian space)

    — the reference point slid a fraction ``bias`` of the way from the centre of the ring to
    the midpoint of the pair.

    ====== ================================= ==========================================
    bias    reference point                   the model is
    ====== ================================= ==========================================
    0       fixation                          **retinotopic**: ``b + a·[f(φ_A)+f(φ_B)]``
    1       the pair's midpoint               **relational**: each object seen from the other
    ====== ================================= ==========================================

    Five parameters — ``baseline``, ``amplitude``, ``theta`` (the mean, i.e. the preferred
    direction), ``width``, and ``bias``. The fitted ``bias`` is this directory's measure of
    relationality: one interpretable number per neuron on a fixed [0, 1] scale.

    Two things to keep straight when reading it:

    * **``bias`` is well determined only where the transform actually moves the angles.**
      For an antipodal pair the midpoint *is* fixation, so ``pos = 0`` for every ``bias`` and
      the trial says nothing about it. About a fifth of 2-object trials are more than 160°
      apart.
    * **``bias`` is bounded to [0, 1] because that is the identifiable range**, not merely
      because it is a weight — see :data:`BIAS_MAX`.
    """

    name = "relational_bias"
    label = "Relational bias"
    params = ("baseline", "theta", "width", "amplitude", "bias")
    bounds = {"bias": (0.0, BIAS_MAX)}

    def rate_2obj(self, p, phi_a, phi_b):
        w = p.get("width")
        tilde_a, tilde_b = bias_angles(phi_a, phi_b, p["bias"])
        return p["baseline"] + p["amplitude"] * (
            self.f(tilde_a, p["theta"], w) + self.f(tilde_b, p["theta"], w)
        )

    def rate_1obj(self, p, phi):
        """``b + a·f(φ)`` — and that is the model's own ``bias < 1`` limit, not a convention.
        With a single object the "midpoint of the objects" is the object itself, so
        ``pos = bias·p_A`` and ``p_A − pos = (1 − bias)·p_A`` points along ``φ_A`` for every
        ``bias < 1``: the transform leaves a lone object's direction unchanged.
        """
        return p["baseline"] + p["amplitude"] * self.f(
            phi, p["theta"], p.get("width")
        )

    def init_params(self, s):
        return dict(
            baseline=s["baseline"],
            theta=s["theta"],
            width=s["width"],
            amplitude=s["amplitude"],
            bias=0.5,
        )


class SoftmaxModel(Model):
    """A soft winner-take-all over the two single-object drives: the paired response is a
    ``beta``-weighted softmax average of ``f(φ_A)`` and ``f(φ_B)``.

        2 object : baseline + amplitude · (f_A·e_A + f_B·e_B) / (e_A + e_B),
                   e_X = exp(beta · f_X)

    ``beta = 0`` is the plain **average** of the two drives, ``beta → +∞`` the **max**
    (winner-take-all), ``beta → −∞`` the **min**.
    """

    name = "softmax"
    label = "Softmax"
    params = ("baseline", "theta", "width", "amplitude", "beta")

    def rate_2obj(self, p, phi_a, phi_b):
        w = p.get("width")
        fa = self.f(phi_a, p["theta"], w)
        fb = self.f(phi_b, p["theta"], w)
        beta = p["beta"]
        # Softmax weights, computed in a max-subtracted form so exp() cannot overflow.
        m = np.maximum(beta * fa, beta * fb)
        ea = np.exp(beta * fa - m)
        eb = np.exp(beta * fb - m)
        return p["baseline"] + p["amplitude"] * (fa * ea + fb * eb) / (ea + eb)

    def init_params(self, s):
        return dict(
            baseline=s["baseline"],
            theta=s["theta"],
            width=s["width"],
            amplitude=s["amplitude"],
            beta=0.0,
        )


class NormalizationModel(Model):
    """Divisive normalisation of the summed drive by itself plus a semi-saturation
    constant ``alpha``, written so that ``amplitude`` is the **low-drive gain**.

        2 object : baseline + amplitude · (f_A + f_B) / (1 + (f_A + f_B)/alpha)

    Large ``alpha`` is near-linear summation — the response tends to
    ``baseline + amplitude·(f_A + f_B)``, i.e. the **retinotopic model with the same
    amplitude** — while small ``alpha`` saturates, the paired response approaching
    ``baseline + amplitude·alpha`` whenever either object drives the neuron at all.

    Why it is written this way. The textbook form is
    ``baseline + a·D/(D + alpha)`` with ``D = f_A + f_B``, whose ``a`` is the *saturated
    asymptote* rather than a gain; that form reaches the retinotopic limit only along
    ``a = amplitude·alpha``, and since both are boxed (:data:`BOUNDS`) the product runs into
    ``amplitude ≤ 500`` before ``alpha`` is large enough for the ratio to be linear. Fits of
    ordinary high-rate neurons then stop at the bound rather than at the data, and the model
    comes out *worse than the retinotopic one on its own training data* — which a strictly
    larger family cannot be. The two forms describe exactly the same curves (``a =
    amplitude·alpha``); this one puts the retinotopic limit inside the box, where the
    optimiser can reach it. See the README §2.1.

    The model is written for a **non-negative** drive, which the von Mises ``f`` always
    is; the denominator is guarded numerically.
    """

    name = "normalization"
    label = "Normalization"
    params = ("baseline", "theta", "width", "amplitude", "alpha")

    def rate_2obj(self, p, phi_a, phi_b):
        w = p.get("width")
        drive = self.f(phi_a, p["theta"], w) + self.f(phi_b, p["theta"], w)
        denom = 1.0 + drive / p["alpha"]
        denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)  # numerical guard
        return p["baseline"] + p["amplitude"] * drive / denom

    def init_params(self, s):
        return dict(
            baseline=s["baseline"],
            theta=s["theta"],
            width=s["width"],
            amplitude=s["amplitude"],
            alpha=1.0,
        )


class InhibitionModel(Model):
    """An excitatory bump minus a suppressive bump at a **different** preferred direction —
    the single-neuron analogue of a tuned suppressive pool, with the two fields tied to the
    same width and the same amplitude so that the model has five parameters like the rest.

        2 object : baseline + amplitude · [(f_E,A + f_E,B) − (f_I,A + f_I,B)]

    Because ``theta_E`` and ``theta_I`` are free, a neuron whose excitation and suppression
    are offset responds best to a particular *separation* between the objects — which can
    look relational without any coordinate transform. That is exactly why it is here.

    .. warning::
       Tying the amplitudes is **not** only a parameter-count change. With equal widths and
       equal amplitudes the two bumps have the same integral, so ``f_E − f_I`` integrates
       to **zero** around the ring: the model's ring-averaged rate is exactly ``baseline``,
       and it can only *redistribute* drive, never add a net excitatory or suppressive one.
       Every other model here (``b + a·f``, ``a ≥ 0``) has a strictly positive mean drive.
       A neuron whose objects raise its mean rate is something this model cannot express at
       all, and it may lose for that reason rather than for anything to do with combination
       rules. Read it as a *lower bound* on what the inhibition family can do at five
       parameters.
    """

    name = "inhibition"
    label = "Inhibition"
    params = ("baseline", "theta_E", "theta_I", "width", "amplitude")

    def rate_1obj(self, p, phi):
        w = p.get("width")
        return p["baseline"] + p["amplitude"] * (
            self.f(phi, p["theta_E"], w) - self.f(phi, p["theta_I"], w)
        )

    def rate_2obj(self, p, phi_a, phi_b):
        w = p.get("width")
        exc = self.f(phi_a, p["theta_E"], w) + self.f(phi_b, p["theta_E"], w)
        inh = self.f(phi_a, p["theta_I"], w) + self.f(phi_b, p["theta_I"], w)
        return p["baseline"] + p["amplitude"] * (exc - inh)

    def init_params(self, s):
        return dict(
            baseline=s["baseline"],
            theta_E=s["theta"],
            theta_I=s["theta"] + np.pi,
            width=s["width"],
            amplitude=s["amplitude"],
        )


class WeightedAverageModel(Model):
    r"""**The retinotopic model with each object's drive weighted by how visible it is.**

        2 object : baseline + amplitude · [ (1 + gamma·d_A)·f(φ_A) + (1 + gamma·d_B)·f(φ_B) ]

    where ``d_A`` is the **signed relative contrast** of object A against object B,

        d_A = (C_A − C_B) / (C_A + C_B),          d_B = −d_A

    and ``C`` is an identity's luminance contrast against the task background, measured from
    the rendered stimulus by :mod:`contrast_lib`. Because ``d_B = −d_A`` exactly, the rate can
    be written with a single covariate:

        baseline + amplitude · [ (f_A + f_B) + gamma·d_A·(f_A − f_B) ]

    — a retinotopic sum plus a term that tilts the drive towards whichever object is the more
    visible of the two.

    ``gamma = 0`` recovers the retinotopic model **exactly**, so the model nests it and the
    held-out ΔLL is attributable to the weighting alone. ``gamma > 0`` means the
    higher-contrast object dominates the response, ``gamma < 0`` the lower-contrast one.

    Five parameters — ``baseline``, ``theta``, ``width``, ``amplitude``, ``gamma`` — the same
    count as every other model here except the retinotopic one, which has four.

    .. note::
       **This model needs a per-trial covariate that the others do not.** ``rate_2obj`` takes
       a fourth argument, ``dctr``, the array of ``d_A`` values for the trials being
       evaluated; :func:`contrast_lib.relative_contrast` computes it from the two object
       identities. It is **required**, not optional: defaulting it to zero would silently
       turn this model into the retinotopic one and the mistake would look like a null
       result. Anything that fits or scores this model has to carry the identities alongside
       the angles.

    ``gamma`` is bounded to ``[-3, 3]``. The weight ``1 + gamma·d`` must stay non-negative,
    and the largest ``|d|`` this stimulus set produces is 0.331 (the b-c pair), so
    ``|gamma| ≤ 1/0.331 = 3.02`` is the range over which no object's drive can be driven
    negative. At the bound the less visible object contributes nothing at all.
    """

    name = "weighted_average"
    label = "Weighted average"
    short_label = "Weighted Avg."
    needs_contrast = True
    params = ("baseline", "theta", "width", "amplitude", "gamma")
    bounds = {"gamma": (-3.0, 3.0)}

    def rate_2obj(self, p, phi_a, phi_b, dctr=None):
        if dctr is None:
            raise ValueError(
                "weighted_average needs the per-trial relative contrast `dctr`; passing "
                "nothing would silently make it the retinotopic model. See "
                "contrast_lib.relative_contrast."
            )
        w = p.get("width")
        fa = self.f(phi_a, p["theta"], w)
        fb = self.f(phi_b, p["theta"], w)
        d = np.asarray(dctr, float)
        return p["baseline"] + p["amplitude"] * (
            (fa + fb) + p["gamma"] * d * (fa - fb)
        )

    def init_params(self, s):
        return dict(
            baseline=s["baseline"],
            theta=s["theta"],
            width=s["width"],
            amplitude=s["amplitude"],
            gamma=0.0,
        )


class ConstantModel(Model):
    """Constant rate — the reference every other model is scored against. Its single
    parameter is fit on the 2-object training trials of each split, so ΔLL is measured
    against the honest "this neuron's mean 2-object rate"."""

    name = "constant"
    label = "Constant"
    params = ("baseline",)

    def rate_1obj(self, p, phi):
        return np.full(np.shape(phi), p["baseline"], float)

    def rate_2obj(self, p, phi_a, phi_b):
        return np.full(np.shape(phi_a), p["baseline"], float)

    def init_params(self, s):
        return dict(baseline=s["mean_fr"])


class BasisFunctionModel(Model):
    r"""**Deneve & Pouget's basis-function unit, written for a pair of objects.**

        2 object : baseline + amplitude · (f_A + f_B) · (1 + beta · cos(2(psi − theta))),
                   beta ∈ [0, 1]

    where psi is the orientation of the chord joining the two objects (:func:`chord_axis`).

    Deneve & Pouget (2003) argue that object-centred behaviour needs no object-centred
    neuron: a unit keeps an ordinary retinotopic receptive field and is *gain-modulated* by a
    property of the object — in their case the orientation of a bar — with the object-centred
    answer appearing only in the population read-out. This is that claim written for a pair,
    and it is the direct competitor to the remapping account: **gain where that one says
    shift**. Across a family of pairs at fixed separation it keeps the preferred direction
    fixed and changes the height of the curve; a remapping model moves the peak.

    ``beta`` is restricted to ``[0, 1]``, so the gain is largest when the pair's chord lies
    **along** the cell's own preferred direction. That is the conjunction the account is
    about — the object variable agreeing with what is in the receptive field — and it is a
    deliberate restriction rather than a bound of convenience: allowing ``beta < 0`` lets the
    model describe the opposite claim, that the response is *cut* when the two objects
    straddle the receptive field, which is suppression between simultaneous objects rather
    than a basis-function conjunction. Fitted freely the two signs split roughly evenly, and
    the neurons on which the free model beat its rivals were overwhelmingly the negative
    ones — so the restriction is what keeps this model a test of the published idea.

    The paper's unit also has its own preferred object orientation, independent of its
    receptive field. Here that would be a sixth parameter, so it is tied: the gain peaks when
    the pair's chord aligns with the cell's own preferred direction.

    ``beta = 0`` is the retinotopic model **exactly**, which is what makes this model's
    held-out ΔLL readable. Note that zero is now a *corner* of the parameter box rather than
    an interior point, which is why the optimiser is started there as well as inside (see
    ``fit_lib._structural_variants``).
    """

    name = "basis_function"
    label = "Basis function"
    short_label = "Basis Func."
    params = ("baseline", "theta", "width", "amplitude", "beta")
    bounds = {"beta": (0.0, 1.0)}

    def rate_2obj(self, p, phi_a, phi_b):
        w = p.get("width")
        drive = self.f(phi_a, p["theta"], w) + self.f(phi_b, p["theta"], w)
        cos2, sin2 = chord_axis(phi_a, phi_b)
        aligned = cos2 * np.cos(2 * p["theta"]) + sin2 * np.sin(2 * p["theta"])
        return p["baseline"] + p["amplitude"] * drive * (
            1.0 + p["beta"] * aligned
        )

    def init_params(self, s):
        return dict(
            baseline=s["baseline"],
            theta=s["theta"],
            width=s["width"],
            amplitude=s["amplitude"],
            beta=0.0,
        )


LAMBDA_PRIORITY = 100.0  # selection sharpness; see PriorityMapModel


class PriorityMapModel(Model):
    r"""**Selection by a retinotopic priority map that need not agree with the receptive
    field.**

        2 object : baseline + amplitude · [ s·f(φ_A) + (1 − s)·f(φ_B) ],
                   s = σ(LAMBDA_PRIORITY · [cos(φ_A − theta_p) − cos(φ_B − theta_p)])

    The neuron keeps an ordinary gaze-centred receptive field and responds to whichever
    object is currently prioritised; ``theta_p`` is the direction the priority map favours.

    Why it is here. Figure 4e's impossibility argument is stated for
    ``R(θ, φ) = f(g(θ), g(φ))`` with ``f`` monotonically increasing in both arguments, and a
    reviewer proposed winner-take-all as a violation of it. It is not: ``max`` is
    monotonically non-decreasing in both arguments, so it *satisfies* the hypothesis, and a
    scan over ``theta`` and ``width`` on figure 4e's own four-corner stimulus finds the
    reversal at exactly zero parameter settings. What does violate it is a selector reading a
    **second map**, misaligned with the receptive field: ``theta_p = theta`` is max-pooling
    and ``theta_p = theta + π`` is min-pooling — neither can reverse — while the reversal
    becomes reachable in between, at a fraction 0.056 of receptive-field orientations by 10°
    of misalignment and 0.494 by 90°. The fitted ``theta_p - theta`` is therefore the whole
    content of the selection account, and this model is the one the relational hypothesis has
    to beat.

    Five parameters — ``baseline``, ``theta``, ``width``, ``amplitude``, ``theta_p`` — the
    same count as every model here but the retinotopic and constant ones.

    .. note::
       ``LAMBDA_PRIORITY`` is a **constant, not a fitted parameter**, and it is 100 for a
       measured reason. A hard ``argmax`` is discontinuous at the tie boundary and leaves
       L-BFGS-B no gradient in ``theta_p``. But a *soft* λ makes this a softmax-weighted
       average whose weights depend on the same variable as the drives, which is
       non-monotone — so a soft λ destroys the reading of ``theta_p = theta`` as max-pooling.
       Measured on the four-corner stimulus, the residual reversal magnitude at that corner
       is 22% of its maximum at λ = 4, 11% at λ = 8, 4.4% at λ = 20 and 0.6% at λ = 100. The
       data are indifferent (a 2.6-nat spread in best achievable log-likelihood across those
       four values, over 24 neurons), so λ is chosen for interpretability. Freeing it would
       be a sixth parameter and a different model — the "soft priority selection" (M3) of
       ``prompts/SELECTION_MODELS.pdf``.

    .. note::
       This model carries **24 structural multi-starts** rather than the three or four the
       others need (:func:`fit_lib._structural_variants`), because at this ``LAMBDA_PRIORITY``
       the gradient in ``theta_p`` is dead almost everywhere. See that function for the
       measurements.
    """

    name = "priority_map"
    label = "Priority map"
    short_label = "Priority"
    params = ("baseline", "theta", "width", "amplitude", "theta_p")

    def rate_2obj(self, p, phi_a, phi_b):
        w = p.get("width")
        fa = self.f(phi_a, p["theta"], w)
        fb = self.f(phi_b, p["theta"], w)
        # Logistic of the priority difference, in the branch-split form so that the
        # exponent is never positive and exp() cannot overflow at LAMBDA_PRIORITY = 100.
        d = LAMBDA_PRIORITY * (
            np.cos(phi_a - p["theta_p"]) - np.cos(phi_b - p["theta_p"])
        )
        e = np.exp(-np.abs(d))
        s = np.where(d >= 0, 1.0 / (1.0 + e), e / (1.0 + e))
        return p["baseline"] + p["amplitude"] * (s * fa + (1.0 - s) * fb)

    def init_params(self, s):
        return dict(
            baseline=s["baseline"],
            theta=s["theta"],
            width=s["width"],
            amplitude=s["amplitude"],
            theta_p=s["theta"],
        )


def rate_2obj(model, params, phi_a, phi_b, dctr=None):
    """One code path for every model's 2-object rate.

    Most models are a function of the two angles alone and ignore ``dctr``; the
    contrast-weighted one requires it. Callers — the fitter, the simulator, the tuning-curve
    plotters — go through this rather than testing model names, so adding another model with
    its own covariate is a change in one place.
    """
    if model.needs_contrast:
        return model.rate_2obj(params, phi_a, phi_b, dctr)
    return model.rate_2obj(params, phi_a, phi_b)


MODEL_CLASSES = {
    "basis_function": BasisFunctionModel,
    "priority_map": PriorityMapModel,
    "relational_bias": RelationalBiasModel,
    "weighted_average": WeightedAverageModel,
    "future_field": FutureFieldModel,
    "softmax": SoftmaxModel,
    "normalization": NormalizationModel,
    "inhibition": InhibitionModel,
    "constant": ConstantModel,
}

#: **Every model there is**, in the default order for tables and figures: the hypotheses of
#: interest first, then the combination-rule alternatives, then the reference every
#: likelihood is scored against.
#:
#: This is also `FitConfig`'s default set, so it is what `run_cache.py` fits. Two models used
#: to be held out of it on the grounds that adding them would refit every neuron — that is no
#: longer true, and was never quite right: the model list is not part of `FitConfig.tag`, so
#: `fit_session` tops up only the models a neuron is missing and leaves the rest untouched.
#: Keeping the list partial meant every caller had to patch around it.
#:
#: Note that `relational_bias` is in here because it has to be *fit*, not because it is one
#: of the models under test: it is the relationality metric's measuring instrument
#: (`relationality_lib`). Figures pass their own model list and leave it out.
MODEL_ORDER = [
    "future_field",
    "relational_bias",
    "softmax",
    "normalization",
    "inhibition",
    "basis_function",
    "priority_map",
    "weighted_average",
    "constant",
]

MODEL_LABELS = {k: MODEL_CLASSES[k].label for k in MODEL_ORDER}

#: Figure labels: the short form where a model has one, the full label otherwise.
MODEL_LABELS_SHORT = {
    k: (MODEL_CLASSES[k].short_label or MODEL_CLASSES[k].label)
    for k in MODEL_ORDER
}

# Colors shared by every figure in this directory. Retinotopic blue and relational orange
# match figure_3/model_comparison and figure_4/relational_continuum.
MODEL_COLORS = {
    "future_field": "tab:orange",
    "softmax": "tab:green",
    "normalization": "tab:red",
    "inhibition": "tab:brown",
    "relational_bias": "tab:cyan",
    "basis_function": "tab:olive",
    "priority_map": "tab:purple",
    "weighted_average": "tab:pink",
    "constant": "tab:gray",
}


def build_models(names=None):
    """``{name: Model}`` for ``names`` (default: everything in ``MODEL_ORDER``)."""
    names = MODEL_ORDER if names is None else list(names)
    return {n: MODEL_CLASSES[n]() for n in names}


# ── Starting values ───────────────────────────────────────────────────────────
def unit_init_stats(unit, default_width=0.8):
    """Cheap summary statistics of one neuron used to seed the optimiser — **computed from
    2-object trials only**.

    ``../neuron_models`` derives these from the 1-object trials, where the preferred
    direction is unambiguous. Here nothing may touch a 1-object trial, so ``theta`` is the
    firing-rate-weighted circular mean over *both* objects of every 2-object trial. On a
    retinotopic neuron the object in the receptive field is the one driving the rate and
    the other is roughly uniform, so the second averages out and the estimate leans towards
    the true preferred direction — diluted, but in the right place. It only has to be in
    the right ballpark: the fit is multi-started at ``n_theta_starts`` directions around
    the whole circle regardless, so a bad ``theta`` here costs nothing.
    """
    t0 = np.asarray(unit["t0"], float)
    t1 = np.asarray(unit["t1"], float)
    fr2 = np.asarray(unit["fr2"], float)
    mean_fr = float(np.mean(fr2)) if len(fr2) else 1.0

    if len(fr2) >= 3 and np.ptp(fr2) > 0:
        w = np.maximum(fr2 - fr2.min(), 0.0)
        z = np.sum(w * (np.exp(1j * t0) + np.exp(1j * t1)))
        theta = float(np.angle(z)) if np.abs(z) > 0 else 0.0
        baseline = float(np.percentile(fr2, 10))
        amplitude = float(
            max(np.percentile(fr2, 90) - baseline, 0.1 * mean_fr, 1e-3)
        )
    else:
        theta, baseline, amplitude = 0.0, mean_fr, max(0.1 * mean_fr, 1e-3)

    return dict(
        theta=theta,
        baseline=baseline,
        amplitude=amplitude,
        width=default_width,
        mean_fr=mean_fr,
    )

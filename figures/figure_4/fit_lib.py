"""Fitting, cross-validation and caching for the 2-object-only model comparison.

**One paradigm.** Every model in ``models_lib`` is fit independently to each neuron by
maximising the Poisson log-likelihood of that neuron's **2-object** spike counts, with
**every parameter free and fit simultaneously**. Nothing is fit in stages, nothing is
frozen from a 1-object fit, and no 1-object trial enters the objective, the scoring or even
the starting values. That is the whole difference from ``../neuron_models``, whose
paradigms 2 and 3 fit most parameters on 1-object trials first (see ``README.md`` §2).

Cross-validation
----------------
``n_bootstraps`` random train/test splits of the neuron's 2-object trials. **The splits are
drawn once per neuron, before any model is fit, and every model sees exactly the same
ones**, so the per-split scores are paired and model differences can be taken split by
split. The per-neuron seed is derived from ``(subject, session, unit)``, so a rerun
reproduces the same splits.

Likelihoods are reported as **nats per trial** (the mean over the scored trials), so they
are comparable across neurons with different trial counts. Every figure plots the
difference against the constant-rate model fit to the *same* trials, because raw per-trial
log-likelihood is dominated by each neuron's firing rate.

Cross-validation is **strict**: each split's fit is multi-started from scratch on its own
training trials, and even the crude initialisation statistics are recomputed from them. The
tempting optimisation — warm-starting each split from a fit to all of the neuron's trials —
is a leak: these likelihoods are multimodal in ``theta``, so a starting point that has seen
the held-out trials biases *which* optimum the training fit lands in, towards the one that
also fits the test set. In ``../neuron_models`` that was measured at +0.0008 nats/trial on
average but landed unevenly across models, so it is simply not offered here.

Optimisation
------------
The models are non-linear and ``theta`` is periodic, so each fit is **multi-started**: the
preferred direction is seeded at ``n_theta_starts`` points around the circle, crossed with
model-specific structural variants (the inhibition model's suppressive bump started
opposite / orthogonal to the excitatory one, the relational model started at and away from
zero remapped drive, the relational-bias model started at each end of its ``alpha`` axis
and in the middle). Every start gets a short coarse L-BFGS-B pass and the best ``n_polish``
are run to convergence.

Runtime
-------
One session is the unit of work — a few minutes, not hours. Results are cached per
(configuration, brain area, session) under
``cache/figures/figure_4/neuron_models/fits/{tag}/``, so re-running a notebook with
an unchanged CONFIG reads the cache instead of refitting.

``run_cache.py`` loops that over every session and additionally writes one combined
``summary.csv.gz`` per configuration, which is what ``population.ipynb`` reads. It is
resumable — a session whose two output files both exist is skipped — and it refreshes the
combined summary after every session, so a run that is still going is already readable.
:func:`load_summary` rebuilds that summary whenever it is older than the session files, so
a partial run can never be silently served as if it were complete, and
:func:`warn_incomplete` says loudly which sessions are still missing.
"""

import os
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import data_lib as dl
import models_lib as ml
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Bumped whenever the cached *numbers* or columns change, so an old cache is never silently
# reused. Note that "numbers" includes a changed parameter *bound*, not only a changed
# column set: version 2 exists because `models_lib.ALPHA_BIAS_MAX` dropped from 1.5 to 1.0,
# which leaves every cached column in place and quietly changes what a fitted `alpha` means.
#   1 — initial
#   2 — relational-bias `alpha` bounded to the identifiable range [0, 1]
#: Default worker count for every parallel fit here. NOT `-1`: this machine has 14 cores and
#: joblib's -1 takes all of them, which makes it unusable and, with BLAS threading inside each
#: worker, is usually slower as well. Callers that know better pass their own.
N_JOBS = 8

CACHE_VERSION = 2


# ── Configuration ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class FitConfig:
    """Everything that changes the numbers. Shared by all models, as specified."""

    fixed_width_hwhm_deg: float | None = (
        None  # None → fit the width; else hand-craft it
    )
    n_bootstraps: int = 10
    test_frac: float = 0.2
    seed: int = 0
    n_theta_starts: int = 8
    n_polish: int = 2
    coarse_maxiter: int = 40
    models: tuple = field(default_factory=lambda: tuple(ml.MODEL_ORDER))

    def __post_init__(self):
        if "constant" not in self.models:
            raise ValueError(
                "the 'constant' model is the reference every likelihood is "
                "measured against — keep it in `models`"
            )

    @property
    def fixed_width(self):
        """True when the width is hand-crafted rather than fit."""
        return self.fixed_width_hwhm_deg is not None

    @property
    def width_value(self):
        """The hand-crafted ``width`` parameter value, or None."""
        if self.fixed_width_hwhm_deg is None:
            return None
        return ml.width_from_hwhm(self.fixed_width_hwhm_deg)

    @property
    def tag(self):
        """Short string identifying this configuration — used in cache file names. The
        leading ``v{CACHE_VERSION}`` means a schema change cannot silently be served a
        stale cache with missing columns."""
        w = (
            "wfit"
            if self.fixed_width_hwhm_deg is None
            else f"w{self.fixed_width_hwhm_deg:g}"
        )
        # 'von_mises' stays in the tag as a literal: it is the only tuning function now,
        # but every cache on disk was written under this name and changing it would miss
        # them all and refit for hours.
        return (
            f"v{CACHE_VERSION}_von_mises_{w}"
            f"_b{self.n_bootstraps}_t{self.test_frac:g}_s{self.seed}"
        )


# ── Objective ─────────────────────────────────────────────────────────────────
def _neg_ll(x, model, free_names, base, d2):
    """Total negative Poisson log-likelihood of the 2-object data
    ``d2 = (phi_a, phi_b, fr, dctr)``, where ``dctr`` is the per-trial covariate only the
    contrast-weighted model reads."""
    p = dict(base)
    for name, value in zip(free_names, x):
        p[name] = value
    with np.errstate(all="ignore"):
        total = float(
            np.sum(
                ml.poisson_ll_per_trial(
                    d2[2], ml.rate_2obj(model, p, d2[0], d2[1], d2[3])
                )
            )
        )
    return -total if np.isfinite(total) else 1e18


def _fit(model, free_names, base, d2, starts, cfg):
    """Multi-start bounded maximum likelihood. ``starts`` is a list of full parameter
    dicts; returns ``(best_param_dict, best_log_likelihood)``."""
    free_names = tuple(free_names)
    if not free_names:  # nothing to fit (can't happen, but be safe)
        return dict(base), -_neg_ll([], model, (), base, d2)

    bounds = [model.param_bounds(n) for n in free_names]
    args = (model, free_names, base, d2)

    def clipped(p):
        return [
            float(np.clip(p[n], *model.param_bounds(n))) for n in free_names
        ]

    # Coarse pass over every start, then polish the best few to convergence.
    coarse = []
    for p0 in starts:
        r = minimize(
            _neg_ll,
            clipped(p0),
            args=args,
            method="L-BFGS-B",
            bounds=bounds,
            options=dict(maxiter=cfg.coarse_maxiter),
        )
        coarse.append((float(r.fun), np.asarray(r.x, float)))
    coarse.sort(key=lambda t: t[0])

    best_fun, best_x = coarse[0]
    for _, x0 in coarse[: max(cfg.n_polish, 1)]:
        r = minimize(
            _neg_ll,
            x0,
            args=args,
            method="L-BFGS-B",
            bounds=bounds,
            options=dict(maxiter=500),
        )
        if float(r.fun) < best_fun:
            best_fun, best_x = float(r.fun), np.asarray(r.x, float)

    out = dict(base)
    out.update(zip(free_names, (float(v) for v in best_x)))
    for name in ml.ANGLE_PARAMS:  # theta is periodic — report it in [0, 2π)
        if name in out:
            out[name] = float(out[name] % (2 * np.pi))
    return out, -best_fun


# ── Starting points ───────────────────────────────────────────────────────────
def _structural_variants(model, p0):
    """Extra starting points a model needs beyond rotating ``theta`` around the circle —
    one per basin its *non-angular* parameters have.

    Every one of these is a parameter the likelihood is genuinely multimodal in, where a
    gradient method started in the wrong basin will not cross to the right one:

    * the inhibition model's excitatory-to-suppressive offset (and ``theta_E == theta_I``
      makes it identically flat, so a start there is a trap rather than a candidate);
    * the relational model's boundary at zero remapped drive, which *is* the retinotopic
      model and must be reachable;
    * the basis-function model's ``beta``, whose zero is the retinotopic model and sits on a
      corner of its box rather than inside it;
    * the relational-bias model's ``bias``, whose two endpoints are the retinotopic and
      relational hypotheses and which interacts with ``theta``;
    * the priority-map model's ``theta_p``, whose gradient is dead almost everywhere and
      which therefore needs a *grid* rather than a handful of basins — 24 of them.
    """
    if model.name == "inhibition":
        return [
            {**p0, "theta_I": p0["theta_E"] + off}
            for off in (np.pi, 0.5 * np.pi, 0.25 * np.pi)
        ]
    if model.name == "future_field":
        return [p0, {**p0, "amplitude_remapped": 0.0}]
    if model.name == "relational_bias":
        # `bias`'s two endpoints are the retinotopic and relational hypotheses, and it
        # interacts with `theta` — moving the reference point rotates the angles the tuning
        # curve sees — so each rotation of `theta` needs its own scan over it.
        return [{**p0, "bias": b} for b in (0.0, 0.5, 1.0)]
    if model.name == "basis_function":
        # beta >= 0, so beta = 0 is a *corner* of the box rather than an interior ridge —
        # L-BFGS-B can sit on a corner and never leave, so start inside it as well.
        return [{**p0, "beta": b} for b in (0.0, 0.3, 0.7)]
    if model.name == "normalization":
        # 1e3 is the alpha bound: the near-linear limit, which IS the retinotopic model at
        # this amplitude and has to be reachable from a start, not just in principle.
        return [{**p0, "alpha": a} for a in (0.3, 10.0, 1e3)]
    if model.name == "priority_map":
        # 24 offsets, not the three or four the other models need, and this is the expensive
        # decision in the model. At `models_lib.LAMBDA_PRIORITY = 100` the gradient in
        # `theta_p` is dead almost everywhere -- the logistic saturates unless the two objects
        # are near-equidistant from the priority direction -- so L-BFGS-B cannot walk
        # `theta_p` and the fit is essentially whichever start was nearest. A coarse grid
        # therefore misses badly: measured against the best fit found by any setting, 8x4 = 32
        # starts lands a mean 4.2 nats short (worst neuron 20), 8x8 = 64 starts 2.2 nats short
        # (worst 16), and only 8x24 = 192 starts converges (0.000, worst 0.000). For
        # comparison the basis-function model at its production 24 starts is 0.06 nats short.
        # A `theta_p` continuation sweep (warm-starting each grid point from the last) was
        # tried and is cheaper but no better. Do not trim this list to save time: an
        # under-converged fit of the model the paper needs to *beat* proves nothing.
        return [
            {**p0, "theta_p": p0["theta"] + 2 * np.pi * i / 24}
            for i in range(24)
        ]
    return [p0]


def _rotated_starts(model, p0, cfg):
    """``p0`` rotated to ``cfg.n_theta_starts`` preferred directions around the circle,
    crossed with the model's structural variants.

    Every angular parameter is rotated by the *same* offset, so a start's internal geometry
    (the inhibition model's excitatory-to-suppressive offset) is preserved and only the
    neuron's overall preferred direction is scanned.
    """
    out = []
    for k in range(cfg.n_theta_starts):
        offset = 2 * np.pi * k / cfg.n_theta_starts
        rotated = {
            name: (value + offset if name in ml.ANGLE_PARAMS else value)
            for name, value in p0.items()
        }
        out.extend(_structural_variants(model, rotated))
    return out


def _apply_width_policy(params, cfg):
    """Resolve the width parameter according to the configuration.

    With a hand-crafted width the key is kept and pinned to the requested value, which is
    what makes it visible in the parameter tables.
    """
    if cfg.width_value is not None:
        params = dict(params)
        for name in ml.WIDTH_PARAMS:
            if name in params:
                params[name] = cfg.width_value
    return params


def _split_stats(unit, tr2):
    """Initialisation statistics from a split's **training** trials only — nothing that
    touched a held-out trial may reach the fit, not even as a starting value.
    """
    sub = dict(unit)
    sub["t0"], sub["t1"], sub["fr2"] = (
        unit["t0"][tr2],
        unit["t1"][tr2],
        unit["fr2"][tr2],
    )
    return ml.unit_init_stats(sub)


def unit_contrast(unit):
    """The per-trial relative contrast for one neuron's 2-object trials, or zeros when the
    identities are absent.

    Only :class:`models_lib.WeightedAverageModel` reads it, but it is computed once per
    neuron and passed to every model, so the fitting code has a single shape.
    """
    n = len(unit["fr2"])
    id0, id1 = unit.get("id0"), unit.get("id1")
    if id0 is None or id1 is None or not n:
        return np.zeros(n)
    import contrast_lib as cl

    return np.asarray(cl.relative_contrast(id0, id1), float)


# ── Cross-validation splits ───────────────────────────────────────────────────
def _unit_seed(unit, cfg):
    key = f"{unit['subject']}/{unit['session']}/{unit['unit']}".encode()
    return (cfg.seed + zlib.crc32(key)) % (2**32)


def make_splits(n, n_bootstraps, test_frac, rng):
    """``n_bootstraps`` random (train_idx, test_idx) splits of ``n`` trials."""
    n_test = int(np.clip(round(test_frac * n), 1, n - 1))
    return [
        (perm[n_test:], perm[:n_test])
        for perm in (rng.permutation(n) for _ in range(n_bootstraps))
    ]


# ── Scoring ───────────────────────────────────────────────────────────────────
def _r2(y, y_pred):
    y = np.asarray(y, float)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot < 1e-12 or len(y) < 2:
        return np.nan
    return float(1.0 - np.sum((y - y_pred) ** 2) / ss_tot)


def _score_2obj(model, p, t0, t1, fr, dctr=None):
    if len(fr) == 0:
        return np.nan, np.nan
    rate = ml.rate_2obj(model, p, t0, t1, dctr)
    return float(np.mean(ml.poisson_ll_per_trial(fr, rate))), _r2(
        fr, ml.rectify(rate)
    )


def _write_atomic(df, path):
    """Write a DataFrame to ``path`` **atomically** — to a temporary file in the same
    directory, then ``os.replace``, so a file exists only once it is complete and an
    interrupted run can never leave a truncated cache that the next run trusts.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    # pandas infers compression from the *file name*, and the temp name ends in `.tmp` —
    # so the compression has to be passed explicitly or the file is written uncompressed
    # under a `.gz` name and every later read raises BadGzipFile.
    compression = "gzip" if path.name.endswith(".gz") else "infer"
    df.to_csv(tmp, index=False, compression=compression)
    os.replace(tmp, path)


# ── Fitting one neuron ────────────────────────────────────────────────────────
def fit_all_trials(unit, cfg, models=None):
    """The **all-trial** parameter fit of every model to one neuron. ``{model_name: params}``.

    This is exactly what :func:`fit_unit` records in its ``unit_rows`` — the fit the tuning
    curves and the parameter tables are drawn from — without the cross-validation splits, so
    it costs one optimisation per model instead of ``n_bootstraps + 1``.

    It exists as its own function so that a fit to *simulated* trials goes through the same
    code path as the fit to real ones: multi-started at the same ``n_theta_starts``
    directions, the same width policy, the same bounds. An analysis that compares a fitted
    parameter across real and synthetic data is only measuring the data if the estimator is
    held fixed, and the way to hold it fixed is to call the same function.
    """
    models = models or ml.build_models(cfg.models)
    stats = ml.unit_init_stats(unit)
    d2 = (unit["t0"], unit["t1"], unit["fr2"], unit_contrast(unit))
    out = {}
    for name, model in models.items():
        p_init = _apply_width_policy(model.init_params(stats), cfg)
        out[name], _ = _fit(
            model,
            model.free_params(cfg.fixed_width),
            p_init,
            d2,
            _rotated_starts(model, p_init, cfg),
            cfg,
        )
    return out


def fit_unit(unit, cfg, models=None):
    """Fit every model to one neuron's 2-object trials. Returns ``(boot_rows, unit_rows)``,
    lists of dicts.

    ``boot_rows`` has one row per (model, split) with the train/test log-likelihoods and
    the parameters that split found; ``unit_rows`` has one row per model with the
    **all-trial** parameter fit — what the tuning-curve panels and the parameter printout
    use.
    """
    models = models or ml.build_models(cfg.models)
    stats = ml.unit_init_stats(unit)
    rng = np.random.default_rng(_unit_seed(unit, cfg))

    t0, t1, fr2 = unit["t0"], unit["t1"], unit["fr2"]
    dctr = unit_contrast(unit)
    n2 = len(fr2)

    # One set of splits per neuron, drawn before any model is fit — every model is scored
    # on exactly the same held-out trials.
    splits = make_splits(n2, cfg.n_bootstraps, cfg.test_frac, rng)

    meta = dict(
        subject=unit["subject"],
        session=unit["session"],
        unit=unit["unit"],
        brain_area=unit["brain_area"],
    )

    # The all-trial fits: what the tuning curves are drawn from. Never scored.
    full = fit_all_trials(unit, cfg, models)

    boot_rows, unit_rows = [], []
    for name, model in models.items():
        free = model.free_params(cfg.fixed_width)
        full_p = full[name]

        for b, (tr, te) in enumerate(splits):
            p0_b = _apply_width_policy(
                model.init_params(_split_stats(unit, tr)), cfg
            )
            p, _ = _fit(
                model,
                free,
                p0_b,
                (t0[tr], t1[tr], fr2[tr], dctr[tr]),
                _rotated_starts(model, p0_b, cfg),
                cfg,
            )

            tr_ll, tr_r2 = _score_2obj(
                model, p, t0[tr], t1[tr], fr2[tr], dctr[tr]
            )
            te_ll, te_r2 = _score_2obj(
                model, p, t0[te], t1[te], fr2[te], dctr[te]
            )
            boot_rows.append(
                dict(
                    **meta,
                    model=name,
                    boot=b,
                    train_ll=tr_ll,
                    test_ll=te_ll,
                    train_r2=tr_r2,
                    test_r2=te_r2,
                    n_train=len(tr),
                    n_test=len(te),
                    **{f"p_{k}": v for k, v in p.items()},
                )
            )

        unit_rows.append(
            dict(
                **meta,
                model=name,
                n_2obj=n2,
                n_params=len(free),
                **{f"p_{k}": v for k, v in full_p.items()},
            )
        )
    return boot_rows, unit_rows


# ── Fitting a session ─────────────────────────────────────────────────────────
def fit_session(
    subject,
    session,
    cfg,
    brain_area=None,
    unit_ids=None,
    n_jobs=N_JOBS,
    verbose=True,
    overwrite=False,
    cache=True,
):
    """Fit every model to every neuron of one session. Returns ``(boot_df, units_df)``.

    Results are cached under ``cache/figures/figure_4/neuron_models/fits/{cfg.tag}/``;
    pass ``overwrite=True`` to refit, or ``cache=False`` to neither read nor write (used
    when ``unit_ids`` restricts the fit to a handful of neurons).

    If the cache exists but was built with a smaller model set, only the **new** models are
    fit and appended. The train/test splits depend on the neuron and the seed, never on
    which models are being fit, so an appended model stays exactly paired with the ones
    already there.
    """
    area_tag = brain_area or "all"
    out_dir = dl.FIT_DIR / cfg.tag / area_tag
    boot_path = out_dir / f"{subject}_{session}_boot.csv.gz"
    unit_path = out_dir / f"{subject}_{session}_units.csv.gz"
    use_cache = cache and unit_ids is None

    cached_boot = cached_units = None
    fit_models = list(cfg.models)
    if (
        use_cache
        and not overwrite
        and boot_path.exists()
        and unit_path.exists()
    ):
        cached_boot = pd.read_csv(boot_path)
        cached_units = pd.read_csv(unit_path)
        have = set(cached_units["model"])
        fit_models = [m for m in cfg.models if m not in have]
        if not fit_models:
            if verbose:
                print(
                    f"{subject}/{session}: read from cache [{cfg.tag}, area={area_tag}]"
                )
            return cached_boot, cached_units
        if verbose:
            print(
                f"{subject}/{session}: cache has {len(have)} models, adding "
                f'{", ".join(fit_models)}'
            )

    units = dl.session_unit_trials(
        subject, session, brain_area=brain_area, verbose=verbose
    )
    if unit_ids is not None:
        units = {u: units[u] for u in unit_ids if u in units}
    if not units:
        raise ValueError(
            f"{subject}/{session}: no units passed the trial-count filter"
            + (f" for brain_area={brain_area}" if brain_area else "")
        )
    if verbose:
        print(
            f"{subject}/{session}: fitting {len(units)} units × {len(fit_models)} models "
            f"× {cfg.n_bootstraps} splits  [{cfg.tag}, area={area_tag}]"
        )

    models = ml.build_models(fit_models)

    def _one(u):
        return fit_unit(u, cfg, models=models)

    if n_jobs == 1:
        results = [_one(u) for u in units.values()]
    else:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs, verbose=5 if verbose else 0)(
            delayed(_one)(u) for u in units.values()
        )

    boot_df = pd.DataFrame([r for boot, _ in results for r in boot])
    units_df = pd.DataFrame([r for _, unit in results for r in unit])
    if cached_boot is not None:
        boot_df = pd.concat([cached_boot, boot_df], ignore_index=True)
        units_df = pd.concat([cached_units, units_df], ignore_index=True)

    if use_cache:
        out_dir.mkdir(parents=True, exist_ok=True)
        # units first, boot second: the resume check requires *both*, so if this is
        # interrupted between the two writes the session is simply refit next time.
        _write_atomic(units_df, unit_path)
        _write_atomic(boot_df, boot_path)
        if verbose:
            print(f"  → {boot_path.relative_to(dl.CACHE_DIR)}")
    return boot_df, units_df


# ── Summaries ─────────────────────────────────────────────────────────────────
LL_COLS = ["train_ll", "test_ll", "train_r2", "test_r2"]
KEYS = ["subject", "session", "unit", "brain_area", "model"]
UNIT_KEYS = ["subject", "session", "unit"]


def summarize(boot_df, units_df=None, reference="constant", score="test_ll"):
    """Per-(neuron, model) summary of the cross-validation results.

    For every column in :data:`LL_COLS` the across-split ``mean`` and ``std`` are reported.
    ΔLL columns (``dtrain_ll``, ``dtest_ll``) are the **paired per-split** difference
    against ``reference`` — paired because every model saw the same splits — with a mean, an
    s.d., and the fraction of splits on which the model beat the reference
    (``dtest_ll_frac_pos``).

    Adds ``best_model`` (per neuron, the ``score``-maximising non-reference model) and
    merges in ``units_df``'s all-trial parameters if given.
    """
    cols = [c for c in LL_COLS if c in boot_df.columns]
    agg = boot_df.groupby(KEYS)[cols].agg(["mean", "std"])
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()

    # Paired ΔLL against the reference model, split by split.
    idx = ["subject", "session", "unit", "brain_area", "boot"]
    ref = boot_df[boot_df["model"] == reference].set_index(idx)
    dparts = []
    for name, g in boot_df.groupby("model"):
        g = g.set_index(idx)
        common = g.index.intersection(ref.index)
        d = pd.DataFrame(index=common)
        for c in ["test_ll", "train_ll"]:
            d[f"d{c}"] = g.loc[common, c] - ref.loc[common, c]
        d["model"] = name
        dparts.append(d.reset_index())
    dall = pd.concat(dparts, ignore_index=True)
    dcols = [c for c in dall.columns if c.startswith("d")]
    dsum = dall.groupby(KEYS, as_index=False).agg(
        **{f"{c}_mean": (c, "mean") for c in dcols},
        **{f"{c}_std": (c, "std") for c in dcols},
        dtest_ll_frac_pos=("dtest_ll", lambda v: float(np.mean(v > 0))),
    )

    out = agg.merge(dsum, on=KEYS, how="left")

    # Winning model per neuron, over the non-reference models.
    score_col = f"{score}_mean"
    cand = out[out["model"] != reference]
    win = cand.loc[
        cand.groupby(KEYS[:-1])[score_col].idxmax(), KEYS[:-1] + ["model"]
    ].rename(columns={"model": "best_model"})
    out = out.merge(win, on=KEYS[:-1], how="left")

    if units_df is not None:
        pcols = [c for c in units_df.columns if c.startswith("p_")] + [
            "n_2obj",
            "n_params",
        ]
        out = out.merge(
            units_df[KEYS + [c for c in pcols if c in units_df.columns]],
            on=KEYS,
            how="left",
        )
    return out


def cache_dir(cfg=None, brain_area=None, tag=None):
    """Directory holding one configuration's fits. ``tag`` overrides the tag derived from
    ``cfg`` — the escape hatch for reading a cache whose directory name does not match the
    current code's tag."""
    if tag is None:
        if cfg is None:
            raise ValueError("pass either cfg or tag")
        tag = cfg.tag
    return dl.FIT_DIR / tag / (brain_area or "all")


def available_tags():
    """``{tag: {area: [sessions]}}`` for every cached configuration on disk."""
    out = {}
    if not dl.FIT_DIR.exists():
        return out
    for tag_dir in sorted(dl.FIT_DIR.iterdir()):
        if not tag_dir.is_dir():
            continue
        areas = {}
        for area_dir in sorted(tag_dir.iterdir()):
            if not area_dir.is_dir():
                continue
            sessions = sorted(
                p.name[: -len("_boot.csv.gz")]
                for p in area_dir.glob("*_boot.csv.gz")
            )
            if sessions:
                areas[area_dir.name] = sessions
        if areas:
            out[tag_dir.name] = areas
    return out


def check_tag(cfg, tag, verbose=True):
    """Warn when a hand-set ``tag`` does not describe the ``cfg`` the notebook is
    advertising.

    Reading a foreign tag is a legitimate thing to do — an older ``CACHE_VERSION``, a
    renamed tree, a cache copied from elsewhere — but every figure title, axis label and
    printed setting comes from ``cfg``, so the two disagreeing means the figures are
    labelled with settings that did not produce them.
    """
    if tag is None or tag == cfg.tag or not verbose:
        return
    print(
        f"⚠  reading cache tag {tag!r}, which is not this CONFIG's tag "
        f"({cfg.tag!r}).\n   The data come from {tag!r}; every label and title below "
        f"comes from the CONFIG.\n   Make sure the CONFIG describes the run that built "
        f"that cache."
    )


def combine_sessions(
    cfg, brain_area=None, sessions=None, verbose=True, tag=None
):
    """Concatenate every cached session's per-neuron summary into one frame and write it to
    ``fits/{tag}/{area}/summary.csv.gz``.

    :func:`summarize` is applied **per session** — so ``best_model`` and the paired ΔLL
    columns are computed within a session, exactly as the single-session notebook computes
    them — and the results are stacked. Returns the combined frame.
    """
    out_dir = cache_dir(cfg, brain_area, tag)
    sessions = sessions or [
        _split_session_name(p.name[: -len("_boot.csv.gz")])
        for p in sorted(out_dir.glob("*_boot.csv.gz"))
    ]
    parts = []
    for subject, session in sessions:
        bp = out_dir / f"{subject}_{session}_boot.csv.gz"
        up = out_dir / f"{subject}_{session}_units.csv.gz"
        if not (bp.exists() and up.exists()):
            if verbose:
                print(f"  {subject}/{session}: no cache — skipped")
            continue
        parts.append(summarize(pd.read_csv(bp), pd.read_csv(up)))
    if not parts:
        raise FileNotFoundError(f"no session caches under {out_dir}")
    out = pd.concat(parts, ignore_index=True)
    _write_atomic(out, out_dir / "summary.csv.gz")
    if verbose:
        n = out[UNIT_KEYS].drop_duplicates().shape[0]
        print(
            f"combined {len(parts)} sessions, {n} neurons "
            f'→ {(out_dir / "summary.csv.gz").relative_to(dl.CACHE_DIR)}'
        )
    return out


def _split_session_name(stem):
    """``'Perle_2022-06-11'`` → ``('Perle', '2022-06-11')``. Sessions contain hyphens and
    subjects do not, so the split is on the **first** underscore."""
    subject, session = stem.split("_", 1)
    return subject, session


def load_summary(cfg, brain_area=None, rebuild="auto", verbose=True, tag=None):
    """The combined per-neuron summary for one configuration — what ``population.ipynb``
    reads.

    The per-session ``*_boot.csv.gz`` / ``*_units.csv.gz`` files are the source of truth;
    ``summary.csv.gz`` is a convenience concatenation of them. ``rebuild``:

    * ``'auto'`` (default) — read the summary only if it exists **and** is newer than every
      session file, otherwise rebuild it from the session files. This is what makes a
      partially finished ``run_cache.py`` run usable, and it is also what stops a summary
      left over from an earlier, shorter run being silently read as if it covered
      everything.
    * ``'never'`` — read the summary as-is and fail if it is absent.
    * ``'always'`` — rebuild unconditionally.
    """
    check_tag(cfg, tag, verbose)
    area = brain_area or "all"
    out_dir = cache_dir(cfg, brain_area, tag)
    path = out_dir / "summary.csv.gz"
    session_files = sorted(out_dir.glob("*_units.csv.gz"))

    if not session_files and not path.exists():
        tags = available_tags()
        listing = (
            "\n".join(
                f"    {t}  →  "
                + ", ".join(
                    f"{a} ({len(ss)} sessions)"
                    for a, ss in sorted(areas.items())
                )
                for t, areas in tags.items()
            )
            or "    (nothing cached yet)"
        )
        raise FileNotFoundError(
            f"No cache for tag {(tag or cfg.tag)!r}, area {area!r}.\n"
            f"Build it with:  python run_cache.py"
            + ("" if brain_area is None else f" --area {brain_area}")
            + f"\n\nConfigurations currently on disk:\n{listing}\n\n"
            "If one of those is the one you want, make the notebook CONFIG match it — the "
            "tag encodes the cache version, tuning, width policy, bootstraps, test "
            "fraction and seed, so any of those differing gives a different tag."
        )

    fresh = (
        path.exists()
        and session_files
        and path.stat().st_mtime
        >= max(f.stat().st_mtime for f in session_files)
    )
    if rebuild == "never" or (rebuild == "auto" and fresh):
        out = pd.read_csv(path)
        warn_incomplete(out, verbose=verbose)
        return out

    if verbose:
        why = (
            "no summary.csv.gz yet"
            if not path.exists()
            else "summary.csv.gz is stale"
        )
        print(
            f"{why} — rebuilding from the {len(session_files)} session file(s) on disk"
        )
    out = combine_sessions(
        cfg, brain_area=brain_area, verbose=verbose, tag=tag
    )
    warn_incomplete(out, verbose=verbose)
    return out


def warn_incomplete(summary, verbose=True):
    """Warn when the loaded cache covers fewer sessions — or fewer **subjects** — than the
    dataset has.

    Worth a loud warning rather than a silent partial result: with an incomplete cache the
    two-monkey figures still render, they just quietly show one monkey, or one monkey's
    twelve sessions against another's two. That looks like a finding and is not one.
    """
    have = {
        (s, ss)
        for s, ss in summary[["subject", "session"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    want = set(dl.fr_sessions()) or set(dl.sessions_with_behavior())
    missing = sorted(want - have)
    if not missing:
        return []
    if verbose:
        by_subject = {}
        for s, ss in missing:
            by_subject.setdefault(s, []).append(ss)
        print(
            f"\n⚠  cache covers {len(have)} of {len(want)} sessions — "
            f"{len(missing)} still missing:"
        )
        for s in sorted(by_subject):
            n_have = sum(1 for a, _ in have if a == s)
            print(
                f"     {dl.subject_title(s)}: {n_have} present, "
                f"{len(by_subject[s])} missing"
            )
        print(
            "   Every per-monkey figure below is drawn from what is present, so the "
            "monkeys are\n   not yet comparable. Finish the cache with "
            "`python run_cache.py`.\n"
        )
    return missing


def restrict_models(
    summary, models, reference="constant", score="test_ll_mean", verbose=True
):
    """Keep only ``models`` (plus the ``reference``) and **recompute ``best_model`` within
    that set**.

    Filtering the rows is not enough on its own. ``best_model`` is written by
    :func:`summarize` as the argmax over *every* model that was fit, so a figure drawn from
    a subset while reading that column counts neurons whose winner is not on the axis: any
    wins-per-model tally would divide by all the selected neurons and its shares would not
    sum to 1. Recomputing makes "only these models" mean the
    same thing everywhere — the comparison is genuinely between the models asked for.

    The ΔLL columns are unaffected: they are all measured against the constant-rate
    reference, which is why the reference is kept regardless of whether it was listed.
    """
    models = [m for m in models if m != reference]
    have = set(summary["model"])
    missing = [m for m in models if m not in have]
    if missing:
        raise KeyError(
            f"models not in this summary: {missing}. Present: "
            f"{sorted(have)}"
        )
    if not models:
        raise ValueError(
            "restrict_models needs at least one non-reference model"
        )

    out = summary[summary["model"].isin(set(models) | {reference})].copy()
    cand = out[out["model"] != reference]
    win = cand.loc[
        cand.groupby(KEYS[:-1])[score].idxmax(), KEYS[:-1] + ["model"]
    ].rename(columns={"model": "best_model"})
    out = out.drop(columns=["best_model"]).merge(win, on=KEYS[:-1], how="left")
    if verbose and len(models) < len(have - {reference}):
        dropped = sorted(have - {reference} - set(models))
        print(
            f'  restricted to {len(models)} models ({", ".join(models)}); '
            f'best_model recomputed within them\n  dropped: {", ".join(dropped)}'
        )
    return out.reset_index(drop=True)

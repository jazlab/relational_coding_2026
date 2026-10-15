r"""Does a fitted model *generate* the observed distribution of relationality?

``figure_relationality.ipynb`` shows one distribution: the relational-bias model's fitted
``bias`` across neurons — how far the point a neuron measures object direction from has slid
from fixation towards the pair's midpoint (see ``relationality_lib`` and
``models_lib.RelationalBiasModel``).
``figure_model_comparison.ipynb`` shows which model wins on held-out likelihood. Neither
answers the question this module is for — **if a model were true, would this population look
the way it does?**

The test is a parametric bootstrap, one neuron at a time:

1. take model :math:`M`'s all-trial fit to neuron :math:`i` (from the cache);
2. predict a rate on each of that neuron's own 2-object trials, at that neuron's own
   :math:`(\phi_A, \phi_B)` and trial count;
3. draw Poisson spike counts from it — a synthetic neuron whose *truth* is :math:`M`;
4. re-fit the relational-bias model to those counts and record ``bias``.

Doing that for every neuron gives, for each model, a synthetic distribution: what the
measurement in ``figure_relationality.ipynb`` would have returned in a world where that
model were true, with this experiment's noise, trial counts and stimulus geometry. The KS
distance between it and the real distribution is how badly the model fails to be a generative
account of the population.

Three things the design is built around.

**The estimator is held fixed.** Real and synthetic values both come from
:func:`fit_lib.fit_all_trials` — same multi-start, same bounds. That is not
a detail: ``bias`` is bounded to [0, 1] and its estimator is biased near those bounds, so the
distribution is as much a property of the estimator as of the neurons, and comparing across a
change of estimator would measure the estimator.

**The relational-bias row is the noise floor, not a result.** Simulating from that model's own
fit and re-measuring is a parametric bootstrap of the real measurement, so its KS distance is
what a *correct* model scores given finite trials. A model is only in trouble to the extent it
exceeds that floor. Read the column against that row, never against zero.

**Only one distribution is being tested.** A model can reproduce it and still be the wrong
model — the statistic is one-dimensional and blind to everything else about the fit. Held-out
likelihood (``figure_model_comparison.ipynb``) is the test of which model explains individual
neurons; this is the test of which model could have produced the population.
"""

import hashlib
import json
import time

import data_lib as dl  # local, in this directory
import fit_lib as fl
import matplotlib.pyplot as plt
import models_lib as ml
import numpy as np
import pandas as pd
import plot_lib as pl
import relationality_lib as rl
from joblib import Parallel, delayed
from scipy import stats

#: Never every core: saturating them makes the machine unusable while a fit runs, so the
#: default leaves some free; pass `n_jobs` explicitly to change it.
N_JOBS = 8


UNIT_KEYS = ["subject", "session", "unit"]
BIAS_CACHE_DIR = dl.CACHE_DIR / "bias_recovery"


# ── Simulating one neuron ─────────────────────────────────────────────────────
def model_params(row, model):
    """A model's parameter dict, pulled out of one ``summary`` row's ``p_*`` columns.

    The summary carries the union of every model's parameters, so the ones this model does
    not have are NaN and are dropped — selecting on ``model.params`` rather than on "every
    ``p_`` column that is finite" means a stray finite value in another model's column
    cannot leak in.
    """
    out = {}
    for name in model.params:
        v = row.get(f"p_{name}", np.nan)
        if v is not None and np.isfinite(v):
            out[name] = float(v)
    return out


def simulate_fr2(model, params, t0, t1, rng, dctr=None):
    """Poisson spike counts from ``model`` on the trials ``(t0, t1)``, returned as a rate.

    The observation model is the one the fits are scored under
    (:func:`models_lib.poisson_ll_per_trial`): mean count ``rectify(rate) · duration``, and
    the returned rate is ``count / duration``, so a synthetic neuron's ``fr2`` is on exactly
    the same footing as a real one's — integer counts over the same window, not a continuous
    rate with Gaussian jitter.

    Note that the *stimuli* are not resampled. Each synthetic neuron inherits the real
    neuron's trials, so trial count and the ring geometry are held fixed and the only thing
    that varies is the spiking. That matters here: ``bias`` is poorly determined by
    near-antipodal pairs, so a neuron's stimulus set is part of how well its ``bias`` can be
    measured at all.

    The window is ``data_lib.DURATION_S``, read at call time.
    """
    duration = dl.DURATION_S
    rate = ml.rectify(ml.rate_2obj(model, params, t0, t1, dctr))
    return rng.poisson(rate * duration) / duration


def _seed(*parts):
    """A deterministic 32-bit seed from the identity of a job, so a rerun of one neuron
    reproduces itself regardless of how the work was chunked or ordered."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:4], "little")


def _unit_job(rows, trials, models, cfg, bias_model, reps, seed0):
    """Every (model, replicate) synthetic ``bias`` for one neuron. Returns a list of dicts.

    ``reps`` is an explicit list of replicate indices rather than a count, so a resumed run can
    ask for only the ones missing from the cache. Each draw is seeded from
    ``(seed0, unit, model, rep)``, so replicate 3 is the same numbers whether it was computed
    in the first run or the fourth.
    """
    t0, t1 = trials["t0"], trials["t1"]
    key = (trials["subject"], trials["session"], trials["unit"])
    dctr = fl.unit_contrast(
        trials
    )  # only the contrast-weighted model reads it
    out = []
    for name, model in models.items():
        row = rows.get(name)
        if row is None:
            continue
        params = model_params(row, model)
        # A missing parameter means the cache row is incomplete and there is nothing to
        # simulate from.
        if any(k not in params for k in model.params):
            continue
        fh = fit_hash(row, model)
        for rep in reps:
            rng = np.random.default_rng(_seed(seed0, *key, name, rep))
            fr2 = simulate_fr2(model, params, t0, t1, rng, dctr=dctr)
            value = rl.unit_bias(dict(t0=t0, t1=t1, fr2=fr2), cfg, bias_model)
            out.append(
                dict(
                    zip(UNIT_KEYS, key),
                    model=name,
                    rep=rep,
                    bias=value,
                    n_2obj=len(fr2),
                    fit_hash=fh,
                )
            )
    return out


# ── The population ────────────────────────────────────────────────────────────
def fit_hash(row, model):
    """A short hash of the fitted parameters one synthetic neuron is simulated from.

    This is what makes the cache safe to keep across a refit. A synthetic ``bias`` is a function
    of exactly three things: the neuron's trials (fixed), the seed (in the key), and *this
    model's fitted parameters for this neuron*. Storing a hash of the third alongside each row
    means a later run can tell, per (neuron, model), whether the fit it was simulated from is
    still the fit in the cache — so refitting one model invalidates that model's rows and
    nothing else, automatically and without anyone having to remember.
    """
    params = model_params(row, model)
    payload = ",".join(f"{k}={params[k]:.6g}" for k in sorted(params))
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def _run_dir(seed):
    """Directory holding one *simulation* configuration's synthetic biases, one file per
    session.

    The key is deliberately small: a fixed tag, the tuning function and the seed. In particular it does not
    hold:

    * ``models`` — rows are per (neuron, model), so dropping a model from the figure should
      not invalidate the others' rows;
    * the neuron selection — it decides which rows are *wanted*, not how any one of them was
      computed, so a stricter cut reads a subset rather than recomputing one;
    * ``n_reps`` — replicates are stored rather than keyed, so asking for 5 after computing 1
      costs the 4 that are missing.

    What guards correctness instead is :func:`fit_hash`, stored per row.
    """
    # 'von_mises' is a literal in the key, as in the fit tag: the cache directory name is the
    # hash of this exact key.
    key = json.dumps(
        dict(tag="bias", tuning="von_mises", seed=seed), sort_keys=True
    )
    return BIAS_CACHE_DIR / hashlib.sha256(key.encode()).hexdigest()[:16]


def synthetic_biases(
    summary,
    models,
    cfg,
    n_reps=1,
    seed=0,
    n_jobs=N_JOBS,
    verbose=True,
):
    """Simulate from each model's fit to each neuron and re-measure ``bias``.

    ``summary`` is the already-selected population (``common.selected_fits`` applied), so the
    synthetic neurons are the same neurons the real distribution is measured over — the
    comparison is paired at the population level, not just in shape.

    Returns one row per (neuron, model, replicate) with the recovered value in a column called
    ``bias``.

    The value is the relational-bias model's fitted ``bias`` (:mod:`relationality_lib`), so
    the synthetic and the real distributions come from the same estimator — which is what
    makes the KS comparison a test of hypotheses rather than of optimisers.

    **The work is cached per session, as each session finishes**, and the cache is keyed on
    what a synthetic ``bias`` actually depends on. This is the slow function in this directory —
    a replicate over the full population is ~5 minutes — so what it can reuse matters:

    * rows land in
      ``cache/figures/figure_4/neuron_models/bias_recovery/{key}/{subject}_{session}.csv.gz``
      the moment a session finishes, so an interrupted run keeps what it had done;
    * one row per (neuron, model, replicate), so **changing the model list, the neuron cut or
      ``n_reps`` reuses everything the new run has in common with the old** — a smaller
      population reads a subset, an extra replicate computes only that replicate;
    * every row stores :func:`fit_hash` — a hash of the fitted parameters it was simulated
      from — so **refitting one model invalidates that model's rows and nothing else**. The
      run reports how many rows it dropped for that reason, which is also the honest way to
      notice that a fit cache has moved under you.

    Delete the directory to force a full recompute.
    """
    units = summary[UNIT_KEYS].drop_duplicates().sort_values(UNIT_KEYS)
    run_dir = _run_dir(seed)
    want = list(range(n_reps))

    model_objs = ml.build_models(models)
    bias_model = ml.build_models((rl.MODEL,))[rl.MODEL]
    by_unit = {k: g.set_index("model") for k, g in summary.groupby(UNIT_KEYS)}

    # What this run wants, and what each (neuron, model) would have to have been simulated
    # from to count as already done.
    wanted_hash = {
        (k[0], k[1], int(k[2]), m): fit_hash(rows.loc[m], model_objs[m])
        for k, rows in by_unit.items()
        for m in models
        if m in rows.index
    }

    sessions = list(units.groupby(["subject", "session"]))
    parts, t_start = [], time.time()
    n_reused = n_stale = n_fresh = 0
    for i, ((subject, session), g) in enumerate(sessions, 1):
        path = run_dir / f"{subject}_{session}.csv.gz"
        have = pd.read_csv(path) if path.exists() else None
        if have is not None:
            # Keep only rows still backed by the fit they were simulated from. A row whose
            # model has since been refit is dropped here and recomputed below; a row for a
            # model this run is not drawing is kept on disk for when it is.
            key_cols = list(
                zip(
                    have["subject"],
                    have["session"],
                    have["unit"].astype(int),
                    have["model"],
                )
            )
            current = np.array(
                [wanted_hash.get(k) for k in key_cols], dtype=object
            )
            ok = np.array(
                [
                    h is None or h == f
                    for h, f in zip(current, have["fit_hash"])
                ]
            )
            n_stale += int((~ok).sum())
            have = have[ok].reset_index(drop=True)

        seen = (
            set(map(tuple, have[["unit", "model", "rep"]].to_numpy()))
            if have is not None
            else set()
        )
        trials_by_unit = dl.session_unit_trials(
            subject, session, verbose=False
        )
        jobs, n_jobs_models = [], 0
        for uid in g["unit"]:
            trials = trials_by_unit.get(int(uid))
            if trials is None:
                continue
            todo = [
                r
                for r in want
                if any((int(uid), m, r) not in seen for m in models)
            ]
            if not todo:
                continue
            rows = {
                m: r
                for m, r in by_unit[(subject, session, int(uid))].iterrows()
            }
            jobs.append(
                delayed(_unit_job)(
                    rows, trials, model_objs, cfg, bias_model, todo, seed
                )
            )
        if verbose and jobs:
            print(
                f"  [{i}/{len(sessions)}] {subject}/{session}: {len(jobs)} neurons to "
                f"compute" + (" (resuming)" if have is not None else ""),
                flush=True,
            )
        fresh = Parallel(n_jobs=n_jobs)(jobs) if jobs else []
        got = pd.DataFrame([r for part in fresh for r in part])
        n_fresh += len(got)
        if have is not None:
            n_reused += len(
                have[have["model"].isin(models) & have["rep"].isin(want)]
            )
        # Drop any row the fresh pass has just replaced, then merge — otherwise a resumed run
        # that recomputed a stale (neuron, model, rep) would keep both copies.
        if have is not None and len(got):
            gone = set(map(tuple, got[["unit", "model", "rep"]].to_numpy()))
            have = have[
                [
                    tuple(r) not in gone
                    for r in have[["unit", "model", "rep"]].to_numpy()
                ]
            ]
        merged = pd.concat(
            [x for x in (have, got) if x is not None and len(x)],
            ignore_index=True,
        )
        if len(merged):
            fl._write_atomic(merged, path)
        parts.append(merged)

    if verbose:
        print(
            f"{n_reused} rows reused from cache, {n_fresh} computed"
            + (
                f", {n_stale} dropped as stale (their model has been refit since)"
                if n_stale
                else ""
            )
        )
    out = pd.concat([p for p in parts if len(p)], ignore_index=True)
    out = out[out["rep"].isin(want) & out["model"].isin(list(models))]
    # ... and to THIS CALL'S NEURONS. Each session's cache file holds every neuron ever
    # simulated for that session, whoever asked for it, so without this the function would
    # return rows for neurons that are not in `summary`, and the real curve would describe one
    # population and the model curves another.
    keys = set(map(tuple, units[UNIT_KEYS].to_numpy()))
    mine = [tuple(r) in keys for r in out[UNIT_KEYS].to_numpy()]
    n_leaked = len(out[UNIT_KEYS].drop_duplicates()) - len(units)
    out = out[mine].reset_index(drop=True)
    if verbose:
        if n_leaked > 0:
            print(
                f"  dropped {n_leaked} cached neuron(s) that are not in this population"
            )
        print(
            f"{len(out)} synthetic neurons in {time.time() - t_start:.0f}s "
            f"({run_dir.relative_to(dl.paths.ROOT)})"
        )
    return out


def ks_table(
    real_biases,
    synth,
    models=None,
    by_subject=False,
    n_boot=0,
):
    """KS distance between the real ``bias`` distribution and each model's synthetic one.

    One row per model: ``ks`` is the two-sample Kolmogorov-Smirnov statistic
    :math:`\\sup_x |F_\\mathrm{model}(x) - F_\\mathrm{data}(x)|` and ``p`` its two-sided
    p-value. With ~2000 neurons a p-value is a test of "identical to the last decimal" and
    will be tiny for every model including the true one — **rank on ``ks``**, and read it
    against the relational-bias row rather than against 0.
    """
    if by_subject:
        # One block per monkey, each an independent KS test against that monkey's own
        # neurons — never against the pooled data, which would fold a difference between the
        # monkeys into every model's distance.
        parts = []
        for subj in subjects(real_biases):
            t = ks_table(
                real_biases[real_biases["subject"] == subj],
                synth,
                models,
                n_boot=n_boot,
            )
            parts.append(t.assign(subject=subj))
        cols = ["subject", "model", "label", "n", "ks"]
        cols += [c for c in ("ks_lo", "ks_hi") if n_boot]
        return pd.concat(parts, ignore_index=True)[
            cols + ["p", "median", "frac_at_0", "frac_at_1"]
        ]

    subj_col = (
        real_biases["subject"].iloc[0]
        if isinstance(real_biases, pd.DataFrame)
        and "subject" in real_biases
        and real_biases["subject"].nunique() == 1
        else None
    )
    real = bias_values(real_biases)
    models = list(models or synth["model"].unique())
    if subj_col is not None:
        synth = synth[synth["subject"] == subj_col]
    rows = []
    for m in models:
        v = synth.loc[synth["model"] == m, "bias"].to_numpy(float)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        res = stats.ks_2samp(v, real)
        rows.append(
            dict(
                model=m,
                label=ml.MODEL_LABELS_SHORT.get(m, m),
                n=len(v),
                ks=float(res.statistic),
                p=float(res.pvalue),
                median=float(np.median(v)),
                frac_at_0=float(np.mean(v <= 1e-9)),
                frac_at_1=float(np.mean(v >= 1 - 1e-9)),
            )
        )
    out = pd.DataFrame(rows)
    if n_boot:
        r = (
            real_biases
            if subj_col is None
            else real_biases[real_biases["subject"] == subj_col]
        )
        band = ecdf_bands(r, synth, models, n_boot=n_boot)
        out = out.merge(
            band["ks_ci"][["model", "ks_lo", "ks_hi"]], on="model", how="left"
        )
    return out.sort_values("ks").reset_index(drop=True)


def bias_values(x):
    """``x`` as a finite 1-D array of biases, whether it arrives as a Series, an array, or the
    DataFrame :func:`relationality_lib.biases` returns."""
    v = np.asarray(x["bias"] if isinstance(x, pd.DataFrame) else x, float)
    return v[np.isfinite(v)]


def subjects(real_biases):
    """The monkeys present, in ``plot_lib``'s canonical order — the same left-to-right order
    ``figure_relationality.ipynb`` puts its panels in."""
    return list(pl.levels(real_biases, "subject"))


def aligned_biases(real_biases, synth, models):
    """Line the real and synthetic biases up **neuron by neuron**.

    Returns ``(index, real[n], {model: A[n, n_reps]})`` over the neurons present in both, in
    one fixed order. The pairing is what makes the bootstrap below a *paired* one: a
    resampled neuron contributes its real bias and its synthetic bias to the same
    iteration, so the neuron-to-neuron variation that both curves share cancels out of their
    difference instead of inflating it.
    """
    r = (
        real_biases.drop_duplicates(UNIT_KEYS).set_index(UNIT_KEYS)["bias"]
        if isinstance(real_biases, pd.DataFrame)
        else None
    )
    if r is None:
        raise TypeError(
            "aligned_biases needs the DataFrame from relationality_lib.biases, "
            "not a bare array — it has to know which neuron is which"
        )
    wide = {
        m: synth[synth["model"] == m].pivot_table(
            index=UNIT_KEYS, columns="rep", values="bias"
        )
        for m in models
    }
    idx = r.index
    for w in wide.values():
        idx = idx.intersection(w.index)
    idx = idx.sort_values()
    return (
        idx,
        r.loc[idx].to_numpy(float),
        {m: w.loc[idx].to_numpy(float) for m, w in wide.items()},
    )


def _ecdf_on(v, grid):
    """ECDF of ``v`` evaluated on ``grid`` — ``F(x) = mean(v <= x)``."""
    return np.searchsorted(np.sort(v), grid, side="right") / len(v)


def ecdf_bands(real_biases, synth, models, n_boot=200):
    """Bootstrap 95% confidence bands for every ECDF, and a 95% CI on every KS distance.

    Each iteration resamples the neurons with replacement *and* draws a random Poisson
    replicate per neuron, so the band carries both the finite-population and the simulation
    noise. The data curve is resampled on the same neuron indices in the same iteration, so
    the KS interval is paired.
    """
    idx, real, A = aligned_biases(real_biases, synth, models)
    n, n_reps = len(idx), next(iter(A.values())).shape[1]
    n_grid, data_key = 201, "__data__"
    grid = np.linspace(0.0, 1.0, n_grid)
    rng = np.random.default_rng(0)

    curves = {m: np.empty((n_boot, n_grid)) for m in models}
    curves[data_key] = np.empty((n_boot, n_grid))
    ks = {m: np.empty(n_boot) for m in models}
    for b in range(n_boot):
        rows = rng.integers(0, n, n)
        reps = rng.integers(0, n_reps, n)
        f_data = _ecdf_on(real[rows], grid)
        curves[data_key][b] = f_data
        for m in models:
            f = _ecdf_on(A[m][rows, reps], grid)
            curves[m][b] = f
            ks[m][b] = np.max(np.abs(f - f_data))

    lo_q, hi_q = 2.5, 97.5
    bands = {
        k: dict(
            lo=np.percentile(v, lo_q, axis=0),
            hi=np.percentile(v, hi_q, axis=0),
            mid=np.median(v, axis=0),
        )
        for k, v in curves.items()
    }
    ks_ci = pd.DataFrame(
        [
            dict(
                model=m,
                ks_lo=float(np.percentile(v, lo_q)),
                ks_hi=float(np.percentile(v, hi_q)),
                ks_boot=float(np.median(v)),
            )
            for m, v in ks.items()
        ]
    )
    return dict(
        grid=grid,
        bands=bands,
        ks_ci=ks_ci,
        n=n,
        n_reps=n_reps,
        data_key=data_key,
    )


def _ecdf(v):
    """Step-plot coordinates of an empirical CDF, drawn flat-then-up so an **atom** shows as
    a vertical jump at its exact location rather than a ramp leading into it. The atoms at
    bias = 0 and 1 are a real feature of this distribution, so this matters.
    """
    v = np.sort(bias_values(v))
    if len(v) == 0:
        return np.array([]), np.array([])
    y = np.arange(1, len(v) + 1) / len(v)
    return np.concatenate([[v[0]], v]), np.concatenate([[0.0], y])


def _ecdf_steps(v, n_grid=None):
    """Step coordinates for drawing an ECDF, optionally evaluated on a regular grid.

    The raw ECDF of this population has one vertex per synthetic neuron — 6500 of them per
    curve, 96 % of consecutive pairs less than 0.001 apart in bias, which is far below one
    pixel at print size. Thousands of sub-pixel risers is not more accurate than a grid, and
    it wrecks a DASHED curve: the pattern restarts at every vertex, so the dashes come out
    ragged and unevenly spaced. Evaluating on ``n_grid`` points draws the same function with
    a hundredth of the vertices.

    The leading ``(grid[0], 0)`` point is what keeps the **atom** at bias = 0 visible: the
    ECDF's value there is already the atom's mass, so without a point at zero the jump would
    be off the bottom of the curve rather than drawn as the vertical riser it is.
    """
    if n_grid is None:
        return _ecdf(v)
    values = bias_values(v)
    if len(values) == 0:
        return np.array([]), np.array([])
    grid = np.linspace(0.0, 1.0, int(n_grid))
    y = _ecdf_on(values, grid)
    return np.concatenate([[grid[0]], grid]), np.concatenate([[0.0], y])


# ── The figure ────────────────────────────────────────────────────────────────
# Extra white space between the legend's contents and its rounded frame, asked for as
# `borderpad` so it lands inside the legend's extent — see the legend block of
# :func:`plot_bias_ecdf`.
_FRAME_PAD = 0.4


#: Points every ECDF curve is drawn on.
ECDF_GRID = 501


def plot_bias_ecdf(*args, **kwargs):
    """:func:`_plot_bias_ecdf`, with dash patterns held at the size they are written in.

    Matplotlib scales a dash pattern by the line's width (``lines.scale_dashes``), which is
    normally what you want — a thick line needs longer dashes to read as dashed. Here it is
    not: ``lw_step`` deliberately draws the curves at different widths so a covered one still
    shows, and with scaling on, TWO dash patterns at four widths look like four patterns. The
    dashes exist only to separate neighbouring greys, so they have to be the same two shapes
    on every curve.
    """
    with plt.rc_context({"lines.scale_dashes": False}):
        return _plot_bias_ecdf(*args, **kwargs)


def _plot_bias_ecdf(
    real_biases,
    synth,
    models,
    colors=None,
    data_color="k",
    figsize=(8.6, 3.2),
    xlabel=r"Relationality ($\alpha$)",
    legend_loc="right",
    legend_ncol=1,
    legend_bbox=None,
    legend_margin=None,
    legend_rounding=0.35,
    legend_borderpad=0.7,
    legend_labelspacing=0.6,
    legend_handlelength=2.2,
    legend_handletextpad=0.8,
    legend_columnspacing=1.2,
    legend_edgecolor="0.6",
    legend_lw=0.8,
    wspace=None,
    styles=None,
    lw_step=0.55,
):
    """Empirical CDFs: one step curve per model, plus the data.

    A model that generates this population lies on the ``data_color`` curve; the widest
    vertical gap between a model's curve and the data's *is* that model's KS statistic. The
    numbers themselves are in :func:`ks_table` — including the relational-bias row, which is
    the noise floor every other model should be read against (see the module docstring), not
    a result.

    There is one panel per monkey, in ``plot_lib``'s canonical order, and **each panel compares that monkey's synthetic neurons against that monkey's own data** —
    not against the pooled distribution. Pooling the reference would let a difference between
    the monkeys show up as a model's misfit in both panels. The two panels are otherwise
    independent replications of the same test, which is the reason to split them: a model that
    fails in one and not the other is not the same finding as one that fails in both.

    Steps are drawn flat-then-up (:func:`_ecdf`), so the atoms at ``bias`` 0 and 1 appear as
    vertical jumps at exactly those values rather than as ramps leading into them.

    The legend is placed by ``legend_loc``: ``'right'`` puts it beside the panels,
    ``'below'`` underneath them, and any matplotlib location string puts it inside the last
    panel. ``legend_bbox`` overrides the anchor point (figure coordinates) and
    ``legend_margin`` how much of the figure the panels keep before it — together they are the
    position control. Spacing is ``legend_borderpad`` (contents to box),
    ``legend_labelspacing`` (between entries), ``legend_handlelength`` (the line samples),
    ``legend_handletextpad`` (sample to label) and ``legend_columnspacing`` (between columns —
    the knob that decides whether a one-row ``'below'`` legend fits the figure's width).
    ``legend_rounding`` is the corner radius of the box, in font-size units.

    ``styles`` is an optional ``{model: linestyle}``; anything not named is solid. Dashing the
    models that share a colour ramp is what makes them separable in print and to a
    colour-blind reader, where four steps of grey are not.

    Every curve is drawn on ``ECDF_GRID`` points instead of one per synthetic neuron
    (:func:`_ecdf_steps`), and as a **polyline** through them rather than a staircase. It is
    a drawing choice, not a smoothing one — the function drawn is the same ECDF, sampled where a printed line
    can resolve it — and it is what makes the dashed curves come out even instead of chewed.

    ``lw_step`` widens the curves earlier in ``models`` by that many points each, so a model
    covered by a later one is still visible under it. Two of these models can predict almost
    the same distribution — the basis-function model IS the retinotopic model at ``beta = 0``, and the
    fitted ``beta`` is small for most neurons — and equal widths make the first of them look
    like it was never drawn. Set to 0 for uniform widths.

    The data curve ("Data") is drawn **behind** the model curves, so the models are never
    hidden where they lie on it. Its legend entry still comes last: the data belongs at the
    bottom of the key whichever layer it is on.

    """
    colors = colors or {}

    def lab(m):
        return ml.MODEL_LABELS_SHORT.get(m, m)

    # A staircase is the exact ECDF, and it is the wrong thing to *draw* here. On a grid its
    # treads are about a pixel wide, so the curve is a fine sawtooth: the edges look ragged
    # and a dash pattern bends around every corner, which is what makes dashed curves look
    # chewed. Between the atoms this distribution is continuous, so a polyline through the
    # same samples is both cleaner and no less true — the only real discontinuities are the
    # atoms at 0 and 1, and the leading vertical from `_ecdf_steps` still draws the one at 0.
    def draw(ax, x, y, **kw):
        return ax.plot(
            x, y, solid_joinstyle="round", dash_joinstyle="round", **kw
        )

    styles = styles or {}
    groups = subjects(real_biases)

    fig, axes = plt.subplots(
        1, len(groups), figsize=figsize, dpi=300, sharey=True, squeeze=False
    )
    axes = axes[0]
    handles, seen = [], []
    for ax, subj in zip(axes, groups):
        r = real_biases[real_biases["subject"] == subj]
        sy = synth[synth["subject"] == subj]
        for k, m in enumerate(models):
            x, y = _ecdf_steps(sy.loc[sy["model"] == m, "bias"], ECDF_GRID)
            if not len(x):
                continue
            # Widths decrease along `models`, so a curve drawn over by a later one still
            # shows underneath it. Two models here predict nearly the same bias distribution
            # — the basis-function model reduces to the retinotopic model at beta = 0 and its fitted
            # beta is small — and without this the earlier of the two is simply invisible,
            # which reads as a missing curve rather than as agreement.
            (h,) = draw(
                ax,
                x,
                y,
                lw=1.8 + lw_step * (len(models) - 1 - k),
                color=colors.get(m),
                ls=styles.get(m, "-"),
                zorder=2,
            )
            if lab(m) not in seen:
                handles.append(h)
                seen.append(lab(m))
        x, y = _ecdf_steps(r, ECDF_GRID)
        # 1.5 puts it under the model curves (2).
        (h,) = draw(
            ax,
            x,
            y,
            lw=2.4,
            color=data_color,
            zorder=1.5,
        )
        if "Data" not in seen:
            handles.append(h)
            seen.append("Data")

        ax.set_title(dl.subject_title(subj), fontsize=14)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0, 1.02)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Cumulative fraction", fontsize=12)

    fig.tight_layout()
    if wspace is not None:
        fig.subplots_adjust(wspace=wspace)
    # One shared key for both panels, placed OUTSIDE them. Curves that all run from (0,0)
    # to (1,1) leave no empty corner big enough for six entries at 12 pt, and
    # FORMATTING.md rules out shrinking the text to make one — so a legend inside a panel
    # would sit on the curves it is explaining.
    # The rounded frame is grown around the legend by `_FRAME_PAD` below, and that growth
    # is NOT part of the legend's reported extent — so `bbox_inches='tight'` measures the
    # figure to the legend's box and slices the border's right edge off the saved PDF.
    # Asking for the padding as `borderpad` instead puts exactly the same white space
    # inside the extent, where the tight bbox can see it, and the box style then adds
    # nothing (pad=0). Same picture, whole border.
    kw = dict(
        fontsize=12,
        ncol=legend_ncol,
        borderpad=legend_borderpad + _FRAME_PAD,
        labelspacing=legend_labelspacing,
        handlelength=legend_handlelength,
        handletextpad=legend_handletextpad,
        columnspacing=legend_columnspacing,
        frameon=True,
        framealpha=1.0,
        edgecolor=legend_edgecolor,
        fancybox=True,
    )
    if legend_loc == "right":
        fig.subplots_adjust(
            right=0.78 if legend_margin is None else legend_margin
        )
        leg = fig.legend(
            handles,
            seen,
            loc="center left",
            bbox_to_anchor=legend_bbox or (0.79, 0.55),
            **kw,
        )
    elif legend_loc == "below":
        fig.subplots_adjust(
            bottom=0.30 if legend_margin is None else legend_margin
        )
        leg = fig.legend(
            handles,
            seen,
            loc="upper center",
            bbox_to_anchor=legend_bbox or (0.5, 0.16),
            **kw,
        )
    else:
        leg = axes[-1].legend(
            handles, seen, loc=legend_loc, bbox_to_anchor=legend_bbox, **kw
        )
    # Rounded corners. `boxstyle` has to be set after the fact — the Legend constructor
    # exposes fancybox as a flag but not its radius.
    fr = leg.get_frame()
    fr.set_boxstyle("round", pad=0.0, rounding_size=legend_rounding)
    fr.set_linewidth(legend_lw)
    return fig, axes

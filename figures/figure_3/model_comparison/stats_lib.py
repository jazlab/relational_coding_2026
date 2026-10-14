r"""Shared inputs, neuron selection, and the receptive-field-orientation figure.

This module owns three things: where the caches live and how to read them, the shared
neuron-selection entry point every figure-3 analysis goes through, and the **2-object phase**
statistics and histograms of `rf_orientation.ipynb`.

It does **not** own the model comparison. That is `comparison_lib.py`, which scores each model
on the sign of the held-out rate orderings it predicts and nothing else. A second, graded
comparison lived here until 2026-09-13 — a Bradley-Terry fit scored by held-out per-comparison
dLL — and was removed when the sign-only one became figure 3's only model comparison.

What survives of that fit is the **phase**. `run_phase_cache.py` still fits each neuron's
2-object tuning vector, because its angle relative to the neuron's own 1-object receptive field
is what panel 3e plots, and the sign-only comparison estimates no bootstrap distribution for
that angle. Its cache is ``phase_units.csv`` / ``phase_draws.csv.gz``; the pairwise machinery
both it and the comparison are built on is :mod:`pairs_lib`.

Two caches are **shared with figure 3's other panels, by path and on purpose**:

    firing_rates/{subject}_{session}.csv.gz    per-trial, per-unit firing rates
    modulation.csv                             the model-agnostic selection statistic

Both are model-free, so every analysis in figure 3 reads the same copy and therefore describes
**exactly the same neurons**.

Everything estimator-dependent lives under ``cache/figures/figure_3/model_comparison/``.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pairs_lib as pairs
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = (_HERE / "../../..").resolve()
_FIGURES = (_HERE / "../..").resolve()
if str(_FIGURES) not in sys.path:
    sys.path.insert(0, str(_FIGURES))
import config  # noqa: E402
import firing_rates as fr_lib  # noqa: E402  the shared per-trial rates
from trial_filters import drop_east_west_arrays  # noqa: E402

_SHARED = _ROOT / "cache/figures/shared_inputs"
FR_DIR = fr_lib.FR_DIR
MODULATION_PATH = _SHARED / "modulation.csv"

#: The per-trial rate columns averaged into the rate every fit here reads: both, i.e. the 2-s
#: stimulus + delay window. `supp_figure_2` sets one of them to refit within one task phase.
#: Read at call time. The estimators are ordinal, so no duration goes with it.
FR_COLUMNS = ("stimulus_fr", "delay_fr")

CACHE_DIR = _ROOT / "cache/figures/figure_3/model_comparison"
UNITS_PATH = CACHE_DIR / "phase_units.csv"
PHASES_PATH = CACHE_DIR / "phase_draws.csv.gz"
PARTS_DIR = CACHE_DIR / "phase_parts"

SUBJECT_ORDER = ["Perle", "Elgar"]
SUBJECT_TITLE = {"Perle": "Monkey P", "Elgar": "Monkey E"}
MODELS = ("retinotopic", "relational", "proximity")
#: The two slope models, which share a phase and a bootstrap of it. Proximity has neither: it
#: carries a preferred separation instead, so anything phase-shaped iterates over this.
PHASE_MODELS = ("retinotopic", "relational")
DISPLAY = {
    "retinotopic": "Retinotopic",
    "relational": "Relational",
    "proximity": "Proximity",
}
SHORT = {
    "retinotopic": "retinotopic",
    "relational": "relational",
    "proximity": "proximity",
}
MODEL_COLORS = {
    "retinotopic": "tab:blue",
    "relational": "tab:orange",
    "proximity": "0.45",
}
_ID_COLS = ["subject", "session", "unit_id", "brain_area"]

# ══════════════════════════════════════════════════════════════════════════════
# Exclusion criteria for the receptive-field-orientation figure.
#
# These are **prerequisites of a measurement**, not a selection on the effect. That figure
# plots an angle *difference*, so it needs both angles to exist; a neuron whose phase is
# undetermined contributes a uniformly random angle at full weight and is pure noise in the
# histogram. They never decide who enters the population — `neuron_selection.py` does that,
# model-agnostically, and is untouched here.
# ══════════════════════════════════════════════════════════════════════════════
#: False-discovery rate for the phase-determinacy test (Benjamini-Hochberg across neurons).
#: Figure 3 used a raw 0.05 here while its population criterion used BH; that inconsistency is
#: not carried over.
PHASE_FDR_Q = 0.05

#: **Whether to apply the shared population criterion** (`figures/neuron_selection.py`,
#: i.e. `config.FDR_Q` and `config.MIN_DEV`) before anything else.
#:
#: On. Every caller in the paper passes ``apply_modulation=True`` explicitly, so this is what
#: they all get; it defaulted to **off** until 2026-09-13, which is how `rf_orientation.ipynb`
#: came to be drawn over 3186 neurons while the panel beside it used 1339. Switching it off
#: grows the population to 3186 and admits neurons fit on as few as `MIN_2OBJ_FIT` = 100
#: 2-object trials, since the criterion also silently carries `config.MIN_2OBJ_TRIALS` = 300 —
#: a neuron below that has no modulation entry to pass.
APPLY_MODULATION_CRITERION = True


# ── Loading ───────────────────────────────────────────────────────────────────
def _require(p, how):
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found — build it with {how}.")
    return p


#: Columns of ``phase_units.csv`` that belonged to the **graded comparison** removed on
#: 2026-09-13 — held-out dLLs, the paired test on their difference, and the winner it named.
#: The fit still writes them because the fit was not re-run for their sake (an hour of compute
#: to delete columns), but nothing reads them and :func:`load_units` drops them on the way in,
#: so a stale ``best_model`` cannot be picked up by accident. The model comparison lives in
#: `comparison_lib.py` and has its own cache.
_RETIRED_COLS = tuple(f"{m}_dll" for m in MODELS) + (
    "oneobj_dll",
    "dll_relational_retinotopic",
    "dll_paired",
    "dll_p",
    "dll_lo",
    "dll_hi",
    "best_model",
    "runner_up_model",
    "margin",
    "margin_p",
    "margin_lo",
    "margin_hi",
    "winner_beats_null",
)


def load_units(path=UNITS_PATH):
    """The per-neuron phase fit, without the retired comparison columns."""
    u = pd.read_csv(_require(path, "`python run_phase_cache.py`"))
    return u.drop(columns=[c for c in _RETIRED_COLS if c in u.columns])


def load_phases(path=PHASES_PATH):
    return pd.read_csv(_require(path, "`python run_phase_cache.py`"))


def select_units(units, apply_modulation=None, verbose=True):
    """Apply the shared, model-agnostic 2-object modulation criterion.

    ``apply_modulation`` defaults to :data:`APPLY_MODULATION_CRITERION`, which is **on**: the
    same ``modulation.csv``, the same ``config.FDR_Q`` and ``config.MIN_DEV`` that every other
    analysis in the paper selects on, so every panel of figure 3 describes one population.
    Switching it off is a diagnostic, not an option a figure should take.
    """
    apply_modulation = (
        APPLY_MODULATION_CRITERION
        if apply_modulation is None
        else apply_modulation
    )
    if not apply_modulation:
        if verbose:
            print(
                f"  modulation criterion: NOT applied ({len(units)} neurons kept) — "
                f"a diagnostic, not what any figure should be drawn over."
            )
        return units.reset_index(drop=True)
    import neuron_selection

    return neuron_selection.filter_units(
        units, unit_col="unit_id", verbose=verbose
    )


def subject_title(s):
    return SUBJECT_TITLE.get(s, s)


def ordered_subjects(u):
    present = set(u["subject"].unique())
    return [s for s in SUBJECT_ORDER if s in present] + sorted(
        present - set(SUBJECT_ORDER)
    )


def fr_sessions():
    """Every session, building any missing firing-rate file first (``figures/firing_rates.py``)."""
    return fr_lib.ensure_all()


def session_unit_trials(subject, session, completed_only=None):
    """One session's firing-rate cache → per unit, its cleaned 1- and 2-object trial arrays.

    Byte-for-byte the computation ``figure_3/model_comparison`` performs, so the two versions
    see identical trials. The over-represented east-west antipodal pair is always dropped.
    """
    completed_only = (
        config.COMPLETED_ONLY if completed_only is None else completed_only
    )
    df = pd.read_csv(fr_lib.session_path(subject, session))
    if completed_only:
        df = df[df["completed"].astype(bool)]
    df = df.assign(_fr=df[list(FR_COLUMNS)].astype(float).mean(axis=1))
    out = {}
    for uid, g in df.groupby("unit"):
        g1 = g[g["num_objects"] == 1]
        fr1 = g1["_fr"].to_numpy(float)
        th1 = g1["object_0_theta"].to_numpy(float) % (2 * np.pi)
        ok1 = np.isfinite(fr1) & np.isfinite(th1)
        g2 = g[g["num_objects"] == 2]
        fr2 = g2["_fr"].to_numpy(float)
        t0 = g2["object_0_theta"].to_numpy(float) % (2 * np.pi)
        t1 = g2["object_1_theta"].to_numpy(float) % (2 * np.pi)
        ok2 = np.isfinite(fr2) & np.isfinite(t0) & np.isfinite(t1)
        t0, t1, fr2 = drop_east_west_arrays(t0[ok2], t1[ok2], fr2[ok2])
        out[int(uid)] = dict(
            t0=t0,
            t1=t1,
            fr2=fr2,
            theta1=th1[ok1],
            fr1=fr1[ok1],
            brain_area=str(g["brain_area"].iloc[0]),
        )
    return out


# ── Building the cache ────────────────────────────────────────────────────────
MIN_2OBJ_FIT = 100  # below this a neuron has too few comparisons to fit at all


def parts_dir_for(cfg, root=PARTS_DIR):
    """Checkpoints live under their own configuration's tag.

    A quick pass (say ``--n-boot 0``, enough for the win counts) and the full build are
    different analyses, and if they shared a directory the full build would find the quick
    one's part files and keep them — silently producing a cache with no bootstrap in it. One
    directory per tag makes that impossible instead of merely discouraged.
    """
    return Path(root) / cfg["tag"]


def build_session(
    subject, session, cfg, parts_dir=None, overwrite=False, verbose=True
):
    """Fit every neuron of one session and checkpoint it atomically.

    Checkpointed per session because the full build takes about an hour: a run that writes one
    file at the end loses everything when interrupted. A part file exists only once complete,
    so a re-run resumes.
    """
    parts_dir = parts_dir_for(cfg) if parts_dir is None else Path(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)
    u_out = parts_dir / f"{subject}_{session}_units.csv"
    p_out = parts_dir / f"{subject}_{session}_phases.csv.gz"
    if not overwrite and u_out.exists() and p_out.exists():
        prev = pd.read_csv(u_out)
        if (
            len(prev)
            and "fit_tag" in prev
            and prev["fit_tag"].iloc[0] != cfg["tag"]
        ):
            print(
                f'  ! {subject}/{session}: checkpoint is [{prev["fit_tag"].iloc[0]}], '
                f'requested [{cfg["tag"]}] — keeping it. Use --force to refit.',
                flush=True,
            )
        return f"{subject}/{session}", len(prev), 0.0

    t_start = time.time()
    units = session_unit_trials(subject, session)
    rows, phase_frames = [], []
    for uid, d in sorted(units.items()):
        if len(d["fr2"]) < MIN_2OBJ_FIT:
            continue
        seed = (
            (hash((subject, session, uid)) % (2**31))
            if cfg["seed"] is None
            else cfg["seed"] + uid
        )
        r = pairs.fit_neuron(
            d["t0"],
            d["t1"],
            d["fr2"],
            d["theta1"],
            d["fr1"],
            sigma_deg=cfg["sigma_deg"],
            n_folds=cfg["n_folds"],
            n_boot=cfg["n_boot"],
            seed=seed,
            max_comparisons=cfg["max_comparisons"],
            with_proximity=cfg.get("with_proximity", True),
            sep_max_comparisons=cfg.get(
                "sep_max_comparisons", pairs.SEP_MAX_COMPARISONS
            ),
        )
        ids = dict(
            subject=subject,
            session=session,
            unit_id=uid,
            brain_area=d["brain_area"],
        )
        boots = {m: r.pop(f"{m}_boot_phase") for m in PHASE_MODELS}
        boots["proximity"] = r.pop(
            "proximity_boot_sep", np.full(cfg["n_boot"], np.nan)
        )
        rows.append({**ids, **r})
        phase_frames.append(
            pd.DataFrame(
                {
                    "bootstrap": np.arange(len(boots["retinotopic"])),
                    **{f"{m}_axis_deg": boots[m] for m in PHASE_MODELS},
                    "proximity_sep_deg": boots["proximity"],
                    **ids,
                }
            )
        )

    units_df = pd.DataFrame(rows)
    if len(units_df):
        units_df["fit_tag"] = cfg["tag"]
        for k in (
            "sigma_deg",
            "n_folds",
            "n_boot",
            "max_comparisons",
            "sep_max_comparisons",
        ):
            units_df[f"param_{k}"] = cfg[k]
    phases_df = (
        pd.concat(phase_frames, ignore_index=True)
        if phase_frames
        else pd.DataFrame()
    )
    _write_atomic(units_df, u_out)
    _write_atomic(phases_df, p_out)
    secs = time.time() - t_start
    if verbose:
        print(
            f"  {subject}/{session}: {len(units_df)} neurons ({secs:.0f} s)",
            flush=True,
        )
    return f"{subject}/{session}", len(units_df), secs


def _write_atomic(df, path):
    """Write atomically, so a file exists only once complete and an interrupted run can never
    leave a truncated cache the next run trusts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(
        tmp,
        index=False,
        compression="gzip" if path.name.endswith(".gz") else "infer",
    )
    os.replace(tmp, path)


def combine_parts(cfg, parts_dir=None, verbose=True):
    """Concatenate the per-session checkpoints into the two tables the notebooks read."""
    parts_dir = parts_dir_for(cfg) if parts_dir is None else Path(parts_dir)
    up = sorted(parts_dir.glob("*_units.csv"))
    if not up:
        raise FileNotFoundError(
            f"no checkpoints in {parts_dir} — run the build first"
        )
    units = pd.concat([pd.read_csv(f) for f in up], ignore_index=True)
    tags = (
        sorted(units["fit_tag"].dropna().unique())
        if "fit_tag" in units
        else []
    )
    if len(tags) > 1:
        raise ValueError(
            f"{parts_dir} holds checkpoints from {len(tags)} configurations "
            f'({", ".join(tags)}). Delete the unwanted ones or rebuild with --force.'
        )
    phases = pd.concat(
        [pd.read_csv(f) for f in sorted(parts_dir.glob("*_phases.csv.gz"))],
        ignore_index=True,
    )
    units = units.sort_values(_ID_COLS).reset_index(drop=True)
    _write_atomic(units, UNITS_PATH)
    _write_atomic(phases, PHASES_PATH)
    if verbose:
        print(f"  wrote {len(units)} neurons to {UNITS_PATH}")
        print(f"  wrote {len(phases)} phase rows to {PHASES_PATH}")
    return units, phases


# ── Receptive-field orientation ───────────────────────────────────────────────
def bh_reject(p, q):
    """Benjamini-Hochberg: boolean mask of the hypotheses rejected at FDR ``q``."""
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    out = np.zeros(len(p), bool)
    if not ok.any():
        return out
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    thresh = q * (np.arange(1, m + 1) / m)
    below = p[order] <= thresh
    if below.any():
        out[order[: np.max(np.where(below)[0]) + 1]] = True
    return out


def phase_eligible(units, model, q=None, verbose=False):
    """Which neurons may contribute a phase for ``model`` — a **measurement prerequisite**.

    The phase must be distinguishable from "no phase". The test is a Wald statistic on the
    two-dimensional ``gamma`` against its **bootstrap** covariance, so it needs no
    composite-likelihood correction, and it is corrected across neurons by Benjamini-Hochberg
    at ``q`` — matching how the population criterion is corrected. It is applied to the row's
    own model **and** to the 1-object reference, because the plotted quantity is their
    difference and either being undetermined ruins it. The reference also needs enough
    1-object trials to exist at all (``config.MIN_1OBJ_TRIALS``).

    There was a second criterion, a ceiling on the bootstrap circular s.d. of the phase, and
    it was removed after being measured: at 30 deg it excluded **zero** neurons on the
    2-object phase and one or two on the 1-object reference, because the determinacy test is
    strictly the harder of the two on this data. Among neurons passing determinacy the phase
    s.d. is median 9.6 deg (retinotopic) and 4.1 deg (relational), with maxima of 27.4 and
    13.9 — the whole surviving distribution sat under the threshold. A precision floor that
    bites would have to be near 15-20 deg, which is a claim about how precise a phase must be
    to belong in a 9 deg bin rather than a prerequisite of measuring one at all.
    """
    q = PHASE_FDR_Q if q is None else q
    ok = bh_reject(units[f"{model}_p"].to_numpy(float), q)
    ok &= bh_reject(units["oneobj_p"].to_numpy(float), q)
    n1 = units["n_1obj"].to_numpy(float)
    ok &= np.isfinite(n1) & (n1 >= config.MIN_1OBJ_TRIALS)
    if verbose:
        print(
            f"  {DISPLAY[model]}: {int(ok.sum())}/{len(units)} neurons have a measurable "
            f"phase (BH q={q}, >= {config.MIN_1OBJ_TRIALS} 1-obj trials)"
        )
    return ok


def phase_shift_values(units, phases, model, q=None, verbose=False):
    """The values behind one row of the orientation histogram: the 2-object phase relative to
    the neuron's own 1-object receptive-field direction, over every bootstrap.

    The **relational** phase is an orientation (period 180 deg), so subtracting a direction
    leaves a 180 deg ambiguity and each neuron contributes **two** values. The retinotopic
    phase is a direction and contributes one. This is the same convention figure 3 uses.
    """
    keep = units[phase_eligible(units, model, q, verbose)]
    if not len(keep):
        return np.zeros(0)
    ref = keep[["subject", "session", "unit_id", "oneobj_phase_deg"]]
    b = phases.merge(ref, on=["subject", "session", "unit_id"], how="inner")
    shift = pairs.wrap_deg(
        b[f"{model}_axis_deg"].to_numpy(float)
        - b["oneobj_phase_deg"].to_numpy(float)
    )
    if model == "relational":
        shift = np.concatenate([shift, pairs.wrap_deg(shift + 180.0)])
    return shift[np.isfinite(shift)]


def phase_shift_per_neuron(units, model, q=None, verbose=False):
    """**One** shift per eligible neuron, from its point estimates.

    :func:`phase_shift_values` returns every bootstrap draw, which is what the histogram
    should show — the spread of each neuron's estimate is part of the picture. It is not what
    a test should be given: those draws are resamples of the same neuron, so counting them as
    observations inflates n fifty-fold (a hundred for the relational row, which also
    contributes each neuron twice by construction) and any p-value computed from them is
    meaningless. The neuron is the unit of independence, so this is what
    :func:`uniformity_test` reads.
    """
    keep = units[phase_eligible(units, model, q, verbose)]
    if not len(keep):
        return np.zeros(0)
    shift = pairs.wrap_deg(
        keep[f"{model}_axis_deg"].to_numpy(float)
        - keep["oneobj_phase_deg"].to_numpy(float)
    )
    return shift[np.isfinite(shift)]


def uniformity_test(shift_deg, period=360.0, n_mc=10000, seed=0, chunk=2000):
    """Is this row's distribution of shifts uniform? Two tests, Monte-Carlo p-values.

    ``period`` is the circle the quantity lives on: 360 deg for the retinotopic shift, which is
    a direction, and **180 for the relational one**, which is an axis. That is not a detail. A
    relational neuron's shift is only defined modulo 180, so the plotted values come in pairs
    180 deg apart and the first harmonic of the pooled set cancels *exactly* — a Rayleigh test
    run on them would return "perfectly uniform" for any data whatsoever. Testing on the
    doubled angle is what makes the question well posed.

    * **Rayleigh** — the resultant length of ``exp(i k theta)``. Sensitive to a single
      concentration, which is the alternative this figure is about: shifts piling up at 0
      (the 2-object phase sits on the 1-object field) or at 90.
    * **Kuiper** — the circular Kolmogorov-Smirnov statistic ``D+ + D-``, invariant to where
      the circle is cut, and sensitive to *any* departure including a bimodal one the Rayleigh
      test would miss.

    Both p-values come from sampling the null directly rather than from an asymptotic formula,
    so they are exact up to Monte-Carlo error and need no large-n assumption; the
    Laplace-smoothed form means the smallest reportable p is ``1 / (n_mc + 1)``.
    """
    x = np.asarray(shift_deg, float)
    x = x[np.isfinite(x)]
    n = len(x)
    out = dict(
        n=n,
        resultant=np.nan,
        rayleigh_p=np.nan,
        kuiper_p=np.nan,
        mean_shift_deg=np.nan,
    )
    if n < 8:
        return out
    u = (x % period) / period  # to [0, 1) on the quantity's own circle
    out["resultant"] = _rayleigh_r(u)
    out["mean_shift_deg"] = float(
        (np.angle(np.mean(np.exp(2j * np.pi * u))) / (2 * np.pi) * period)
        % period
    )
    obs_r, obs_v = out["resultant"], _kuiper_v(u)
    rng = np.random.default_rng(seed)
    ge_r = ge_v = 0
    for lo in range(0, n_mc, chunk):
        m = min(chunk, n_mc - lo)
        null = rng.random((m, n))
        ge_r += int(
            (np.abs(np.mean(np.exp(2j * np.pi * null), axis=1)) >= obs_r).sum()
        )
        ge_v += int((_kuiper_v(null) >= obs_v).sum())
    out["rayleigh_p"] = (1 + ge_r) / (n_mc + 1)
    out["kuiper_p"] = (1 + ge_v) / (n_mc + 1)
    return out


def _rayleigh_r(u):
    """Mean resultant length of angles given as fractions of a turn."""
    return float(np.abs(np.mean(np.exp(2j * np.pi * np.asarray(u, float)))))


def _kuiper_v(u):
    """Kuiper's ``V = D+ + D-``; accepts one sample or a ``(m, n)`` stack of them."""
    u = np.sort(np.atleast_2d(np.asarray(u, float)), axis=-1)
    n = u.shape[-1]
    i = np.arange(1, n + 1)
    d_plus = np.max(i / n - u, axis=-1)
    d_minus = np.max(u - (i - 1) / n, axis=-1)
    v = d_plus + d_minus
    return (
        float(v[0]) if v.shape[0] == 1 and u.ndim == 2 and v.size == 1 else v
    )


def uniformity_report(units, models=PHASE_MODELS, q=None, n_mc=10000, seed=0):
    """:func:`uniformity_test` per subject and model, as a table."""
    rows = []
    for s in ordered_subjects(units):
        sub = units[units["subject"] == s]
        for m in models:
            period = 180.0 if m == "relational" else 360.0
            r = uniformity_test(
                phase_shift_per_neuron(sub, m, q), period, n_mc, seed
            )
            rows.append(
                dict(
                    subject=subject_title(s),
                    model=DISPLAY[m],
                    period_deg=period,
                    n_neurons=r["n"],
                    mean_shift_deg=round(r["mean_shift_deg"], 1),
                    resultant=round(r["resultant"], 3),
                    rayleigh_p=r["rayleigh_p"],
                    kuiper_p=r["kuiper_p"],
                )
            )
    return pd.DataFrame(rows)


# ── Plotting ──────────────────────────────────────────────────────────────────
def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _mean_ci(d, level=0.95, n_boot=2000, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    if len(d) < 2:
        return np.nan, np.nan
    means = np.mean(d[rng.integers(0, len(d), size=(n_boot, len(d)))], axis=1)
    lo, hi = np.percentile(means, [50 * (1 - level), 100 - 50 * (1 - level)])
    return float(lo), float(hi)


def _violin(
    ax,
    data,
    colors,
    point_size=4,
    point_alpha=0.3,
    violin_alpha=0.25,
    jitter=0.22,
    mean_halfwidth=0.3,
    mean_linewidth=1.6,
    ci_level=0.95,
    seed=0,
):
    parts = ax.violinplot(data, showmeans=False, showextrema=False)
    for body, c in zip(parts["bodies"], colors):
        body.set_facecolor(c)
        body.set_alpha(violin_alpha)
    rng = np.random.default_rng(seed)
    for i, (d, c) in enumerate(zip(data, colors)):
        d = np.asarray(d, float)
        d = d[np.isfinite(d)]
        if not len(d):
            continue
        try:
            wgt = gaussian_kde(d)(d)
            wgt = wgt / wgt.max()
        except Exception:
            wgt = np.ones(len(d))
        ax.scatter(
            (i + 1) + wgt * rng.uniform(-jitter, jitter, len(d)),
            d,
            s=point_size,
            color=c,
            linewidths=0,
            alpha=point_alpha,
            zorder=3,
        )
        m = float(np.mean(d))
        lo, hi = _mean_ci(d, ci_level, rng=rng)
        ax.errorbar(
            i + 1,
            m,
            yerr=[[m - lo], [hi - m]],
            fmt="none",
            ecolor="k",
            elinewidth=1.0,
            capsize=4,
            capthick=1.0,
            zorder=6,
        )
        ax.hlines(
            m,
            i + 1 - mean_halfwidth,
            i + 1 + mean_halfwidth,
            color="k",
            lw=mean_linewidth,
            zorder=5,
        )


def _pull_columns_together(fig, axes, shift_in):
    """Slide column *c* left by ``c * shift_in`` INCHES, after the layout has settled.

    Constrained layout's ``wspace`` is a *minimum*: the real gap here is set by the right
    column's y tick labels and by what the left column hangs past its own right edge (the
    rotated model names, the ``N =`` text), and those exceed any ``wspace`` worth asking for --
    at ``wspace = 0, w_pad = 0`` the gap only falls from 0.128 of the figure width to 0.106. So
    closing it means overriding the engine rather than asking it: draw once to let it settle,
    freeze it, then move the columns.

    The shift leaves the right side of the figure empty; ``bbox_inches='tight'`` crops that at
    save time, so the PDF comes out tighter than the inline preview. Shifting far enough will
    run the right column's tick labels into the left column's -- nothing prevents it, since
    going past what the engine allows is the whole point of the knob.
    """
    if not shift_in:
        return
    fig.canvas.draw()  # let constrained layout finish before overriding it
    fig.set_layout_engine("none")
    dx = float(shift_in) / fig.get_size_inches()[0]
    for row in axes:
        for ci, ax in enumerate(row):
            pos = ax.get_position()
            ax.set_position([pos.x0 - ci * dx, pos.y0, pos.width, pos.height])


def plot_phase_histograms(
    units,
    phases,
    models=PHASE_MODELS,
    n_bins=40,
    figsize=(5.7, 4.1),
    share_y="row",
    q=None,
    null_line=True,
    verbose=True,
):
    """The 2-object phase relative to each neuron's own 1-object receptive field, as
    histograms: rows are models, columns subjects. Returns ``(fig, axes)``.

    ``n_bins`` is the number of bin **edges**, so the histogram has ``n_bins - 1`` bins (40
    edges, 39 bins, as the caption says). ``null_line`` draws the horizontal rule at the density
    a uniform distribution would have over those bins, i.e. the level the histogram is being read
    against. It defaults to True because that is what this function has always drawn; every
    figure in the paper now passes False, so the default is a legacy the callers override.
    """
    subjects = ordered_subjects(units)
    models = list(models)
    edges = np.linspace(-180, 180, n_bins)
    nb = len(edges) - 1
    fig, axes = plt.subplots(
        len(models),
        len(subjects),
        figsize=figsize,
        squeeze=False,
        sharex=True,
        sharey={"row": "row", "all": "all", None: False}[share_y],
        layout="constrained",
    )
    if verbose:
        print(
            "Phase relative to the 1-object receptive field (all bootstraps):"
        )
    tops = np.zeros(axes.shape)
    for ci, s in enumerate(subjects):
        sub = units[units["subject"] == s]
        for ri, m in enumerate(models):
            if verbose:
                print(f"  {subject_title(s)} —", end=" ")
            v = phase_shift_values(sub, phases, m, q, verbose)
            ax = axes[ri][ci]
            if len(v):
                counts, _ = np.histogram(v, bins=edges)
                ax.hist(
                    v,
                    bins=edges,
                    weights=np.full(len(v), 1.0 / len(v)),
                    color=MODEL_COLORS[m],
                    edgecolor="none",
                )
                tops[ri, ci] = max(counts.max() / len(v), 1.0 / nb)
            else:
                tops[ri, ci] = 3.0 / nb
                ax.text(
                    0.5,
                    0.5,
                    "no neurons pass",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="0.5",
                )
            if null_line:
                ax.axhline(1.0 / nb, color="0.35", lw=0.9, zorder=5)
            ax.set_xlim(-180, 180)
            ax.set_xticks(range(-180, 181, 90))
            if ri == 0:
                ax.set_title(subject_title(s))
            if ri == len(models) - 1:
                ax.set_xlabel("1-obj vs 2-obj orientation")
            if ci == 0:
                ax.set_ylabel(f"{DISPLAY[m]} density")
            _despine(ax)
    if share_y == "row":
        tops = np.repeat(
            tops.max(axis=1, keepdims=True), len(subjects), axis=1
        )
    for ri in range(axes.shape[0]):
        for ci in range(axes.shape[1]):
            axes[ri][ci].set_ylim(0, tops[ri, ci] * 1.05)
    return fig, axes


def exclusion_report(units, q=None):
    """How many neurons each prerequisite removes, and why — printed beside the figure so the
    cost of the criteria is visible rather than implied."""
    q = PHASE_FDR_Q if q is None else q
    rows = []
    for m in PHASE_MODELS:  # proximity has a separation, not a phase
        det_2 = bh_reject(units[f"{m}_p"].to_numpy(float), q)
        det_1 = bh_reject(units["oneobj_p"].to_numpy(float), q)
        n1 = units["n_1obj"].to_numpy(float) >= config.MIN_1OBJ_TRIALS
        rows.append(
            dict(
                model=DISPLAY[m],
                n=len(units),
                determinate_2obj=int(det_2.sum()),
                determinate_1obj=int(det_1.sum()),
                enough_1obj_trials=int(n1.sum()),
                all_criteria=int((det_2 & det_1 & n1).sum()),
            )
        )
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# The tuning insets above `plot_phase_histograms`
#
# Ideal neurons at a given shift, drawn as 2-object chords inside their 1-object annulus, one
# above each shift on the x axis. They live here rather than in a notebook because three
# notebooks draw them -- `model_comparison/rf_orientation.ipynb` and the per-area and
# single-unit supplements -- and the supplements are that figure re-drawn over a subset, so a
# difference in how the insets are built would be a difference the reader has to discount.
#
# NOTE the one substantive difference from an earlier version of this code. There, a relational inset
# is built at `rf + shift - 90` because the plotted value is `1/2*chi - phi + 90`. Here the
# estimator already returns `relational_axis_deg = (phase + 180)/2 % 180` -- a midpoint
# direction -- and the histogram plots `axis_deg - oneobj_phase_deg`, so an inset is built at
# `rf + shift` for BOTH models. Checked by simulation: build an ideal neuron at `rf + s`, run
# it back through `pairs_lib.fit_neuron`, and it measures as `s` to within ~1 degree.
# ══════════════════════════════════════════════════════════════════════════════
from matplotlib.colors import Normalize  # noqa: E402


def _boosted(cmap=None, value=1.0, saturation=1.0, blend=0.5, n=256):
    """`coolwarm` with its hot end lifted in HSV at constant hue -- the same red the rings use."""
    from matplotlib.colors import (
        LinearSegmentedColormap,
        hsv_to_rgb,
        rgb_to_hsv,
    )

    cmap = plt.cm.coolwarm if cmap is None else cmap
    t = np.linspace(0.0, 1.0, n)
    rgba = cmap(t)
    hsv = rgb_to_hsv(rgba[:, :3])
    w = np.clip((t - (1.0 - blend)) / blend, 0.0, 1.0)[:, None]
    target = np.stack(
        [hsv[:, 0], np.full(n, float(saturation)), np.full(n, float(value))], 1
    )
    rgba[:, :3] = hsv_to_rgb((1.0 - w) * hsv + w * target)
    return LinearSegmentedColormap.from_list("boosted_coolwarm", rgba)


def _ideal_2obj(model, t0, t1, axis_deg, baseline=10.0, amplitude=0.5):
    """The ideal neuron's 2-object rate. `axis_deg` is the histogram's own coordinate."""
    if model == "retinotopic":
        mu = np.radians(axis_deg)
        return baseline + amplitude * (np.cos(t0 - mu) + np.cos(t1 - mu))
    chi = np.radians(2.0 * axis_deg - 180.0)
    return baseline + amplitude * np.cos(t0 + t1 - chi)


def _ideal_1obj(theta, rf_deg, baseline=10.0, amplitude=0.5):
    """The 1-object reference the shift is measured from."""
    return baseline * np.exp(
        amplitude * np.cos(np.asarray(theta, float) - np.radians(rf_deg))
    )


def draw_shift_inset(
    ax,
    model,
    shift_deg,
    rf_deg=90.0,
    sep_deg=90.0,
    n_orientations=12,
    radius=0.30,
    annulus_radius=0.44,
    annulus_width=0.11,
    linewidth=1.3,
    cmap=None,
    cross_size=4.0,
):
    """One inset: an ideal neuron at this shift, as chords inside its 1-object annulus.

    The annulus is the same in every inset of a row -- it is the reference the shift is
    measured from -- and the chords are that neuron's 2-object tuning. The two are on
    **separate colour scales**: a 1-object rate and a 2-object rate are not comparable, so
    each spans the full blue-to-red of its own range.
    """
    cmap = _boosted() if cmap is None else cmap
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.plot(
        0.5,
        0.5,
        marker="+",
        color="k",
        markersize=cross_size,
        markeredgewidth=1.3 * cross_size / 7,
        zorder=4,
    )

    th = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    r1 = _ideal_1obj(th, rf_deg)
    t = np.append(th, th[0] + 2 * np.pi)
    c = np.append(r1, r1[0])
    TH, RR = np.meshgrid(
        t,
        np.array(
            [
                annulus_radius - annulus_width / 2,
                annulus_radius + annulus_width / 2,
            ]
        ),
    )
    ax.pcolormesh(
        0.5 + RR * np.cos(TH),
        0.5 + RR * np.sin(TH),
        np.vstack([c, c]),
        cmap=cmap,
        norm=Normalize(float(r1.min()), float(r1.max())),
        shading="gouraud",
        zorder=1,
    )

    mids = 2 * np.pi * np.arange(n_orientations) / n_orientations
    half = np.radians(sep_deg) / 2
    fr = _ideal_2obj(model, mids - half, mids + half, rf_deg + shift_deg)
    norm = Normalize(float(fr.min()), float(fr.max()))
    for k, (m, v) in enumerate(zip(mids, fr)):
        a, b = m - half, m + half
        x0, y0 = 0.5 + radius * np.cos(a), 0.5 + radius * np.sin(a)
        x1, y1 = 0.5 + radius * np.cos(b), 0.5 + radius * np.sin(b)
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        col = cmap(norm(v))
        # Split at the midpoint so every chord passes in front of its clockwise neighbour and
        # behind its counter-clockwise one -- no seam at the wrap-around.
        ax.plot(
            [xm, x1],
            [ym, y1],
            color=col,
            lw=linewidth,
            solid_capstyle="round",
            zorder=2 + k / (n_orientations + 1),
        )
        ax.plot(
            [xm, x0],
            [ym, y0],
            color=col,
            lw=linewidth,
            solid_capstyle="round",
            zorder=3 + k / (n_orientations + 1),
        )
    return ax


def add_shift_insets(
    fig,
    axes,
    models,
    shifts_deg=(-90, 0, 90, 180),
    rf_deg=90.0,
    size_in=0.55,
    y=1.02,
    title_pad=40,
    hspace=0.42,
    connector_color="0.75",
    connector_lw=0.9,
    **kw,
):
    """One `draw_shift_inset` per shift, above every panel, at that shift's own x coordinate.

    `hspace` is opened on the layout engine BEFORE the draw, and it is not cosmetic: an inset
    sits above its own panel, so without a gap between the rows the lower row's insets land
    inside the upper row's histogram. That is what happened on the first attempt here.

    Everything else is placed after the draw, once the panels have settled, and the layout
    engine is then switched off -- it would otherwise reflow at draw time and leave the
    insets behind.

    `connector_color` draws the line tying each inset to its value on the x axis, across the
    histogram and up through the gap. Without it the insets are a row of pictures near an
    axis; with it each one is pinned to a shift.
    """
    eng = fig.get_layout_engine()
    if eng is not None and hspace is not None:
        eng.set(hspace=hspace)
    fig.canvas.draw()
    fig.set_layout_engine("none")
    for ri, m in enumerate(models):
        for ci in range(axes.shape[1]):
            ax = axes[ri][ci]
            bb = ax.get_window_extent().transformed(
                fig.dpi_scale_trans.inverted()
            )
            wf, hf = size_in / bb.width, size_in / bb.height
            lo, hi = ax.get_xlim()
            for sh in shifts_deg:
                xf = (sh - lo) / (hi - lo) - wf / 2
                if connector_color:
                    # Behind the bars inside the panel, and on up to the inset's lower edge.
                    ax.axvline(
                        sh, color=connector_color, lw=connector_lw, zorder=0
                    )
                    ax.plot(
                        [sh, sh],
                        [1.0, y],
                        transform=ax.get_xaxis_transform(),
                        color=connector_color,
                        lw=connector_lw,
                        clip_on=False,
                        zorder=0,
                    )
                inax = ax.inset_axes([xf, y, wf, hf])
                # An inset may overhang the row above when `hspace` is small. Lift it
                # clear so it draws OVER that panel instead of under it.
                inax.set_zorder(ax.get_zorder() + 10)
                inax.patch.set_visible(False)
                draw_shift_inset(inax, m, sh, rf_deg=rf_deg, **kw)
            if ri == 0 and ax.get_title():
                ax.set_title(ax.get_title(), pad=title_pad)

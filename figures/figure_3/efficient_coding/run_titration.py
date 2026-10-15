"""The titration sweep. Writes `dataframes/<name>.csv` for `titration.ipynb`.

    python regenerate.py                       # the real entry point: 12 of these in parallel
    python run_titration.py --dry-run          # pool sizes and K, trains nothing
    python run_titration.py --force            # ignore checkpoints and recompute

What the sweep is
-----------------
Sweep point ``x`` is a **target weighted mean rank** of the relationality coordinate rho.
``lam(x)`` (:func:`lam_for_mean_rank`) turns it into an exponential tilt, and
:func:`tilt_positions` draws ``K`` distinct units from the WHOLE pool without replacement with
probability ~ ``exp(lam * rank)``. An MLP is then trained to read the two object angles off
those units' simulated responses, and the mean per-object decode error is recorded.

Three properties carry the result:

1. **rho is the angle in the two models' held-out excess accuracies** (the model comparison's
   score), and neurons must clear the shared modulation criterion of ``figures/config.py``
   and have both models beat chance out of sample (``min_fit=0.0``, ``SELECTION``) before
   entering the pool. Without the modulation cut the neurons neither model explains land at
   rho = 0 and masquerade as retinotopic, and they are not equally common in the two animals,
   so the sweeps would not be measuring the same thing.
2. **The tilt is on rho ranked within subject**, so a sweep point means the same quantile of
   each animal's population, and ``lam(0.5) = 0`` makes x = 0.5 *exactly* the unbiased
   population -- a landmark fixed by construction rather than estimated.
3. **Every row records the realized composition** of the population actually drawn
   (``mean_rho_inputs``), so the curve is plotted against a proportion that exists in the data
   instead of against the sampling parameter.

Two things are fixed and are deliberately not knobs, because varying either would make the
curve's points incomparable: ``N_SAMPLES`` decoder inputs at every point (a point drawing more
neurons does not also get a bigger model), and ``K`` distinct units at every point
(:func:`matched_k`), so no point is measured with more neurons than another.
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path

for _v in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
_REPO = (
    Path(__file__).resolve().parents[3]
)  # repository root, where paths.py lives
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
import pool_lib as pool  # noqa: E402
import selection_lib as sel  # noqa: E402

import paths  # noqa: E402

# ── The sweep ──────────────────────────────────────────────────────────────────
SWEEP_START, SWEEP_STOP, SWEEP_STEP = 0.1, 0.9, 0.1
N_REPEATS = 20

N_SAMPLES = 500  # bootstrap inputs = the model's input dimension, fixed
SAMPLE_SIGMA_DEG = 10.0
FIRING_RATE_SCALE = 0.1
PRECOMPUTE_N_GRID = 36
#: Which neurons are eligible — the same dict `titration_lib.SELECTION` uses, so the pool the
#: figure describes is the pool that was fitted.
SELECTION = dict(exclusion="config", min_fit=0.0)

MIN_SEP_DEG, N_ANGLES, TARGET_SIGMA_DEG = 80.0, 96, 12.0
HIDDEN_SIZES, BATCH_SIZE, N_STEPS, LR = [64, 64], 128, 1000, 2e-3
N_TEST, MIN_PEAK_SEP_DEG = 2000, 30.0
SEED = 0

SWEEP = np.round(np.arange(SWEEP_START, SWEEP_STOP + 1e-9, SWEEP_STEP), 4)


def matched_k(data):
    """`K`: the smallest wedge over the **whole** sweep — 186 (Monkey P) / 62 (Monkey E).

    Every sweep point draws exactly this many distinct units, the extremes included, so no point
    on the curve is measured with more neurons than another. Taking K from a sub-range instead
    would leave the extremes either short or drawn at their own size; both make the curve's
    points incomparable, which is why the range is not a parameter.
    """
    return min(len(sel.wedge_positions(data["rank"], x)[0]) for x in SWEEP)


def mean_rank_of_tilt(lam):
    """Mean of the density proportional to ``exp(lam*r)`` on the unit interval.

    Ranks are (i - 0.5)/n by construction, i.e. uniform on [0, 1] to within 1/n — so this
    continuous form is the mean rank a tilt of strength ``lam`` produces, and it depends on
    **no data at all**. That is what lets the sweep quote one lambda grid for both animals.
    ``m(0) = 1/2`` exactly, which is why lambda = 0 is the unbiased population.
    """
    if abs(lam) < 1e-6:
        return (
            0.5 + lam / 12.0
        )  # the series, to keep the expression finite at 0
    if lam > 60:  # exp(-lam) underflows; the limit is 1 - 1/lam
        return 1.0 - 1.0 / lam
    if lam < -60:  # ... and -1/lam on the other side
        return -1.0 / lam
    return 1.0 / (1.0 - np.exp(-lam)) - 1.0 / lam


def lam_for_mean_rank(x):
    """Invert :func:`mean_rank_of_tilt`: the tilt whose weighted mean rank is ``x``.

    ``m`` is strictly increasing, so this is a plain bisection. ``x = 0.5`` returns exactly 0.
    The grid for x = 0.1 ... 0.9 comes out antisymmetric: -10.00, -4.80, -2.67, -1.23, 0,
    +1.23, +2.67, +4.80, +10.00.

    Note this targets the **weighted pool**, not the draw. Drawing ``k`` distinct units without
    replacement caps the realized mean rank at ``[k/2n, 1 - k/2n]`` whatever lambda is, so the
    realized value falls short of ``x`` at the ends. That is reported, not corrected: the
    figure's axis is the realized composition either way.
    """
    if abs(x - 0.5) < 1e-12:
        return 0.0
    from scipy.optimize import brentq

    return float(
        brentq(
            lambda L: mean_rank_of_tilt(L) - float(x),
            -200.0,
            200.0,
            xtol=1e-10,
        )
    )


def tilt_positions(rho, lam, k, rng):
    """``k`` distinct units drawn **without replacement** with probability ~ exp(lam * rho).

    Two properties carry it:

    * **Every unit has weight at every point.** There is no boundary, so the composition moves
      smoothly and no neuron is in or out by a hair's breadth.
    * **The number of distinct units is exactly ``k`` at every point**, the extremes included,
      so the curve is never confounded with neuron count.

    A consequence worth knowing before reading the curve: the *spread* of rho among the drawn
    units peaks at ``lam = 0`` -- concentrating weight to move the mean necessarily narrows it
    -- so a one-parameter tilt does not remove the mean/diversity confound, it only removes the
    hard boundary.

    Implemented by the Gumbel top-k trick: adding i.i.d. Gumbel noise to ``log w`` and taking
    the largest ``k`` keys is exactly successive sampling without replacement proportional to
    ``w``, in one vectorised pass rather than ``k`` renormalisations.
    """
    z = lam * (np.asarray(rho, float) - float(np.mean(rho)))
    keys = z + rng.gumbel(size=len(z))
    k = min(int(k), len(z))
    return np.argpartition(-keys, k - 1)[:k]


def run_point(data, subject, x, i, k, lam, cfg):
    """``N_REPEATS`` trainings at one sweep point. Returns its rows and the pool size.

    Every unit is eligible at every ``lam``, and the draw is always ``k`` distinct units, so no
    point is ever short and every point on the curve is comparable to every other.
    """
    n_pool = len(data["rho"])
    # The tilt is on the WITHIN-ANIMAL RANK of rho, which is what makes lam mean the same thing
    # in both animals and lam = 0 exactly the unbiased population (see lam_for_mean_rank).
    tilt_stat = data["rank"]
    boot_rng = np.random.default_rng(SEED + 1000 * i)
    train_rng = np.random.default_rng(SEED + 1 + 1000 * i)
    rows = []
    for r in range(N_REPEATS):
        sub_pos = tilt_positions(
            tilt_stat, lam, k, boot_rng
        )  # redrawn every repeat
        unit_rows, shifts = pool.sample_bootstrap_table_inputs(
            sub_pos, N_SAMPLES, PRECOMPUTE_N_GRID, boot_rng
        )
        neural_fn = pool.make_table_neural_fn(
            data["tables"],
            unit_rows,
            shifts,
            firing_rate_scale=FIRING_RATE_SCALE,
        )
        torch.manual_seed(SEED + 100 * i + r)
        model = pool.StimulusRepresentationModel(
            n_neurons=N_SAMPLES,
            n_angles=N_ANGLES,
            hidden_sizes=tuple(HIDDEN_SIZES),
        )
        pool.train_model(model, neural_fn, cfg, train_rng)
        err = pool.evaluate_error(
            model,
            neural_fn,
            cfg,
            train_rng,
            n_test=N_TEST,
            min_peak_sep_deg=MIN_PEAK_SEP_DEG,
        )
        rows.append(
            dict(
                subject=subject,
                # The sweep point x, which is the target weighted mean rank.
                sweep_fraction=float(x),
                repeat=r,
                lam=float(lam),
                pool_size=n_pool,
                # Always K: every point is drawn at exactly K distinct units.
                n_units=len(sub_pos),
                # The x axis the figure actually uses: what the decoder received, not what was
                # asked for. `_inputs` is over the N_SAMPLES bootstrap draws (with replacement,
                # so it is the composition the model saw); `_units` is over distinct neurons.
                mean_rho_inputs=float(np.mean(data["rho"][unit_rows])),
                mean_rho_units=float(np.mean(data["rho"][sub_pos])),
                mean_rank_inputs=float(np.mean(data["rank"][unit_rows])),
                mean_rank_units=float(np.mean(data["rank"][sub_pos])),
                error_deg=float(err.mean()),
                rms_deg=float(np.sqrt((err**2).mean())),
                mse_deg2=float((err**2).mean()),
                median_deg=float(np.median(err)),
            )
        )
        del model, neural_fn, err
    gc.collect()
    return rows, n_pool


def stamp_params(df, lams, k):
    """What this run was, written onto every row, so a CSV is self-describing.

    These are all constants of the module rather than knobs, but they are still worth recording:
    `titration.ipynb` prints them as its provenance line, and a CSV that outlives a change to one
    of these constants says so itself instead of being silently averaged with a newer one.
    """
    df = df.copy()
    df["param_exclusion"] = SELECTION["exclusion"]
    df["param_min_fit"] = SELECTION["min_fit"]
    df["param_tilt_on"] = "rank"
    df["param_lams"] = ",".join(f"{v:g}" for v in lams)
    df["param_n_units_matched"] = int(k)
    df["param_objective"] = "bump"
    df["param_hidden"] = ",".join(str(h) for h in HIDDEN_SIZES)
    df["param_n_samples"] = N_SAMPLES
    df["param_rotation_bins"] = (
        PRECOMPUTE_N_GRID  # orientations each unit can be drawn at
    )
    df["param_repeats"] = N_REPEATS
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", default=None, help="output stem in dataframes/")
    ap.add_argument(
        "--out-dir",
        default=None,
        help="write the results here instead of "
        "cache/figures/figure_3/efficient_coding/dataframes/",
    )
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--repeats", type=int, default=N_REPEATS)
    ap.add_argument(
        "--points",
        default=None,
        help="comma-separated sweep points to run (default all). Parts are per "
        "(subject, point), so several processes can share a parts directory.",
    )
    ap.add_argument(
        "--subject",
        default=None,
        choices=sel.SUBJECTS,
        help="fit one subject only. The two are independent, and a training does "
        "not scale past ~2 torch threads, so one process per subject is a "
        "real halving of wall time where more threads is not. Checkpoints "
        "are per (subject, sweep point) in the same parts directory, so the "
        "processes cannot collide; finish with --combine-only.",
    )
    ap.add_argument(
        "--combine-only",
        action="store_true",
        help="rebuild the CSV from whatever parts exist, fitting nothing",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="pool sizes and K; trains nothing",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if not 1 <= args.threads <= 10:
        raise SystemExit("--threads must be 1..10: leave the machine usable")
    torch.set_num_threads(args.threads)
    globals()["N_REPEATS"] = args.repeats

    # One lambda per sweep point, from that point's target mean rank. The map is
    # data-independent, so this grid is identical for both animals and lam(0.5) = 0 exactly.
    lams = [lam_for_mean_rank(x) for x in SWEEP]
    print(
        "sweep point (target mean rank) -> lambda:  "
        + "  ".join(f"{x:g}->{l:+.2f}" for x, l in zip(SWEEP, lams))
    )
    name = args.name or "results"
    out_dir = Path(
        args.out_dir
        or (paths.FIGURES_CACHE / "figure_3/efficient_coding/dataframes")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = out_dir / f"{name}_parts"
    parts.mkdir(exist_ok=True)

    cfg = dict(
        min_sep_deg=MIN_SEP_DEG,
        n_angles=N_ANGLES,
        target_sigma_deg=TARGET_SIGMA_DEG,
        batch_size=BATCH_SIZE,
        n_steps=N_STEPS,
        lr=LR,
    )

    if args.combine_only:
        frames = [pd.read_csv(f) for f in sorted(parts.glob("*.csv"))]
        frames = [f for f in frames if len(f)]
        if not frames:
            raise SystemExit(f"no parts in {parts}")
        res = pd.concat(frames, ignore_index=True)
        res.to_csv(out_dir / f"{name}.csv", index=False)
        print(
            f'combined {len(frames)} parts -> {out_dir / f"{name}.csv"} ({len(res)} rows)'
        )
        for s_, g in res.groupby("subject"):
            print(
                f'  {sel.subject_title(s_)}: {g["sweep_fraction"].nunique()} sweep points'
            )
        return

    all_rows = []
    wanted = (
        None if not args.points else [float(v) for v in args.points.split(",")]
    )
    t_start = time.time()
    for subject in [args.subject] if args.subject else sel.SUBJECTS:
        print(f"\n{sel.subject_title(subject)}")
        data = pool.subject_pool(
            subject,
            sample_sigma_deg=SAMPLE_SIGMA_DEG,
            n_grid=PRECOMPUTE_N_GRID,
            verbose=True,
            with_tables=not args.dry_run,
            **SELECTION,
        )
        # K is the smallest rank wedge over the sweep (see matched_k); the wedges are only used
        # to size it, never to select.
        sizes = [len(sel.wedge_positions(data["rank"], x)[0]) for x in SWEEP]
        k = matched_k(data)
        print(
            f"  wedge sizes {sizes}  -> K = {k} (every point, the extremes included)"
            + (
                ""
                if args.dry_run
                else f'  | eval trials/neuron median {np.median(data["n_eval_trials"]):.0f}'
            )
        )
        if args.dry_run:
            continue

        # One part per (subject, sweep point), written only once complete, so a killed run
        # resumes at the point it died and several processes can share a parts directory
        # (`--points` splits the sweep across them; see regenerate.py).
        for i, x in enumerate(SWEEP):
            if wanted is not None and not any(
                abs(x - p) < 1e-9 for p in wanted
            ):
                continue
            part = parts / f"{subject}_{x:.2f}.csv"
            if part.exists() and not args.force:
                print(f"  x={x:.2f}  (checkpoint)", flush=True)
                continue
            t1 = time.time()
            rows, npool = run_point(data, subject, x, i, k, lams[i], cfg)
            df = stamp_params(pd.DataFrame(rows), lams, k)
            tmp = part.with_suffix(".csv.tmp")
            df.to_csv(tmp, index=False)
            os.replace(tmp, part)  # a part exists only once it is complete
            all_rows.append(df)
            print(
                f'  x={x:.2f}  pool={npool:4d}  mean error={df["error_deg"].mean():5.1f}°'
                f"  ({time.time() - t1:.0f}s)",
                flush=True,
            )
        del data
        gc.collect()

    if args.dry_run:
        return
    if all_rows:
        res = pd.concat(all_rows, ignore_index=True)
        res.to_csv(out_dir / f"{name}.csv", index=False)
        print(
            f'\nwrote {out_dir / f"{name}.csv"} ({len(res)} rows)  '
            f"[{(time.time() - t_start) / 60:.1f} min] — finish with --combine-only"
        )


if __name__ == "__main__":
    main()

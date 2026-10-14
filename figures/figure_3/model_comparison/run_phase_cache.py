"""Build the **phase fit** `rf_orientation.ipynb` reads. Resumable and checkpointed.

    python run_phase_cache.py --sessions 1     # smoke test; prints a projection of the full build
    python run_phase_cache.py --n-jobs 8       # everything
    python run_phase_cache.py --combine-only   # re-combine existing checkpoints, fit nothing
    python run_phase_cache.py --force          # discard checkpoints and refit

One checkpoint per session, written atomically, so a part file exists only once complete and
an interrupted run resumes from the sessions already finished.

What it fits, per neuron, is the 2-object tuning vector ``gamma`` of :mod:`pairs_lib` — and
what panel 3e needs from it is the **angle** of that vector relative to the neuron's own
1-object receptive field, with a bootstrap distribution for it (``phase_draws.csv.gz``) and a
Wald test of determinacy. The model comparison is not here: that is `run_comparison_cache.py`,
which scores models on the sign of the orderings they predict and estimates no phase bootstrap.
This script still writes the retired graded comparison's dLL columns because it has not been
re-run since they stopped being read; `stats_lib.load_units` drops them.

**This does not rebuild the firing rates or `modulation.csv`.** Both are model-free and shared
by every figure-3 analysis.

Cost is set almost entirely by `--n-boot`; the fits are exact and nearly free. Measured on
real data at the default comparison cap, per neuron and over 3186 neurons on 8 workers:

    --n-boot   0   0.5 s/neuron    ~4 min   (no bootstrap: no phase histogram either)
    --n-boot  20   1.1 s/neuron    ~9 min
    --n-boot  50   2.1 s/neuron   ~15 min   (the default)
    --n-boot 200   7.1 s/neuron   ~48 min

Only `rf_orientation.ipynb` reads the bootstrap. If you want just the win counts, `--n-boot 0`
is the whole job.

Never `--n-jobs -1`: this machine has 14 cores and a build has to leave it usable.
"""

import argparse
import os
import shutil
import time

for _v in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import pairs_lib as pairs  # noqa: E402
import stats_lib as sl  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--sessions", type=int, default=None, help="first N only (smoke test)"
    )
    ap.add_argument(
        "--stride",
        type=int,
        default=1,
        help="take every Nth session. Sessions are sorted and the first 12 are "
        "one animal, so a --sessions limit alone gives a single-monkey "
        "sample; a stride covers both.",
    )
    ap.add_argument("--sigma-deg", type=float, default=pairs.SIGMA_DEG)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument(
        "--n-boot",
        type=int,
        default=pairs.N_BOOT,
        help="trial resamples per neuron. The knob that sets the runtime — "
        "the fits are exact and nearly free, the bootstrap is not. "
        "See pairs_lib.N_BOOT for what the default is and is not "
        "enough for. 0 skips it entirely, which is all the model "
        "comparison needs (rf_orientation does need it).",
    )
    ap.add_argument(
        "--max-comparisons", type=int, default=pairs.MAX_COMPARISONS
    )
    ap.add_argument(
        "--sep-max-comparisons",
        type=int,
        default=pairs.SEP_MAX_COMPARISONS,
        help="comparisons kept for the proximity profile. Lower than the others "
        "because the profile is re-run inside every bootstrap resample, and "
        "precision is set by the trial count, not the comparison count.",
    )
    ap.add_argument(
        "--no-proximity",
        action="store_true",
        help="fit only the two slope models, as before proximity existed.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--combine-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if not 1 <= args.n_jobs <= 10:
        raise SystemExit(
            "--n-jobs must be between 1 and 10: leave the machine usable"
        )

    cfg = dict(
        sigma_deg=args.sigma_deg,
        n_folds=args.n_folds,
        n_boot=args.n_boot,
        max_comparisons=args.max_comparisons,
        seed=args.seed,
        with_proximity=not args.no_proximity,
        sep_max_comparisons=args.sep_max_comparisons,
    )
    cfg["tag"] = (
        f"v1_ordinal_s{args.sigma_deg:g}_k{args.n_folds}_b{args.n_boot}"
        f"_m{args.max_comparisons}_seed{args.seed}"
        + (f"_prox{args.sep_max_comparisons}" if cfg["with_proximity"] else "")
    )

    if args.combine_only:
        print(f"combining checkpoints in {sl.parts_dir_for(cfg)}")
        sl.combine_parts(cfg)
        return

    parts = sl.parts_dir_for(cfg)
    if args.force and parts.exists():
        shutil.rmtree(parts)
    parts.mkdir(parents=True, exist_ok=True)

    sessions = sl.fr_sessions()[:: max(args.stride, 1)]
    if args.sessions:
        sessions = sessions[: args.sessions]
    done = {f.name[: -len("_units.csv")] for f in parts.glob("*_units.csv")}
    todo = [s for s in sessions if f"{s[0]}_{s[1]}" not in done]

    print(f'config: {cfg["tag"]}')
    print(
        f"{len(todo)} of {len(sessions)} sessions to fit, {args.n_jobs} workers"
        + (
            f" ({len(sessions) - len(todo)} already checkpointed)"
            if done
            else ""
        ),
        flush=True,
    )

    t0 = time.time()
    if todo:
        if args.n_jobs == 1 or len(todo) == 1:
            results = [sl.build_session(s, ss, cfg) for s, ss in todo]
        else:
            from joblib import Parallel, delayed

            results = Parallel(n_jobs=min(args.n_jobs, len(todo)))(
                delayed(sl.build_session)(s, ss, cfg) for s, ss in todo
            )
            for name, n, secs in results:
                print(f"  {name}: {n} neurons ({secs:.0f} s)", flush=True)
        fitted = [r for r in results if r[2] > 0]
        print(
            f"\n{(time.time() - t0) / 60:.1f} min for {len(fitted)} sessions",
            flush=True,
        )
        if fitted and len(sessions) < len(sl.fr_sessions()):
            per = sum(r[2] for r in fitted) / len(fitted)
            n_all = len(sl.fr_sessions())
            print(
                f"projection: ~{per:.0f} s/session x {n_all} / {args.n_jobs} workers "
                f"~= {per * n_all / args.n_jobs / 60:.0f} min for the full build"
            )

    have = {f.name[: -len("_units.csv")] for f in parts.glob("*_units.csv")}
    missing = [s for s in sessions if f"{s[0]}_{s[1]}" not in have]
    if missing:
        print(
            f"\n{len(missing)} sessions still missing — not combining. Re-run to continue; "
            f"nothing already fit will be refit."
        )
        return
    print("\ncombining:")
    units, _ = sl.combine_parts(cfg)
    print(f"\n{len(units)} neurons fit. Next: open model_comparison.ipynb.")


if __name__ == "__main__":
    main()

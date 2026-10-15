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

**This does not rebuild the firing rates or `modulation.csv`.** Both are model-free and shared
by every figure-3 analysis.

Only the neurons that pass the shared modulation criterion (`neuron_selection.py`, 1339) are
fitted, since `rf_orientation.ipynb` reads no other. Cost is set almost entirely by
`--n-boot`; the fits are exact and nearly free. Measured per neuron on real data at the
default comparison cap (before the retired graded fits were removed, so now an upper bound):

    --n-boot   0   0.5 s/neuron
    --n-boot  20   1.1 s/neuron
    --n-boot  50   2.1 s/neuron   (the default; ~6 min for 1339 neurons on 8 workers)
    --n-boot 200   7.1 s/neuron

`rf_orientation.ipynb` (figure 3e) is the only reader of this cache, and it needs the
bootstrap.

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
        "--n-boot",
        type=int,
        default=pairs.N_BOOT,
        help="trial resamples per neuron. The knob that sets the runtime — "
        "the fits are exact and nearly free, the bootstrap is not. "
        "See pairs_lib.N_BOOT for what the default is and is not "
        "enough for. 0 skips it entirely (rf_orientation needs it).",
    )
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--combine-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.n_jobs < 1:
        raise SystemExit("--n-jobs must be at least 1")

    cfg = dict(
        sigma_deg=pairs.SIGMA_DEG,
        n_boot=args.n_boot,
        max_comparisons=pairs.MAX_COMPARISONS,
        seed=0,
    )
    # The "_k5" is historical (the fit once also ran a 5-fold cross-validation); it is kept so
    # the tag, and so the checkpoint directory, stays what the published cache was built under.
    cfg["tag"] = (
        f"ordinal_s{cfg['sigma_deg']:g}_k5_b{cfg['n_boot']}"
        f"_m{cfg['max_comparisons']}_seed{cfg['seed']}"
    )

    if args.combine_only:
        print(f"combining checkpoints in {sl.parts_dir_for(cfg)}")
        sl.combine_parts(cfg)
        return

    parts = sl.parts_dir_for(cfg)
    if args.force and parts.exists():
        shutil.rmtree(parts)
    parts.mkdir(parents=True, exist_ok=True)

    sessions = sl.fr_sessions()
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
    print(f"\n{len(units)} neurons fit. Next: open rf_orientation.ipynb.")


if __name__ == "__main__":
    main()

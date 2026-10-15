"""Fit every model on every session, then re-measure relationality on model-generated data.

    python run_cache.py                    # both stages, 8 workers
    python run_cache.py --n-jobs 6         # fewer workers
    python run_cache.py --stage fits       # just the fits
    python run_cache.py --sessions 2       # smoke test on the first two sessions

Two stages, both **resumable and incremental** — stop this at any point and re-run it, and
it picks up where it left off:

  1. **Fits.** One session at a time. A session whose cache already holds a model is not
     refit for that model: `fit_lib.fit_session` reads what is on disk, compares the model
     list, and fits only what is missing. The cross-validation splits are seeded per neuron
     before any model is fit, so a model added later is scored on exactly the same held-out
     trials as the ones already cached.
  2. **Relationality recovery.** For each model, spikes are simulated from that model's own
     fit to each neuron, on that neuron's own trials, and the relational-bias `bias` is
     re-measured from them. This is what says whether the measured distribution of `bias` is
     one a non-relational model could have produced, and it is what fills the cache
     `figure_ks_stats.ipynb` reads — run this and that notebook draws in seconds instead of
     refitting thousands of neurons. Rows are cached per
     (neuron, model, replicate) with a fingerprint of the fit they came from, so this too
     computes only what is missing.

`--n-jobs` defaults to 8, which leaves a typical workstation usable; `figures/build_caches.py`
passes 12, for a long unattended run where throughput matters more.

**Resuming.** Both stages are resumable at *session* granularity: `fit_session` writes a
session's fits only once that session finishes, so re-running after an interruption redoes
at most the session that was in flight and skips every session already on disk. Just re-run
the same command.
"""

import argparse
import os
import time

# Cap the BLAS threads BEFORE numpy is imported, here and in every worker this spawns —
# otherwise macOS Accelerate takes every core inside each of `--n-jobs` workers.
for _v in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import common  # noqa: E402
import data_lib as dl  # noqa: E402
import fit_lib  # noqa: E402
import ks_lib  # noqa: E402
import models_lib as ml  # noqa: E402
import relationality_lib as rl  # noqa: E402

#: Every model.
ALL_MODELS = tuple(ml.MODEL_ORDER)

N_REPS, SIM_SEED = 5, 0


def fit_all(cfg, sessions, n_jobs, verbose=True):
    """Stage 1 — every model on every session, topping up whatever the cache lacks."""
    for i, (subject, session) in enumerate(sessions, 1):
        t = time.time()
        print(f"\n[{i}/{len(sessions)}] {subject}/{session}", flush=True)
        fit_lib.fit_session(
            subject, session, cfg, n_jobs=n_jobs, verbose=verbose
        )
        print(f"    {time.time() - t:.0f} s", flush=True)
    fit_lib.load_summary(cfg, rebuild="always", verbose=verbose)


def recover_biases(cfg, n_jobs, n_reps=N_REPS, seed=SIM_SEED, verbose=True):
    """Stage 2 — the relationality of model-generated data, for every model.

    The population is `common.selected_fits`, the shared model-agnostic criterion, and it is
    not a knob: `figure_ks_stats.ipynb` reads these rows for exactly those neurons, and any
    other population here would leave the figure refitting the difference on every run.

    `relational_bias` is in the model list because the figure needs a row for it too: it is
    the metric's own model, and its synthetic distribution is the noise floor the others are
    read against.
    """
    models = tuple(m for m in ALL_MODELS if m != "constant")
    summary = common.selected_fits(models, verbose=False)
    n = summary[ks_lib.UNIT_KEYS].drop_duplicates().shape[0]
    print(
        f"\nrecovery: {len(models)} models x {n_reps} replicates x {n} neurons "
        f"= {len(models) * n_reps * n:,} rows",
        flush=True,
    )

    real = rl.biases(summary, verbose=False)
    print(
        f'real bias: median {real["bias"].median():.3f}, n = {len(real):,}',
        flush=True,
    )
    synth = ks_lib.synthetic_biases(
        summary,
        models,
        cfg,
        n_reps=n_reps,
        seed=seed,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    print(f"\n{len(synth):,} synthetic neurons, {len(real):,} real")
    print(
        ks_lib.ks_table(
            real,
            synth,
            models,
            by_subject=True,
            n_boot=200,
        )
        .round(3)
        .to_string()
    )
    return synth


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--n-jobs",
        type=int,
        default=8,
        help="worker processes",
    )
    ap.add_argument(
        "--stage", choices=("all", "fits", "recovery"), default="all"
    )
    ap.add_argument(
        "--sessions", type=int, default=None, help="first N sessions only"
    )
    ap.add_argument("--n-reps", type=int, default=N_REPS)
    args = ap.parse_args()
    if args.n_jobs < 1:
        raise SystemExit("--n-jobs must be at least 1")

    cfg = fit_lib.FitConfig(models=ALL_MODELS)
    sessions = dl.fr_sessions()
    if args.sessions:
        sessions = sessions[: args.sessions]

    t0 = time.time()
    print(
        f"cache tag: {cfg.tag}\n{len(ALL_MODELS)} models, {len(sessions)} sessions, "
        f"{args.n_jobs} workers",
        flush=True,
    )

    if args.stage in ("all", "fits"):
        print("\n=== 1/2 fits ===", flush=True)
        fit_all(cfg, sessions, args.n_jobs)
    if args.stage in ("all", "recovery"):
        print(
            "\n=== 2/2 relationality of model-generated data ===", flush=True
        )
        recover_biases(cfg, args.n_jobs, n_reps=args.n_reps)

    print(f"\ntotal {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()

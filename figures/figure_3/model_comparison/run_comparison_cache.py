#!/usr/bin/env python
"""Build figure 3's model-comparison cache: `comparison_units.csv`.

    python run_comparison_cache.py                     # the full build
    python run_comparison_cache.py --sessions 2        # a quick look
    python run_comparison_cache.py --combine-only      # rebuild the CSV from existing parts
    python run_comparison_cache.py --boot-range 500:1000   # extend the bootstrap (see below)

Each model is scored on the **orderings it predicts** and on nothing else -- see
`comparison_lib`'s module docstring for the estimator and for what the graded alternative it
replaced was doing. Checkpointed per session, so an interrupted run resumes.

**No comparison cap.** Every comparison of every neuron is used; the fit is O(N log N) in rows
and can afford it. The graded comparison that preceded this one capped at 400,000 comparisons
per neuron for the slope models and 100,000 for proximity because it could not, and 52% / 82%
of neurons hit those caps. That cap is gone with it: the precision of these statistics is set
by the number of *trials*, not of comparisons, so it was never buying accuracy.

**The bootstrap is extendable.** Replicate ``b`` is seeded from ``b`` alone, so
``--boot-range 500:1000`` after a ``0:500`` run yields exactly the draws one 1000-replicate run
would have produced. Draws are stored per session as `.npz` under `comparison_boot/` and every
p-value is recomputed from the union of the chunks on disk, so nothing has to be refit to add
replicates. Re-run `model_comparison.ipynb` afterwards and nothing else.
"""
import argparse
import os
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

import comparison_lib as cl  # noqa: E402
import numpy as np  # noqa: E402
import pairs_lib as pairs  # noqa: E402
import pandas as pd  # noqa: E402
import stats_lib as sl  # noqa: E402

PARTS_DIR = sl.CACHE_DIR / "comparison_parts"
BOOT_DIR = sl.CACHE_DIR / "comparison_boot"


def build_session(subject, session, cfg, overwrite=False, verbose=True):
    """Fit every neuron of one session and checkpoint it atomically.

    Two files per session: the summary CSV, and an `.npz` of the per-replicate bootstrap draws
    for the replicate range this run covered. The second is what makes a bootstrap extendable:
    replicate b is seeded from b alone, so `--boot-range 500:1000` after a 0:500 run yields
    exactly the draws one 1000-replicate run would have produced.
    """
    parts = Path(cfg["parts_dir"])
    parts.mkdir(parents=True, exist_ok=True)
    boot_dir = Path(cfg["boot_dir"])
    boot_dir.mkdir(parents=True, exist_ok=True)
    lo, hi = cfg["boot_lo"], cfg["boot_hi"]
    boot_path = boot_dir / f"{subject}_{session}_boot_{lo:05d}_{hi:05d}.npz"
    out_path = parts / f"{subject}_{session}_fit.csv"
    if out_path.exists() and not overwrite:
        prev = pd.read_csv(out_path)
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
    units = sl.session_unit_trials(subject, session)
    rows, draws, uids = [], [], []
    for uid, d in sorted(units.items()):
        if len(d["fr2"]) < sl.MIN_2OBJ_FIT:
            continue
        seed = cfg["seed"] + uid
        r, dr = cl.fit_neuron_sign(
            d["t0"],
            d["t1"],
            d["fr2"],
            sigma_deg=cfg["sigma_deg"],
            n_folds=cfg["n_folds"],
            seed=seed,
            boot_lo=lo,
            boot_hi=hi,
            max_comparisons=cfg["max_comparisons"],
            sep_max_comparisons=cfg["sep_max_comparisons"],
        )
        rows.append(
            dict(
                subject=subject,
                session=session,
                unit_id=uid,
                brain_area=d["brain_area"],
                n_2obj=len(d["fr2"]),
                **r,
            )
        )
        draws.append(dr)
        uids.append(uid)
    df = pd.DataFrame(rows)
    if uids:
        np.savez_compressed(
            boot_path,
            unit_id=np.asarray(uids, int),
            boot_lo=lo,
            boot_hi=hi,
            **{
                k: np.stack([d[k] for d in draws]).astype(np.float32)
                for k in cl.BOOT_STATS
            },
        )
    if len(df):
        df["fit_tag"] = cfg["tag"]
        for k in (
            "sigma_deg",
            "n_folds",
            "n_boot",
            "max_comparisons",
            "sep_max_comparisons",
        ):
            df[f"param_{k}"] = cfg[k]
    sl._write_atomic(df, out_path)
    secs = time.time() - t_start
    if verbose:
        print(
            f"  {subject}/{session}: {len(df)} neurons ({secs:.0f} s)",
            flush=True,
        )
    return f"{subject}/{session}", len(df), secs


def recombine_boot(df, boot_dir):
    """Recompute every bootstrap p-value from the union of all replicate chunks on disk.

    A chunk file holds the per-replicate draws for one session over one replicate range. Because
    replicate `b` is seeded from `b` alone, the union of ranges [0, 500) and [500, 1000) IS the
    1000-replicate bootstrap — so the p-values are recomputed from the concatenation, never
    averaged across chunks. Chunks that overlap are de-duplicated by replicate index, so
    re-running a range is harmless.
    """
    boot_dir = Path(boot_dir)
    files = sorted(boot_dir.glob("*_boot_*.npz"))
    if not files:
        print(
            f"  (no bootstrap chunks in {boot_dir}; keeping this run's p-values)"
        )
        return df
    by_session = {}
    for f in files:
        stem = f.name[: -len(".npz")]
        head, lo, hi = stem.rsplit("_", 2)
        subject, session = head[: -len("_boot")].split("_", 1)
        by_session.setdefault((subject, session), []).append(
            (int(lo), int(hi), f)
        )

    updates, n_rep = {}, set()
    for (subject, session), chunks in by_session.items():
        seen, per_unit = set(), {}
        for lo, hi, f in sorted(chunks):
            if (lo, hi) in seen:
                continue
            seen.add((lo, hi))
            z = np.load(f)
            uids = z["unit_id"]
            for row, uid in enumerate(uids):
                d = per_unit.setdefault(
                    int(uid), {k: [] for k in cl.BOOT_STATS}
                )
                for k in cl.BOOT_STATS:
                    d[k].append(z[k][row])
        for uid, d in per_unit.items():
            joined = {k: np.concatenate(v) for k, v in d.items()}
            updates[(subject, session, uid)] = cl.p_from_draws(joined)
            n_rep.add(max(len(v) for v in joined.values()))

    keys = list(zip(df["subject"], df["session"], df["unit_id"]))
    cols = sorted({c for u in updates.values() for c in u})
    for c in cols:
        df[c] = [updates.get(k, {}).get(c, np.nan) for k in keys]
    df["beats_null_p"] = [
        df.at[i, f"{m}_beats_null_p"] if isinstance(m, str) else np.nan
        for i, m in enumerate(df["best_model"])
    ]
    print(
        f"  recombined {len(files)} bootstrap chunk file(s); "
        f"replicates per neuron: {sorted(n_rep)}"
    )
    return df


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--sessions",
        type=int,
        default=None,
        help="fit only the first N sessions",
    )
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--sigma-deg", type=float, default=pairs.SIGMA_DEG)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument(
        "--n-boot",
        type=int,
        default=cl.N_BOOT,
        help="trial resamples per neuron. Nothing is refitted inside one, so this "
        "is cheap; it sets the resolution of every p-value.",
    )
    ap.add_argument(
        "--boot-range",
        type=str,
        default=None,
        metavar="LO:HI",
        help="compute only replicates [LO, HI). Replicate b is seeded from b "
        "alone, so --boot-range 500:1000 after a 0:500 run yields exactly "
        "the draws a single 1000-replicate run would have produced; re-run "
        "model_comparison.ipynb afterwards and nothing else.",
    )
    # Uncapped is the only build. The cap existed because the graded comparison could not
    # afford every comparison of every neuron; the sign-only fit can, so there is no reason to
    # subsample and no second build to keep straight. 0 means "no cap"; the flags remain so a
    # capped run can still be done as a check, but nothing published uses one.
    ap.add_argument("--max-comparisons", type=int, default=0)
    ap.add_argument("--sep-max-comparisons", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--n-jobs",
        type=int,
        default=6,
        help="joblib workers, capped at 10 to leave the machine usable",
    )
    ap.add_argument("--combine-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if not 1 <= args.n_jobs <= 10:
        raise SystemExit(
            "--n-jobs must be between 1 and 10: leave the machine usable"
        )

    tag = (
        "uncapped"
        if not (args.max_comparisons or args.sep_max_comparisons)
        else "capped"
    )
    parts_dir = PARTS_DIR
    if args.boot_range:
        lo, hi = (int(x) for x in args.boot_range.split(":"))
    else:
        lo, hi = 0, args.n_boot
    cfg = dict(
        sigma_deg=args.sigma_deg,
        n_folds=args.n_folds,
        n_boot=args.n_boot,
        boot_lo=lo,
        boot_hi=hi,
        max_comparisons=args.max_comparisons,
        sep_max_comparisons=args.sep_max_comparisons,
        seed=args.seed,
        tag=tag,
        parts_dir=str(parts_dir),
        boot_dir=str(BOOT_DIR),
    )

    sessions = sl.fr_sessions()[:: args.stride]
    if args.sessions:
        sessions = sessions[: args.sessions]

    if not args.combine_only:
        print(
            f"{len(sessions)} sessions, {args.n_jobs} workers, [{tag}], "
            f"replicates [{lo}, {hi})",
            flush=True,
        )
        t0 = time.time()
        if args.n_jobs == 1 or len(sessions) == 1:
            done = [
                build_session(s, ses, cfg, overwrite=args.force)
                for s, ses in sessions
            ]
        else:
            from joblib import Parallel, delayed

            done = Parallel(n_jobs=min(args.n_jobs, len(sessions)))(
                delayed(build_session)(s, ses, cfg, args.force)
                for s, ses in sessions
            )
        n = sum(d[1] for d in done)
        print(
            f"\n{(time.time() - t0) / 60:.1f} min for {len(done)} sessions, {n} neurons",
            flush=True,
        )

    frames = [
        pd.read_csv(p) for p in sorted(Path(parts_dir).glob("*_fit.csv"))
    ]
    frames = [f for f in frames if len(f)]
    if not frames:
        raise SystemExit(f"no parts in {parts_dir}")
    df = pd.concat(frames, ignore_index=True)
    df = recombine_boot(df, Path(cfg["boot_dir"]))
    out = cl.UNITS_PATH
    sl._write_atomic(df, out)
    print(f"wrote {out}  ({len(df)} neurons)")
    for m in cl.MODELS:
        a = df[f"{m}_acc"]
        print(
            f"  {m:12s} held-out accuracy: median {a.median():.4f}  "
            f"> chance in {100 * (a > cl.CHANCE).mean():.0f}% of neurons"
        )
    print("  argmax:", df["best_model"].value_counts().to_dict())


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Build figure 3's model-comparison cache: `comparison_units.csv`.

    python run_comparison_cache.py                     # the full build
    python run_comparison_cache.py --sessions 2        # a quick look
    python run_comparison_cache.py --combine-only      # rebuild the CSV from existing parts

Each model is scored on the **orderings it predicts** and on nothing else -- see
`comparison_lib`'s module docstring for the estimator. Checkpointed per session, so an
interrupted run resumes.

**No comparison cap.** Every comparison of every neuron is used; the fit is O(N log N) in rows
and can afford it, and the precision of these statistics is set by the number of *trials*, not
of comparisons.

The per-replicate bootstrap draws are stored per session as `.npz` under `comparison_boot/`,
and every p-value in the combined table is computed from them.
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

#: The one configuration the comparison is built with. Written to every row as ``param_*``
#: columns; ``max_comparisons = sep_max_comparisons = 0`` means no comparison cap.
PARAMS = dict(
    sigma_deg=pairs.SIGMA_DEG,
    n_folds=5,
    max_comparisons=0,
    sep_max_comparisons=0,
)
SEED = 0
TAG = "uncapped"


def build_session(
    subject,
    session,
    parts_dir=PARTS_DIR,
    boot_dir=BOOT_DIR,
    n_boot=cl.N_BOOT,
    overwrite=False,
    verbose=True,
):
    """Fit every selected neuron of one session (:func:`stats_lib.selected_unit_ids`) and
    checkpoint it atomically.

    Two files per session: the summary CSV, and an `.npz` of the per-replicate bootstrap draws
    (replicates ``0..n_boot-1``), named ``{subject}_{session}_boot_00000_{n_boot:05d}.npz``.
    """
    parts = Path(parts_dir)
    parts.mkdir(parents=True, exist_ok=True)
    boot_dir = Path(boot_dir)
    boot_dir.mkdir(parents=True, exist_ok=True)
    boot_path = boot_dir / f"{subject}_{session}_boot_{0:05d}_{n_boot:05d}.npz"
    out_path = parts / f"{subject}_{session}_fit.csv"
    if out_path.exists() and not overwrite:
        return f"{subject}/{session}", len(pd.read_csv(out_path)), 0.0

    t_start = time.time()
    units = sl.session_unit_trials(subject, session)
    selected = sl.selected_unit_ids(subject, session)
    rows, draws, uids = [], [], []
    for uid, d in sorted(units.items()):
        if uid not in selected or len(d["fr2"]) < sl.MIN_2OBJ_FIT:
            continue
        r, dr = cl.fit_neuron_sign(
            d["t0"],
            d["t1"],
            d["fr2"],
            sigma_deg=PARAMS["sigma_deg"],
            n_folds=PARAMS["n_folds"],
            seed=SEED + uid,
            n_boot=n_boot,
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
    # One draws file per session: drop any left by an earlier run with another replicate count.
    for old in boot_dir.glob(f"{subject}_{session}_boot_*.npz"):
        if old != boot_path:
            old.unlink()
    if uids:
        np.savez_compressed(
            boot_path,
            unit_id=np.asarray(uids, int),
            boot_lo=0,
            boot_hi=n_boot,
            **{
                k: np.stack([d[k] for d in draws]).astype(np.float32)
                for k in cl.BOOT_STATS
            },
        )
    if len(df):
        df["fit_tag"] = TAG
        params = dict(PARAMS, n_boot=n_boot)
        for k in (
            "sigma_deg",
            "n_folds",
            "n_boot",
            "max_comparisons",
            "sep_max_comparisons",
        ):
            df[f"param_{k}"] = params[k]
    sl._write_atomic(df, out_path)
    secs = time.time() - t_start
    if verbose:
        print(
            f"  {subject}/{session}: {len(df)} neurons ({secs:.0f} s)",
            flush=True,
        )
    return f"{subject}/{session}", len(df), secs


def recombine_boot(df, boot_dir):
    """Recompute every bootstrap p-value from the per-session draws files on disk.

    Each session has one file, ``{subject}_{session}_boot_{lo}_{hi}.npz``, holding the
    per-replicate draws of its neurons. A neuron with no draws on disk gets NaN p-values.
    """
    boot_dir = Path(boot_dir)
    files = sorted(boot_dir.glob("*_boot_*.npz"))
    if not files:
        print(
            f"  (no bootstrap draws in {boot_dir}; keeping this run's p-values)"
        )
        return df
    updates, n_rep, seen = {}, set(), set()
    for f in files:
        head = f.name[: -len(".npz")].rsplit("_", 2)[0]
        subject, session = head[: -len("_boot")].split("_", 1)
        if (subject, session) in seen:
            raise ValueError(
                f"{boot_dir} holds more than one draws file for {subject}/{session}; "
                f"delete the stale one or rebuild that session with --force."
            )
        seen.add((subject, session))
        z = np.load(f)
        for row, uid in enumerate(z["unit_id"]):
            joined = {k: z[k][row] for k in cl.BOOT_STATS}
            updates[(subject, session, int(uid))] = cl.p_from_draws(joined)
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
        f"  recombined {len(files)} bootstrap draws file(s); "
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
    ap.add_argument(
        "--n-boot",
        type=int,
        default=cl.N_BOOT,
        help="trial resamples per neuron. Nothing is refitted inside one, so this "
        "is cheap; it sets the resolution of every p-value.",
    )
    ap.add_argument(
        "--n-jobs",
        type=int,
        default=6,
        help="joblib workers",
    )
    ap.add_argument("--combine-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.n_jobs < 1:
        raise SystemExit("--n-jobs must be at least 1")

    sessions = sl.fr_sessions()
    if args.sessions:
        sessions = sessions[: args.sessions]

    if not args.combine_only:
        print(
            f"{len(sessions)} sessions, {args.n_jobs} workers, [{TAG}], "
            f"{args.n_boot} replicates",
            flush=True,
        )
        t0 = time.time()
        kw = dict(n_boot=args.n_boot, overwrite=args.force)
        if args.n_jobs == 1 or len(sessions) == 1:
            done = [build_session(s, ses, **kw) for s, ses in sessions]
        else:
            from joblib import Parallel, delayed

            done = Parallel(n_jobs=min(args.n_jobs, len(sessions)))(
                delayed(build_session)(s, ses, **kw) for s, ses in sessions
            )
        n = sum(d[1] for d in done)
        print(
            f"\n{(time.time() - t0) / 60:.1f} min for {len(done)} sessions, {n} neurons",
            flush=True,
        )

    frames = [pd.read_csv(p) for p in sorted(PARTS_DIR.glob("*_fit.csv"))]
    frames = [f for f in frames if len(f)]
    if not frames:
        raise SystemExit(f"no parts in {PARTS_DIR}")
    df = pd.concat(frames, ignore_index=True)
    df = recombine_boot(df, BOOT_DIR)
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

"""Build `cache/figures/shared_inputs/modulation.csv`, the neuron-selection statistic.

    python run_modulation_cache.py                # all sessions, 6 workers, resumable
    python run_modulation_cache.py --n-jobs 4
    python run_modulation_cache.py --force        # ignore the parts already on disk

One row per neuron with at least `config.MIN_2OBJ_TRIALS` 2-object trials; the statistic is
defined in `modulation_lib.py` and `figures/neuron_selection.py`, which reads this table and
decides which neurons enter every population figure. The parameters used are recorded as
`param_*` columns. Takes ~10 min on 6 workers.

Each session is checkpointed to `shared_inputs/modulation_parts/`, so an interrupted run
resumes from the sessions already finished. Builds the shared firing rates first if they
are missing.
"""

import argparse
import os
import shutil
import time
from pathlib import Path

# Cap the BLAS threads BEFORE numpy is imported, in this process and in every worker it
# spawns — otherwise macOS Accelerate takes every core inside each of `--n-jobs` workers.
for _v in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import modulation_lib as ml  # noqa: E402
import pandas as pd  # noqa: E402
import stats_lib as sl  # noqa: E402

PARTS_DIR = sl.MODULATION_PATH.parent / "modulation_parts"
METRIC_COLS = [
    "n_2obj",
    "mean_count",
    "dev",
    "cv_r2",
    "const_cv_r2",
    "cw_cv_r2",
    "ioa_cv_r2",
    "null_mean",
    "null_std",
    "z",
    "p",
]


def session_rows(subject, session, p):
    """One session's neurons."""
    units = sl.session_unit_trials(subject, session)
    rows = []
    for uid, d in sorted(units.items()):
        # Only the 2-object floor: the statistic uses 2-object trials alone.
        if len(d["fr2"]) < p["min_2obj_trials"]:
            continue
        m = ml.neuron_modulation(
            d["t0"], d["t1"], d["fr2"], p, seed=p["seed"] + uid
        )
        m.update(
            subject=subject,
            session=session,
            unit_id=uid,
            brain_area=d["brain_area"],
        )
        rows.append(m)
    return pd.DataFrame(rows)


def build_session(subject, session, p, parts_dir):
    """Compute one session and write its checkpoint atomically. Returns (name, n, seconds)."""
    out = parts_dir / f"{subject}_{session}.csv"
    if out.exists():
        return f"{subject}/{session}", len(pd.read_csv(out)), 0.0
    t0 = time.time()
    df = session_rows(subject, session, p)
    tmp = out.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, out)  # a part file exists only once it is complete
    return f"{subject}/{session}", len(df), time.time() - t0


def combine(p, parts_dir, out_path):
    """Concatenate the checkpoints, add the `param_*` columns, write the table."""
    parts = sorted(parts_dir.glob("*.csv"))
    df = pd.concat([pd.read_csv(f) for f in parts], ignore_index=True)
    for k, v in p.items():
        df[f"param_{k}"] = v
    df = df.reindex(
        columns=sl._ID_COLS + METRIC_COLS + [f"param_{k}" for k in p]
    )
    df = df.sort_values(sl._ID_COLS[:3]).reset_index(drop=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  wrote {len(df)} neurons to {out_path}")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--n-jobs", type=int, default=6, help="parallel workers (default 6)"
    )
    ap.add_argument(
        "--sessions", type=int, default=None, help="first N only (smoke test)"
    )
    ap.add_argument(
        "--force", action="store_true", help="discard existing checkpoints"
    )
    args = ap.parse_args()

    p = ml.mod_params()
    sessions = sl.fr_sessions()
    if args.sessions:
        sessions = sessions[: args.sessions]

    if args.force and PARTS_DIR.exists():
        shutil.rmtree(PARTS_DIR)
    PARTS_DIR.mkdir(parents=True, exist_ok=True)

    done = {f.stem for f in PARTS_DIR.glob("*.csv")}
    todo = [s for s in sessions if f"{s[0]}_{s[1]}" not in done]
    print(
        f"modulation cache: {p}\n{args.n_jobs} workers, {len(todo)} of {len(sessions)} "
        f"sessions to compute",
        flush=True,
    )

    t0 = time.time()
    if todo:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=min(args.n_jobs, len(todo)), verbose=0)(
            delayed(build_session)(subj, sess, p, PARTS_DIR)
            for subj, sess in todo
        )
        for name, n, secs in results:
            print(f"  {name}: {n} neurons ({secs:.0f} s)", flush=True)
    print(f"{(time.time() - t0) / 60:.1f} min of computation", flush=True)

    if args.sessions:
        print("--sessions given: not combining a partial table")
        return
    print("combining:")
    df = combine(p, PARTS_DIR, sl.MODULATION_PATH)
    floor = 1.0 / (1 + p["n_shuffles"])
    print(
        f"\nsmallest attainable p = {floor:.4f}; "
        f'{int((df["p"] <= 0.05).sum())} of {len(df)} neurons at raw p <= 0.05'
    )


if __name__ == "__main__":
    main()

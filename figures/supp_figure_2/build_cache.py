"""Build every cache the supp_figure_2 notebooks read: figures 3 and 4's fits, once per phase.

    python build_cache.py                              # everything, both phases, 8 workers
    python build_cache.py --phases delay               # one phase
    python build_cache.py --stage fits                 # one stage
    python build_cache.py --sessions 2 --stride 12     # smoke test: two sessions, both monkeys

Three stages, run in this order, each **resumable** — stop it at any point and re-run the same
command, and it skips what is already on disk:

  comparison  figure 3's sign-only model comparison  (`run_comparison_cache.build_session`)
              checkpointed per session; 1000 bootstrap replicates
  fits        figure 4's model fits                  (`fit_lib.fit_unit`)
              checkpointed **per neuron**
              — a large session is most of an hour of fitting, too much to lose
  recovery    figure 4's KS synthetic distributions  (`ks_lib._unit_job`)
              5 replicates, checkpointed per neuron

Every stage fits only the neurons passing the shared criterion — the 1339 the figures draw —
as the main caches do.

Everything lands under ``cache/figures/supp_figure_2/{phase}/``; nothing in the main
figures' caches is read as a fit or written. The firing rates are the main figures' own — both
phase columns are already in them.

**Why this script parallelises instead of the libraries.** The phase is a set of module
globals (see `common.use_phase`), and a joblib worker imports the libraries afresh with the
default 2-s window. Every library call that would fan out is therefore made with one job, from
inside a worker that has called `use_phase` first. Each stage checks that its workers actually
saw the phase before trusting what they return.

``--n-jobs`` defaults to 8; `figures/build_caches.py` passes its own (12).
"""

import argparse
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

_HERE = str(Path(__file__).resolve().parent)
sys.path.insert(0, _HERE)
import common  # noqa: E402  (sets sys.path for everything below)

# `common` puts figure_4/ ahead of this directory, and figure_4 has a `common.py` of its own.
# The parent has already imported the right one, but a joblib worker imports afresh from the
# parent's sys.path, so this directory must come first or the workers get figure 4's.
sys.path.remove(_HERE)
sys.path.insert(0, _HERE)

import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

N_REPS, SIM_SEED = 5, 0  # must match figure_ks_stats.ipynb


def _write_atomic(obj, path):
    """Pickle ``obj`` to ``path`` via a temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pd.to_pickle(obj, tmp)
    os.replace(tmp, path)


def _phase_state():
    """What the libraries in *this* process are set to — returned by every worker so the
    parent can check the phase reached it."""
    import data_lib as dl
    import stats_lib as sl

    return (tuple(sl.FR_COLUMNS), tuple(dl.FR_COLUMNS), float(dl.DURATION_S))


def _expect(phase, state, where):
    col = common.PHASE_COLUMN[phase]
    want = ((col,), (col,), common.PHASE_DURATION_S)
    if tuple(state) != want:
        raise RuntimeError(
            f"{where}: a worker ran with {state}, expected {want} for "
            f"{phase!r}. The phase did not reach it; nothing it returned is "
            f"usable."
        )


def _run(jobs, n_jobs):
    if not jobs:
        return []
    if n_jobs == 1:
        return [f(*a, **k) for f, a, k in jobs]
    return Parallel(n_jobs=n_jobs, verbose=5)(jobs)


# ── Figure 3: the sign-only model comparison ──────────────────────────────────
def _comparison_cfg(phase):
    """`run_comparison_cache.build_session`'s keyword arguments for this phase."""
    p = common.phase_paths(phase)
    return dict(
        parts_dir=str(p["comparison_parts"]),
        boot_dir=str(p["comparison_boot"]),
        n_boot=common.COMPARISON_N_BOOT,
    )


def _comparison_session(phase, subject, session, cfg):
    common.use_phase(phase)
    import run_comparison_cache as rcc

    return (
        rcc.build_session(subject, session, **cfg, verbose=False),
        _phase_state(),
    )


def stage_comparison(phase, sessions, n_jobs):
    import run_comparison_cache as rcc
    import stats_lib as sl

    p = common.use_phase(phase)
    cfg = _comparison_cfg(phase)
    t0 = time.time()
    out = _run(
        [
            delayed(_comparison_session)(phase, s, ss, cfg)
            for s, ss in sessions
        ],
        n_jobs,
    )
    for (name, n, secs), state in out:
        _expect(phase, state, "comparison")
        if secs:
            print(f"  {name}: {n} neurons ({secs:.0f} s)", flush=True)
    print(f"  {(time.time() - t0) / 60:.1f} min", flush=True)

    frames = [
        pd.read_csv(f) for f in sorted(p["comparison_parts"].glob("*_fit.csv"))
    ]
    frames = [f for f in frames if len(f)]
    if len(frames) < len(sl.fr_sessions()):
        print(
            f"  {len(frames)} of {len(sl.fr_sessions())} sessions done — not combining yet"
        )
        return
    df = rcc.recombine_boot(
        pd.concat(frames, ignore_index=True), p["comparison_boot"]
    )
    sl._write_atomic(df, p["comparison_units"])
    print(
        f'  wrote {p["comparison_units"]} ({len(df)} neurons); argmax '
        f'{df["best_model"].value_counts().to_dict()}'
    )


# ── Figure 4: the model fits ──────────────────────────────────────────────────
def _passing():
    import neuron_selection

    keep = neuron_selection.passing_units(verbose=False)
    return {
        (s, ss, int(u))
        for s, ss, u in keep[["subject", "session", "unit_id"]].to_numpy()
    }


def _fit_neuron(phase, unit, cfg, out_path):
    common.use_phase(phase)
    import fit_lib as fl

    boot, rows = fl.fit_unit(unit, cfg)
    _write_atomic((boot, rows, _phase_state()), out_path)
    return _phase_state()


def stage_fits(phase, sessions, n_jobs):
    import data_lib as dl
    import fit_lib as fl

    p = common.use_phase(phase)
    cfg = fl.FitConfig(models=common.FIT_MODELS)
    keep = _passing()
    parts = p["fit_parts"] / cfg.tag

    jobs, todo = [], {}
    for subject, session in sessions:
        units = dl.session_unit_trials(subject, session, verbose=False)
        uids = sorted(u for u in units if (subject, session, u) in keep)
        todo[(subject, session)] = uids
        for u in uids:
            path = parts / f"{subject}_{session}_{u}.pkl"
            if not path.exists():
                jobs.append(delayed(_fit_neuron)(phase, units[u], cfg, path))
    n_all = sum(len(v) for v in todo.values())
    print(
        f"  {len(jobs)} of {n_all} neurons to fit x {len(cfg.models)} models "
        f"[{cfg.tag}]",
        flush=True,
    )
    t0 = time.time()
    for state in _run(jobs, n_jobs):
        _expect(phase, state, "fits")
    if jobs:
        print(
            f"  {(time.time() - t0) / 60:.1f} min, "
            f"{(time.time() - t0) * min(n_jobs, len(jobs)) / len(jobs):.0f} s/neuron",
            flush=True,
        )

    # Assemble the per-neuron parts into the per-session files `fit_lib.load_summary` reads.
    out_dir = fl.cache_dir(cfg)
    for (subject, session), uids in todo.items():
        files = [parts / f"{subject}_{session}_{u}.pkl" for u in uids]
        if not uids or not all(f.exists() for f in files):
            continue
        boot, rows = [], []
        for f in files:
            b, r, state = pd.read_pickle(f)
            _expect(phase, state, f"fits {f.name}")
            boot += b
            rows += r
        fl._write_atomic(
            pd.DataFrame(rows), out_dir / f"{subject}_{session}_units.csv.gz"
        )
        fl._write_atomic(
            pd.DataFrame(boot), out_dir / f"{subject}_{session}_boot.csv.gz"
        )
    n_sessions = len(list(out_dir.glob("*_units.csv.gz")))
    print(f"  {n_sessions} session files in {out_dir}")
    if n_sessions:
        fl.load_summary(fl.FitConfig(), rebuild="always", verbose=True)


# ── Figure 4: the KS test's synthetic distributions ───────────────────────────
def _recover_neuron(phase, rows, trials, models, cfg, reps, out_path):
    common.use_phase(phase)
    import ks_lib as ks
    import models_lib as ml
    import relationality_lib as rl

    objs = ml.build_models(models)
    bias_model = ml.build_models((rl.MODEL,))[rl.MODEL]
    out = ks._unit_job(rows, trials, objs, cfg, bias_model, reps, SIM_SEED)
    _write_atomic((out, _phase_state()), out_path)
    return _phase_state()


def stage_recovery(phase, sessions, n_jobs):
    import data_lib as dl
    import fit_lib as fl
    import ks_lib as ks

    p = common.use_phase(phase)
    # What figure_ks_stats.ipynb asks for: the compared models plus the metric's own, read
    # through the same selection, under the default FitConfig whose tag keys the cache.
    models = tuple(m for m in common.FIT_MODELS if m != "constant")
    summary = common.selected_fits(phase, models, verbose=False)
    cfg = fl.FitConfig()
    parts = p["bias_parts"] / f"seed{SIM_SEED}"
    reps = list(range(N_REPS))
    wanted = {(s, ss) for s, ss in sessions}

    jobs, by_session = [], {}
    for (subject, session), g in summary.groupby(["subject", "session"]):
        if (subject, session) not in wanted:
            continue
        trials_by_unit = dl.session_unit_trials(
            subject, session, verbose=False
        )
        for uid, rows_df in g.groupby("unit"):
            uid = int(uid)
            path = parts / f"{subject}_{session}_{uid}.pkl"
            by_session.setdefault((subject, session), []).append(path)
            if path.exists():
                continue
            rows = {m: r for m, r in rows_df.set_index("model").iterrows()}
            jobs.append(
                delayed(_recover_neuron)(
                    phase, rows, trials_by_unit[uid], models, cfg, reps, path
                )
            )
    n_all = sum(len(v) for v in by_session.values())
    print(
        f"  {len(jobs)} of {n_all} neurons to simulate x {len(models)} models x "
        f"{N_REPS} replicates",
        flush=True,
    )
    t0 = time.time()
    for state in _run(jobs, n_jobs):
        _expect(phase, state, "recovery")
    if jobs:
        print(f"  {(time.time() - t0) / 60:.1f} min", flush=True)

    # Write the per-session files `ks_lib.synthetic_biases` reads, in its own format.
    run_dir = ks._run_dir(SIM_SEED)
    for (subject, session), files in by_session.items():
        if not all(f.exists() for f in files):
            continue
        rows = []
        for f in files:
            out, state = pd.read_pickle(f)
            _expect(phase, state, f"recovery {f.name}")
            rows += out
        fl._write_atomic(
            pd.DataFrame(rows), run_dir / f"{subject}_{session}.csv.gz"
        )
    # The check that the notebook will find everything: one in-process call, which must
    # reuse every row and compute none.
    if len(sessions) == len(dl.fr_sessions()):
        synth = ks.synthetic_biases(
            summary,
            models,
            cfg,
            n_reps=N_REPS,
            seed=SIM_SEED,
            n_jobs=1,
            verbose=True,
        )
        print(f"  {len(synth)} synthetic rows in {run_dir}")


STAGES = dict(
    comparison=stage_comparison,
    fits=stage_fits,
    recovery=stage_recovery,
)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--phases",
        nargs="+",
        choices=common.PHASES,
        default=list(common.PHASES),
    )
    ap.add_argument(
        "--stage", nargs="+", choices=list(STAGES), default=list(STAGES)
    )
    ap.add_argument(
        "--sessions", type=int, default=None, help="first N sessions only"
    )
    ap.add_argument(
        "--stride",
        type=int,
        default=1,
        help="every Nth session. The first 12 are one monkey, so a --sessions "
        "limit alone samples one animal; a stride covers both.",
    )
    ap.add_argument("--n-jobs", type=int, default=8)
    args = ap.parse_args()
    if args.n_jobs < 1:
        raise SystemExit("--n-jobs must be at least 1")

    import stats_lib as sl

    sessions = sl.fr_sessions()[:: args.stride]
    if args.sessions:
        sessions = sessions[: args.sessions]

    t0 = time.time()
    for stage in [s for s in STAGES if s in args.stage]:
        for phase in args.phases:
            print(
                f"\n=== {stage} / {phase} — {len(sessions)} sessions, {args.n_jobs} workers "
                f"===",
                flush=True,
            )
            STAGES[stage](phase, sessions, args.n_jobs)
    print(f"\ntotal {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()

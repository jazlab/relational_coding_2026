"""Build every cache the supp_figure_2 notebooks read: figures 3 and 4's fits, once per phase.

    python build_cache.py                              # everything, both phases, 8 workers
    python build_cache.py --phases delay               # one phase
    python build_cache.py --stage fits                 # one stage
    python build_cache.py --sessions 2 --stride 11     # smoke test: two sessions, both monkeys

Four stages, run in this order, each **resumable** — stop it at any point and re-run the same
command, and it skips what is already on disk:

  comparison  figure 3's sign-only model comparison  (`run_comparison_cache.build_session`)
              checkpointed per session; 1000 bootstrap replicates
  phase       figure 3's 2-object phase fit          (`stats_lib.build_session`)
              checkpointed per session; 50 bootstraps
  fits        figure 4's model fits                  (`fit_lib.fit_unit`)
              checkpointed **per neuron**
              — a large session is most of an hour of fitting, too much to lose
  recovery    figure 4's KS synthetic distributions  (`ks_lib._unit_job`)
              5 replicates, checkpointed per neuron

Every stage fits only the neurons passing the shared criterion — the 1339 the figures draw,
not the 3186 the main caches hold.

Everything lands under ``cache/figures/supp_figure_2/{phase}/``; nothing in the main
figures' caches is read as a fit or written. The firing rates are the main figures' own — both
phase columns are already in them.

**Why this script parallelises instead of the libraries.** The phase is a set of module
globals (see `common.use_phase`), and a joblib worker imports the libraries afresh with the
default 2-s window. Every library call that would fan out is therefore made with one job, from
inside a worker that has called `use_phase` first. Each stage checks that its workers actually
saw the phase before trusting what they return.

Never every core: ``--n-jobs`` defaults to 8 of this machine's 14 and refuses more than 10.
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

N_REPS, SIM_SEED = 5, 0  # must match figure_ks_stats.ipynb


def _write_atomic(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if path.suffix == ".pkl":
        pd.to_pickle(df, tmp)
    else:
        df.to_csv(
            tmp,
            index=False,
            compression="gzip" if path.name.endswith(".gz") else None,
        )
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


def _only_passing(keep):
    """Restrict figure 3's per-session loader, in this process, to ``keep``.

    `run_comparison_cache.build_session` and `stats_lib.build_session` fit every unit the
    loader returns, which is all 3186; only the 1339 passing the shared criterion are ever
    drawn. Both look the loader up as `stats_lib.session_unit_trials` at call time, so
    wrapping it there is enough. Each unit's fit is seeded from its own id, so dropping the
    others changes nothing about the units that are kept.
    """
    import stats_lib as sl

    load = getattr(
        sl.session_unit_trials, "__wrapped__", sl.session_unit_trials
    )

    def session_unit_trials(subject, session, *a, **k):
        units = load(subject, session, *a, **k)
        return {
            u: d for u, d in units.items() if (subject, session, u) in keep
        }

    session_unit_trials.__wrapped__ = load
    sl.session_unit_trials = session_unit_trials


# ── Figure 3: the sign-only model comparison ──────────────────────────────────
def _comparison_cfg(phase):
    import pairs_lib as pairs

    p = common.paths(phase)
    n = common.COMPARISON_N_BOOT
    return dict(
        sigma_deg=pairs.SIGMA_DEG,
        n_folds=5,
        n_boot=n,
        boot_lo=0,
        boot_hi=n,
        max_comparisons=0,
        sep_max_comparisons=0,
        seed=0,
        tag="uncapped",
        parts_dir=str(p["comparison_parts"]),
        boot_dir=str(p["comparison_boot"]),
    )


def _comparison_session(phase, subject, session, cfg, keep):
    common.use_phase(phase)
    _only_passing(keep)
    import run_comparison_cache as rcc

    return (
        rcc.build_session(subject, session, cfg, verbose=False),
        _phase_state(),
    )


def stage_comparison(phase, sessions, n_jobs):
    import run_comparison_cache as rcc
    import stats_lib as sl

    p = common.use_phase(phase)
    cfg = _comparison_cfg(phase)
    keep = _passing()
    t0 = time.time()
    out = _run(
        [
            delayed(_comparison_session)(phase, s, ss, cfg, keep)
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


# ── Figure 3: the 2-object phase fit ──────────────────────────────────────────
def _phase_cfg():
    import pairs_lib as pairs

    cfg = dict(
        sigma_deg=pairs.SIGMA_DEG,
        n_folds=5,
        n_boot=pairs.N_BOOT,
        max_comparisons=pairs.MAX_COMPARISONS,
        seed=0,
        with_proximity=True,
        sep_max_comparisons=pairs.SEP_MAX_COMPARISONS,
    )
    # The tag `run_phase_cache.py` writes, so the checkpoints say what built them.
    cfg["tag"] = (
        f"v1_ordinal_s{cfg['sigma_deg']:g}_k{cfg['n_folds']}_b{cfg['n_boot']}"
        f"_m{cfg['max_comparisons']}_seed{cfg['seed']}"
        f"_prox{cfg['sep_max_comparisons']}"
    )
    return cfg


def _phase_session(phase, subject, session, cfg, parts_dir, keep):
    common.use_phase(phase)
    _only_passing(keep)
    import stats_lib as sl

    return (
        sl.build_session(
            subject, session, cfg, parts_dir=parts_dir, verbose=False
        ),
        _phase_state(),
    )


def stage_phase(phase, sessions, n_jobs):
    import stats_lib as sl

    p = common.use_phase(phase)
    cfg = _phase_cfg()
    parts = p["phase_parts"] / cfg["tag"]
    keep = _passing()
    t0 = time.time()
    out = _run(
        [
            delayed(_phase_session)(phase, s, ss, cfg, parts, keep)
            for s, ss in sessions
        ],
        n_jobs,
    )
    for (name, n, secs), state in out:
        _expect(phase, state, "phase")
        if secs:
            print(f"  {name}: {n} neurons ({secs:.0f} s)", flush=True)
    print(f"  {(time.time() - t0) / 60:.1f} min", flush=True)

    have = {f.name[: -len("_units.csv")] for f in parts.glob("*_units.csv")}
    missing = [s for s in sl.fr_sessions() if f"{s[0]}_{s[1]}" not in have]
    if missing:
        print(f"  {len(missing)} sessions still to fit — not combining yet")
        return
    sl.combine_parts(
        cfg, parts_dir=parts
    )  # to sl.UNITS_PATH, set by use_phase


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


def stage_recovery(phase, sessions, n_jobs, n_reps=N_REPS):
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
    reps = list(range(n_reps))
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
        f"{n_reps} replicates",
        flush=True,
    )
    t0 = time.time()
    for state in _run(jobs, n_jobs):
        _expect(phase, state, "recovery")
    if jobs:
        print(f"  {(time.time() - t0) / 60:.1f} min", flush=True)

    # Write the per-session files `ks_lib.synthetic_biases` reads, in its own format.
    run_dir = ks._run_dir(cfg, SIM_SEED, tag="bias")
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
            n_reps=n_reps,
            seed=SIM_SEED,
            n_jobs=1,
            cache=True,
            resume=True,
            min_dll=None,
            verbose=True,
        )
        print(f"  {len(synth)} synthetic rows in {run_dir}")


STAGES = dict(
    comparison=stage_comparison,
    phase=stage_phase,
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
    if not 1 <= args.n_jobs <= 10:
        raise SystemExit(
            "--n-jobs must be between 1 and 10: leave the machine usable"
        )

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

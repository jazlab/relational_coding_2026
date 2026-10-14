"""Build every cache the figures read, from the DANDI data, in dependency order.

    python build_caches.py                    # everything (several hours)
    python build_caches.py --n-jobs 4         # fewer parallel workers
    python build_caches.py --from modulation  # resume from a step
    python build_caches.py --only figure_4    # one step
    python build_caches.py --dry-run          # print the commands

Needs the DANDI behavior and spike-sorting files in `cache/dandi_data/` (see the top-level
README). Each step is a script you can also run on its own from the directory shown by
`--dry-run`; every one of them checkpoints its work, so re-running after an interruption
picks up where it stopped.

Steps:
    ring          behavior_processing/run_cache_data.py           seconds
    firing_rates  figures/firing_rates.py                         ~2 min
    modulation    figure_3/model_comparison/run_modulation_cache  ~10 min
    comparison    figure_3/model_comparison/run_comparison_cache  ~15 min
    phase         figure_3/model_comparison/run_phase_cache       ~15 min
    figure_4      figure_4/run_cache                              several hours
    supp_figure_2 supp_figure_2/build_cache                       ~4 hours

Times are for 6-8 workers. Every other cache (decoders, response profiles, pool tables, the
single-unit table, ...) is built automatically the first time a notebook needs it.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

_FIGURES = Path(__file__).resolve().parent
_ROOT = _FIGURES.parent

# (name, working directory, script and fixed arguments, whether it takes --n-jobs)
STEPS = [
    ("ring", _ROOT / "behavior_processing", ["run_cache_data.py"], False),
    ("firing_rates", _FIGURES, ["firing_rates.py"], False),
    (
        "modulation",
        _FIGURES / "figure_3/model_comparison",
        ["run_modulation_cache.py"],
        True,
    ),
    (
        "comparison",
        _FIGURES / "figure_3/model_comparison",
        ["run_comparison_cache.py"],
        True,
    ),
    (
        "phase",
        _FIGURES / "figure_3/model_comparison",
        ["run_phase_cache.py"],
        True,
    ),
    ("figure_4", _FIGURES / "figure_4", ["run_cache.py"], True),
    ("supp_figure_2", _FIGURES / "supp_figure_2", ["build_cache.py"], True),
]
NAMES = [s[0] for s in STEPS]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--n-jobs", type=int, default=6, help="parallel workers (default 6)"
    )
    ap.add_argument(
        "--from", dest="start", choices=NAMES, help="start at this step"
    )
    ap.add_argument("--only", choices=NAMES, help="run just this step")
    ap.add_argument(
        "--dry-run", action="store_true", help="print the commands only"
    )
    args = ap.parse_args()

    steps = STEPS
    if args.only:
        steps = [s for s in STEPS if s[0] == args.only]
    elif args.start:
        steps = STEPS[NAMES.index(args.start) :]

    for name, cwd, cmd, takes_jobs in steps:
        cmd = (
            [sys.executable]
            + cmd
            + (["--n-jobs", str(args.n_jobs)] if takes_jobs else [])
        )
        print(
            f'\n=== {name}: (cd {cwd.relative_to(_ROOT)} && {" ".join(cmd[1:])})',
            flush=True,
        )
        if args.dry_run:
            continue
        t0 = time.time()
        result = subprocess.run(cmd, cwd=cwd)
        if result.returncode != 0:
            raise SystemExit(
                f"step {name!r} failed (exit {result.returncode}); fix it and "
                f"resume with `python build_caches.py --from {name}`"
            )
        print(
            f"=== {name}: done in {(time.time() - t0) / 60:.1f} min",
            flush=True,
        )


if __name__ == "__main__":
    main()

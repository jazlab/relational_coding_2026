#!/usr/bin/env python
"""Regenerate `dataframes/results_k20_b100.csv`, the fit `titration.ipynb` reads.

    python regenerate.py                  # ~25 min on 10 cores
    python regenerate.py --jobs 8         # fewer workers
    python regenerate.py --dry-run        # print the commands and exit

The notebook never calls this — it reads the CSVs this writes. Nothing here needs re-running
unless the fits themselves change.

**What is being fit.** Sub-populations are drawn from the whole pool by an exponential tilt on
each neuron's within-animal rank of rho: ``w ~ exp(lambda * rank)``, K distinct units without
replacement. The sweep variable is the *target weighted mean rank*, and ``run_titration.py``
solves lambda for each sweep point by inverting ``m(lambda) = 1/(1 - e^-lambda) - 1/lambda``.
Ranks are uniform on [0, 1] by construction, so that inverse uses no data: one lambda grid serves
both animals and ``lambda(0.5) = 0`` exactly, which is what makes 0.5 the unbiased population.

Runtime is ~3.3 s per training: 9 points x 2 animals x 100 repeats = 1800 trainings. Response
tables are cached under `cache/figures/figure_3/pool_tables/`, so only the first run pays the
127 ms/unit to build them.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DF = HERE / "dataframes"
PY = sys.executable
SWEEP = HERE / "run_titration.py"

#: The animals. K -- distinct units per sweep point -- is not set here and is not a knob: it is
#: the smallest wedge over the whole sweep, which `run_titration.matched_k` derives as 160
#: (Monkey P) / 51 (Monkey E). Every point on the curve is drawn at that size or skipped, so
#: the curve cannot be confounded with neuron count.
SUBJECTS = ("Perle", "Elgar")
REPEATS = 100  # trainings per sweep point
SWEEP_NAME = "results_k20_b100"
#: Sweep points per worker. Six groups x two animals = 12 workers, which is the intended shape.
GROUPS = ["0.1,0.2", "0.3,0.4", "0.5,0.6", "0.7", "0.8", "0.9"]

BASE = ["--out-dir", str(DF)]


def cmd(subject, points, name, repeats):
    return [
        PY,
        str(SWEEP),
        "--threads",
        "1",
        "--subject",
        subject,
        "--points",
        points,
        "--repeats",
        str(repeats),
        "--name",
        name,
    ] + BASE


def run_pool(jobs, cmds, label, dry):
    """Run `cmds` with at most `jobs` at a time, refilling as each finishes.

    A refilling pool rather than fixed batches with a barrier: the animals' jobs differ in cost
    (Monkey P's pool is three times Monkey E's), so a barrier every `jobs` commands would idle
    most of the workers waiting for the slowest.
    """
    if dry:
        for c in cmds:
            print("   " + " ".join(c[1:]))
        return
    t0 = time.time()
    running, queue, done = [], list(cmds), 0
    log = open(f"/tmp/regen_{label}.log", "w")
    while queue or running:
        while queue and len(running) < jobs:
            running.append(
                subprocess.Popen(
                    queue.pop(0), stdout=log, stderr=subprocess.STDOUT
                )
            )
        time.sleep(1.0)
        for p in running[:]:
            if p.poll() is not None:
                running.remove(p)
                done += 1
                print(
                    f"\r   {label}: {done}/{len(cmds)} workers done "
                    f"({time.time() - t0:.0f} s)",
                    end="",
                    flush=True,
                )
    log.close()
    print(
        f"\r   {label}: {done}/{len(cmds)} workers done in {time.time() - t0:.0f} s"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--jobs", type=int, default=10, help="parallel workers (default 10)"
    )
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    DF.mkdir(exist_ok=True)

    print(f"sweep: {SWEEP_NAME}  ({REPEATS} repeats)")
    run_pool(
        a.jobs,
        [cmd(s, g, SWEEP_NAME, REPEATS) for g in GROUPS for s in SUBJECTS],
        "sweep",
        a.dry_run,
    )
    if not a.dry_run:
        subprocess.run(
            [
                PY,
                str(SWEEP),
                "--combine-only",
                "--name",
                SWEEP_NAME,
                "--repeats",
                str(REPEATS),
            ]
            + BASE,
            check=True,
            stdout=subprocess.DEVNULL,
        )

    print("done ->", DF)


if __name__ == "__main__":
    main()

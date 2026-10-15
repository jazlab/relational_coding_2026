"""The titration's candidate pool and its response tables.

The model-free machinery lives beside this file, in `decoder_lib`: the kernel-regression
response tables, the bootstrap draw, the MLP and its training loop. Those do not know a model
exists — `precompute_unit_tables` is a swap-invariant circular-Gaussian regression of real
firing rates onto real (theta0, theta1) pairs. Which neurons enter the pool, and where each
sits on the relationality axis, is `../selection_lib.py`'s job: it applies the shared
criterion of `figures/config.py` (``FDR_Q`` + ``MIN_DEV`` on the kernel smoother) and places
each surviving neuron on the axis built from the model comparison's held-out accuracies.
:func:`subject_pool` puts the two together for one subject.
"""

import hashlib
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_FIGURE_3 = _HERE.parent
for _p in (
    str(_HERE),
    str(_FIGURE_3),
    str(_FIGURE_3.parent),
    str(_FIGURE_3 / "model_comparison"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import comparison_lib as cl  # noqa: E402  the model comparison rho is built from
import decoder_lib as dec  # noqa: E402  the model-free tables + read-out
import selection_lib as sel  # noqa: E402
import stats_lib as sl  # noqa: E402

_REPO = (
    Path(__file__).resolve().parents[3]
)  # repository root, where paths.py lives
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
import paths  # noqa: E402

#: Response-table cache for the titration.
TABLE_CACHE = paths.FIGURES_CACHE / "figure_3/pool_tables"

# Re-exported so `run_titration.py` never has to know where they came from.
precompute_unit_tables = dec.precompute_unit_tables
sample_bootstrap_table_inputs = dec.sample_bootstrap_table_inputs
make_table_neural_fn = dec.make_table_neural_fn
StimulusRepresentationModel = dec.StimulusRepresentationModel
train_model = dec.train_model
evaluate_error = dec.evaluate_error


def selection_units(verbose=True, exclusion="config", min_fit=None):
    """The pool, with the titration coordinate ``rho`` and its within-subject ``rho_rank``.

    ``rho`` is the polar angle in the two models' held-out excess accuracies, and the default
    ``exclusion='config'`` is the shared modulation criterion of `figures/config.py` and
    nothing else — direction from the models, inclusion from a model-agnostic test.
    ``min_fit`` additionally requires both models to fit at least that well. See
    `../selection_lib.py`.

    The coordinate is computed on the same trials the decoder is later scored on.
    """
    return sel.prepared_pool(
        cl.load_units(), exclusion=exclusion, verbose=verbose, min_fit=min_fit
    )


def load_pool_trials(units_df):
    """``[(t0, t1, fr), ...]`` in row order — every 2-object trial of each neuron."""
    cache = {}
    out = []
    for _, row in units_df.iterrows():
        key = (row["subject"], row["session"])
        if key not in cache:
            cache[key] = sl.session_unit_trials(*key)
        d = cache[key][int(row["unit_id"])]
        out.append((d["t0"], d["t1"], d["fr2"]))
    return out


def subject_pool(
    subject,
    sample_sigma_deg=10.0,
    n_grid=36,
    verbose=True,
    with_tables=True,
    exclusion="config",
    min_fit=None,
):
    """One subject's pool, its wedge coordinates and its response tables.

    Returns ``dict(pop_df, rho, rank, tables, n_eval_trials)``. ``rho`` is **raw** (the
    figure's axis) and ``rank`` is the within-subject rank the sampling is tilted on — see
    `selection_lib` for why those are different quantities. ``with_tables=False`` returns the
    pool alone (``tables=None``), without reading any per-trial data.
    """
    pool = selection_units(
        verbose=verbose, exclusion=exclusion, min_fit=min_fit
    )
    pool = pool[pool["subject"] == subject].reset_index(drop=True)
    pool = sel.add_rank(pool)  # re-rank within the single subject
    if not with_tables:  # reading every session's trials costs minutes
        return dict(
            pop_df=pool,
            rho=pool["rho"].to_numpy(float),
            rank=pool["rho_rank"].to_numpy(float),
            tables=None,
            n_eval_trials=np.zeros(len(pool), int),
        )
    # The tables depend only on WHICH neurons are in the pool (and sigma and the grid) -- not on
    # rho, not on any sweep parameter. Building them is the single slowest step (127 ms/unit:
    # 101 s for Monkey P) and every worker process would pay it independently, so they are
    # cached on the unit list and paid once per pool.
    key = hashlib.md5(
        repr(
            (
                subject,
                float(sample_sigma_deg),
                int(n_grid),
                [
                    (str(a), int(b))
                    for a, b in zip(pool["session"], pool["unit_id"])
                ],
            )
        ).encode()
    ).hexdigest()[:16]
    cached = TABLE_CACHE / f"pool_{subject}_{key}.npz"
    TABLE_CACHE.mkdir(parents=True, exist_ok=True)
    if cached.exists():
        tables = np.load(cached)["tables"]
        if verbose:
            print(f"  tables from cache: {cached.name}  [{tables.shape}]")
        trials = load_pool_trials(pool)
    else:
        trials = load_pool_trials(pool)
        tables = precompute_unit_tables(
            trials, sample_sigma_deg, n_grid=n_grid
        )
        # The temp name carries the PID: several workers of a cold-cache run would otherwise
        # write the SAME `.tmp.npz`, and the first `replace` would move the file out from under
        # the others. The rename itself is what makes the cache atomic, so concurrent readers
        # never see a partial file.
        tmp = cached.with_suffix(f".tmp{os.getpid()}.npz")
        np.savez_compressed(tmp, tables=tables)
        tmp.replace(cached)
        if verbose:
            print(f"  tables cached -> {cached.name}")
    return dict(
        pop_df=pool,
        rho=pool["rho"].to_numpy(float),
        rank=pool["rho_rank"].to_numpy(float),
        tables=tables,
        n_eval_trials=np.array([len(t[0]) for t in trials]),
    )

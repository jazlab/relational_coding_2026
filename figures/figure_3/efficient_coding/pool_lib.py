"""The v2 candidate pool and its response tables.

Everything model-free lives beside this file (`decoder_lib`) rather than
copied: the kernel-regression response tables, the bootstrap draw, the MLP and its training
loop. Those do not know a model exists — `precompute_unit_tables` is a swap-invariant
circular-Gaussian regression of real firing rates onto real (theta0, theta1) pairs — so
copying them would only create two things to keep in step. What changes in v2 is which
neurons enter the wedge, and that lives in `../selection_lib.py`.

The one thing that must be rebuilt here is the **pool**. Figure 3's
`selection_lib` applies the shared criterion out of `figures/config.py`
(``FDR_Q`` + ``MIN_DEV`` on the kernel smoother) and places each surviving neuron on the
relationality axis built from the model comparison's held-out accuracies.
:func:`subject_pool` is that substitution and nothing else.
"""

import hashlib
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_V2 = _HERE.parent  # figures/figure_3
for _p in (
    str(_HERE),
    str(_V2),
    str(_V2.parent),
    str(_V2 / "model_comparison"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import comparison_lib as cl  # noqa: E402  the model comparison rho is built from
import decoder_lib as dec  # noqa: E402  the model-free tables + read-out
import selection_lib as sel  # noqa: E402
import stats_lib as sl  # noqa: E402

#: Response-table cache for the titration. (A separate `wedge_tables/` belonged to
#: `../efficient_coding_schematic/` until 2026-09-13, when that figure stopped using tables.
TABLE_CACHE = _HERE.parent.parent.parent / "cache/figures/figure_3/pool_tables"

# Re-exported so `run_titration.py` never has to know where they came from.
precompute_unit_tables = dec.precompute_unit_tables
subsample_pool_positions = dec.subsample_pool_positions
sample_bootstrap_table_inputs = dec.sample_bootstrap_table_inputs
make_table_neural_fn = dec.make_table_neural_fn
StimulusRepresentationModel = dec.StimulusRepresentationModel
train_model = dec.train_model
evaluate_error = dec.evaluate_error


def selection_units(
    verbose=True,
    coordinate="angle",
    exclusion="config",
    min_fit=None,
    min_fit_mode="both",
    min_fit_sum=None,
    sig_q=None,
    sig_use_fdr=True,
):
    """The pool, with the titration coordinate ``rho`` and its within-subject ``rho_rank``.

    Defaults are the adopted scheme: ``coordinate='prob'`` (``P(relational | data)``) and
    ``exclusion='config'`` (the shared modulation criterion of `figures/config.py`, and
    nothing else). See `../selection_lib.py` for why those two belong together — direction from
    the models, inclusion from a model-agnostic test.

    The coordinate is computed on the same trials the decoder is later scored on. A
    selection/evaluation trial split was built to answer that objection and removed on
    2026-09-13 with the graded comparison it refit; see `../selection_lib.py`.
    """
    units = cl.load_units()
    return sel.prepared_pool(
        units,
        coordinate=coordinate,
        exclusion=exclusion,
        verbose=verbose,
        min_fit=min_fit,
        min_fit_mode=min_fit_mode,
        min_fit_sum=min_fit_sum,
        sig_q=sig_q,
        sig_use_fdr=sig_use_fdr,
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
    coordinate="angle",
    exclusion="config",
    min_fit=None,
    min_fit_mode="both",
    shuffle_seed=None,
    min_fit_sum=None,
    sig_q=None,
    sig_use_fdr=True,
    cache_tables=True,
):
    """One subject's pool, its wedge coordinates and its response tables.

    Returns ``dict(pop_df, rho, rank, tables)``. ``rho`` is **raw** (the figure's axis) and
    ``rank`` is the within-subject rank the wedge is taken on — see `selection_lib` for why
    those are different quantities.
    """
    pool = selection_units(
        verbose=verbose,
        coordinate=coordinate,
        exclusion=exclusion,
        min_fit=min_fit,
        min_fit_mode=min_fit_mode,
        min_fit_sum=min_fit_sum,
        sig_q=sig_q,
        sig_use_fdr=sig_use_fdr,
    )
    pool = pool[pool["subject"] == subject].reset_index(drop=True)
    if shuffle_seed is not None:
        # The null: permute rho ACROSS UNITS within this subject, then re-rank. Same pool, same
        # tables, same K, same rho distribution -- only the pairing between a unit's
        # relationality and its tuning is destroyed. Everything downstream is untouched, so any
        # curvature that survives is what the procedure invents from a meaningless label.
        rng = np.random.default_rng(shuffle_seed)
        pool = pool.assign(rho=rng.permutation(pool["rho"].to_numpy(float)))
    pool = sel.add_rank(pool)  # re-rank within the single subject
    if (
        not with_tables
    ):  # --dry-run only inspects the wedges; reading every session's
        return dict(
            pop_df=pool,
            rho=pool["rho"].to_numpy(float),  # trials costs minutes
            rank=pool["rho_rank"].to_numpy(float),
            tables=None,
            n_eval_trials=np.zeros(len(pool), int),
        )
    # The tables depend only on WHICH neurons are in the pool (and sigma, the grid, and the
    # trial split) -- not on rho, not on the shuffle, not on any sweep parameter. Building them
    # is the single slowest step (127 ms/unit: 101 s for Monkey P) and every worker process was
    # paying it independently. Cache on the unit list so it is paid once per pool, ever.
    cached = None
    if cache_tables:
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
    if cached is not None and cached.exists():
        tables = np.load(cached)["tables"]
        if verbose:
            print(f"  tables from cache: {cached.name}  [{tables.shape}]")
        trials = load_pool_trials(pool)
    else:
        trials = load_pool_trials(pool)
        tables = precompute_unit_tables(
            trials, sample_sigma_deg, n_grid=n_grid
        )
        if cached is not None:
            # The temp name carries the PID. Without it every worker of a cold-cache run
            # writes the SAME `.tmp.npz`, and the first `replace` moves the file out from
            # under the others -- which killed 3 of 12 workers on 2026-09-13 with a
            # FileNotFoundError on the rename. The rename itself is what makes the cache
            # atomic, so concurrent readers still never see a partial file.
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


# ── Faster training: the Poisson draw in bulk ─────────────────────────────────
#: Steps generated per call. Memory is chunk x batch x n_samples floats -- 20 x 128 x 500 is
#: about 5 MB, so this is free; going much larger buys nothing once the per-call overhead is
#: amortised.
BATCH_CHUNK = 20

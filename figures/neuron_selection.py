"""The single neuron-inclusion criterion, shared by every population analysis.

One question decides whether a neuron enters any population figure:

    **Is this neuron's firing rate on 2-object trials predictable from the stimulus,
    beyond what chance would give?**

It is answered **without reference to any model of the tuning** — no cosine, no receptive
field, no retinotopic-vs-relational anything — so that selecting the population cannot
prejudge the comparison the figures then make. That circularity is the whole reason the
criterion is model-agnostic: figure 3 and figure 4 ask which *model* explains a neuron, and
a population picked by how well one of those models fits would answer part of that question
before the figure is drawn.

The criterion, exactly
----------------------
For each neuron, using only its completed 2-object trials (the over-represented east-west
antipodal pair excluded, ≥ ``MIN_2OBJ_TRIALS`` trials required). *Only* 2-object trials: a
neuron's 1-object trial count has no bearing on whether it enters a population.

1. **The statistic.** Fit an exchange-symmetric Gaussian **kernel smoother** on the
   ``(θ0, θ1)`` torus and score it out of sample. The distance between two stimuli is the
   smaller of the direct and object-swapped sums of squared circular distances, so the
   smoother cannot depend on which object was labelled 0; the bandwidth is
   ``BANDWIDTH_DEG``. The score is the **held-out Poisson deviance explained**,

       dev = 1 − D(y_test, kernel prediction) / D(y_test, train mean)

   where ``D`` is the Poisson deviance ``2 Σ [y log(y/µ) − (y − µ)]`` of the observed spike
   counts over the 2-second stimulus + delay window. It is computed by ``N_FOLDS``-fold
   cross-validation, repeated ``N_CV_REPEATS`` times with re-randomised folds and averaged.
   Held-out is essential: a kernel smoother can fit any in-sample scatter, and only its
   out-of-sample score says whether there is structure to fit.

   Being a smoother rather than a parametric tuning curve, it will find *any* reproducible
   dependence on the stimulus pair — one bump, two bumps, a ridge along the sum, a ridge
   along the difference, or a shape nobody has a name for.

2. **The null.** Permute the firing rates across that neuron's own trials and recompute
   ``dev``, ``N_SHUFFLES`` times. Permutation breaks the stimulus-response association while
   leaving the neuron's rate distribution, its trial count and the stimulus set exactly as
   they were, so the null carries the neuron's own noise and the experiment's own stimulus
   sampling. No distributional assumption is made about spike counts.

3. **The p-value.** ``p = (1 + #{null dev ≥ observed dev}) / (1 + N_SHUFFLES)``, the
   standard permutation p-value. Its smallest possible value is ``1/(1 + N_SHUFFLES)``,
   which is why ``N_SHUFFLES`` has to be comfortably larger than the reciprocal of the
   threshold — see the warning in :func:`significance`.

4. **The threshold.** ``3011`` neurons (those with enough 2-object trials) are tested at
   once, so a raw 5% cut would admit ~150 by chance. The p-values are therefore corrected across the whole population by
   **Benjamini-Hochberg**, controlling the false discovery rate at ``FDR_Q``: at most that
   fraction of the neurons *declared* modulated are expected to be false positives. The
   correction is computed over every tested neuron once, so the population is one fixed list
   that does not change when a figure looks at a subset of it.

Everything above is already computed by ``figure_3/model_comparison/run_modulation_cache.py``
(the statistic is in ``modulation_lib.py``) and stored per neuron in ``modulation.csv`` as
``dev`` and ``p``.
This module does not recompute it; it reads that table and applies steps 3-4.

Using it
--------
    import neuron_selection
    keep = neuron_selection.passing_units()            # subject / session / unit_id
    df = neuron_selection.filter_units(df, unit_col='unit')   # any frame with those keys

`figure_3`, `figure_4`, `supp_figure_1` and `supp_figure_2` all route their population
selection through here, so the four figures are drawn over exactly the same neurons.
"""

from pathlib import Path

import config
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent

#: Where `run_modulation_cache.py` writes. There used to be a second, byte-different but
#: numerically identical copy under the superseded figure's cache, tried as a fallback. It was
#: removed: a fallback into a directory scheduled for deletion is the kind of reference that
#: changes the population silently rather than failing, and this file defines the population
#: every figure in the paper selects on.
MODULATION_PATH = _ROOT / "cache/figures/shared_inputs/modulation.csv"

# ── The criterion's parameters ────────────────────────────────────────────────
# They live in `config.py`, and are read from it **at call time**, never copied into module
# constants or default arguments here. That is what lets `config.reload()` reach a running
# notebook: a `FDR_Q = config.FDR_Q` at this level would freeze the value at import and the
# edit would silently not take effect. Every public function below therefore defaults its
# thresholds to `None` and resolves them when it runs.

_ID = ["subject", "session", "unit_id"]


def load_modulation(path=None):
    """The per-neuron modulation table, with `dev`, `p` and the parameters behind them."""
    if path is None:
        path = MODULATION_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. It is the paper-wide modulation criterion, written by the "
                "`python run_modulation_cache.py` in figures/figure_3/model_comparison/."
            )
    return pd.read_csv(path)


def benjamini_hochberg(p, q=None):
    """Boolean mask of the p-values rejected by Benjamini-Hochberg at level `q`.

    Sort the `n` p-values ascending; find the largest `k` with `p_(k) <= q·k/n`; reject
    everything at or below `p_(k)`. NaN p-values are never rejected.
    """
    q = config.FDR_Q if q is None else q
    p = np.asarray(p, float)
    finite = np.isfinite(p)
    ps = np.sort(p[finite])
    n = len(ps)
    if n == 0:
        return np.zeros_like(p, dtype=bool)
    below = ps <= q * np.arange(1, n + 1) / n
    if not below.any():
        return np.zeros_like(p, dtype=bool)
    cutoff = ps[np.max(np.where(below)[0])]
    return finite & (p <= cutoff)


def significance(
    modulation=None, q=None, use_fdr=None, min_dev=None, verbose=True
):
    """The modulation table with a boolean `modulated` column added.

    Two cuts, both model-agnostic. **Significance** — is there any reproducible stimulus
    dependence — then **effect size**: `dev >= min_dev`, how much of the neuron's held-out
    Poisson deviance the kernel smoother actually explains.

    `min_dev` is the knob for keeping fewer neurons. Once a large fraction of the population
    sits on the p resolution floor `1/(1 + n_shuffles)`, Benjamini-Hochberg rejects all of
    them or none, so tightening `q` steps off a cliff instead of trimming; `dev` is
    continuous and keeps discriminating past that point.

    Warns when the permutation p-value cannot resolve the requested threshold: with
    `n_shuffles` permutations the smallest attainable p is `1/(1 + n_shuffles)`, so a
    threshold below that admits nobody, and a threshold *at* it means the test is
    resolution-limited rather than measuring anything finer.
    """
    q = config.FDR_Q if q is None else q
    use_fdr = (config.CORRECTION == "fdr") if use_fdr is None else use_fdr
    min_dev = config.MIN_DEV if min_dev is None else min_dev

    df = load_modulation() if modulation is None else modulation.copy()
    n_shuffles = (
        int(df["param_n_shuffles"].iloc[0])
        if "param_n_shuffles" in df
        else None
    )

    p = df["p"].to_numpy(float)
    sig = benjamini_hochberg(p, q) if use_fdr else (np.isfinite(p) & (p <= q))
    dev = df["dev"].to_numpy(float)
    df["modulated"] = (
        sig & (np.isfinite(dev) & (dev >= min_dev)) if min_dev > 0 else sig
    )

    if verbose:
        rule = (
            f"Benjamini-Hochberg FDR q = {q:g}"
            if use_fdr
            else f"raw p <= {q:g}"
        )
        if min_dev > 0:
            rule += f", deviance explained >= {min_dev:g}"
        n_keep = int(df["modulated"].sum())
        print(
            f"2-object modulation ({rule}): {n_keep} of {len(df)} neurons "
            f"({n_keep / len(df):.1%})"
        )
        if min_dev > 0:
            print(
                f"  significance alone would keep {int(sig.sum())}; the deviance floor "
                f"drops a further {int(sig.sum()) - n_keep}"
            )
        if n_shuffles is not None:
            floor = 1.0 / (1 + n_shuffles)
            # Judged on SIGNIFICANCE alone. Testing the final set would make the deviance
            # floor trigger this: it removes the weakly-driven neurons, which are exactly the
            # ones with the larger p-values, so the survivors can all sit at the floor while
            # the significance test itself discriminates perfectly well. More shuffles would
            # not help in that case, so saying so would be wrong.
            kept_p = p[sig]
            if len(kept_p) and np.nanmax(kept_p) <= floor + 1e-12:
                print(
                    f"  ⚠ resolution-limited: every selected neuron sits at the smallest "
                    f"p this cache can produce ({floor:.4f}, from n_shuffles = "
                    f"{n_shuffles}). Rebuild the modulation cache with a larger "
                    f"n_shuffles for a threshold that discriminates."
                )
    return df


def passing_units(q=None, use_fdr=None, min_dev=None, verbose=True):
    """`subject` / `session` / `unit_id` of every neuron that passes — the population."""
    df = significance(q=q, use_fdr=use_fdr, min_dev=min_dev, verbose=verbose)
    return df.loc[df["modulated"], _ID].reset_index(drop=True)


def filter_units(
    frame,
    unit_col="unit_id",
    q=None,
    use_fdr=None,
    min_dev=None,
    verbose=True,
    what="neurons",
):
    """Keep the rows of `frame` whose neuron passes the criterion.

    `unit_col` names the neuron-id column, which is `unit_id` in figure 3's tables and
    `unit` in figure 4's. The join is on (subject, session, that column).
    """
    keep = passing_units(
        q=q, use_fdr=use_fdr, min_dev=min_dev, verbose=verbose
    )
    if unit_col != "unit_id":
        keep = keep.rename(columns={"unit_id": unit_col})
    before = frame[["subject", "session", unit_col]].drop_duplicates().shape[0]
    out = frame.merge(keep, on=["subject", "session", unit_col], how="inner")
    if verbose:
        after = (
            out[["subject", "session", unit_col]].drop_duplicates().shape[0]
        )
        print(f"  modulation criterion: {before} -> {after} {what}")
    return out.reset_index(drop=True)

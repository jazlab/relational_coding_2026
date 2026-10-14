"""Shared setup for the supp_figure_9 notebooks: style, saving, and the single-unit cut.

This directory is figures 3 and 4's population results recomputed over **well-isolated single
units only**, pooled across recording sites — the complement to `supp_figure_1`, which splits
the same five results by brain area and keeps every unit. The question
it answers is whether the results are carried by multi-unit clusters, whose spikes may come
from several neurons with different tuning and whose apparent selectivity is therefore a
mixture.

**Where the label comes from.** Each unit in the spike-sorting NWB files carries a
``quality`` label, ``good`` (a well-isolated single unit) or ``mua`` (a multi-unit
cluster). Nothing downstream carries it: neither `phase_units.csv`, nor the shared
`firing_rates/` tables, nor figure 4's fit summaries has a quality column, so it is read
here from the NWB unit tables once (via `phys_processing/spike_counts.py`) and cached as a
table of ``(subject, session, unit, probe, quality)``.

Unit ids are assigned per probe, so in principle two probes in one session could use the same
id; they do not. Every unit has a distinct ``(subject, session, unit)``, which
is what makes the join to the downstream tables safe, and :func:`unit_quality` re-checks it
every time it rebuilds rather than trusting that.

**The selection criterion is not recomputed.** As in `supp_figure_1`, the shared 2-object
modulation criterion is applied first, over the whole population, and the single-unit cut comes
after: the Benjamini-Hochberg correction runs over every neuron tested, so which neurons pass
must not depend on how they are later split. What *is* recomputed within the cut is everything
that is a property of the population being drawn — which model wins, the model posteriors and
elpd differences,
the bootstrap resampling.

Because it is the same analysis it is the same code: importing this module puts
`../figure_3/model_comparison` and `../figure_4` on `sys.path`, and the notebooks then import
their libraries. Nothing is reimplemented and no cache is rebuilt.
"""

import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
_FIGURES = _HERE.parent
_ROOT = _FIGURES.parent

for _p in (
    _FIGURES,
    _FIGURES / "figure_3" / "model_comparison",
    _FIGURES / "figure_4",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import config  # noqa: E402,F401  the one place selection criteria are set
from style import despine, save, set_style  # noqa: E402,F401
from trial_filters import (  # noqa: E402,F401
    drop_east_west,
    drop_east_west_arrays,
    east_west_mask,
)

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from phys_processing import spike_counts as spike_counts_lib  # noqa: E402

#: Where the recovered table is kept. It is a property of the recordings rather than of this
#: figure, so it lives with
#: the other shared inputs rather than under this directory's own cache.
QUALITY_PATH = _ROOT / "cache/figures/shared_inputs/unit_quality.csv"

#: The label this directory keeps. The only other value is 'mua'.
GOOD = "good"


def unit_quality(force=False, verbose=True):
    """``(subject, session, unit, probe, quality)`` for every unit in the spike-sorting data.

    Read from the NWB unit tables, which are the only place the sorting label survives, and
    cached as a CSV.
    """
    if QUALITY_PATH.exists() and not force:
        return pd.read_csv(QUALITY_PATH)
    df = pd.concat(
        [
            spike_counts_lib.units(subject, session).assign(
                subject=subject, session=session
            )
            for subject, session in spike_counts_lib.sessions()
        ],
        ignore_index=True,
    )
    df = (
        df[["subject", "session", "unit", "probe", "quality"]]
        .sort_values(["subject", "session", "unit"])
        .reset_index(drop=True)
    )
    dup = df.duplicated(["subject", "session", "unit"]).sum()
    if dup:
        raise ValueError(
            f"{dup} units share a (subject, session, unit) key across probes. The join to the "
            f"downstream tables, which carry no probe column, would then be ambiguous."
        )
    QUALITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(QUALITY_PATH, index=False)
    if verbose:
        n = df.groupby(["subject", "quality"]).size()
        print(f"built {QUALITY_PATH.name}: {len(df)} units\n{n.to_string()}")
    return df


def add_quality(units, unit_col="unit", verbose=True):
    """Join the sorting label onto a table keyed on ``(subject, session, <unit_col>)``.

    Raises if any row fails to match: an unlabelled neuron silently dropped by the cut below
    would change the population without saying so.
    """
    q = unit_quality(verbose=False)[["subject", "session", "unit", "quality"]]
    out = units.merge(
        q,
        left_on=["subject", "session", unit_col],
        right_on=["subject", "session", "unit"],
        how="left",
        suffixes=("", "_q"),
    )
    missing = int(out["quality"].isna().sum())
    if missing:
        raise ValueError(
            f"{missing} of {len(out)} rows have no sorting label; the cut would "
            f"drop them silently."
        )
    if "unit_q" in out.columns:
        out = out.drop(columns=["unit_q"])
    return out


def good_only(units, unit_col="unit", verbose=True):
    """Keep the well-isolated single units. One group, so the notebooks read like the
    per-area ones with the loop collapsed."""
    u = add_quality(units, unit_col=unit_col)
    kept = (
        u[u["quality"] == GOOD]
        .drop(columns=["quality"])
        .reset_index(drop=True)
    )
    if verbose:
        n_all = len(u[["subject", "session", unit_col]].drop_duplicates())
        n_good = len(kept[["subject", "session", unit_col]].drop_duplicates())
        per = (
            kept.drop_duplicates(["subject", "session", unit_col])
            .groupby("subject")
            .size()
        )
        print(
            f"single units: {n_good} of {n_all} neurons kept "
            f"({100 * n_good / max(n_all, 1):.0f}%)"
        )
        for s, n in per.items():
            print(f"  {s}: {n}")
    return kept


def selected_fits_good(models, verbose=True):
    """Figure 4's selected fits, restricted to single units, then re-restricted to ``models``.

    Same order as `supp_figure_1.selected_fits_by_area`: the shared criterion first, the cut
    second, the model restriction last, so ``best_model`` is recomputed within the population
    actually drawn. Returned as a one-element ``[(label, fits)]`` so the notebooks keep the
    shape of the per-area ones.
    """
    import fit_lib
    import neuron_selection

    summary = fit_lib.load_summary(fit_lib.FitConfig(), verbose=False)
    summary = neuron_selection.filter_units(
        summary, unit_col="unit", verbose=verbose
    )
    rows = good_only(summary, unit_col="unit", verbose=verbose)
    rows = fit_lib.restrict_models(rows, models, verbose=False)
    return [(GOOD, rows)]

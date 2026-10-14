"""Shared setup for the supp_figure_1 notebooks: style, saving, and the split by brain area.

This directory is **figure 3's and figure 4's population results, once per recording area**
instead of pooled — over FEF neurons and DMFC neurons separately, so a reader can see that
neither result is carried by one area. Five notebooks:

* from figure 3 — `model_wins_and_probability`, `rf_orientation`;
* from figure 4 — `figure_model_comparison`, `figure_relationality`, `figure_ks_stats`.

(`supp_figure_2/` held the figure-4 three until 2026-09-12, when the two directories were
merged. Its notebooks and PDFs kept their names, so nothing that referenced them by filename
had to change.)

Because it is the same analysis, it is the same code: importing this module puts **both**
`../figure_3/model_comparison` and `../figure_4` on `sys.path`, and the notebooks then import
`stats_lib` from the first and `fit_lib` / `plot_lib` / `ks_lib` / `relationality_lib` from the
second. Nothing is reimplemented and no cache is rebuilt — those libraries read
`cache/figures/figure_3/model_comparison` and `cache/figures/figure_4/neuron_models`,
which took an hour and several hours to fit respectively. A copy here would be a second
definition of the same models, free to drift from the figures this one supplements.
FORMATTING.md's self-containedness rule is about plotting style, which each notebook owns.

.. note::
   **The two source directories share exactly one module name, `run_cache.py`, and neither
   supplement imports it** — checked when they were merged. Everything else is distinct, so
   having both on the path cannot silently serve the wrong module.

   One real coupling does exist: `figure_4/data_lib.py` does ``from common import
   drop_east_west_arrays``, and by the time it runs *this* module is what `common` resolves
   to. The re-export below is therefore load-bearing, not convenience — deleting it breaks
   figure 4's data loading from here.

**The area filter comes first.** :func:`selected_fits_by_area` restricts to an area *before*
restricting the model set, so ``best_model`` and everything downstream are judged within that
area — the analysis rerun on that population, not the pooled analysis with rows hidden at the
end. Who is *included* no longer depends on any model: the shared, model-agnostic modulation
criterion is computed once over the whole population, so the Benjamini-Hochberg correction does
not depend on how the neurons are later split.

**Ported to the pairwise estimator on 2026-09-11**, when `figure_3_deprecated/` was deleted,
and to the sign-only model comparison on 2026-09-12. Until the first of those the figure-3
notebooks imported the *parametric* `stats_lib`, so the supplement reported a different model
from the main text.
"""

import sys
from pathlib import Path

from matplotlib import (  # noqa: F401  (imported for notebook convenience)
    pyplot as plt,
)

_HERE = Path(__file__).resolve().parent

# Figure 3's model comparison, and figure 4's model fitting. Imported rather than
# copied, for the reasons in the module docstring.
MODEL_COMPARISON = (_HERE / "../figure_3/model_comparison").resolve()
FIGURE_4 = (_HERE / "../figure_4").resolve()
for _p in (MODEL_COMPARISON, FIGURE_4):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# The shared neuron-inclusion criterion and the east-west antipodal exclusion both live one
# level up, in `figures/`. The exclusion is applied to every 2-object analysis here without
# exception; see ../trial_filters.py.
_FIGURES = _HERE.parent
if str(_FIGURES) not in sys.path:
    sys.path.insert(0, str(_FIGURES))
import config  # noqa: E402,F401  the one place selection criteria are set
from trial_filters import (  # noqa: E402,F401
    drop_east_west,
    drop_east_west_arrays,
    east_west_mask,
)

AREAS = ("FEF", "DMFC")  # one figure each, in this order

# The style block, the PDF writer and `despine` live in `figures/style.py` and are
# re-exported here, so every notebook keeps calling `common.set_style()` while there is only
# one definition of what the style IS. See FORMATTING.md.
from style import despine, save, set_style  # noqa: E402,F401


def by_area(units, areas=AREAS):
    """``[(area, that area's neurons), ...]``, reporting what each one holds.

    The figure-3 notebooks' entry point. Every cut in those figures is applied *within* an
    area, so the split happens first and the printed counts are the ones the figures are
    drawn from.
    """
    import stats_lib as sl

    out = []
    for area in areas:
        rows = units[units["brain_area"] == area]
        counts = ", ".join(
            f'{sl.subject_title(s)} {len(rows[rows["subject"] == s])}'
            for s in sl.ordered_subjects(rows)
        )
        print(f"{area}: {len(rows)} neurons ({counts})")
        out.append((area, rows))
    return out


def selected_fits_by_area(models, areas=AREAS):
    """``[(area, that area's selected fits), ...]`` — the figure-4 notebooks' entry point.

    Per area: keep that area's neurons, apply the shared 2-object modulation criterion of
    ``figures/neuron_selection.py``, then restrict to ``models`` (recomputing which of them
    wins). The area filter comes first, so ``best_model`` is judged within the area.

    The criterion itself is computed once over the **whole** population, not per area: the
    Benjamini-Hochberg correction is over every neuron tested, so which neurons pass does not
    depend on how they are later split.
    """
    import fit_lib
    import neuron_selection

    # One configuration — the one the cache was built with; there is only one on disk.
    summary = fit_lib.load_summary(fit_lib.FitConfig(), verbose=False)
    summary = neuron_selection.filter_units(
        summary, unit_col="unit", verbose=True
    )

    out = []
    for area in areas:
        rows = summary[summary["brain_area"] == area]
        rows = fit_lib.restrict_models(rows, models, verbose=False)
        n_kept = len(rows[["subject", "session", "unit"]].drop_duplicates())
        print(f"{area}: {n_kept} neurons enter the figure")
        out.append((area, rows))
    return out

"""Shared setup for the figure_4 notebooks: style, cache paths, and the model fits.

Figure 4 has two halves. Two notebooks draw **schematics** and need no data at all
(`computational_model.ipynb`, `future_field_proof.ipynb`). The rest read one cache — the
per-neuron model fits that `run_cache.py` built — and the three that draw **population**
results (`figure_model_comparison`, `figure_relationality`, `figure_ks_stats`) all begin the
same way: load the fits and restrict them to the models the figure compares. That opening is
here so it is written once and every panel is guaranteed to be about the same neurons. The
remaining two (`figure_tuning`, `tuning_plots`) draw a single neuron and select nothing.

Refitting takes hours, so `run_cache.py` is incremental: it tops up the models missing from
`cache/figures/figure_4/neuron_models/` and leaves the rest alone. Deleting the cache to
force a full refit would move every number by however much the optimiser is
non-deterministic.
"""

import sys
from pathlib import Path

from matplotlib import pyplot as plt

# The shared neuron-inclusion criterion and the east-west antipodal exclusion both live one
# level up, in `figures/`. The exclusion is applied to every 2-object analysis here
# without exception; see ../trial_filters.py.
_FIGURES = Path(__file__).resolve().parent.parent
if str(_FIGURES) not in sys.path:
    sys.path.insert(0, str(_FIGURES))
import config  # noqa: E402  the one place selection criteria are set

# The style block, the PDF writer and `despine` live in `figures/style.py` and are
# re-exported here, so every notebook keeps calling `common.set_style()` while there is only
# one definition of what the style IS. See FORMATTING.md.
from style import despine, save, set_style  # noqa: F401

# Re-exported, not merely imported: `data_lib` and the notebooks reach the east-west
# antipodal exclusion through this module, so removing these names breaks them. It is applied
# to every 2-object analysis here without exception; see ../trial_filters.py.
from trial_filters import (  # noqa: E402,F401
    drop_east_west,
    drop_east_west_arrays,
    east_west_mask,
)

# Cache paths, subject names and the angle helpers are NOT re-declared here. `data_lib` owns
# them, every caller reaches them as `dl.`, and the copy this module used to keep was both
# unused and subtly wrong — it built its paths from a *relative* root, so it would have
# resolved somewhere else the moment anything ran from another directory.

# Every figure here now draws the SAME population — the neurons that pass the shared
# 2-object modulation criterion of `figures/neuron_selection.py`. The model set still
# differs between figures (it is the comparison each one makes), but it no longer changes
# who is included, because selection no longer depends on the models.


def selected_fits(models, verbose=True):
    """The fits every population figure starts from: one row per (neuron, model).

    Two steps, in this order:

    1. **the 2-object modulation criterion** — the shared, model-agnostic test in
       ``figures/neuron_selection.py``. A neuron enters if its 2-object firing rate is
       predictable from the stimulus beyond chance, judged by a cross-validated kernel
       smoother against a within-neuron permutation null and corrected across the whole
       population by Benjamini-Hochberg. Nothing about any *model* of the tuning enters,
       which is the point: figure 4 asks which model explains these neurons, so a
       population chosen by how well one of those models fits would answer part of that
       question before the figure is drawn.
    2. **restrict to `models`**, recomputing which model wins within that set. ``best_model``
       as cached is the argmax over every model that was fit, so a figure drawn over a
       subset while reading that column would count neurons whose winner is not on its axis.

    An earlier version cut neurons by their best model's held-out ΔLL. That is gone: it
    selected the population by the very fits the figures then compare.
    """
    import fit_lib
    import neuron_selection

    # One configuration, the one the cache was built with. It is not a knob: a different
    # tuning function or split count is a different cache, and there is only one on disk.
    summary = fit_lib.load_summary(fit_lib.FitConfig(), verbose=verbose)
    n_all = summary[["subject", "session", "unit"]].drop_duplicates().shape[0]
    summary = neuron_selection.filter_units(
        summary, unit_col="unit", verbose=verbose
    )
    summary = fit_lib.restrict_models(summary, models, verbose=verbose)
    if verbose:
        n = summary[["subject", "session", "unit"]].drop_duplicates().shape[0]
        print(f"{n} of {n_all} neurons enter the figure ({n / n_all:.0%})")
    return summary

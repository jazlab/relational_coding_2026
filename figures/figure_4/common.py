"""Shared setup for the figure_4 notebooks: style, cache paths, and the model fits.

Figure 4 has two halves. Two notebooks draw **schematics** and need no data at all
(`computational_model.ipynb`, `future_field_proof.ipynb`). The rest read one cache — the
per-neuron model fits that `run_cache.py` built — and the three that draw **population**
results (`figure_model_comparison`, `figure_relationality`, `figure_ks_stats`) all begin the
same way: load the fits and restrict them to the models the figure compares. That opening is
here so it is written once and every panel is guaranteed to be about the same neurons. The
remaining two (`figure_example_neurons`, `figure_example_model_neurons`) draw single neurons.

Refitting takes hours, so `run_cache.py` is incremental: it tops up the models missing from
`cache/figures/figure_4/neuron_models/` and leaves the rest alone. Deleting the cache to
force a full refit would move every number by however much the optimiser is
non-deterministic.
"""

import sys
from pathlib import Path

# The shared neuron-inclusion criterion, the style block and the east-west antipodal
# exclusion live one level up, in `figures/`.
_FIGURES = Path(__file__).resolve().parent.parent
if str(_FIGURES) not in sys.path:
    sys.path.insert(0, str(_FIGURES))

# The style block and the PDF writer live in `figures/style.py` and are re-exported here, so
# every notebook calls `common.set_style()` while there is only one definition of what the
# style IS. See FORMATTING.md.
from style import save, set_style  # noqa: E402,F401

# Cache paths, subject names and the angle helpers are not declared here: `data_lib` owns
# them and every caller reaches them as `dl.`.

# Every figure here draws the SAME population: the neurons that pass the shared 2-object
# modulation criterion of `figures/neuron_selection.py`. The model set differs between
# figures (it is the comparison each one makes), but it does not change who is included,
# because selection does not depend on the models.


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

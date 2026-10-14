"""Shared setup for the `figure_3` notebooks — the style block, the cache tree, saving.

Figure 3's model comparison is **pairwise**: for two trials that share an object, each
hypothesis predicts which one fired more, and a model is scored on the fraction of held-out
orderings it calls correctly. Nothing about baseline rate, modulation depth, receptive-field
width, link function or noise model enters. See `model_comparison/comparison_lib.py`'s module
docstring for the method and `model_comparison/pairs_lib.py` for how the comparisons are built.

The style block, the PDF writer and `despine` come from `figures/style.py`, so there is one
definition of what the paper looks like rather than a copy per directory.
"""

import sys
from pathlib import Path

from matplotlib import (  # noqa: F401  (imported for notebook convenience)
    pyplot as plt,
)

# The east-west antipodal exclusion, shared by every figure in figures/ (see
# ../trial_filters.py). It is applied to every 2-object analysis without exception.
_FIGURES = Path(__file__).resolve().parent.parent
if str(_FIGURES) not in sys.path:
    sys.path.insert(0, str(_FIGURES))
import config  # noqa: E402,F401  the one place selection criteria are set
from trial_filters import drop_east_west_arrays  # noqa: E402,F401
from trial_filters import drop_east_west, east_west_mask

ROOT = Path("../../..")  # the paper root, from a figure_3 subdirectory
CACHE = ROOT / "cache/figures/figure_3"
RING_CSV = ROOT / "cache/behavior_processing/ring.csv"

SUBJECTS = ["Perle", "Elgar"]
SUBJECT_LABELS = {"Perle": "Monkey P", "Elgar": "Monkey E"}

from style import despine, save, set_style  # noqa: E402,F401

"""Shared setup for the `figure_3` notebooks: the style block and saving.

Figure 3's model comparison is **pairwise**: for two trials that share an object, each
hypothesis predicts which one fired more, and a model is scored on the fraction of held-out
orderings it calls correctly. Nothing about baseline rate, modulation depth, receptive-field
width, link function or noise model enters. See `model_comparison/comparison_lib.py`'s module
docstring for the method and `model_comparison/pairs_lib.py` for how the comparisons are built.

The style block and the PDF writer come from `figures/style.py`, so there is one definition of
what the paper looks like rather than a copy per directory.
"""

import sys
from pathlib import Path

_FIGURES = Path(__file__).resolve().parent.parent
if str(_FIGURES) not in sys.path:
    sys.path.insert(0, str(_FIGURES))

from style import save, set_style  # noqa: E402,F401

"""Shared setup for the supp_figure_3 notebooks: style and saving.

This directory is the **discriminability-gradient** analysis — which direction of the
two-object stimulus space each population tells apart best, at every stimulus — for the
recorded neurons (`gradients.ipynb`) and for idealized ones (`gradients_ideal.ipynb`).

The populations, their trials and the selection rule are figure 3's, imported rather than copied
(`gradients_lib` puts the figure-3 directories it needs on `sys.path`), so the selection rule
cannot drift from the figure this one supplements.
"""

import sys
from pathlib import Path

# The style block and the PDF writer live one level up, in `figures/style.py`.
_FIGURES = Path(__file__).resolve().parent.parent
if str(_FIGURES) not in sys.path:
    sys.path.insert(0, str(_FIGURES))

from style import save, set_style  # noqa: E402,F401

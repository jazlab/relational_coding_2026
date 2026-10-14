"""Shared setup for the supp_figure_3 notebooks: style and saving.

This directory is the **discriminability-gradient** analysis — which direction of the
two-object stimulus space each population tells apart best, at every stimulus — for the
recorded neurons (`gradients.ipynb`, `gradients_schematic.ipynb`) and for idealized ones
(`gradients_ideal.ipynb`). It moved here from `figure_3/gradients/` on 2026-10-05.

The populations, their trials and the decoder are figure 3's, imported rather than copied
(`gradients_lib` puts the figure-3 directories it needs on `sys.path`), so the selection rule
and the read-out cannot drift from the figure this one supplements.
"""

import sys
from pathlib import Path

from matplotlib import (  # noqa: F401  (imported for notebook convenience)
    pyplot as plt,
)

_HERE = Path(__file__).resolve().parent

# The shared neuron-inclusion criterion and the east-west antipodal exclusion live one level
# up, in `figures/`, as does the style block.
_FIGURES = _HERE.parent
if str(_FIGURES) not in sys.path:
    sys.path.insert(0, str(_FIGURES))
import config  # noqa: E402,F401  the one place selection criteria are set

# The style block, the PDF writer and `despine` live in `figures/style.py` and are
# re-exported here, so every notebook keeps calling `common.set_style()` while there is only
# one definition of what the style IS.
from style import despine, save, set_style  # noqa: E402,F401
from trial_filters import (  # noqa: E402,F401
    drop_east_west,
    drop_east_west_arrays,
    east_west_mask,
)

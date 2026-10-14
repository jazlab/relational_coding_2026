"""Shared setup for the supp_figure_6 notebooks: style, and the paths to figure 4's code.

This directory asks whether figure 4's future-field model still produces the relational spectrum
when the receptive field is a 2-D Gaussian in *space* (so it depends on distance, not only on
direction) and when remapping is *convergent* (Zirnsak et al., 2014). See `ff_distance_lib.py`.
"""

import sys
from pathlib import Path

from matplotlib import (  # noqa: F401  (imported for notebook convenience)
    pyplot as plt,
)

_HERE = Path(__file__).resolve().parent

# The shared selection criteria and the style block live one level up, in `figures/`.
_FIGURES = _HERE.parent
if str(_FIGURES) not in sys.path:
    sys.path.insert(0, str(_FIGURES))
import config  # noqa: E402,F401  the one place selection criteria are set
from style import despine, save, set_style  # noqa: E402,F401

# Figure 4's `data_lib` does `from common import drop_east_west_arrays`, and once this
# directory's `common` is imported it is the one Python finds — so it must re-export that too.
from trial_filters import (  # noqa: E402,F401
    drop_east_west,
    drop_east_west_arrays,
    east_west_mask,
)

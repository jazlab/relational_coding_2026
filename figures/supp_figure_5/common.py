"""Shared setup for the supp_figure_5 notebooks: style and saving.

This directory shows how figure 4's relationality metric — the relational-bias model's fitted
``bias`` — is read off a neuron: the measured tuning, the goodness of fit as a function of
relationality, and the tuning the best-fitting model predicts. See `profile_lib.py`.
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

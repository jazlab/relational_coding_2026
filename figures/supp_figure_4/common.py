"""Shared setup for the supp_figure_4 notebooks: style and saving.

This directory asks how figure 3's motor decoder (`figure_3/motor_decoding/decoder_v2.ipynb`)
does on **1-object** stimuli, as a function of where the object sits on the ring. See
`decoder_cache.py`.
"""

import sys
from pathlib import Path

from matplotlib import (  # noqa: F401  (imported for notebook convenience)
    pyplot as plt,
)

_HERE = Path(__file__).resolve().parent

# The shared neuron-inclusion criterion and the style block live one level up, in
# `figures/`.
_FIGURES = _HERE.parent
if str(_FIGURES) not in sys.path:
    sys.path.insert(0, str(_FIGURES))
import config  # noqa: E402,F401  the one place selection criteria are set
from style import despine, save, set_style  # noqa: E402,F401

# Figure 3's `population_lib` does `from common import drop_east_west_arrays`, and once this
# directory's `common` is imported it is the one Python finds — so it must re-export that too.
from trial_filters import (  # noqa: E402,F401
    drop_east_west,
    drop_east_west_arrays,
    east_west_mask,
)

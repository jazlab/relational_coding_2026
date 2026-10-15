"""Shared setup for the supp_figure_6 notebooks: style and saving.

This directory asks whether figure 4's future-field model still produces the relational spectrum
when the receptive field is a 2-D Gaussian in *space* (so it depends on distance, not only on
direction) and when remapping is *convergent* (Zirnsak et al., 2014). See `ff_distance_lib.py`.
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# The style block and the east-west exclusion live one level up, in `figures/`.
_FIGURES = _HERE.parent
if str(_FIGURES) not in sys.path:
    sys.path.insert(0, str(_FIGURES))
from style import save, set_style  # noqa: E402,F401

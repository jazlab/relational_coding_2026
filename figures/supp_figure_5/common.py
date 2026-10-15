"""Shared setup for the supp_figure_5 notebooks: style and saving.

This directory shows how figure 4's relationality metric — the relational-bias model's fitted
``bias`` — is read off a neuron: the measured tuning, the goodness of fit as a function of
relationality, and the tuning the best-fitting model predicts. See `profile_lib.py`.
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# The style block and the east-west exclusion live one level up, in `figures/`.
_FIGURES = _HERE.parent
if str(_FIGURES) not in sys.path:
    sys.path.insert(0, str(_FIGURES))
from style import despine, save, set_style  # noqa: E402,F401

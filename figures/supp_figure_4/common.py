"""Shared setup for the supp_figure_4 notebooks: style and saving.

This directory asks how figure 3's motor decoder (`figure_3/motor_decoding/decoder.ipynb`)
does on **1-object** stimuli, as a function of where the object sits on the ring. See
`decoder_cache.py`.
"""

import sys
from pathlib import Path

# The style block and the PDF writer live one level up, in `figures/style.py`.
_FIGURES = Path(__file__).resolve().parent.parent
if str(_FIGURES) not in sys.path:
    sys.path.insert(0, str(_FIGURES))

from style import save, set_style  # noqa: E402,F401

"""Shared setup for the supp_figure_7 notebooks: the style block and saving.

Both notebooks here make PDFs and nothing else, so what they share is small.
"""

import sys
from pathlib import Path

# `figures/` on the path, resolved from THIS FILE rather than the working directory, so
# `style` resolves however the notebook was launched.
_FIGURES = Path(__file__).resolve().parent.parent
if str(_FIGURES) not in sys.path:
    sys.path.insert(0, str(_FIGURES))

# The style block, the PDF writer and `despine` live in `figures/style.py` and are
# re-exported here, so every notebook keeps calling `common.set_style()` while there is only
# one definition of what the style IS. See FORMATTING.md.
from style import despine, save, set_style  # noqa: F401

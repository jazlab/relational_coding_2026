"""Figure 2's tuning rings, reused for figure 4's example neurons and supp figure 5.

This module deliberately contains **no** tuning logic of its own: every firing rate,
prototype set, color scale and ring drawer comes from ``figure_2/psth_lib.py`` (added to
``sys.path`` below), so the rings drawn from it are the same objects as the right-hand
column of the single-unit PSTH figures, with the same parameters. ``psth_lib.build_rows``
gives three rows per unit

    row 0 : 1 object            — a continuous shaded annulus over object direction
    row 1 : ``pl.SEP1_DEG`` apart — a dense family of same-separation chords
    row 2 : 180° apart          — the antipodal family (spans half a turn)

Color is the mean firing rate over ``FR_WINDOW_MS`` (stimulus + delay), and each row is
normalized to **its own** firing-rate range.
"""

import sys
from pathlib import Path

import numpy as np

# The PSTH toolkit of figure 2 — reused as-is (see the module docstring).
#
# It has to be imported with figure 2's directory FIRST on the path, because `psth_lib`
# does `import common` and figure 4 has a `common` of its own. Both are then put back:
# `psth_lib` keeps a direct reference to the module object it imported, so restoring
# `sys.modules['common']` afterwards leaves it working and leaves `import common` in this
# directory meaning what it should.
_PSTH_DIR = (Path(__file__).resolve().parent / "../figure_2").resolve()
_saved_common = sys.modules.pop("common", None)
sys.path.insert(0, str(_PSTH_DIR))
try:
    import psth_lib as pl  # noqa: E402  (path set up above)
finally:
    sys.path.remove(str(_PSTH_DIR))
    sys.modules.pop("common", None)
    if _saved_common is not None:
        sys.modules["common"] = _saved_common

#: The colormap every panel is drawn in: coolwarm with its hot end lifted in HSV at constant
#: hue. Every panel's scale runs the whole map, so the extreme red is worth setting
#: deliberately — this brightens it without changing which red it is, where cutting the map
#: short would only fade it towards the pale centre.
FIRING_RATE_CMAP = pl.boosted_cmap()


#: The binning of the rings, shared with ``figures/figure_2/psth.ipynb``.
#: Every value here is that notebook's, so the panels are directly comparable — except
#: ``SMOOTH_HALF_WIN_MS``, which is 250 there. That one is temporal, and a mean over a 2 s
#: window barely feels it, but the two figures are not identical until it is reconciled.
SIMILARITY_WINDOW_DEG = (
    15.0  # a trial joins a prototype within this angular distance
)
SMOOTH_HALF_WIN_MS = (
    200.0  # temporal smoothing half-window of the firing rates
)
FR_WINDOW_MS = (0, 2000)  # stimulus + delay, stopping before the response


def unit_rows(subject, session, unit_id, orientation_deg=0.0):
    """``psth_lib.build_rows`` for one unit, keeping only the tuning rows.

    A trial joins a prototype when **both** of its objects are within
    ``SIMILARITY_WINDOW_DEG`` of their own slot.
    """
    _, _, rows = pl.build_rows(
        subject,
        session,
        unit_id,
        orientation_deg=orientation_deg,
        similarity_window_deg=SIMILARITY_WINDOW_DEG,
        smooth_half_win_ms=SMOOTH_HALF_WIN_MS,
        share_2obj_color_scale=True,
    )
    return rows


def draw_tuning_panel(ax, row, rotate_deg=0.0):
    """One tuning ring in :data:`FIRING_RATE_CMAP`: the 1-object row as a shaded annulus,
    the 2-object rows as a dense family of ``row['sep']``-apart chords (as in
    ``psth_lib.plot_rows``'s right-hand column).

    ``rotate_deg`` turns the drawn ring counter-clockwise by that many degrees: every
    sample direction moves, and so both ends of every chord with it, which is a rigid
    rotation of the picture. Nothing else changes — the conditions, their firing rates and
    their color scale are what they were."""
    phis = np.asarray(row["phis"], float) + np.radians(rotate_deg)
    if row["kind"] == "chord":
        pl.draw_dense_2obj(
            ax,
            phis,
            row["sep"],
            row["frs"],
            row["norm"],
            radius=0.43,
            cmap=FIRING_RATE_CMAP,
            lw=1.5,
        )
    else:
        pl.draw_1obj_annulus(
            ax,
            phis,
            row["frs"],
            row["norm"],
            radius=0.43,
            width=0.1,
            cmap=FIRING_RATE_CMAP,
            smooth_deg=10.0,
        )

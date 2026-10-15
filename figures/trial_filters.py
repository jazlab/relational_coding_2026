"""Stimulus-set filters shared by every figure in `figures/`.

Right now that is one filter, applied everywhere: dropping the **east–west antipodal**
2-object configuration.

Why it is dropped
-----------------
The ring task's 2-object stimuli are otherwise sampled broadly, but one configuration — one
object due east (0°) and the other due west (180°) — was presented of the order of a thousand
times per session, far more often than any other. Left in, it dominates every average that
pools over configurations: an anchor condition at 0° or 180° is mostly that one pair, a
scatter of the stimulus set is one dense blob, and a fit weights that geometry far above the
rest of the ring. Dropping it costs about 6% of the 2-object trials and makes what remains a
roughly even sample of the configuration space.

The rule is a match to within `config.EAST_WEST_TOL_DEG` on both objects, in **either object order**, so it
removes the repeated pair and nothing else — a genuinely different stimulus that happens to
be near-antipodal (say 3° and 179°) is kept.

There is deliberately **no option to turn this off**. It is a property of the stimulus set,
not an analysis choice, and every analysis in `figures/` applies it.
"""

import config
import numpy as np

THETA_COLUMNS = ("object_0_theta", "object_1_theta")


def _circ_dist(a, b):
    """Unsigned circular distance between angles, in [0, pi]."""
    return np.abs(
        (np.asarray(a, float) - np.asarray(b, float) + np.pi) % (2 * np.pi)
        - np.pi
    )


def east_west_mask(theta_0, theta_1):
    """`True` where a trial is the east–west antipodal pair, in either object order.

    Angles are in radians. A 1-object trial (whose second angle is NaN) is never east–west,
    which falls out of the comparison rather than needing a special case.
    """
    # Read from `config` at call time (never copied here), so `config.reload()` reaches a
    # running notebook.
    tol = np.radians(config.EAST_WEST_TOL_DEG)
    t0, t1 = np.asarray(theta_0, float), np.asarray(theta_1, float)
    return ((_circ_dist(t0, 0.0) <= tol) & (_circ_dist(t1, np.pi) <= tol)) | (
        (_circ_dist(t0, np.pi) <= tol) & (_circ_dist(t1, 0.0) <= tol)
    )


def drop_east_west(df):
    """The rows of `df` that are not the east–west pair.

    The object angles are read from `THETA_COLUMNS`, in radians. Rows missing either angle —
    1-object trials — are kept.
    """
    theta_0, theta_1 = (df[c].to_numpy(float) for c in THETA_COLUMNS)
    return df[~east_west_mask(theta_0, theta_1)]


def drop_east_west_arrays(theta_0, theta_1, *others):
    """The same filter over parallel arrays: `(theta_0, theta_1, *others)` with the
    east–west trials removed from each."""
    keep = ~east_west_mask(theta_0, theta_1)
    return (
        np.asarray(theta_0)[keep],
        np.asarray(theta_1)[keep],
        *(np.asarray(o)[keep] for o in others),
    )

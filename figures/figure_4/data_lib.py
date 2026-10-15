"""Firing rates and trial loading for the 2-object-only single-neuron model comparison.

The per-trial firing rates are the shared table built by ``figures/firing_rates.py``, the
same one the figure-3 model comparison reads, so the neurons and rates entering this analysis
are the ones entering that comparison. Points specific to this directory:

* **its own cache tree** for fits, ``cache/figures/figure_4/neuron_models/``;
* **firing rates are built lazily, one session at a time** — :func:`session_unit_trials`
  builds the session it was asked for if that session's file is missing, so nothing here
  ever costs more than the one session being looked at;
* **``CACHE_MIN_1OBJ_TRIALS = 0``.** Nothing in this directory fits, scores or initialises on
  1-object trials, so requiring a neuron to have them would drop neurons for no reason.
  1-object trials are still *loaded*, because the tuning-curve panels draw them as a
  check on a receptive field that was never fit to them.

Reads the shared per-trial firing rates (built by ``figures/firing_rates.py`` from the
DANDI spike-sorting files and ``ring.csv``)

    cache/figures/shared_inputs/firing_rates/{subject}_{session}.csv.gz

and writes everything it computes under

    cache/figures/figure_4/neuron_models/
        fits/{tag}/all/*.csv.gz                   per-neuron model fits (fit_lib)
        bias_recovery/{key}/*.csv.gz              the KS test's synthetic biases (ks_lib)

Per unit and trial, spikes are summed over 0-1000 ms (``stimulus_fr``) and 1000-2000 ms
(``delay_fr``) after stimulus onset and converted to Hz. The two objects' **identities**
(``'a'``, ``'b'`` or ``'c'``) are carried through alongside their angles, because
:class:`models_lib.WeightedAverageModel` weights each object by its luminance contrast and
needs to know which object it is looking at. The analysis uses
``fr = mean(stimulus_fr, delay_fr)``, so ``fr · DURATION_S`` is exactly the integer spike
count over the 2-s stimulus + delay window — which is what the Poisson likelihood in
``models_lib`` is evaluated on.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = (_HERE / "../..").resolve()  # paper root (figures/figure_4/)

_FIGURES = _ROOT / "figures"
if str(_FIGURES) not in sys.path:
    sys.path.insert(0, str(_FIGURES))
import firing_rates as fr_lib  # noqa: E402  the shared per-trial firing-rate table
from trial_filters import drop_east_west_arrays  # noqa: E402

# Everything under here is built by this directory's own code — `run_cache.py` for the fits.
# The fit cache is keyed by a tag encoding every setting that changes the numbers, so a
# configuration change gets its own tree rather than silently overwriting one.
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import paths  # noqa: E402

CACHE_DIR = paths.FIGURES_CACHE / "figure_4/neuron_models"
FR_DIR = fr_lib.FR_DIR
FIT_DIR = CACHE_DIR / "fits"

DURATION_S = 2.0  # the analysis window is stimulus + delay = 2 s

#: Measured tuning curves of the example-neuron figures: trials whose two objects both lie
#: within this many degrees of a sampled stimulus are averaged, and the curve is smoothed
#: with a circular Gaussian of this width.
SIMILARITY_WINDOW_DEG = 15.0
LINE_SMOOTH_DEG_2OBJ = 20.0

#: The per-trial rate columns averaged into ``fr``. Both by default, which with
#: ``DURATION_S = 2.0`` is the 2-s stimulus + delay window every figure here uses.
#: `supp_figure_2` sets one column and ``DURATION_S = 1.0`` to refit within one task phase;
#: both are read at call time, so they must be changed together and before any fit.
FR_COLUMNS = ("stimulus_fr", "delay_fr")

SUBJECT_ORDER = ["Perle", "Elgar"]  # Monkey P first, Monkey E second
SUBJECT_TITLE = {"Perle": "Monkey P", "Elgar": "Monkey E"}

# Minimum trial counts for a neuron to get a firing-rate cache built at all. These are NOT
# the population criterion and deliberately do not share its names: `config.MIN_2OBJ_TRIALS`
# (200) is the floor that decides who enters a figure, and these two only decide who is worth
# caching. They are far more permissive on purpose — the cache is built once and subset later.
CACHE_MIN_1OBJ_TRIALS = 0
CACHE_MIN_2OBJ_TRIALS = 30


def build_firing_rates(sessions=None, verbose=True):
    """Build the shared firing-rate files (see ``figures/firing_rates.py``)."""
    return fr_lib.build_sessions(sessions, verbose=verbose)


def fr_sessions():
    """Every ``(subject, session)``, building any firing-rate file that is missing first."""
    return fr_lib.ensure_all()


def sessions_with_behavior():
    """Every ``(subject, session)`` this analysis can be run on."""
    return fr_lib.sessions()


def session_unit_trials(subject, session, verbose=True):
    """One session's firing-rate cache → ``{unit: trial dict}``, building that cache first
    if it does not exist yet (one session, ~5 s). Only completed trials are used.

    Each value is ``dict(theta1, fr1, t0, t1, fr2, id0, id1, brain_area, subject, session,
    unit)`` where ``theta1``/``fr1`` are the 1-object trials (object angle in rad, rate in
    Hz), ``t0``/``t1``/``fr2`` the 2-object trials, and ``id0``/``id1`` those trials' object
    identities — ``'a'``, ``'b'`` or ``'c'`` — which only
    :class:`models_lib.WeightedAverageModel` reads. The firing rate is the mean of the stimulus
    and delay windows, so ``fr · DURATION_S`` is the integer spike count over the 2-s
    window. **Only ``t0``/``t1``/``fr2`` are ever fit or scored in this directory**;
    ``theta1``/``fr1`` exist for the tuning-curve panels.

    The over-represented east-west antipodal stimulus pair (one object at 0 deg and the
    other at 180 deg) is always dropped -- see `figures/trial_filters.py` for why and
    for the exact rule; there is no option to keep it.
    """
    path = FR_DIR / f"{subject}_{session}.csv.gz"
    if not path.exists():
        if verbose:
            print(
                f"{subject}/{session}: building the firing-rate cache (one session)…"
            )
        build_firing_rates([(subject, session)], verbose=verbose)
        if not path.exists():
            raise FileNotFoundError(
                f"{subject}/{session} has no 1-/2-object behaviour — nothing to fit. "
                f"`data_lib.sessions_with_behavior()` lists the usable sessions."
            )
    df = pd.read_csv(path)
    df = df[df["completed"].astype(bool)]
    df = df.assign(_fr=df[list(FR_COLUMNS)].astype(float).mean(axis=1))
    out = {}
    for uid, g in df.groupby("unit"):
        g1 = g[g["num_objects"] == 1]
        fr1 = g1["_fr"].to_numpy(float)
        th1 = g1["object_0_theta"].to_numpy(float) % (2 * np.pi)
        ok1 = np.isfinite(fr1) & np.isfinite(th1)
        th1, fr1 = th1[ok1], fr1[ok1]

        g2 = g[g["num_objects"] == 2]
        fr2 = g2["_fr"].to_numpy(float)
        t0 = g2["object_0_theta"].to_numpy(float) % (2 * np.pi)
        t1 = g2["object_1_theta"].to_numpy(float) % (2 * np.pi)
        id0 = g2["object_0_id"].to_numpy(object)
        id1 = g2["object_1_id"].to_numpy(object)
        ok2 = np.isfinite(fr2) & np.isfinite(t0) & np.isfinite(t1)
        t0, t1, fr2, id0, id1 = t0[ok2], t1[ok2], fr2[ok2], id0[ok2], id1[ok2]
        # the repeated east-west pair, dropped in every analysis (see figures/trial_filters.py)
        t0, t1, fr2, id0, id1 = drop_east_west_arrays(t0, t1, fr2, id0, id1)

        if (
            len(fr1) < CACHE_MIN_1OBJ_TRIALS
            or len(fr2) < CACHE_MIN_2OBJ_TRIALS
        ):
            continue
        out[int(uid)] = dict(
            theta1=th1,
            fr1=fr1,
            t0=t0,
            t1=t1,
            fr2=fr2,
            id0=id0,
            id1=id1,
            brain_area=str(g["brain_area"].iloc[0]),
            subject=subject,
            session=session,
            unit=int(uid),
        )
    return out


def notebook_literal(path, var):
    """The value of a **literal** assignment to ``var`` in another notebook's code cells.

    It reads the notebook file, walks its
    code cells for ``var = <literal>`` and evaluates that literal with ``ast.literal_eval``.
    A CONFIG cell is exactly the case this works for, and reading a setting out of the
    notebook that owns it is what stops two figures drifting apart — nothing is copied, so a
    change there is picked up here.

    Raises ``KeyError`` if no such assignment exists (a typo'd name is a mistake, not a
    default), and ``ValueError`` if the value is not a literal.
    """
    import ast
    import json

    nb = json.loads(Path(path).read_text())
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        try:
            tree = ast.parse("".join(cell["source"]))
        except SyntaxError:  # a cell with magics, say
            continue
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == var for t in node.targets
            ):
                return ast.literal_eval(node.value)
    raise KeyError(
        f"no literal assignment to {var!r} in any code cell of {path}"
    )


def units_from_notebook(path):
    """The ``(subject, session, unit)`` triples of the ``UNITS`` list in another notebook's
    CONFIG cell, read from the notebook file itself (:func:`notebook_literal`).

    Reading that list rather than copying it means the two figures cannot drift apart:
    comment a unit out there and it stops being marked here. Entries may carry extra fields
    (an orientation), which are dropped; only the first three are the neuron's identity.
    """
    return [tuple(e[:3]) for e in notebook_literal(path, "UNITS")]


def subject_title(s):
    return SUBJECT_TITLE.get(s, f"Monkey {s[0]}")

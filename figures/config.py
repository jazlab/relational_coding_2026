"""**The** selection criteria, for every figure in `figures/`.

Every rule that decides which neurons and which trials enter an analysis lives here, and
nowhere else. Change a number in this file and every figure that depends on it changes
together — which is the point: the alternative is the same threshold written into a dozen
notebook CONFIG cells, drifting apart until two panels of one figure are drawn over
different populations. (They were. `figure_4`'s three model panels used `MIN_DLL` 0.01,
0.01 and 0.1, so they showed 1,903, ~1,880 and 839 neurons.)

Who reads what
--------------
    NEURON SELECTION      `neuron_selection.py`  -> figure_3, figure_4, supp_1, supp_2
                          (2-object trials only — a 1-object trial floor is a prerequisite
                           of `rf_orientation`, not of entering a population)
    THE STATISTIC         `figure_3/model_comparison/modulation_lib.py`
                          -> `run_modulation_cache.py`, which writes `modulation.csv`
    TRIAL SELECTION       `trial_filters.py`, and each figure's `common.py`

Changing a threshold under NEURON SELECTION or TRIAL SELECTION only requires re-running the
notebooks. Changing anything under THE STATISTIC requires **rebuilding `modulation.csv`**
(`python run_modulation_cache.py --force` in `figure_3/model_comparison/`, ~10 min on 6
workers) —
the p-values themselves change, not just where they are cut.
"""

# ══════════════════════════════════════════════════════════════════════════════
# NEURON SELECTION — who enters a population analysis
# ══════════════════════════════════════════════════════════════════════════════
# One criterion, applied everywhere: is this neuron's 2-object firing rate predictable from
# the stimulus, beyond chance? It is model-agnostic on purpose — figures 3 and 4 ask which
# *model* explains these neurons, so a population picked by how well one of those models
# fits would answer part of that question before the figure is drawn. The full definition is
# in `neuron_selection.py`.

#: How the permutation p-value is thresholded.
#:   'fdr'  — Benjamini-Hochberg across every tested neuron at `FDR_Q`. ~3000 neurons are
#:            tested at once, so an uncorrected 5% cut would admit ~150 by chance.
#:   'raw'  — threshold the p-value directly at `FDR_Q`, no correction. For comparison only.
CORRECTION = "fdr"

#: The cutoff: the false-discovery rate when CORRECTION = 'fdr', the raw p ceiling otherwise.
#: It cannot go below 1/(1 + N_SHUFFLES) — no permutation test can produce a smaller p.
FDR_Q = 0.05

#: An **effect-size** floor, applied on top of the significance test above: a neuron enters
#: only if the kernel smoother explains at least this fraction of its held-out Poisson
#: deviance. `0.0` disables it, keeping every significant neuron.
#:
#: This is the knob to reach for when you want FEWER neurons. `FDR_Q` cannot do it: 1516 of
#: the 3011 tested neurons sit exactly on the p resolution floor (1/201 = 0.0050), so Benjamini-
#: Hochberg rejects either all of them or none — below q = 0.0099 the significant population
#: drops straight from 1516 to 0. Significance also stops being informative once it saturates;
#: `dev` keeps discriminating, because it measures HOW STRONGLY the stimulus drives the
#: neuron rather than how confident we are that it does at all.
#:
#: It stays model-agnostic — `dev` comes from the kernel smoother, not from any tuning model
#: — so raising it does not prejudge which model wins in figures 3 and 4.
#:
#: For scale, among the 1339 neurons currently selected: median dev = 0.077, quartiles
#: 0.040 / 0.171, 90th percentile 0.310, max 0.749. Without this floor (`MIN_DEV = 0`)
#: 1801 neurons would pass.
MIN_DEV = 0.02

# ══════════════════════════════════════════════════════════════════════════════
# THE STATISTIC — how the p-value above is computed  (changing these needs a rebuild)
# ══════════════════════════════════════════════════════════════════════════════
#: Width of the exchange-symmetric Gaussian kernel smoother on the (theta_0, theta_1) torus.
#: Narrower sees finer structure and generalises worse; wider is the reverse.
BANDWIDTH_DEG = 30.0

#: Cross-validation of the smoother's held-out Poisson deviance explained.
N_FOLDS = 5
N_CV_REPEATS = 10  # re-randomise the folds this many times and average

#: Permutations of the firing rates across a neuron's own trials, forming the null. The
#: smallest p-value obtainable is 1/(1 + N_SHUFFLES), so this has to sit comfortably below
#: FDR_Q — at the old value of 50 the BH cutoff landed exactly on the 1/51 = 0.0196 floor,
#: i.e. the test was resolution-limited rather than measuring anything finer. It is the
#: expensive knob: the null is recomputed this many times per neuron.
N_SHUFFLES = 200

#: A neuron with fewer 2-object trials than this is never tested, so it can never pass. This
#: is the trial-count floor on every population figure, and now the only one: the criterion is
#: computed from 2-object trials alone, so nothing about a neuron's 1-object trial count
#: decides whether it enters a population. (It used to. See MIN_1OBJ_TRIALS below.)
#:
#: Raising it is not free: it is applied when `modulation.csv` is BUILT, so a change here
#: needs `python run_modulation_cache.py --force` before it reaches any figure. It also moves
#: the Benjamini-Hochberg correction, which is computed over exactly the neurons tested — a
#: higher floor means fewer tests and a slightly less severe correction.
MIN_2OBJ_TRIALS = 300

#: Seeds the folds and the permutations, so a rebuild reproduces the same p-values.
SEED = 0

# ══════════════════════════════════════════════════════════════════════════════
# TRIAL SELECTION — which trials enter, before any neuron is scored
# ══════════════════════════════════════════════════════════════════════════════
#: Only completed trials — the animal held fixation through the delay and responded.
COMPLETED_ONLY = True

#: The east-west antipodal 2-object pair (one object at 0 deg, the other at 180 deg) was
#: presented of the order of a thousand times per session and is dropped everywhere. A match
#: this close, in either object order, counts as that repeated pair; a genuinely different
#: near-antipodal stimulus (3 deg and 179 deg, say) is kept. There is no switch to keep it:
#: it is a property of the stimulus set, not an analysis choice.
EAST_WEST_TOL_DEG = 0.05

#: Figure 1 only — responses that never left the centre are treated as uncommitted rather
#: than as errors. In units of the display, where the stimulus ring has radius 0.35.
MIN_RESPONSE_R = 0.25

#: Figure 1's repulsion analysis — beyond this the response reports the *other* object,
#: which is a swap rather than a biased estimate of the target, and is analysed separately.
MAX_ERROR_DEG = 30

#: The post-hoc on-target criterion: a response is scored **correct** when its stored endpoint
#: lands within this Euclidean distance of the cued object, in display units. This is what
#: `figure_1/responses_aligned.ipynb` labels its points by, via `figure_1/common.add_correct`.
#:
#: It is scored offline and so is not identical to the reward the animal received: the online
#: window was applied to the running eye trace under real-time constraints, while the stored
#: `response_position` comes from a slightly better offline saccade detection. Over the trials
#: that figure plots the two agree on ~99.4%; the residue is trials sitting within a hair of
#: the boundary either way. The notebook prints the comparison every time it runs.
#:
#: It was 0.18, which disagreed with the old upstream `load_ring_data.py` (0.2) — so 0.2 it is,
#: which is also the closest single cut to the record (0.197 is marginally closer).
REWARD_THRESHOLD = 0.2

# ══════════════════════════════════════════════════════════════════════════════
# MEASUREMENT PREREQUISITES — not population selection
# ══════════════════════════════════════════════════════════════════════════════
#: `rf_orientation` plots a 2-object phase measured *relative to* each neuron's 1-object
#: receptive field, so a neuron whose 1-object tuning fit is undetermined would contribute a
#: phase relative to a random direction. This is a requirement of that measurement, not a
#: statement about which neurons are modulated — which is why it is separate from the
#: criterion above. `None` switches it off.
P_THRESH_1OBJ = 0.05

#: `rf_orientation` again — each ROW of that figure shows the 2-object phase under ONE
#: model, so a neuron whose fit *under that row's model* is undetermined has no phase to
#: place. It is per-row and model-specific, which is exactly why it is a prerequisite of the
#: measurement and not the population criterion: it never decides who is in the population,
#: only who has a measurable phase on a given axis. `None` switches it off.
P_THRESH_2OBJ = 0.05

# `MIN_R2_1OBJ` and `MIN_R2_2OBJ` lived here until 2026-09-11: effect-size floors on the
# parametric fits' R-squared, the `MIN_DEV` idea applied to `rf_orientation`'s prerequisites.
# They went with `figure_3_deprecated/`, whose `stats_lib` was their only consumer paper-wide.
# `rf_orientation` has no R-squared to floor; its prerequisite is a Wald test on the fitted
# tuning vector, and a precision floor in degrees was measured against it and removed as
# inert.

#: `rf_orientation` measures a 2-object phase **relative to** each neuron's 1-object
#: receptive field, so it needs enough 1-object trials for that field to be worth measuring
#: from. It is a prerequisite of *that measurement*, not a population criterion — which is
#: where it now lives.
#:
#: It used to gate the modulation cache instead, and so silently decided who entered every
#: population figure. That was wrong twice over: the 2-object criterion never touches
#: 1-object data, and the gate removed 282 neurons that had ample 2-object trials (up to 528)
#: purely for having fewer than 100 1-object ones.
MIN_1OBJ_TRIALS = 100

#: `rf_per_anchor` (figure 2) fits one tuning curve per anchor condition; an anchor with
#: fewer contributing trials than this gets no fit.
MIN_TRIALS_PER_ANCHOR = 5

#: The swap/lapse mixture in figure 1 is fit per session; a session with fewer completed
#: 2-object trials than this is not fit. (None currently falls below it.)
MIN_TRIALS_PER_SESSION = 50


# ══════════════════════════════════════════════════════════════════════════════
# RELOADING — so an edit to this file reaches a notebook that is already running
# ══════════════════════════════════════════════════════════════════════════════
def reload():
    """Re-read this file into the live kernel, and return the refreshed module.

    Python caches imported modules, so editing `config.py` does **nothing** to a kernel that
    has already imported it — a notebook cell re-run after the edit keeps using the old
    numbers, silently. `importlib.reload` re-executes this file into the *same* module
    object, so every module holding `import config` sees the new values immediately, with no
    kernel restart.

    That only works because nothing downstream copies a value out of here at import time.
    Every consumer reads `config.X` at call time (`neuron_selection`, `trial_filters`, each
    figure's `common`), and the functions that take a threshold as an argument default it to
    `None` and resolve it against this module when they run. Keep it that way: a
    module-level `FOO = config.FOO`, or a default argument `def f(x=config.FOO)`, freezes the
    value at import and reintroduces exactly the staleness this exists to prevent.

    `common.set_style()` calls this, so any notebook whose first cell calls it picks up an
    edit to this file on the next run of that cell.
    """
    import importlib
    import sys

    return importlib.reload(sys.modules[__name__])

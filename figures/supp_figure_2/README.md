# `supp_figure_2/` — the same results, split by task phase

Figures 3 and 4's population results computed **once on the stimulus period and once on the
delay period** instead of over the whole trial. The companion to `supp_figure_1/` (split by
brain area) and `supp_figure_9/` (single units only): the same five results, drawn the same
way. The question: is the relational code present while the objects are on screen, or only
once they have to be held in memory — and does the ranking of the models change between the
two?

    model_wins_and_probability_{stimulus,delay}.pdf   figure 3's model comparison
    rf_orientation_{stimulus,delay}.pdf               figure 3's 2-object phase vs the 1-object RF
    figure_model_comparison_{stimulus,delay}.pdf      figure 4's delta-elpd model comparison
    figure_relationality_{stimulus,delay}.pdf         figure 4's distribution of relationality
    figure_ks_stats_{stimulus,delay}.pdf              figure 4's KS test of each model as a generator

## The two windows

Every figure-3 and figure-4 fit reads a per-trial rate that is the mean of two cached columns,
`stimulus_fr` (spikes 0–1000 ms after stimulus onset, objects visible) and `delay_fr`
(1000–2000 ms, objects gone; the cue comes later and is central, so nothing about the objects
is on screen). The main figures fit that mean — the 2-s stimulus + delay window. Here each
phase is fit on its own column. Nothing new is extracted from the spikes: both columns are
already in the shared firing-rate caches.

**This is a refit, not a subset**, which is the difference from the other two supplements: a
different window is different data, so every model is fit again per phase. The libraries are
the main figures' own; three module-level switches, set by `common.use_phase`, select the
window:

| switch | default | per phase |
|---|---|---|
| `figure_3/model_comparison/stats_lib.FR_COLUMNS` | both columns | one |
| `figure_4/data_lib.FR_COLUMNS` | both columns | one |
| `figure_4/data_lib.DURATION_S` | 2.0 s | 1.0 s |

The duration matters for figure 4 only: its Poisson likelihood turns a rate into a spike count
with it, and a 1-s window scored as 2 s would double every count. Figure 3's estimators are
ordinal and have no duration. (`models_lib.poisson_ll_per_trial` read the duration as a
default argument bound at import, which no switch could reach; it now reads it at call time.
With the defaults nothing in the main figures changes.)

`use_phase` also repoints every cache path at `cache/figures/supp_figure_2/{phase}/`, so a
phase fit can never be written to, or read from, the main figures' caches.

## Order of operations

The shared 2-object modulation criterion is computed **once, on the 2-s window**, and applied
first — the same order `supp_figure_1` and `supp_figure_9` use, for the same reason: the
Benjamini-Hochberg correction runs over every neuron tested, so who is included must not
depend on how the data are later split. Both phases are therefore drawn over **the same
neurons as the main figures**, and a difference between the two panels is a difference between
phases, not between populations. A neuron selective in only one phase is in both panels.

Everything downstream is recomputed within each phase: every fit, which model wins, the
per-neuron posterior, the reference model and every elpd difference, `rf_orientation`'s
phase-determinacy cut (and the 1-object receptive field it is measured against, fit in the
same window), and the KS test's synthetic distributions, which are simulated over the same
1-s window the fits were.

**Reading figure 4's panels across phases.** A 1-s window holds half the spikes of the 2-s
one, so per-trial likelihood differences are smaller and noisier than the main figure's.
Compare the *ordering* of the models, not their distances from the reference.

## Building the caches

    python build_cache.py                             # everything; resumable
    python build_cache.py --sessions 2 --stride 12    # smoke test, one session per monkey

| stage | what | population | checkpoint |
|---|---|---|---|
| `comparison` | figure 3's sign-only comparison, 1000 bootstrap replicates | criterion-passing | per session |
| `phase` | figure 3's 2-object phase fit, 50 bootstraps | criterion-passing | per session |
| `fits` | figure 4's fits: the five compared models, `relational_bias`, `constant` | criterion-passing | per neuron |
| `recovery` | figure 4's KS synthetic distributions, 5 replicates | criterion-passing | per neuron |

Measured on 8 workers, per phase: `comparison` 13 min, `phase` ~25 min (one session per
worker, so the three largest Monkey P sessions set the time), `fits` 55 min (~20 s/neuron),
`recovery` 30 min. About 4 h for both phases. Run stages separately (`--stage fits --phases
delay`) when a single invocation has to stay under a time limit; each one resumes.

Checked on 2026-10-05: every stage, in both phases, holds exactly the main figures' 1,339
`(subject, session, unit)` keys, and the KS notebook reads every synthetic row from the cache
(33,475 per phase: 1,339 neurons × 5 models × 5 replicates) and computes none.

Figure 3's settings are the ones the main caches were built with, read off their `fit_tag` and
`param_*` columns. Figure 4's `softmax` and `priority_map` are in the main cache but in no
figure here, so they are not refit.

**Why the script parallelises instead of the libraries.** The switches are module globals,
and a joblib worker imports the libraries afresh with the defaults — the 2-s window —
whatever the parent set. So every library call that would fan out is made with one job, from
inside a worker that has called `use_phase` itself, and every worker reports the switches it
ran with; a stage refuses to use results from a worker that did not see the phase. The same is
why `figure_ks_stats.ipynb` runs with `N_JOBS = 1`: it should only ever read the cache.

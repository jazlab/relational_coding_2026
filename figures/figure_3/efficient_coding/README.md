# `efficient_coding/` — the titration result

Is the recorded population's mix of retinotopic and relational neurons near the mix that best
supports reading out a two-object stimulus?

**Open `titration.ipynb`.** It is the whole of this directory's output: the titration curve, and
the one test of whether any composition decodes better than the recorded one. Everything else
here exists to produce the CSV it reads.

## Method

Sub-populations are drawn from the whole pool by an exponential tilt on each neuron's
within-animal **rank** of ρ: `w ∝ exp(λ · rank)`, K distinct units without replacement by Gumbel
top-k. The sweep variable *x* is the **target weighted mean rank**, with λ(x) inverting
`m(λ) = 1/(1 − e^−λ) − 1/λ`. Ranks are uniform on [0, 1] by construction, so that inverse uses no
data — one λ grid serves both animals, and **λ(0.5) = 0 exactly**. x = 0.5 *is* the unbiased
population, which is what makes "could the animal decode better by shifting its mix?" a test
against a landmark fixed by construction rather than estimated.

The pool is every neuron passing the shared modulation criterion of `figures/config.py` whose
retinotopic and relational models both beat chance out of sample (`min_fit=0.0`): 1239 neurons.

Decoder: bump read-out, hidden [64, 64], 500 Poisson inputs, 36 rotation replicas, 100 repeats
per sweep point. K — distinct units per sweep point — is 186 (Monkey P) / 62 (Monkey E), the
smallest wedge over the whole sweep, and is **derived, not chosen**: every point is drawn at that
size, so no two points on a curve differ in neuron count.

## Result

The figure is the two animals' curves on one axis, both with an interior minimum. The notebook's
verdict cell asks the direct question — *could the animal decode better by shifting its mix?* —
by testing every sweep point against x = 0.5 (Welch, Holm-corrected within animal):

| | Monkey P (n = 928, K = 186) | Monkey E (n = 311, K = 62) |
|---|---|---|
| points significantly **worse** than unbiased | 5 of 8 | 4 of 8 |
| points significantly **better** than unbiased | **0 of 8** | **0 of 8** |
| best challenger | x = 0.4, by 0.46% of its error, Holm p = 1.000 | x = 0.6, by 1.08%, Holm p = 0.519 |
| error at the unbiased point | 10.09° | 13.28° |
| pool mean ρ | 0.573 [0.559, 0.587] | 0.545 [0.522, 0.568] |

**No sweep point is significantly better than the unbiased population in either animal.** That is
the claim, and it is the one the notebook prints. Every number above comes out of a cell in
`titration.ipynb`.

## Files

    titration.ipynb     the result: the curve, and the test against the unbiased population
    titration.pdf       the figure it writes
    titration_lib.py    the λ solver, the curve statistics, the figure, the caching
    regenerate.py       refits the titration from scratch (also run by figures/build_caches.py)
    run_titration.py    the sweep driver that regenerate.py calls
    pool_lib.py  decoder_lib.py            the pool, the response tables, the read-out

Caches, under `cache/figures/figure_3/efficient_coding/` at the repository root:

    dataframes/         the fit: results.csv, written by regenerate.py
    derived/            derived quantities only, built by the notebook; safe to delete

## Re-running

The notebook never refits. Its own expensive steps — the neuron pool and the curve's realized-x
table — are cached in `derived/`, so **changing `INSET_SEED` redraws the example insets and touches
nothing else**. `titration_lib.clear_cache()` drops the derived files and leaves the fit alone.

To refit from scratch (~25 min with the default 10 workers):

    python regenerate.py              # --jobs N, --dry-run

Response tables are cached under `cache/figures/figure_3/pool_tables/`, so only the first run
pays the 127 ms/unit to build them. The fit is deterministic in its seeds: re-running reproduces
`results.csv` value for value.

## What this does not establish

* the `cv_r2` gradient along ρ is untouched;
* independent Poisson noise, and the task's own stimulus distribution;
* capacity is probed only downward — the decoder sits ~3× above the Cramér–Rao bound, so part of
  a difference between compositions could be learnability rather than information;
* the ρ coordinate was chosen among several candidate coordinates (a forking path), and the
  convex result depends on using the fit-strength angle rather than the model probability;
* the tilt still confounds mean composition with ρ-*spread*, though more weakly than a hard rank
  wedge would; only a two-parameter tilt would remove it;
* only the *sampling family* is fixed here; the read-out's objective is not varied;
* the curve is not tested against a shuffled-ρ null. What stands is the verdict above.

# `supp_figure_9/` — the same results over single units only

Figures 3 and 4's population results recomputed over **well-isolated single units**, pooled
across recording sites. The complement to `supp_figure_1/`, which splits the same four results
by brain area and keeps every unit. The question: is any of this carried by multi-unit
clusters, whose spikes may come from several neurons with different tuning?

    model_wins_and_probability_good.pdf   figure 3's model comparison
    figure_model_comparison_good.pdf      figure 4's delta-elpd model comparison
    figure_relationality_good.pdf         figure 4's distribution of relationality
    figure_ks_stats_good.pdf              figure 4's KS test of each model as a generator

## The label

Each unit in the spike-sorting NWB files carries a `quality` label, **`good`** (a well-isolated
single unit) or **`mua`** (a multi-unit cluster). Nothing downstream carries it — not
`phase_units.csv`, not the shared `firing_rates/` tables, not figure 4's fit summaries — so
`common.unit_quality()` reads it from the NWB unit tables (via `phys_processing/spike_counts.py`)
and caches it as `cache/figures/shared_inputs/unit_quality.csv` (3,186 units: 1,181 good, 2,005
mua). The join is only safe if `(subject, session, unit)` is unique; it is, and the builder
re-checks rather than assuming.

## Order of operations

The shared 2-object modulation criterion is applied **first**, over the whole population, and
the single-unit cut second — the same order `supp_figure_1` uses for its area split, and for the
same reason: the Benjamini-Hochberg correction runs over every neuron tested, so which neurons
pass must not depend on how they are later split. Everything that is a property of the
population actually drawn *is* recomputed within the cut: which model wins, the per-neuron
posterior over the figure-3 models, the reference model and every elpd difference, every
interval drawn or printed, the bootstrap resampling.

**672 of 1,339 neurons (50%) are single units** — 556 of Monkey P's 998 and 116 of Monkey E's
341. Monkey E's panel is therefore the thin one; at 116 neurons it is comparable to
`supp_figure_1`'s Elgar/DMFC panel (86).

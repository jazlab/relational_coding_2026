# Cleanups before open-sourcing

Proposed simplifications to `behavior_processing/` and `phys_processing/`, so that a user
can go from the DANDI data to the figures without friction.

**Status (2026-10-13):** the `behavior_processing/` items (except shipping `ring.csv`) and
all `phys_processing/` items are done; see "What was done" at the bottom. The table below
describes the layout *before* the cleanup.

## What the figures actually read

The figures need only two preprocessing outputs: `cache/behavior_processing/ring.csv` and
`cache/phys_processing/spikes_to_trials/spikes_per_trial/`. Nothing in `figures/` or
`task_visualization/` reads anything else that `behavior_processing/` or `phys_processing/`
produces.

| Output | Read by |
|---|---|
| `ring.csv` | figure_1, figure_2, figure_3, figure_4 |
| `spikes_per_trial/` | figure_2, figure_3, figure_4, supp_figure_9 (supp 9 only walks the folder tree to recover good/mua labels) |
| `dandi_data/behavior/*.nwb` (opened directly) | `figure_1/trial_structure.ipynb`, `supp_figure_7/binding_capacity/stimulus_frame.ipynb`, both `task_visualization` demos |
| `dandi_data/spikesorting/*.nwb` (opened directly) | supp_figure_8, `task_visualization/demo_replay_trials_with_neurons.ipynb` |
| `selectivity/`, `receptive_fields/`, `*_phase_firing_rates/`, `psths/`, `unit_summary_plots/` | nothing |
| `ring_before_reward.csv`, `load_ring_data.py` | nothing (one comment in `figures/config.py` mentions `load_ring_data`) |

## behavior_processing/

- [x] **Merge `add_reward_to_ring.py` into `run_cache_data.py`.** It patches in three
      columns (`reward_duration`, `reward_time`, `correct`) that the main script could read
      directly from the NWB trials table. Its "in-place, row for row" design protects
      existing caches, but a public user builds from scratch and doesn't need that. Then
      delete `cache/behavior_processing/ring_before_reward.csv`.
- [x] **Fix the write path.** `run_cache_data.py` writes to `../cache/behavior/`, but every
      consumer reads `cache/behavior_processing/`. Several phys scripts have the same
      mismatch and read `cache/behavior/ring.csv` (e.g.
      `phys_processing/spikes_to_trials/run_spikes_per_trial.py:33`,
      `phys_processing/psths/run_ring.py:32`, `selectivity/run_cache_selectivity.py`). As
      written, the documented pipeline can't run end to end.
- [x] **Delete `load_ring_data.py`.** Its 0.2 distance-threshold `correct` is the
      reconstruction that `add_reward_to_ring.py` exists to replace.
- [x] **Drop the leftover triangle-task code:** the `object_2_*` columns, the 3-object skip,
      and the `num_objects` loop up to 3. `download_dandi_data.py` already fetches only ring
      sessions.
- [x] **Speed (minor):** `_get_session_df` builds one DataFrame per trial and concatenates
      each into the table, which is quadratic. Build a list of dicts and make one DataFrame
      at the end.
- [ ] **Consider shipping `ring.csv` directly.** It is 29 MB, while the behavior NWBs are
      5.8 GB (~200 MB per session), and only 3–4 notebooks open them.

## phys_processing/

- [x] **Delete `selectivity/` and `receptive_fields/`.** They are byte-identical copies of
      each other, and no figure uses them; the paper's neuron selection now comes from
      `modulation.csv` via `figures/neuron_selection.py`. They would also crash on NumPy ≥ 2.0,
      which removed `np.trapz` (`auc.py:23`).
- [x] **Delete `spikes_to_trials/run_firing_rates_per_phase.py`.**
      `figures/figure_4/data_lib.build_firing_rates` replaced it. It is also internally
      inconsistent: `_PHASE` defaults to `"stimulus"`, but the docstring says "delay", and
      `selectivity/` reads `delay_phase_firing_rates`.
- [x] **Move `psths/` and `unit_summary_plots/` to an optional "explore the data" section**
      (or drop them). They make browsing PDFs and no figure uses them; figure 2 has its own
      `psth_lib.py`.
- [x] **Replace the `spikes_per_trial` cache. This is the biggest win.**
  - It is 6,372 pickles totalling 7.9 GB, each holding dense 10 ms bins as Python float
    lists (~3 MB per unit).
  - The spike-sorting NWBs it comes from total only 1.8 GB.
  - Option A: one compressed integer array per session (likely ~100× smaller).
  - Option B (simpler): a shared `load_spike_counts(subject, session, unit)` that bins spike
    times straight from the NWB, using `obs_trials`, `completed`, and the trial times from
    `ring.csv`. Binning one unit takes milliseconds, so the preprocessing step disappears.
  - Either option also stores the good/mua label as a column, so supp_figure_9 no longer
    needs to walk folders to recover it.
  - Before switching, check the new counts against the existing pickles. In particular,
    `_get_spike_counts` keeps a spike exactly at `trial_end` and drops one exactly at
    `trial_start`.
- [x] **Define the timing constants once.** `BIN_SIZE=10ms` and `LEAD_IN=200ms` are set
      separately in `phys_processing/spikes_to_trials/constants.py`,
      `figures/figure_2/common.py`, and `figures/figure_4/data_lib.py` (whose comment reads
      "must match the upstream spike-count pipeline").

## Problems on the figures/ side

Outside the two preprocessing directories, but a user hits them in the same pipeline.

- [x] **`modulation.csv` can't be rebuilt from this repo.** `figures/neuron_selection.py`
      says it is written by "the `--modulation` stage of `run_cache.py`", and
      `figures/config.py` says `rebuild_modulation.py`. Neither exists, and
      `figure_4/run_cache.py` has no such stage. Every population figure depends on this
      file.
- [x] **Same for `cache/figures/shared_inputs/firing_rates/`.**
      `figure_3/model_comparison/stats_lib.py` says to run `run_phase_cache.py` to build it,
      but that script's docstring says it does *not* rebuild it. Its content is
      `cache/figures/figure_4/neuron_models/firing_rates/` minus the two identity columns,
      with identical row counts. That makes it a redundant second copy with no builder.
      Point both readers at one cache.
- [ ] **Unused files to leave out of the OSF bundle:**
      `cache/figures/_removed_old_supp_figure_8_modulation/`,
      `shared_inputs/modulation_nshuffles*.csv`, and `shared_inputs/modulation_parts/` (only
      `figures/docs/make_cache_doc.py` mentions them). `figure_3/efficient_coding/titration_old.ipynb`
      looks like a leftover too.

## Cross-cutting

- [ ] **Paths:** every script uses CWD-relative paths like `Path("../../cache")`, so it only
      works if run from its own folder. Use paths relative to the script's own location, as
      `add_reward_to_ring.py` already does, or one shared `paths.py`.
- [x] **DANDI key** (done 2026-10-13): `download_dandi_data.py` requires `DANDI_API_KEY.txt`, but dandiset
      000620 is public, so the token can be optional. That removes a setup step and the
      temptation to commit a key. The real key file is in the repo root; it's gitignored,
      but double-check before `git init` and pushing.
- [ ] **requirements.txt:** `pynwb`, `pandas`, `numpy` and `matplotlib` arrive only as side
      effects of other installs. List them explicitly with tested versions.

## Suggested end state

```
python download_dandi_data.py --modality=spikesorting   # behavior NWBs optional
python behavior_processing/build_ring_table.py          # ring.csv, reward included
python figures/build_caches.py                          # firing rates -> modulation -> fits
```

`phys_processing/` shrinks to one spike-count loader (or one small script), plus an
optional `explore/` folder for the PSTH and unit summary plots.

## Decide before cutting

- [ ] **Methods text:** does it still describe the AUC selectivity criterion? If so, keep one
      copy of `selectivity/` rather than deleting both.
- [ ] **Check before swapping:** confirm a regenerated `ring.csv` matches the current one row
      for row, and that the new spike counts match the existing pickles.

## What was done (2026-10-13)

- `behavior_processing/run_cache_data.py` now writes `cache/behavior_processing/ring.csv`
  directly, including `reward_duration`, `reward_time` and `correct`. Paths are relative to
  the script, the triangle columns are gone, and it builds the table in one pass (~5 s).
  Its output was checked against the previous `ring.csv`: identical in every shared column;
  the only difference is the three dropped all-NaN `object_2_*` columns. The cached
  `ring.csv` was replaced with this output.
- Deleted `add_reward_to_ring.py`, `load_ring_data.py`, `ring_before_reward.csv`.
- `phys_processing/` is now one module, `spike_counts.py`, which bins spikes on the fly from
  the spike-sorting NWBs (option B above). Checked against every cached pickle: 3,186 units,
  3,344,332 trials, 0 mismatches. `figure_2/common.py`, `figure_4/data_lib.py` and
  `supp_figure_9/common.py` use it; `BIN_MS`/`LEAD_IN_MS` are defined only there. Figure 4's
  firing-rate files and supp 9's `unit_quality.csv` rebuild byte-identically.
- Deleted `selectivity/`, `receptive_fields/`, and `spikes_to_trials/`.
- Moved `task_visualization/`, `psths/` and `unit_summary_plots/` into `data_exploration/`.
  Their outputs now go to `cache/data_exploration/`. `figures/` points at the new renderer path.
- `cache/phys_processing/spikes_to_trials/spikes_per_trial/` (7.9 GB) is no longer read by
  anything and can be deleted.

## Figures-side fixes (2026-10-13)

Goal: with only the DANDI data and this code, every cache can be rebuilt and every figure
except supp_figure_8 (which needs the MRI scans) reproduced.

- **One firing-rate table.** `figures/firing_rates.py` builds
  `cache/figures/shared_inputs/firing_rates/` (with the object-identity columns) and is read by
  both figure 3 (`stats_lib`) and figure 4 (`data_lib`). Missing sessions are built on demand,
  so figure 3's builders and figure 4's `run_cache.py` can no longer run on a partial table.
  Rebuilt files match both old copies (22/22 sessions); figure 3's trial arrays are unchanged.
  `cache/figures/figure_4/neuron_models/firing_rates/` was deleted.
- **`modulation.csv` builder restored** as `figure_3/model_comparison/modulation_lib.py` +
  `run_modulation_cache.py`, ported from `multi_object_memory_paper` commit `c1c13a9`. A full
  rebuild reproduces the shipped table exactly (3,011 neurons, every column, 4.6 min on 10
  workers).
- **`figures/build_caches.py`** runs every builder in order (`--dry-run`, `--from`, `--only`).
- **Spot checks of the other builders on Perle 2022-06-07:** figure 3's comparison (bootstrap
  draws identical; recombined table equal to 6e-13 relative) and phase cache (identical), and
  figure 4's fits (identical up to 4e-31). supp_figure_2's builder was import-checked only.
- **Comparison bootstrap.** The shipped comparison cache has replicates 0–999, but
  `comparison_lib.N_BOOT` is 2000, so a default rebuild would not reproduce it.
  `build_caches.py` passed `--boot-range 0:1000`, which reproduces it exactly. Later the same
  day `N_BOOT` was lowered to 1000, so a default rebuild now matches without the flag.
- `figure_1/trial_structure` and `supp_figure_7/.../stimulus_frame` now use a ring trial (see
  TODO.md); stale "run `run_cache.py`" / `rebuild_modulation.py` messages were corrected.
- Not needed by any figure (safe to leave out of a cache bundle):
  `cache/figures/_removed_old_supp_figure_8_modulation/`, `shared_inputs/modulation_nshuffles*.csv`,
  `figure_3/model_comparison/{paired_bootstrap,separation_grid}_validation.csv`,
  `figure_4/neuron_models/{baseline_future_field_preflight.csv,priority_map_run.log}`.

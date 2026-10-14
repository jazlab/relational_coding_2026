# Context for Claude sessions

This repo holds the code for a paper by Nick Watters et al. (2026): neurophysiology from
monkey frontal cortex (DMFC and FEF; monkeys Perle = "Monkey P", Elgar = "Monkey E") during
a 1- and 2-object working memory task on a ring ("ring task"), plus computational models of
it. **The goal right now is to open-source it** so that a user can download the data from
DANDI (dandiset 000620), rebuild every cache, and reproduce every figure, and so the process
is smooth.

The directory was renamed from `watters_relational_neurons_2026/` to `relational_coding_2026/`
on 2026-10-13. It is not a git repo yet.

## Read these first

- `README.md`: the public-facing instructions (setup, DANDI download, exploring the data,
  rebuilding caches, the figure-by-notebook table). Rewritten 2026-10-13.
- `TODO.md`: the user's open task list.
- `CLEANUPS.md`: the audit of the preprocessing code, what was changed, and how each change
  was verified.

## Layout and data flow

```
DANDI NWB files (cache/dandi_data/{behavior,spikesorting}/, via download_dandi_data.py)
  ├─ behavior_processing/run_cache_data.py  -> cache/behavior_processing/ring.csv (1 row/trial)
  └─ phys_processing/spike_counts.py        -> per-trial 10 ms spike counts, binned on the fly
                                               from the NWB (a library; no cache)
figures/firing_rates.py                     -> cache/figures/shared_inputs/firing_rates/
figures/figure_3/model_comparison/run_modulation_cache.py
                                            -> cache/figures/shared_inputs/modulation.csv
                                               (neuron selection, read by figures/neuron_selection.py)
figure_3 run_comparison_cache.py / run_phase_cache.py, figure_4 run_cache.py,
supp_figure_2 build_cache.py                -> cache/figures/<figure>/...
figures/build_caches.py                     runs all of the above in order (--dry-run, --from, --only)
```

- `figures/`: one directory per figure (figure_1..4, supp_figure_1..10) with notebooks and
  `*_lib.py` modules. `figures/config.py` holds every neuron/trial selection threshold.
- `data_exploration/`: `task_visualization/` (trial replay gif/video; its renderer is also
  imported by figure_1 and supp_figure_7, and its `resources/` by figure_4's `contrast_lib`),
  `psths/`, `unit_summary_plots/`. Not needed for figures.
- Other caches (supp_figure_4 decoders, supp_figure_5 profiles, pool tables, supp_figure_9's
  `unit_quality.csv`, ...) build lazily when a notebook first needs them.
- supp_figure_8 (MRI figure): its code is in `gitignore/supp_figure_8/` (gitignored, not
  published); `figures/supp_figure_8/` keeps only `images/` and the `.ai`/`.png`. It needs `cache/mris/`, which is NOT on DANDI; it will come from an
  OSF bundle. It is the one figure exempt from "rebuild from DANDI only".

## How to work here

- Python env with everything installed: `/opt/miniconda3/envs/wm_paper/bin/python` (3.12).
  It lacks `nbclient`/`nbconvert` and `kaleido`. `requirements.txt` was rewritten from the
  actual imports but has not been test-installed in a fresh env.
- Run scripts and notebooks from their own directory (notebooks use cwd-relative paths).
- **Do not re-run the whole pipeline end to end** (the user said so). Verify changes with
  targeted checks: rebuild one session (Perle 2022-06-07 is the smallest, 14 units) or one
  table into the scratchpad and compare against the existing cache.
- Before replacing a cache file or deleting code, show the replacement reproduces the old
  result. Every change so far was verified this way (numbers are in CLEANUPS.md).
- Don't overwrite the saved figure PDFs or notebook outputs without asking. They still come
  from the old repo, and the user updates the Illustrator `.ai` files by hand. To test a
  notebook, execute its code cells with its output path redirected to a scratch directory.
- `DANDI_API_KEY.txt` in the root is a real secret (gitignored). Never print or commit it.
- History: this code was copied from `../multi_object_memory_paper/` (a git repo, mostly
  `figures_v2/`). Deleted code can be recovered there with `git show`; for example the
  modulation builder came from commit `c1c13a9`.

## Done on 2026-10-13 (all verified; details in CLEANUPS.md)

- `ring.csv` builder: reward columns merged in, triangle-task code and wrong paths removed.
  Output is identical to the old table.
- `phys_processing/` reduced to `spike_counts.py`. It matches all 6,372 old spike-count
  pickles exactly. `selectivity/`, `receptive_fields/` and `spikes_to_trials/` were deleted
  (unused by figures).
- `task_visualization/`, `psths/` and `unit_summary_plots/` moved into `data_exploration/`.
- One shared firing-rate table, `figures/firing_rates.py`, read by figures 3 and 4. It
  builds missing sessions on demand.
- `modulation.csv` builder restored (`modulation_lib.py` + `run_modulation_cache.py`). A full
  rebuild reproduces the shipped table exactly.
- `figures/build_caches.py` added. Figure 3's comparison and phase builders and figure 4's
  fitter were spot-checked on one session and reproduce the cache. supp_figure_2's
  `build_cache.py` was only import-checked.
- `figure_1/trial_structure.ipynb` and `supp_figure_7/binding_capacity/stimulus_frame.ipynb`
  now use Perle 2022-06-11 trial 200 (the old trial was from a non-ring session). Both run.
- README, `requirements.txt`, TODO.md and CLEANUPS.md updated; README gif generated at
  `data_exploration/task_visualization/gifs/Perle_2022-06-11_trial_200.gif`.

## Done later on 2026-10-13

- Bootstrap: `comparison_lib.N_BOOT` lowered to 1000 (what the shipped cache uses);
  `build_caches.py` no longer passes `--boot-range`. The shipped `comparison_units.csv` still
  says `param_n_boot = 2000` (metadata only; its p-values come from 1000 replicates).
- DANDI key optional in `download_dandi_data.py` (env var or file). The env's `dandi` 0.71.3 is
  rejected by the server; `requirements.txt` now says `dandi >= 0.74.0`.
- LICENSE year 2026. Stale neuron counts in `config.py` / `neuron_selection.py` fixed (3,011
  tested, 1,516 at the p floor, 1,801 significant, 1,339 selected; 3,186 is the number *fit*).
- Deleted `titration_old.ipynb`, `figures/docs/` and `figures/doc_lib.py`, and
  `cache/phys_processing/` (the old spike-count pickles).
- References to the old repo removed; no code refers to the triangle dataset.
- black + isort run over all `.py` (line length 79). All 80 modules import; `test_pairs.py` passes.

## Open items

1. **README placeholders:** paper title, GitHub URL, OSF URL and archive name, and the BibTeX
   citation (search for `PAPER TITLE`, `GITHUB_URL`, `OSF_URL`, `CACHE_ARCHIVE_NAME`,
   `CITATION`, `TODO`). Ask the user; do not invent them.
2. **Figure panels to redo by the user:** see TODO.md.
3. **No figure notebook has been run since the copy from the old repo**, apart from
   `trial_structure`, `stimulus_frame` and the data_exploration demos. A targeted pass that
   runs each notebook (outputs redirected) would find any remaining breakage, including any
   caused by the black/isort pass.
4. **Absolute paths** in `figures/FORMATTING.md` and five notebook outputs: listed in TODO.md.
5. **`figure_3 copy.ai` / `.png`**: ask the user which version is current.
6. **Cache housekeeping** (ask before deleting). Not read by any figure; keep out of the OSF
   bundle: `cache/figures/_removed_old_supp_figure_8_modulation/`,
   `shared_inputs/modulation_nshuffles*.csv`,
   `figure_3/model_comparison/{paired_bootstrap,separation_grid}_validation.csv`,
   `figure_4/neuron_models/{baseline_future_field_preflight.csv,priority_map_run.log}`.
7. **Smaller cleanups:** stale markdown headers in some notebooks; test-install
   `requirements.txt` in a fresh env; scripts use cwd-relative paths.
8. **Before publishing:** `git init`, `pre-commit install`, and decide whether this `CLAUDE.md`
   ships with the repo.

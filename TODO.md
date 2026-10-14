# TODO

Left over from copying `multi_object_memory_paper/figures_v2/` here as `figures/` and
removing the triangle dataset. Nothing was re-run during the copy, so every saved figure,
PDF and notebook output is still the one made in the old repo.

## Figures that need a new example trial

The example trial in these figures came from Perle 2022-06-01, a triangle session that is
not in the ring dataset. On 2026-10-13 both notebooks were pointed at **Perle 2022-06-11,
trial 200** (a completed, rewarded 2-object trial; the same one as the README gif) and both
run. Their saved PDFs and outputs were not overwritten.

- [ ] `figures/figure_1/trial_structure.ipynb`: check you're happy with trial 200, re-run, and
      replace the panel in `figure_1.ai`.
- [ ] `figures/supp_figure_7/binding_capacity/stimulus_frame.ipynb`: re-run and update
      `supp_figure_7.ai`.
- [ ] Update any caption or methods text that names "Monkey P, session 2022-06-01, trial 41".
      (`figures/figure_1/prompts/make_docs.py` and `figures/supp_figure_7/make_methods_doc.py`,
      named here before, are not in this repo.)

## MRI probe figure (supp_figure_8)

`cache/dandi_data/spikesorting/` now has only the ring sessions, so a re-run draws fewer
probe tracks than the current figure.

- [ ] Re-run `gitignore/supp_figure_8/probes_on_mri.ipynb` and `probes_3d.ipynb`. Their saved
      outputs, `images/`, and `cache/Perle/probes_3d.gif` still show every session.
- [ ] The supp_figure_8 code moved to `gitignore/supp_figure_8/` (gitignored) on 2026-10-13;
      only `images/`, `supp_figure_8.ai` and `.png` stay in `figures/supp_figure_8/`.
      `probes_on_mri.ipynb` still saves into `figures/supp_figure_8/images/`. The top-level
      `README.md` (line 51 and the S8 row of the figure table) still points at the notebooks
      in `supp_figure_8/`; reword it once you decide how the MRI figure is presented.
- [ ] Update `supp_figure_8.ai` and any caption or methods text that counts sessions or probes.
- [ ] The MRIs now live in `cache/mris/`, copied from `../mehrdad/monkey_stuff/mris/`. Only
      the two scans `subjects.py` uses were copied.

## Task visualization and demos

All four demos now point at Perle 2022-06-11 (a ring session) and have their outputs
cleared. None of them has been run since the change.

- [x] `data_exploration/task_visualization/demo_replay_trial.ipynb`: ran it (trial 200 is a
      completed, rewarded 2-object trial) and put the gif in `task_visualization/README.md`
      and the top-level `README.md`.
- [ ] `data_exploration/task_visualization/demo_replay_trials_with_neurons.ipynb`: `UNIT_IDS` and `TRIALS`
      were chosen for the old triangle session (2022-05-31); pick units and trials from a
      ring session. It now filters to 2-object trials, since ring sessions have no 3-object
      trials. Make a new video. The old mp4 and its YouTube link were removed from the
      READMEs.
- [ ] `data_exploration/psths/demo.ipynb` and `data_exploration/unit_summary_plots/demo.ipynb`:
      unit 100 was picked without looking. Both run; choose a good unit. The old README
      images (`phys_processing/readme_visuals/`) showed triangle data and were deleted.

## Pipeline (`behavior_processing/`, `phys_processing/`)

Cleaned up on 2026-10-13; see `CLEANUPS.md`. `run_cache_data.py` was re-run and reproduces
`ring.csv`; `phys_processing/spike_counts.py` reproduces every cached spike-count pickle.

- [x] `download_dandi_data.py` downloads only the ring sessions, and the API key is now
      optional (env var `DANDI_API_KEY` or `DANDI_API_KEY.txt`); anonymous access to dandiset
      000620 was checked against the DANDI API. Note: the `wm_paper` env has `dandi` 0.71.3,
      which the DANDI server now rejects (it needs >= 0.74.0), so `pip install -U dandi` there
      before downloading. `requirements.txt` was bumped to `dandi >= 0.74.0`.
- [x] `cache/phys_processing/spikes_to_trials/` (7.9 GB) was deleted on 2026-10-13. It was an
      APFS clone of the old repo's copy, so the space comes back only once that copy goes too.

## Repo housekeeping

- [ ] The top-level `README.md` was rewritten on 2026-10-13 but has placeholders for the
      paper title, GitHub URL, OSF URL and archive name, and the citation (search for
      `TODO`, `PAPER TITLE`, `GITHUB_URL`, `OSF_URL`, `CACHE_ARCHIVE_NAME`, `CITATION`).
- [x] `.gitignore` no longer lists old-repo paths (checked 2026-10-13).
- [ ] This directory is not a git repo yet.
- [ ] `cache/figures/shared_inputs/unit_quality.csv` was filtered to ring-session units
      (8085 → 3186 rows). It rebuilds itself if deleted.
- [ ] The copied `cache/` holds only what `figures/` reads: `cache/figures/` (was
      `cache/figures_v2/`), `ring.csv` and `ring_before_reward.csv`, `spikes_per_trial` for
      the ring sessions, and the ring-session DANDI NWBs. Intermediate pipeline outputs
      (`psths/`, `selectivity/`, `unit_summary_plots/`, per-phase firing rates) were not
      copied.
- [ ] Every file was copied with APFS clones (`cp -c`), so the copy takes almost no extra
      disk space until you edit files. The disk has about 12 GB free. If you move this
      directory to another volume, it will need its full ~17 GB there.
- [x] References to the old repo (`figures_v1/`, `revisit/`, `model_comparison_v1/`) were
      removed from code, docstrings and notebook sources on 2026-10-13. No code refers to the
      triangle dataset (the remaining "triangle"s are mesh triangles and marker shapes).
- [ ] `figures/figure_3/figure_3 copy.ai` and `figure_3 copy.png` are newer than
      `figure_3.ai` / `.png` (5 Oct 19:15 vs 14:49). Keep whichever is current and delete the
      other before publishing.
- [ ] **Remove absolute paths.** Found on 2026-10-13 (outside `cache/`):
      - `figures/FORMATTING.md` lines 73 and 75 (source): `/opt/miniconda3/bin/python` and
        `/opt/miniconda3/envs/wm_paper/bin/python` as kernel names. Reword as "a Python env with
        `requirements.txt` installed".
      - Notebook **outputs** printing `/Users/nicholaswatters/.../relational_coding_2026/cache/...`:
        `figures/figure_4/figure_ks_stats.ipynb`, `figures/supp_figure_1/figure_ks_stats.ipynb`,
        `figures/supp_figure_2/figure_ks_stats.ipynb`, `figures/supp_figure_9/figure_ks_stats.ipynb`
        (the `bias_recovery/<hash>` cache path), and
        `figures/figure_3/motor_decoding_schematic/rf_response_map.ipynb` (`shared_inputs`).
        Best fix: print paths relative to the repo root in the code that emits them, then
        re-run. supp_figure_2's output also shows a stale `cache/figures/supp_figure_6/...` path.
      - `figures/figure_3/efficient_coding_schematic/decoding.ipynb` output: sklearn warnings
        that print `/opt/miniconda3/envs/wm_paper/lib/python3.12/site-packages/...`. Clear
        that output or silence the warning.
      - Re-check with `grep -rIn -E "/Users/|/opt/miniconda3" --exclude-dir=cache .`
- [x] black (24.1.1) and isort (5.13.2) were run over every `.py` file on 2026-10-13 with the
      `.pre-commit-config.yaml` settings (line length 79). Every module still imports from its
      own directory, `test_pairs.py` passes, and isort only reordered import lines. Once this
      is a git repo, `pre-commit install` keeps it that way. Notebooks were not formatted.

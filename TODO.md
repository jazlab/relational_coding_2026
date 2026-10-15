# TODO

Open work before open-sourcing this repo.

## For Nick

- [ ] OSF bundle: decide what it holds, then upload it. The choices: all of `cache/figures/`
      or only the tables the notebooks read (leaving out per-session checkpoints such as
      `phase_parts/`, `comparison_parts/`, `fit_parts/`), and whether to include `ring.csv`
      (15 MB), which spares most users the 5.8 GB behavior download.
- [ ] README: fill in `OSF_URL` and `CACHE_ARCHIVE_NAME` ("Option A: download the precomputed
      caches"), replace `CITATION (BibTeX)` ("Reference"), and delete the `<!-- TODO ... -->`
      comment at the top.
- [ ] `pre-commit install`, then commit.

## For Claude

- [ ] Check that the re-run notebooks print the same numbers as the outputs saved in git
      before the rebuild, apart from the expected changes (counts of 3186 fitted neurons now
      say 1339; supp 4's settings line lists two fewer keys; the titration verdict says
      "trails" for Monkey E).
- [ ] The saved outputs of `data_exploration/task_visualization/demo_replay_trial.ipynb` and
      `demo_replay_trials_with_neurons.ipynb` print absolute `/Users/...` paths: make the
      code print repo-relative ones, then Nick re-runs both. (The sklearn warnings in
      `figure_3/efficient_coding_schematic/decoding.ipynb`, which show the conda path, stay.)

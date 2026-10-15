# TODO

Open work before open-sourcing this repo.

## For Nick

### README
- [ ] OSF: upload the cache bundle, then fill in `OSF_URL` and `CACHE_ARCHIVE_NAME` in
      `README.md` ("Option A: download the precomputed caches").
- [ ] Citation: replace `CITATION (BibTeX)` in `README.md` ("Reference"), and delete the
      `<!-- TODO ... -->` comment at the top.

### Cache and figures
- [ ] Rebuild the caches from scratch (`figures/build_caches.py`; see the README).
- [ ] Then re-run every notebook in Jupyter, which regenerates the PDFs and the saved outputs.
      Expected changes in printed text only: lines that counted all 3186 fitted neurons now
      say 1339 (the builders fit only the neurons that pass the selection), and supp 4's
      settings line lists two fewer keys. Figures and statistics should be unchanged.
- [ ] **Check the Illustrator files.** All 14 `.ai` files link only to panels under
      `relational_coding_2026/figures/`, so they pull in the panels as they are now. Open each,
      check every panel still sits right (changed panels: figure 1 `trial_structure`, supp 7
      `stimulus_frame`/`cue_frame`, supp 8 probe images, the supp 1/2/9 KS and relationality
      panels), and re-export the matching `.png`. Pre-relink backups:
      `gitignore/ai_backup_2026-10-15/`.

### Housekeeping
- [ ] Delete the backups once `cache/` and the figures check out: `gitignore_cache/` (old
      caches), `gitignore/cache_backup_2026-10-15_models/` (figure 4's fit and bias caches
      before the softmax/priority_map rows were removed),
      `gitignore/cache_backup_2026-10-15_cleanups/` (the caches before the in-place column and
      tag migrations) and `gitignore/ai_backup_2026-10-15/`. All are gitignored and read by no
      code.

## For Claude

- [ ] After the re-run, check that no saved output contains an absolute path:
      `grep -rIn -E "/Users/|/opt/miniconda3" --exclude-dir=cache --exclude-dir=gitignore --exclude-dir=gitignore_cache .`
      (the code now prints repo-relative paths).

## Packaging
- [ ] OSF bundle: decide what it holds (see the README's "Option A"). A fresh rebuild writes
      only what the code reads, so the stale files of the old cache are gone.
- [ ] `pre-commit install`.

# Formatting conventions for every figure in `figures/`

All figure-generating notebooks under `figures/` follow the rules below.

## Style

The style lives in one place, `figures/style.py`, and each figure directory's `common.py`
re-exports it. Every notebook calls it in its first cell:

```python
import common
common.set_style()
```

`set_style()` sets Arial, 14 pt titles, 12 pt labels, ticks and legends, 300 dpi, and
`pdf.fonttype = 42` (PDF text stays editable in Illustrator). It also calls
`config.reload()`, so re-running the first cell picks up an edit to `figures/config.py`
without restarting the kernel.

**No font size below 12 pt, anywhere.** That includes text drawn inside axes (trial
counts, annotations, in-panel labels), tick labels, and anything passed an explicit
`fontsize=`. If a label doesn't fit at 12 pt, make the figure bigger or move the label —
don't shrink the text.

If a plot uses seaborn, call `sns.despine(...)` only — **do not** use
`sns.set_theme(...)`, which silently overrides the rcParams set by `set_style()` (or call
`set_style()` again after it).

## Despine

**Despine every data plot with real axes** — scatterplots, line plots, bar plots,
histograms: no top or right spine, keep left and bottom.

```python
style.despine(ax)           # figures/style.py; some `common.py` files re-export it
# or, with seaborn imported: sns.despine(fig)
```

This does not apply to schematic / spatial panels that carry no ticks (ring plots, task
frames, aligned-response clouds) — those either keep a full frame box or have no spines
at all, whichever the figure calls for.

## Saving

Every generated figure is saved as a **PDF next to the notebook that made it**, with a
descriptive file name, overwriting any previous version:

```python
common.save(fig, 'my_figure.pdf')   # savefig(..., bbox_inches='tight', pad_inches=0.01)
```

## Figure size

Each notebook exposes the figure size(s) as a **parameter in a CONFIG cell at the top**
(e.g. `FIGSIZE = (4.2, 4.6)`), one per generated figure, so the size can be retuned
without touching plotting code.

## Code organisation

- Each figure directory has a `common.py` (style, cache locations, shared constants) and,
  where needed, `*_lib.py` modules next to the notebooks that use them.
- Code used by several figures lives directly in `figures/`: `style.py`, `config.py`
  (every neuron and trial selection threshold), `neuron_selection.py`, `trial_filters.py`
  and `firing_rates.py`. Figure directories import these by putting `figures/` on
  `sys.path`.
- A figure may also import another figure's library rather than copying it (for example,
  supplementary figures 1, 2 and 9 build on `figure_3/model_comparison/` and `figure_4/`).

## Paths / kernels

Every shared location comes from `paths.py` in the repository root; never write a `../`
path. A notebook that needs one finds `paths.py` by walking up from its folder (see the
`paths.py` docstring for the four lines), then uses e.g. `paths.RING_CSV` or
`paths.FIGURES / 'figure_3'`. Figures are saved next to the notebook by bare file name.

Every notebook runs on a Python env with `requirements.txt` installed. Only the notebooks that
render task frames need `moog-games`.

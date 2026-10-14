# Formatting conventions for every figure in `figures/`

Note-to-self (and to future notebooks): **all** figure-generating notebooks under
`figures/` follow the rules below. Copy the style block verbatim into each new
notebook.

## Style block (put it in the imports cell of every notebook)

```python
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12     # legend entries at 12 pt
plt.rcParams['figure.dpi'] = 300
plt.rcParams['pdf.fonttype'] = 42        # keep PDF text editable in Illustrator
```

**No font size below 12 pt, anywhere.** That includes text drawn inside axes (trial
counts, annotations, in-panel labels), tick labels, and anything passed an explicit
`fontsize=`. If a label doesn't fit at 12 pt, make the figure bigger or move the label —
don't shrink the text.

If a plot uses seaborn, call `sns.despine(...)` only — **do not** use
`sns.set_theme(...)`, which silently overrides the rcParams above (or re-apply the
block after it).

## Despine

**Despine every data plot with real axes** — scatterplots, line plots, bar plots,
histograms: no top or right spine, keep left and bottom.

```python
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
# or, with seaborn imported: sns.despine(fig)
```

This does not apply to schematic / spatial panels that carry no ticks (ring plots, task
frames, aligned-response clouds) — those either keep a full frame box or have no spines
at all, whichever the figure calls for.

## Saving

Every generated figure is saved as a **PDF next to the notebook that made it**, with a
descriptive file name, overwriting any previous version:

```python
fig.savefig('my_figure.pdf', bbox_inches='tight', pad_inches=0.01)
```

## Figure size

Each notebook exposes the figure size(s) as a **parameter in a CONFIG cell at the top**
(e.g. `FIGSIZE = (4.2, 4.6)`), one per generated figure, so the size can be retuned
without touching plotting code.

## Self-containedness

Notebooks in `figures/` do **not** import library code from
`new_exploration/paper_draft_0/`. Anything shared lives in a `*.py` file in the same
directory as the notebook that uses it (copied/adapted as needed).

## Paths / kernels

Notebooks compute paths relative to the paper root
(`figures/figure_<n>/<sub>/` is three levels down, so `'../../../cache/...'`).

- Notebooks that only need numpy/pandas/scipy/matplotlib/seaborn run on the
  **Python (base)** kernel (`/opt/miniconda3/bin/python`).
- Notebooks that render task frames need `moog` + `pynwb` → **Python (wm_paper)**
  kernel (`/opt/miniconda3/envs/wm_paper/bin/python`).

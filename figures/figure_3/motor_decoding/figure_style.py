"""How `decoder_figures.ipynb`'s two figures LOOK — sizes, colours, markers, spacing.

Nothing here changes a number. Every constant is a drawing decision: the figure would carry
the same values with any of them set differently. They live out of the notebook so that its
CONFIG cell holds only what changes the *result* — which cache, which populations, which
bins, which error bar — and a knob you turn while looking at the data is never buried in
forty lines of marker sizes.

The notebook imports this as `fs` and reads `fs.NAME`. Edit a value here and re-run the
notebook's import cell (it calls `importlib.reload`, so no kernel restart).
"""

# ══════════════════════════════════════════════════════════════════════════════
# Where the PDFs go
# ══════════════════════════════════════════════════════════════════════════════
# These names are what `../figure_3.ai` links to; renaming one orphans the link.
OUTPUT_RINGS_1OBJ = "decoder_rings_1obj.pdf"
OUTPUT_RINGS_2OBJ = "decoder_rings_2obj.pdf"
OUTPUT_ERROR = "response_error.pdf"  # ONE figure, one subplot per monkey

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — the ring heatmaps
# ══════════════════════════════════════════════════════════════════════════════
# All the rings of a figure live on ONE axis (no subplots), so RING_SPACING sets the gap
# directly and no ring can be clipped at a panel edge.
FIGSIZE_RINGS = (
    4.0,
    1.1,
)  # the row's aspect is fixed by RING_SPACING, so this only
# sets the scale — extra white space is cropped by
# bbox_inches='tight'
RING_CMAP = "coolwarm"
RING_SPACING = 2.6  # distance between ring centers, in ring radii: 2.0 = rings
# exactly touching, 2.6 = the old one-subplot-per-ring
# layout at wspace = 0. Below ~2.35 the stimulus dots of
# neighboring rings start to overlap — they are drawn, not
# cut
RING_MARGIN = 0.33  # blank space kept around the whole row, in ring radii
# (0.16 is where the stimulus dots sit)

STIM_MARK_COLOR = "0.5"  # stimulus objects, drawn as dots
STIM_MARK_SIZE = 130  # ... their dot area, in points^2
STIM_SPOKE_LW = None  # ... width of the line running out to each dot from the
# annulus, in points; None or 0 draws the dots alone
STIM_RING = True  # the stimulus ring itself: a dashed circle through the
STIM_RING_COLOR = "k"  # dots, at the radius they sit at, as in
STIM_RING_LW = 0.7  # ../efficient_coding/ and this directory's schematics. It
STIM_RING_ALPHA = 0.35  # says where an object COULD have been, which the dots
# alone do not

FIX_MARK_SIZE = 7  # fixation cross at each ring's center: its size, in points
# (as in figure_2/psth)
FIX_MARK_LINEWIDTH = 1.3  # ... and its stroke width, in points

PEAK_MARK_RGB = (
    0.15,
    0.70,
    0.15,
)  # decoder's inferred locations: dot color, RGB
PEAK_MARK_MARKER = "x"  # ... drawn as a cross, as in
# ../efficient_coding/decoding_example.ipynb, so a decode
# that lands on an object marks its dot instead of covering
# it
PEAK_MARK_SIZE = 50  # ... marker area in points^2
PEAK_MARK_EDGE = (
    2.0  # ... stroke weight in points. An unfilled marker ('x', '+')
)
# is drawn in PEAK_MARK_RGB alone and this is its whole
# weight; a filled one ('o', '*') keeps the stimulus dots'
# thin black edge

# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — response error vs. inter-object angle
# ══════════════════════════════════════════════════════════════════════════════
FIGSIZE_ERROR = (3.0, 4.8)  # one subject per row, so tall rather than wide
ERROR_STACK = (
    "vertical"  # 'vertical' = one subject per row (they share the x axis,
)
# so only the bottom one is labelled); 'horizontal' = side
# by side
ERROR_HSPACE = (
    0.25  # vertical gap between the subject subplots when stacked, as
)
# a fraction of the average height
ERROR_WSPACE = (
    0.23  # horizontal gap between them side by side, as a fraction of
)
# the average subplot width. None = whatever tight_layout
# picks; a number overrides it (0 = panels touching)
ERROR_SHARE_YLIM_SUBJECTS = False  # share y across the two monkeys
ERROR_YLIM = (
    None  # (lo, hi) in degrees, applied to EVERY panel — which is what
)
# made the two monkeys share limits even with
# ERROR_SHARE_YLIM_SUBJECTS False. None = each panel scaled to
# its OWN real series, so the panels are independent and a
# control that runs off one — as the additive null does where
# it merges the two objects — cannot flatten the rest.
# Set a tuple again to pin both panels to one range.
ERROR_BIN_LABELS_DEG = (
    80.0,
    130.0,
    180.0,
    230.0,
    280.0,
)  # which bin ticks get a label;
# every bin is ticked, labelling them all is unreadable

ERROR_MARKER = "o"  # every series uses the same marker and line style, and the
ERROR_LINESTYLE = "-"  # figure carries no legend by default
ERROR_POINT_SIZE = 8  # the individual swept conditions, drawn as an x-jittered
ERROR_POINT_ALPHA = 0.5  # cloud behind the bin means
ERROR_JITTER_FRAC = 0.35  # jitter half-width, as a fraction of bin width
ERROR_JITTER_SEED = 0
SHOW_CLOUD = False  # draw that cloud, behind the UNBIASED bin means. The other
# series are bin means only — four clouds in one panel is
# not a figure.

ADDITIVE_COLOR = "0.6"  # the additive null, in gray behind the real response
# (the decoded series takes figure_1/repulsion_bias'
# per-bin rainbow, from decoder_lib.bin_colors)

LEGEND = False  # one key for the whole figure, placed OUTSIDE the panels:
LEGEND_ANCHOR = (
    0.5,
    -0.01,
)  # anchor in FIGURE coordinates (x = 0.5 is centred, y = 0
LEGEND_LOC = (
    "upper center"  # the bottom edge) and which part of the box lands there.
)
LEGEND_NCOL = 2  # Every series runs corner to corner, so there is no free
LEGEND_FONTSIZE = 12  # corner inside a panel to put a key in.

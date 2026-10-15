"""How `decoder.ipynb`'s figures LOOK — sizes, colours, markers, spacing.

Nothing here changes a number. Every constant is a drawing decision: the figure would carry
the same values with any of them set differently. They live out of the notebook so that its
CONFIG cell holds only what changes the *result*, and a knob you turn while looking at the
data is never buried in forty lines of marker sizes.

The notebook imports this as `fs` and reads `fs.NAME`.
"""

# ══════════════════════════════════════════════════════════════════════════════
# Where the PDFs go
# ══════════════════════════════════════════════════════════════════════════════
# These names are what `../figure_3.ai` links to; renaming one orphans the link.
OUTPUT_RINGS_1OBJ = "decoder_rings_1obj.pdf"
OUTPUT_RINGS_2OBJ = "decoder_rings_2obj.pdf"
OUTPUT_ERROR = "response_error.pdf"  # ONE figure, one subplot per monkey

# ══════════════════════════════════════════════════════════════════════════════
# The ring heatmaps
# ══════════════════════════════════════════════════════════════════════════════
# All the rings of a figure live on ONE axis (no subplots), so RING_SPACING sets the gap
# directly and no ring can be clipped at a panel edge.

# The row's aspect is fixed by RING_SPACING, so this only sets the scale — extra white space
# is cropped by bbox_inches='tight'.
FIGSIZE_RINGS = (4.0, 1.1)
RING_CMAP = "coolwarm"
# Distance between ring centers, in ring radii: 2.0 = rings exactly touching. Below ~2.35 the
# stimulus dots of neighboring rings start to overlap — they are drawn, not cut.
RING_SPACING = 2.6
# Blank space kept around the whole row, in ring radii (0.16 is where the stimulus dots sit).
RING_MARGIN = 0.33

STIM_MARK_COLOR = "0.5"  # stimulus objects, drawn as dots
STIM_MARK_SIZE = 130  # ... their dot area, in points^2
# The stimulus ring itself: a dashed circle through the dots, at the radius they sit at, as in
# ../efficient_coding_schematic/. It says where an object COULD have been, which the dots
# alone do not.
STIM_RING_COLOR = "k"
STIM_RING_LW = 0.7
STIM_RING_ALPHA = 0.35

# Fixation cross at each ring's center (as in figure_2/psth): its size and stroke width, in
# points.
FIX_MARK_SIZE = 7
FIX_MARK_LINEWIDTH = 1.3

# The decoder's inferred locations: colour (RGB), drawn as a cross so a decode that lands on
# an object marks its dot instead of covering it.
PEAK_MARK_RGB = (0.15, 0.70, 0.15)
PEAK_MARK_MARKER = "x"
PEAK_MARK_SIZE = 50  # marker area in points^2
# Stroke weight in points. An unfilled marker ('x', '+') is drawn in PEAK_MARK_RGB alone and
# this is its whole weight; a filled one ('o', '*') keeps the stimulus dots' thin black edge.
PEAK_MARK_EDGE = 2.0

# ══════════════════════════════════════════════════════════════════════════════
# Response error vs. inter-object angle
# ══════════════════════════════════════════════════════════════════════════════
# One subject per row; they share the x axis, so only the bottom one is labelled.
# Vertical gap between the subject subplots, as a fraction of the average height.
ERROR_HSPACE = 0.25
# Which bin ticks get a label; every bin is ticked, labelling them all is unreadable.
ERROR_BIN_LABELS_DEG = (80.0, 130.0, 180.0, 230.0, 280.0)

# Every series uses the same marker and line style, and the figure carries no legend.
ERROR_MARKER = "o"
ERROR_LINESTYLE = "-"

# The additive null, in gray behind the real response (the decoded series takes
# figure_1/repulsion_bias' per-bin rainbow, from decoder_lib.bin_colors).
ADDITIVE_COLOR = "0.6"

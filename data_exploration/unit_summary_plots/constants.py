"""Constants."""

import colorsys

################################################################################
####  COLOR SCHEME FOR TASK PHASES
################################################################################

PHASE_COLORS = [
    ("stimulus", colorsys.hsv_to_rgb(0.28, 1.0, 1.0)),  # stimulus onset
    ("delay", colorsys.hsv_to_rgb(0.4, 1.0, 1.0)),  # delay onset
    ("cue", colorsys.hsv_to_rgb(0.52, 1.0, 1.0)),  # cue onset
    ("response", colorsys.hsv_to_rgb(0.64, 1.0, 1.0)),  # response onset
    ("feedback", colorsys.hsv_to_rgb(0.76, 1.0, 1.0)),  # feedback onset
]

################################################################################
####  TRIAL TIME RANGE FOR RASTER AND PSTH PLOTS
################################################################################

# How long (seconds) before stimulus onset to begin raster plot
T_BEFORE_STIMULUS_ONSET = 0.5
# How long (seconds) after stimulus onset to end raster plot
T_AFTER_STIMULUS_ONSET = 3.25

################################################################################
####  PSTH PARAMETERS
################################################################################

PSTH_BOOTSTRAP_NUM = 100
PSTH_BIN_WIDTH = 0.03
# Number of bins in half the range of the tooth smoothing function for PSTHs
PSTH_KERNEL_HALF_WIDTH = 5

################################################################################
####  INTER-SPIKE INTERVAL PLOT PARAMETERS
################################################################################

ISI_BIN_WIDTH = 0.001
ISI_MAX = 0.1

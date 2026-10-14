# `supp_figure_10` — where the PSTH prototypes' trials actually sat

Figure 2's PSTH colours each condition by the mean firing rate over every trial whose **two
objects were both within 15° of their prototype slot**. The prototype is a *label for a cloud
of real stimuli*, not a stimulus that was ever shown, so the supplement it supports is: how far
is the cloud's centre from the label?

**There is deliberately no `common.py` here.** `figure_2/psth_lib.py` does `from common import
...`, and a `common.py` in this directory would shadow figure 2's and break it — the notebook
puts `../figure_2` on the path and imports that one instead. Everything about the prototypes,
the 15° window and the unordered pair matching therefore comes from figure 2's own code rather
than a copy, which is the only way this figure can be a statement about what figure 2 did.

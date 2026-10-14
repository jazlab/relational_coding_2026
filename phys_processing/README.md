# Physiology Processing

This directory holds one module, `spike_counts.py`, which turns the DANDI
spike-sorting NWB files into per-trial binned spike counts. Every figure that uses
neural data gets its spikes from here.

There is nothing to run ahead of time. Counts are binned on the fly (about a
second per session), so there is no intermediate cache. Each trial's counts are
10 ms bins starting 200 ms before stimulus onset and ending at feedback onset.
Only completed trials on which the unit was observed are included.

It needs:
- the spike-sorting NWB files in `cache/dandi_data/spikesorting/`
  (`python download_dandi_data.py --modality=spikesorting` from the repo root);
- `cache/behavior_processing/ring.csv`, for the trial event times (see
  `../behavior_processing/`).

Example:
```python
import sys
sys.path.insert(0, "path/to/repo/root")
from phys_processing import spike_counts as spike_counts_lib

spike_counts_lib.sessions()                       # [(subject, session), ...]
spike_counts_lib.units("Perle", "2022-06-11")     # unit, probe, quality per unit
counts, trials = spike_counts_lib.unit_spike_counts("Perle", "2022-06-11", 100)
```

`counts[i]` is the count vector for trial number `trials[i]`. `probe` is `s0`
(DMFC) or `vprobe0`/`vprobe1` (FEF), and `quality` is `good` (single unit) or
`mua` (multi-unit).

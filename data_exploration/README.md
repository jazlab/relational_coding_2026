# Data Exploration

Tools for getting a feel for the data. None of this is needed to reproduce the
paper's figures. Run each notebook from its own directory.

## task_visualization/

Replays trials as the monkey saw them, with the monkey's gaze drawn post hoc as a
green cross.

- `demo_replay_trial.ipynb` writes a gif of one trial to `gifs/`.
- `demo_replay_trials_with_neurons.ipynb` writes an mp4 of several trials, with
  spike rasters and a click for each spike from a few neurons.

Needs the behavior NWB file for the session (and, for the second notebook, the
spike-sorting NWB file) in `cache/dandi_data/`. The renderer needs the `moog-games`
package. `figures/` also borrows this renderer to draw task frames.

## psths/

PSTHs of single units, split by object positions and number of objects.

- `demo.ipynb` plots one unit.
- `python run_ring.py` plots every unit labeled `good` by spike sorting and
  saves the PDFs to `cache/data_exploration/psths/`.

Needs `cache/behavior_processing/ring.csv` and the spike-sorting NWB files. Spike
counts come from `phys_processing/spike_counts.py`.

## unit_summary_plots/

One-page summaries of each unit: mean waveform, ISI distribution, spike raster
and PSTH.

- `demo.ipynb` plots one unit.
- `python run.py` plots every unit and saves the PNGs to
  `cache/data_exploration/unit_summary_plots/`.

Needs the behavior and spike-sorting NWB files.

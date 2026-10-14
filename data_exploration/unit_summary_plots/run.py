"""Run this script to generate summary plots for all units.

Usage:
$ python run.py

This generates a summary plot for each unit and saves it to
"cache/data_exploration/unit_summary_plots/$SUBJECT/$SESSION/$PROBE/$QUALITY/$UNIT.png"

This script requires the following data:
- Behavior and spike sorting data in "cache/dandi_data/". This can be
  downloaded from DANDI by running `$ python download_dandi_data.py` from the
  repo root.
"""

from pathlib import Path

import numpy as np
import plot_isi_distribution
import plot_psth
import plot_raster
import plot_waveform_mean
from matplotlib import pyplot as plt
from pynwb import NWBHDF5IO

_ROOT = Path(__file__).resolve().parents[2]
_BEHAVIOR_DATA_DIR = _ROOT / "cache/dandi_data/behavior"
_SPIKESORTING_DATA_DIR = _ROOT / "cache/dandi_data/spikesorting"
_WRITE_DIR = _ROOT / "cache/data_exploration/unit_summary_plots"


def _session_unit_summary_plots(
    behavior_file: Path,
    spikesorting_file: Path,
) -> None:
    """Generate and save unit summary plots for a single session.

    Args:
        behavior_file: Path to the behavior NWB file.
        spikesorting_file: Path to the spikesorting NWB file.
    """
    # Load phase times per trial
    behavior_read_io = NWBHDF5IO(behavior_file, mode="r", load_namespaces=True)
    behavior_nwbfile = behavior_read_io.read()
    trials_data = behavior_nwbfile.intervals["trials"]
    trial_phase_times = list(
        zip(
            trials_data.start_time,
            trials_data.phase_stimulus_time,
            trials_data.phase_delay_time,
            trials_data.phase_cue_time,
            trials_data.phase_response_time,
            trials_data.phase_reveal_time,
            trials_data.stop_time,
        )
    )
    trial_completed = [not x for x in trials_data.broke_fixation]
    trial_phase_times = [
        x for x, c in zip(trial_phase_times, trial_completed) if c
    ]
    print(f"Number of trials = {len(trial_phase_times)}")

    # Get subject and session and make write directory
    subject = behavior_nwbfile.subject.subject_id
    session = behavior_nwbfile.session_id
    write_dir = _WRITE_DIR / subject / session
    write_dir.mkdir(parents=True, exist_ok=True)

    # Load neural data
    spikesorting_read_io = NWBHDF5IO(
        spikesorting_file, mode="r", load_namespaces=True
    )
    spikesorting_nwbfile = spikesorting_read_io.read()
    ecephys = spikesorting_nwbfile.processing["ecephys"]
    units = ecephys.data_interfaces["units"]
    electrodes_df = spikesorting_nwbfile.electrodes.to_dataframe()
    print(f"Number of units = {len(units)}")

    # Iterate through units and generate unit summary plots
    for unit_index in range(len(units)):
        if unit_index % 20 == 0:
            print(f"Processing unit {unit_index} / {len(units)}")

        # Load spike times and amplitudes
        spike_times = units.spike_times_index[unit_index]
        mean_waveform = units.mean_waveform_index[unit_index]
        mean_waveform_electrodes = np.copy(
            units.mean_waveform_electrodes_index[unit_index]
        )
        electrodes_group = units.electrodes_group[unit_index]
        quality = units.quality[unit_index]
        obs_trials = units.obs_trials_index[unit_index]
        obs_trials = [x for x, c in zip(obs_trials, trial_completed) if c]

        # Create figure and axes
        fig = plt.figure(figsize=(8, 8))
        gridspec = fig.add_gridspec(
            2, 2, height_ratios=(2, 1), hspace=0.3, wspace=0.3
        )
        ax_raster = fig.add_subplot(gridspec[0, 0])
        ax_psth = fig.add_subplot(gridspec[1, 0])
        ax_waveform = fig.add_subplot(gridspec[0, 1])
        ax_inter_spike_interval = fig.add_subplot(gridspec[1, 1])

        # Plot raster
        spike_times_per_obs_trial = plot_raster.plot_raster(
            ax_raster, spike_times, trial_phase_times, obs_trials
        )

        # Plot firing rate
        plot_waveform_mean.plot_waveform_mean(
            ax_waveform,
            mean_waveform,
            mean_waveform_electrodes,
            electrodes_df,
            electrodes_group,
        )

        # Plot PSTH
        plot_psth.plot_psth(ax_psth, spike_times_per_obs_trial)

        # Plot ISI distribution
        plot_isi_distribution.plot_isi_distribution(
            ax_inter_spike_interval, spike_times
        )

        # Create title for plot
        plt.suptitle(
            f"Unit {unit_index} - Quality: {quality}",
            weight="bold",
            fontsize=14,
        )

        # Save and close figure
        fig_path = write_dir / electrodes_group / quality / f"{unit_index}.png"
        if not fig_path.parent.exists():
            fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path)
        plt.close(fig)


def main():
    """Generate and save unit summary plots."""
    # Append each session to the dataframe
    for behavior_subject_dir in sorted(_BEHAVIOR_DATA_DIR.iterdir()):
        if behavior_subject_dir.name.startswith("."):
            continue
        print(f"\nProcessing {behavior_subject_dir.name}\n")
        for behavior_file in sorted(behavior_subject_dir.iterdir()):
            print(f"\nProcessing {behavior_file.name}\n")
            if behavior_file.name.startswith("."):
                continue
            spikesorting_file_name = behavior_file.name.replace(
                "behavior+task", "spikesorting"
            )
            spikesorting_file = (
                _SPIKESORTING_DATA_DIR
                / behavior_subject_dir.name
                / spikesorting_file_name
            )
            if not spikesorting_file.exists():
                print(f"No spikesorting file found for {behavior_file}")
                continue
            _session_unit_summary_plots(behavior_file, spikesorting_file)


if __name__ == "__main__":
    main()

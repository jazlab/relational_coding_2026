"""Function for making raster plot."""

import constants
import numpy as np
from matplotlib import lines as mlp_lines


def plot_raster(ax, spike_times, trial_phase_times, obs_trials):
    """Make raster plot in provided axes.

    Args:
        ax: Matplotlib Axes object to render plot in.
        spike_times: Array of spike times for a unit.
        trial_phase_times: List of phase times per trial. Each element is a list
            of times [start, fixation, stimulus, delay, cue, response, reveal].
    """
    spikes_raster_x = []
    spikes_raster_y = []

    phase_times_raster_x = []
    phase_times_raster_y = []
    phase_times_raster_colors = []

    spike_times_per_trial = []

    num_trials = len(trial_phase_times)
    spike_index = 0
    for trial_num in range(num_trials):
        # Skip if neuron was not observed during this trial
        if not obs_trials[trial_num]:
            continue

        # Get data for this trial
        phase_times = trial_phase_times[trial_num]
        t_trial_stop = phase_times[-1]
        t_stimulus = phase_times[1]

        # Append relative phase times to raster data
        for i, t in enumerate(phase_times[1:-1]):
            phase_times_raster_x.append(t - t_stimulus)
            phase_times_raster_y.append(trial_num)
            phase_times_raster_colors.append(constants.PHASE_COLORS[i][1])

        # Update spikes raster data based on spike times
        next_trial = False
        spike_times_this_trial = []
        while not next_trial:
            if spike_index >= len(spike_times):
                # No more spikes for this neuron, so move on to next trial
                next_trial = True
                continue

            spike_t = spike_times[spike_index]

            spike_too_late = (
                spike_t > t_trial_stop
                or spike_t > t_stimulus + constants.T_AFTER_STIMULUS_ONSET
            )
            if spike_too_late:
                # Move on to next trial
                next_trial = True
                continue

            if (
                trial_num < len(trial_phase_times) - 1
                and spike_t > trial_phase_times[trial_num + 1][0]
            ):
                # Trial is completed, so move on to next trial
                next_trial = True
                continue

            if spike_t < t_stimulus - constants.T_BEFORE_STIMULUS_ONSET:
                # Have yet to reach beginning of trial, so move on to next spike
                spike_index += 1
                continue

            # Append the spike time to raster data
            spikes_raster_x.append(spike_t - t_stimulus)
            spikes_raster_y.append(trial_num)
            spike_index += 1

            # Append spike time to spike_times_this_trial if necessary
            spike_times_this_trial.append(spike_t - t_stimulus)

        # Append spike_times_this_trial to spike_times_per_trial
        spike_times_per_trial.append(spike_times_this_trial)

    # Scatter spike times and phases
    ax.scatter(spikes_raster_x, spikes_raster_y, c="k", s=0.2)
    ax.scatter(
        phase_times_raster_x,
        phase_times_raster_y,
        c=phase_times_raster_colors,
        s=0.2,
    )

    # Create axis labels
    ax.set_xlabel("Time within trial (sec)", fontsize=10)
    ax.set_ylabel("Trial number", fontsize=12)
    ax.set_xlim(
        -0.1 - constants.T_BEFORE_STIMULUS_ONSET,
        0.1 + constants.T_AFTER_STIMULUS_ONSET,
    )
    ax.set_ylim(-10, num_trials + 10)
    ax.set_title("Raster Plot", fontsize=12, weight="bold", y=1.05)

    # Create legend
    legend_handles = [
        mlp_lines.Line2D(
            [],
            [],
            marker=".",
            color=color,
            markerfacecolor=color,
            markersize=3.5,
            label=label,
            linewidth=0.0,
        )
        for label, color in constants.PHASE_COLORS
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.45, 1.06),
        ncol=5,
        fancybox=True,
        markerscale=3,
        borderpad=0.0,
        columnspacing=0.6,
        handletextpad=0.0,
        handlelength=1.5,
        fontsize=9,
    )

    return spike_times_per_trial

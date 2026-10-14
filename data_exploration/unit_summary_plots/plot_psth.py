"""Function for making PSTH plot."""

import constants
import numpy as np
import pandas as pd
import seaborn as sns


def _times_to_hist(x, bin_edges, kernel):
    """Convert spike times to smooth histogram."""
    hist, _ = np.histogram(x, bin_edges)
    hist = np.convolve(hist, kernel, mode="same")

    # Remove edge effects from convolution
    end_pad = (len(kernel) - 1) // 2
    beginning_pad = len(kernel) - 1 - end_pad
    beginning_normalization = np.cumsum(kernel)[-beginning_pad - 1 : -1]
    end_normalization = np.cumsum(kernel[::-1])[-end_pad - 1 : -1][::-1]
    hist[:beginning_pad] /= beginning_normalization
    hist[-end_pad:] /= end_normalization

    return hist


def _bootstrap_psth(spike_hists_per_trial):
    """Bootstrap to compute errorbars."""
    all_psths = []
    num_trials = len(spike_hists_per_trial)
    sample_size = int(np.ceil(0.5 * num_trials))
    for _ in range(constants.PSTH_BOOTSTRAP_NUM):
        trial_inds = np.random.choice(
            num_trials, size=sample_size, replace=True
        )
        psth = np.mean(spike_hists_per_trial[trial_inds], axis=0)
        all_psths.append(psth)
    return np.array(all_psths)


def plot_psth(ax, spike_times_per_trial):
    """Plot PSTH on given axes.

    Args:
        ax: Matplotlib axis to plot on.
        spike_times_per_trial: List of spike times for each trial.
    """
    bin_edges = np.arange(
        -constants.T_BEFORE_STIMULUS_ONSET,
        constants.PSTH_BIN_WIDTH + constants.T_AFTER_STIMULUS_ONSET,
        constants.PSTH_BIN_WIDTH,
    )
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # Get bootstrap PSTHs
    kernel = np.linspace(0.0, 1.0, constants.PSTH_KERNEL_HALF_WIDTH)
    kernel = np.concatenate([kernel, kernel[:-1][::1]])
    kernel /= np.sum(kernel)
    spike_hists_per_trial = np.array(
        [_times_to_hist(x, bin_edges, kernel) for x in spike_times_per_trial]
    )
    psths = _bootstrap_psth(spike_hists_per_trial)
    psths /= constants.PSTH_BIN_WIDTH  # Convert to units of seconds

    # Create dataframe with PSTHs
    data_df_dict = {
        "Time within trial (s)": np.tile(bin_centers, psths.shape[0]),
        "Firing rate (Hz)": np.ravel(psths),
    }
    data_df = pd.DataFrame(data_df_dict)

    # Plot PSTH
    sns.lineplot(
        ax=ax,
        data=data_df,
        x="Time within trial (s)",
        y="Firing rate (Hz)",
        color=[0.9, 0.1, 0.1],
        errorbar="sd",
        linewidth=3,
    )

    # Axis handling
    ax.set_xlim(
        -0.1 - constants.T_BEFORE_STIMULUS_ONSET,
        0.1 + constants.T_AFTER_STIMULUS_ONSET,
    )
    ax.set_title("Peri-Stimulus Time Histogram", fontsize=12, weight="bold")

    return

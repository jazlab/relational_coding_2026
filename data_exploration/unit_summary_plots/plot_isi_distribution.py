"""Function for making inter-spike interval distribution plot."""

import constants
import numpy as np


def plot_isi_distribution(ax, spike_times):
    """Plot inter-spike interval distribution.

    Args:
        ax: Matplotlib axis to plot on.
        spike_times: List of spike times for a unit.
    """
    isi = spike_times[1:] - spike_times[:-1]
    bin_edges = np.arange(
        0.0,
        constants.ISI_MAX + constants.ISI_BIN_WIDTH,
        constants.ISI_BIN_WIDTH,
    )

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    isi_hist, _ = np.histogram(isi, bin_edges)
    isi_hist = isi_hist.astype(float) / len(isi)
    ax.bar(bin_centers, isi_hist, width=constants.ISI_BIN_WIDTH)
    ax.set_xlim(0.0, constants.ISI_MAX)
    ax.set_title(
        "Inter-Spike Interval Distribution", fontsize=12, weight="bold"
    )
    ax.set_xlabel("ISI (ms)")

    return

"""Function for making waveform plot."""

import numpy as np


def plot_waveform_mean(
    ax,
    mean_waveform,
    mean_waveform_electrodes,
    electrodes_df,
    electrodes_group,
    x_span=10,
    y_span=40,
):
    """Make waveform plot in provided axis.

    Args:
        ax: Matplotlib axis to plot on.
        mean_waveform: Mean waveform data for the unit.
        mean_waveform_electrodes: Electrode indices corresponding to the mean
            waveform.
        electrodes_df: DataFrame containing electrode information.
        electrodes_group: Group name for the electrodes.
        x_span: Horizontal span of the waveform plot (default is 10).
        y_span: Vertical span of the waveform plot (default is 40).
    """

    # Return without plotting if no mean_waveform
    if np.any(np.isnan(mean_waveform)):
        return

    # Reshape mean_waveform and mean_waveform_electrodes
    n_channels = len(np.unique(mean_waveform_electrodes))
    mean_waveform = np.reshape(mean_waveform, (-1, n_channels))
    mean_waveform_electrodes = np.reshape(
        mean_waveform_electrodes, (-1, n_channels)
    )

    # Sanity check that mean_waveform_electrodes is sorted
    same_rows = np.all(
        mean_waveform_electrodes == mean_waveform_electrodes[:1, :]
    )
    if not same_rows:
        raise ValueError(
            "mean_waveform_electrodes does not have all identical rows. "
            "Something went wrong with the reshaping."
        )

    # Remove time dimension from mean_waveform_electrodes
    mean_waveform_electrodes = mean_waveform_electrodes[0]

    # Get channel positions
    df = electrodes_df.loc[electrodes_df.group_name == electrodes_group]
    electrode_x_pos = [
        df.loc[df.shank_electrode_number == x].rel_x.values[0]
        for x in mean_waveform_electrodes
    ]
    electrode_y_pos = [
        df.loc[df.shank_electrode_number == x].rel_y.values[0]
        for x in mean_waveform_electrodes
    ]
    electrode_positions = np.stack([electrode_x_pos, electrode_y_pos], axis=1)

    # Normalize waveform to be between -1 and 1
    wf_spread = np.max(mean_waveform) - np.min(mean_waveform)

    # Plot waveform
    timesteps_per_waveform = mean_waveform.shape[0]
    x_axis = np.linspace(-x_span, x_span, timesteps_per_waveform)
    for chan in range(electrode_positions.shape[0]):
        wf_chan = mean_waveform[:, chan]
        wf_chan = (wf_chan - np.min(wf_chan)) / wf_spread
        x = electrode_positions[chan, 0] + x_axis
        y = electrode_positions[chan, 1] + y_span * wf_chan
        ax.plot(x, y, color="k", linewidth=3)

    # Set axis y-limit
    ymin = np.min(electrode_positions[:, 1]) - y_span
    ymax = np.max(electrode_positions[:, 1]) + y_span
    ax.set_ylim([ymin, ymax])

    # Set axis labels and title
    ax.set_xlabel("X-position (um)")
    ax.set_ylabel("Y-position (um)")
    ax.set_title("Waveform", fontsize=12, weight="bold", y=1.05)

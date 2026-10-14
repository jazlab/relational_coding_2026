"""Run this script to process and cache behavior data.

Usage:
$ python run_cache_data.py

This writes the dataframe `cache/behavior_processing/ring.csv`, with one row per
trial of every session in `cache/dandi_data/behavior/`.

Note: For all angular data columns (e.g. "response_theta", "object_0_theta",
...), the angle has zero at due right and increases counterclockwise in radians,
like polar coordinates for complex numbers.

The `correct` column is the reward the task actually delivered online
(`reward_duration > 0`), not a reconstruction from response distance.

This script requires the behavior dandi data to be downloaded and cached, which
you can do by running `$ python download_dandi_data.py --modality=behavior` from
the root of this repo.
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _ROOT / "cache/dandi_data/behavior"
_WRITE_PATH = _ROOT / "cache/behavior_processing/ring.csv"
_NUM_OBJECTS = 2  # Ring-task trials have one or two objects
_DATA_COLUMNS = (
    [
        "subject",
        "session",
        "trial_num",
        "completed",
        "time_fixation_onset",
        "time_stimulus_onset",
        "time_delay_onset",
        "time_cue_onset",
        "time_response_onset",
        "time_feedback_onset",
        "time_iti_onset",
        "response_time",
        "response_r",
        "response_theta",
        "response_x",
        "response_y",
        "target_object_index",
        "num_objects",
    ]
    + [
        f"object_{i}_{key}"
        for i in range(_NUM_OBJECTS)
        for key in ["r", "theta", "x", "y", "id"]
    ]
    + [
        "reward_duration",
        "reward_time",
        "correct",
    ]
)


def _cartesian_to_polar(x: float, y: float) -> tuple[float, float]:
    """Convert Cartesian coordinates to polar coordinates."""
    r = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
    theta = np.arctan2(y - 0.5, x - 0.5) % (2 * np.pi)
    return r, theta


def _str_to_bool(s: str) -> bool:
    """Convert a string to a boolean."""
    if s.lower() == "true":
        return True
    elif s.lower() == "false":
        return False
    else:
        raise ValueError(f"Cannot convert string {s} to boolean")


def _get_session_df(nwbfile: NWBHDF5IO) -> pd.DataFrame:
    """Get a dataframe of session data from an NWB file.

    Args:
        nwbfile: NWB file object.

    Returns:
        session_df: DataFrame of session data.
    """
    subject = nwbfile.subject.subject_id
    session = nwbfile.session_id
    trial_df = nwbfile.trials.to_dataframe()

    # Ensure all values of delay_object_blanks are False
    if trial_df.delay_object_blanks.any():
        raise ValueError("delay_object_blanks contains True values")

    # Convert list columns to lists
    trial_df.stimulus_object_target = trial_df.stimulus_object_target.apply(
        lambda x: [_str_to_bool(s.strip()) for s in x[1:-1].split(",")]
    )

    rows = []
    for trial_num, row in trial_df.iterrows():
        completed = not row.broke_fixation and np.isfinite(row.response_time)
        trial_data = dict(
            subject=subject,
            session=session,
            trial_num=trial_num,
            completed=completed,
            time_fixation_onset=row.phase_fixation_time,
            time_stimulus_onset=row.phase_stimulus_time,
            time_delay_onset=row.phase_delay_time,
            time_cue_onset=row.phase_cue_time,
            time_response_onset=row.phase_response_time,
            time_feedback_onset=row.phase_reveal_time,
            time_iti_onset=row.phase_iti_time,
            response_time=row.response_time,
            reward_duration=row.reward_duration,
            reward_time=row.reward_time,
        )

        # Add response data
        if row.response_position is not None:
            response_r, response_theta = _cartesian_to_polar(
                row.response_position[0], row.response_position[1]
            )
            trial_data["response_r"] = response_r
            trial_data["response_theta"] = response_theta
            trial_data["response_x"] = row.response_position[0]
            trial_data["response_y"] = row.response_position[1]

        # Add object data
        trial_data["target_object_index"] = np.argwhere(
            row.stimulus_object_target
        )[0, 0]
        obj_positions = ast.literal_eval(row.stimulus_object_positions)
        obj_identities = ast.literal_eval(row.stimulus_object_identities)
        trial_data["num_objects"] = len(obj_positions)
        for i, (position, identity) in enumerate(
            zip(obj_positions, obj_identities)
        ):
            obj_r, obj_theta = _cartesian_to_polar(position[0], position[1])
            trial_data[f"object_{i}_r"] = obj_r
            trial_data[f"object_{i}_theta"] = obj_theta
            trial_data[f"object_{i}_x"] = position[0]
            trial_data[f"object_{i}_y"] = position[1]
            trial_data[f"object_{i}_id"] = identity

        rows.append(trial_data)

    # Columns absent from a trial (e.g. object_1_* on 1-object trials) are NaN
    return pd.DataFrame(rows, columns=_DATA_COLUMNS)


def build_ring_table() -> pd.DataFrame:
    """Dataframe of every trial of every session in the behavior data dir."""
    session_dfs = []
    for subject_dir in sorted(_DATA_DIR.iterdir()):
        if subject_dir.name.startswith("."):
            continue
        for session_file in sorted(subject_dir.iterdir()):
            if session_file.name.startswith("."):
                continue
            with NWBHDF5IO(session_file, "r") as io:
                print(f"Processing {session_file}")
                session_dfs.append(_get_session_df(nwbfile=io.read()))

    df = pd.concat(session_dfs, ignore_index=True)
    df["correct"] = np.isfinite(df["reward_duration"]) & (
        df["reward_duration"] > 0
    )
    return df


def main():
    """Generate and save behavior data."""
    df = build_ring_table()
    print(f"Writing ring data to {_WRITE_PATH}")
    if _WRITE_PATH.exists():
        print(f"Overwriting existing file {_WRITE_PATH}")
    _WRITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(_WRITE_PATH, index=False)


if __name__ == "__main__":
    main()

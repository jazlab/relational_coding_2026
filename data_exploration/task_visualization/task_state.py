"""Function for creating a task state from stimulus information.

Do not change any of the constants or values in functions in this file. These
values were used in the original task design.

Usage:
    stim_positions = [[0.85, 0.5], [0.325, 0.197], [0.325, 0.803]]
    stim_identities = ["a", "b", "c"]
    stim_target = [True, False, False]
    state = get_task_state(stim_positions, stim_identities, stim_target)
    # state is an OrderedDict with sprites for the task state and can be
    # rendered using the MOOG renderer.
"""

import collections

import numpy as np
from moog import shapes, sprite

# Constants for the task. Do not change these. These values were used in the
# original task design.
_CUE_SCALE = 0.16
_PREY_SCALE = 0.083
_FOVEAL_WINDOW_RADIUS = 0.12
_WALL_THICKNESS = 0.05


def _get_walls():
    """Create wall sprites for the task."""
    wall_v_shape = np.array(
        [
            [-0.5 * _WALL_THICKNESS, -1],
            [-0.5 * _WALL_THICKNESS, 2],
            [0.5 * _WALL_THICKNESS, 2],
            [0.5 * _WALL_THICKNESS, -1],
        ]
    )
    wall_h_shape = np.array(
        [
            [-1, -0.5 * _WALL_THICKNESS],
            [2, -0.5 * _WALL_THICKNESS],
            [2, 0.5 * _WALL_THICKNESS],
            [-1, 0.5 * _WALL_THICKNESS],
        ]
    )

    wall_shapes = [
        np.array([-0.01, 0.0]) + wall_v_shape,
        np.array([1.01, 0.0]) + wall_v_shape,
        np.array([0.0, -0.01]) + wall_h_shape,
        np.array([0.0, 1.01]) + wall_h_shape,
    ]

    wall_sprites = [
        sprite.Sprite(x=0.0, y=0.0, shape=s, c0=0.0, c1=0.0, c2=0.05)
        for s in wall_shapes
    ]

    return wall_sprites


def get_task_state(
    stim_positions: list[list[float]],
    stim_identities: list[str],
    stim_target: list[bool],
):
    """Create MOOG task state from stimulus information.

    Args:
        stim_positions: List of stimulus positions, each a list of [x, y]
            coordinates.
        stim_identities: List of stimulus identities, each a string.
        stim_target: List of booleans indicating whether the stimulus is a
            target.

    Returns:
        state: An OrderedDict containing the sprites for the task state.
    """
    # Create prey
    prey = []
    prey_shape = shapes.circle_vertices(_PREY_SCALE, num_sides=30)
    for p, i, t in zip(stim_positions, stim_identities, stim_target):
        prey_sprite = sprite.Sprite(
            shape=np.copy(prey_shape),
            x=p[0],
            y=p[1],
            c0=0.0,
            c1=0.0,
            c2=1.0,
            opacity=0,
            metadata={"id": i, "target": t},
        )
        prey.append(prey_sprite)

    # Create fixation cross
    fixation_shape = 0.05 * np.array(
        [
            [-5, 1],
            [-1, 1],
            [-1, 5],
            [1, 5],
            [1, 1],
            [5, 1],
            [5, -1],
            [1, -1],
            [1, -5],
            [-1, -5],
            [-1, -1],
            [-5, -1],
        ]
    )
    fixation = sprite.Sprite(
        x=0.5,
        y=0.5,
        shape=fixation_shape,
        scale=0.1,
        c0=0.0,
        c1=0.0,
        c2=1.0,
        opacity=255,
    )

    # Create cue sprite
    cue_metadata = {"target_prey_ind": np.argwhere(stim_target)[0][0]}
    cue = sprite.Sprite(
        x=0.5,
        y=0.5,
        shape="square",
        scale=_CUE_SCALE,
        c0=0.0,
        c1=0.0,
        c2=1.0,
        opacity=0,
        metadata=cue_metadata,
    )

    # Create foveal window sprite
    fw_shape = shapes.circle_vertices(0.13)  # Radius here is unused
    fw = sprite.Sprite(
        x=-0.5,
        y=0.5,
        shape=fw_shape,
        c0=0.0,
        c1=0.0,
        c2=0.0,
        opacity=0,
        metadata={"radius": _FOVEAL_WINDOW_RADIUS},
    )

    # Create eye sprite
    eye = sprite.Sprite(
        x=0.5,
        y=0.5,
        shape=fixation_shape,
        scale=0.07,
        c0=0.33,
        c1=1.0,
        c2=0.8,
        opacity=0,
    )

    broke_fixation_screen = sprite.Sprite(
        x=0.5,
        y=0.5,
        shape="square",
        scale=2.0,
        c0=0.0,
        c1=0.5,
        c2=0.4,
        opacity=0,
    )

    state = collections.OrderedDict(
        [
            ("fw", [fw]),
            ("prey", prey),
            ("fixation", [fixation]),
            ("cue", [cue]),
            ("broke_fixation_screen", [broke_fixation_screen]),
            ("walls", _get_walls()),
            ("eye", [eye]),
        ]
    )

    return state

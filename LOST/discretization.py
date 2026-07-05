"""Discretization of the continuous MountainCarContinuous-v0 spaces.

The environment is continuous in both observations and actions:

- Observation: ``[position, velocity]`` with position in ``[-1.2, 0.6]``
  and velocity in ``[-0.07, 0.07]``.
- Action: a single throttle value in ``[-1.0, 1.0]``.

Tabular Q-Learning needs a finite state set and a finite action set, so both
spaces are mapped onto uniform grids. The grid resolutions are the main
design choice explored in the experiments: coarse grids alias very different
physical situations into the same table cell, while fine grids blow up the
number of cells that must each be visited many times to be learned.
"""

import numpy as np

# Physical bounds of MountainCarContinuous-v0 (see Gymnasium source).
POSITION_RANGE = (-1.2, 0.6)
VELOCITY_RANGE = (-0.07, 0.07)
ACTION_RANGE = (-1.0, 1.0)


class Discretizer:
    """Uniform-grid discretizer for observations and actions.

    Parameters
    ----------
    position_bins : int
        Number of grid edges for the position dimension.
    velocity_bins : int
        Number of grid edges for the velocity dimension.
    num_actions : int
        Number of discrete throttle values, evenly spaced in ``[-1, 1]``.
    """

    def __init__(self, position_bins=20, velocity_bins=20, num_actions=5):
        self.position_bins = position_bins
        self.velocity_bins = velocity_bins
        self.num_actions = num_actions
        self.x_space = np.linspace(*POSITION_RANGE, position_bins)
        self.vel_space = np.linspace(*VELOCITY_RANGE, velocity_bins)
        self.actions = np.linspace(*ACTION_RANGE, num_actions)

    def state(self, obs):
        """Map a continuous observation to a discrete ``(x_bin, vel_bin)`` pair."""
        x, vel = obs
        return int(np.digitize(x, self.x_space)), int(np.digitize(vel, self.vel_space))

    def action(self, action_idx):
        """Map a discrete action index to the continuous action the env expects."""
        return np.array([self.actions[action_idx]], dtype=np.float32)

    @property
    def q_table_shape(self):
        # np.digitize returns values in [0, len(edges)], hence the +1.
        return (self.position_bins + 1, self.velocity_bins + 1, self.num_actions)

    @property
    def num_states(self):
        return (self.position_bins + 1) * (self.velocity_bins + 1)

    def describe(self):
        return (
            f"{self.position_bins}x{self.velocity_bins} state grid "
            f"({self.num_states} states), {self.num_actions} actions"
        )

import math

import numpy as np
from gym.spaces import Box, Discrete, Dict


class ConvolutionalObservationManager:
    def __init__(self, num_cells: int, cell_shape=(20, 47)):
        self.num_cells = num_cells
        self.cell_shape = cell_shape

    def get_observation_space(self):
        return Dict({
            'position': Discrete(self.num_cells),
            'time_per_step': Box(low=0, high=math.inf, shape=(1,)),
            'cells': Box(low=0, high=math.inf, shape=self.cell_shape)
        })

    def get_observation(self, position_x: int, position_y: int, time_per_step: float, time_per_pick: float,
                        cells: np.ndarray):
        width = cells.shape[1]
        return {
            'position': position_y * width + position_x,
            'time_per_step': time_per_step,
            'cells': cells
        }


class ObservationManager:

    def get_observation_space(self):
        return Box(0, np.inf, shape=(940,))

    def get_observation(self, position_x: int, position_y: int, time_per_step: float, time_per_pick: float,
                        cells: np.ndarray):
        width = cells.shape[1]
        position = position_y * width + position_x
        time_per_step: time_per_step
        obs = np.concatenate([cells.reshape(-1), [position, time_per_step]], dtype=np.float32)
        return obs

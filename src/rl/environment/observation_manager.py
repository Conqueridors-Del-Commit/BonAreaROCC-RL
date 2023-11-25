import math

import numpy as np
from gymnasium.spaces import Box, Discrete, Dict


class ConvolutionalObservationManager:
    def __init__(self, num_cells: int):
        self.num_cells = num_cells

    def get_observation_space(self):
        return Dict({
            'position': Discrete(self.num_cells),
            'time_per_step': Box(low=0, high=math.inf, shape=(1,)),
            'cells': Box(low=0, high=math.inf, shape=(self.num_cells,))
        })

    def get_observation(self, position_x: int, position_y: int, time_per_step: float, time_per_pick: float,
                        cells: np.ndarray):
        width = cells.shape[1]
        return {
            'position': position_y * width + position_x,
            'time_per_step': time_per_step,
            'cells': cells
        }

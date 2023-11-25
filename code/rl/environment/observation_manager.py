import math
from gymnasium.spaces import Box, Discrete, Dict


class ConvolutionalObservationManager:
    def __init__(self, num_cells: int):
        self.num_cells = num_cells

    def get_observation_space(self):
        return Dict({
            'position': Discrete(self.num_cells),
            'time_per_step': Box(low=0, high=math.inf, shape=(1,)),  # TODO: check this
            'time_per_pick': Box(low=0, high=math.inf, shape=(1,)),
            'cells': Box(low=0, high=math.inf, shape=(self.num_cells,))
        })

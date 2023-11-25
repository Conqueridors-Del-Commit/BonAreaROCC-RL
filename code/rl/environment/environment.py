from code.rl.environment.observation_manager import ConvolutionalObservationManager
from code.rl.environment.reward_manager import RewardManager

import gymnasium as gym
import pandas as pd
from gymnasium.spaces import Discrete


class Environment(gym.Env):
    def __init__(self, obs_manager, reward_manager, cells, picking_positions, ticket_items):
        self.observation_manager = obs_manager
        self.reward_manager = reward_manager

        self.cells = cells
        self.picking_positions = picking_positions
        self.ticket_items = ticket_items

        self.observation_space = self.observation_manager.get_observation_space()
        self.action_space = Discrete(5)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        raise NotImplementedError


class EnvironmentBuilder:
    def __init__(self, obs_mode, reward_mode, planogram_csv_path, ticket_path):
        self.obs_mode = obs_mode
        self.reward_mode = reward_mode
        self.cells = self._load_cells(planogram_csv_path)
        self.picking_positions = self._load_picking_positions(planogram_csv_path)
        self.ticket_items = self._load_ticket(ticket_path)

    def build(self):
        if self.obs_mode == 1:
            obs_manager = ConvolutionalObservationManager(self.num_cells)
        else:
            raise NotImplementedError

        if self.reward_mode:
            reward_manager = RewardManager()
        else:
            raise NotImplementedError

        return Environment(obs_manager=obs_manager, reward_manager=reward_manager,
                           cells=self.cells, picking_positions=self.picking_positions, ticket_items=self.ticket_items)

    def _load_cells(self, planogram_csv_path: str):
        products = pd.read_csv(planogram_csv_path, delimiter=';')
        self.num_cells = products.shape[0]
        prods = products['description'].unique()
        prods_dict = {prods[i]: i for i in range(len(prods))}
        return prods_dict

    def _load_picking_positions(self, planogram_csv_path: str):
        products = pd.read_csv(planogram_csv_path, delimiter=';')
        # Get indexes where picking_x is not null
        picking_positions = products[products['picking_x'].notnull()].index.tolist()
        final_picking_positions = {}
        for pos in picking_positions:
            product_idx = self.cells[products['description'][pos]]
            final_picking_positions[product_idx] = (
                products['picking_x'][pos], products['picking_y'][pos])
        return final_picking_positions

    def _load_ticket(self, ticket_path: str):
        ticket = pd.read_csv(ticket_path, delimiter=';')
        final_ticket = {}
        for index, row in ticket.iterrows():
            final_ticket[row['article_id']] = row['quantity']
        return final_ticket


if __name__ == "__main__":
    env = EnvironmentBuilder(
        ticket_path='data/data/test_ticket.csv',
        planogram_csv_path='data/data/planogram_table.csv',
        obs_mode=1,
        reward_mode=1).build()

import copy

from code.rl.environment.observation_manager import ConvolutionalObservationManager
from code.rl.environment.reward_manager import RewardManager

import gymnasium as gym
import pandas as pd
import numpy as np
import json
from gymnasium.spaces import Discrete


class Environment(gym.Env):
    def __init__(self, obs_manager, reward_manager, picking_positions, ticket_items, planogram_csv_path,
                 customer_properties_csv, article_grouping_map):
        self.observation_manager = obs_manager
        self.reward_manager = reward_manager

        self.picking_positions = picking_positions
        self.ticket_items = ticket_items
        self.base_map = self._build_map(planogram_csv_path)
        self.customer_properties_df = pd.read_csv(customer_properties_csv, delimiter=';')
        self.article_grouping = article_grouping_map

        # TODO: action mask
        self.observation_space = self.observation_manager.get_observation_space()
        self.action_space = Discrete(5)

        self.action_map = {
            1: (0, -1),  # UP
            2: (0, 1),  # DOWN
            3: (-1, 0),  # LEFT
            4: (1, 0)  # RIGHT
        }

        self.reset()

    def reset(self):
        self.customer_pos_x = 28
        self.customer_pos_y = 19
        self.time_per_step, self.time_per_pick = self._get_customer_properties(self.customer_properties_df)
        self.map = np.copy(self.base_map)
        self.pending_items = copy.deepcopy(self.ticket_items)
        for item in self.ticket_items.keys():
            pos_x, pos_y = self.picking_positions[item]
            self.map[pos_y - 1, pos_x - 1] = self.article_grouping[item] + 4

        observation = self.observation_manager.get_observation(self.customer_pos_x, self.customer_pos_y,
                                                               self.time_per_step, self.time_per_pick, self.map)
        return observation, {}

    def step(self, action):
        if action == 0:
            for article, position in self.picking_positions.items():
                if (position[0] - 1) == self.customer_pos_x and (position[1] - 1) == self.customer_pos_y:
                    self.pending_items[article] -= 1
                    if self.pending_items[article] == 0:
                        self.map[self.customer_pos_y, self.customer_pos_x] = 0

            observation = self.observation_manager.get_observation(self.customer_pos_x, self.customer_pos_y,
                                                                   self.time_per_step, self.time_per_pick, self.map)

            return observation, 0, False, False, {}
        else:
            # move agent
            self.customer_pos_x += self.action_map[action][0]
            self.customer_pos_y += self.action_map[action][1]

            done = self.map[self.customer_pos_y, self.customer_pos_x] == 2

            observation = self.observation_manager.get_observation(self.customer_pos_x, self.customer_pos_y,
                                                                   self.time_per_step, self.time_per_pick, self.map)
            return observation, 0, done, done, {}

    def _build_map(self, planogram_csv_path: str):
        """
        Creates a matrix representing the map that the agent will see. We distinguish between points
        where the agent can walk on and points where he can't.
        """
        planogram_df = pd.read_csv(planogram_csv_path, delimiter=';')
        width = planogram_df['x'].max()
        height = planogram_df['y'].max()
        src_map = np.zeros(shape=(height, width))
        cell_type_dict = {"paso": 0, "paso-entrada": 2, "paso-salida": 3}
        for idx, row in planogram_df.iterrows():
            cell_item = row["description"]
            x = row["x"] - 1
            y = row["y"] - 1
            if cell_item in cell_type_dict:
                src_map[y, x] = cell_type_dict[cell_item]
            else:
                src_map[y, x] = 1

        return src_map

    def _get_customer_properties(self, customer_properties):
        sample = customer_properties.sample(1)
        return sample["step_seconds"], sample["picking_offset"]

    def render(self):
        for row in self.map:
            for cell in row:
                print(cell, end=' ')
            print()


class EnvironmentBuilder:
    def __init__(self, obs_mode, reward_mode, planogram_csv_path, ticket_path, customer_properties_path, grouping_path):
        self.obs_mode = obs_mode
        self.reward_mode = reward_mode
        self.planogram_csv_path = planogram_csv_path
        self.customer_properties_path = customer_properties_path
        self.cells = self._load_cells(planogram_csv_path)
        self.picking_positions = self._load_picking_positions(planogram_csv_path)
        self.ticket_items = self._load_ticket(ticket_path)
        self.grouping_map = self._load_article_grouping(grouping_path)

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
                           picking_positions=self.picking_positions, ticket_items=self.ticket_items,
                           planogram_csv_path=self.planogram_csv_path,
                           customer_properties_csv=self.customer_properties_path,
                           article_grouping_map=self.grouping_map)

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
            final_picking_positions[products['description'][pos]] = (
                int(products['picking_x'][pos]), int(products['picking_y'][pos]))
        return final_picking_positions

    def _load_ticket(self, ticket_path: str):
        ticket = pd.read_csv(ticket_path, delimiter=';')
        final_ticket = {}
        for index, row in ticket.iterrows():
            final_ticket[row['article_id']] = row['quantity']
        return final_ticket

    def _load_article_grouping(self, grouping_path: str):
        with open(grouping_path, 'r') as f:
            return json.load(f)


if __name__ == "__main__":
    env = EnvironmentBuilder(
        ticket_path='data/data/test_ticket.csv',
        planogram_csv_path='data/data/planogram_table.csv',
        customer_properties_path='data/data/hackathon_customers_properties.csv',
        grouping_path='data/data/article_group.json',
        obs_mode=1,
        reward_mode=1).build()

    env.step(1)
    env.step(1)

    env.step(3)
    env.step(3)
    env.step(3)
    env.step(0)

    env.render()

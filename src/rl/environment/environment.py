import copy

from src.rl.environment.observation_manager import ConvolutionalObservationManager, ObservationManager
from src.rl.environment.reward_manager import RewardManager

import gym
import tensorflow as tf
import pandas as pd
import numpy as np
import json
from gym.spaces import Discrete

class Environment(gym.Env):
    def __init__(self, obs_manager, reward_manager, picking_positions, ticket_items, planogram_csv_path,
                 customer_properties_csv, article_grouping_map):
        self.observation_manager = obs_manager
        self.reward_manager = reward_manager

        self._picking_positions = picking_positions
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

    def reset(self, seed=None):
        # set the start position
        self.customer_pos_x = 28
        self.customer_pos_y = 19
        # set the current time_per_step, and time_per_pick
        self.time_per_step, self.time_per_pick = self._get_customer_properties(self.customer_properties_df)
        # set start map situation from base map
        self.map = np.copy(self.base_map)
        self.pending_items = copy.deepcopy(self.ticket_items)
        # fill the articles to peek in the map
        self.picking_positions = copy.deepcopy(self._picking_positions)
        for item in self.ticket_items.keys():
            pos_x, pos_y = self.picking_positions[item]
            self.map[pos_y, pos_x] = self.article_grouping[item] + 4
        # get current observation
        observation = self.observation_manager.get_observation(self.customer_pos_x, self.customer_pos_y,
                                                               self.time_per_step, self.time_per_pick, self.map)
        return observation

    def step(self, action):
        # update the enviroment in function of the action
        if action == 0:
            article = self._pick_action()
            if article:
                # level 1 optimal get all units in one step
                self.pending_items.pop(article)
                self.picking_positions.pop(article)
                self.map[self.customer_pos_y, self.customer_pos_x] = 0
        else:
            # move agent
            self.customer_pos_x, self.customer_pos_y = self._movement_action(action)
        # generate the new observation
        observation = self.observation_manager.get_observation(self.customer_pos_x, self.customer_pos_y,
                                                               self.time_per_step, self.time_per_pick, self.map)
        # compute the reward
        reward = self.reward_manager.compute_reward(action, self.time_per_pick, self.time_per_step, self.pending_items)
        # check if is the end of the episode
        done = (self.map[self.customer_pos_y, self.customer_pos_x] == 2)
        return observation, reward, done, {}

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
        # fill the map with the fixed elements
        for idx, row in planogram_df.iterrows():
            cell_item = row["description"]
            x = row["x"] - 1
            y = row["y"] - 1
            # choice the value of the cell in function of the type
            if cell_item in cell_type_dict:
                src_map[y, x] = cell_type_dict[cell_item]
            else:
                src_map[y, x] = 1
        return src_map

    def _movement_action(self, action):
        customer_pos_x = self.customer_pos_x + self.action_map[action][0]
        customer_pos_y = self.customer_pos_y + self.action_map[action][1]
        return customer_pos_x, customer_pos_y

    def _pick_action(self):
        article_pick = None
        for article, position in self.picking_positions.items():
            if (position[0]) == self.customer_pos_x and (position[1]) == self.customer_pos_y:
                article_pick = article
                break
        if article_pick:
            return article_pick if article_pick in self.pending_items else None

    def _get_customer_properties(self, customer_properties):
        sample = customer_properties.sample(1)
        return sample["step_seconds"].item(), sample["picking_offset"].item()

    def render(self):
        for row in self.map:
            for cell in row:
                print(cell, end=' ')
            print()

    def get_valid_actions(self, **kargs):
        valid_actions = [True] if self._pick_action() else [False]
        for action in self.action_map.keys():
            nex_x, nex_y = self._movement_action(action)
            if nex_x < 0 or nex_x >= self.base_map.shape[1]:
                valid_actions.append(False)
                continue
            if nex_y < 0 or nex_y >= self.base_map.shape[0]:
                valid_actions.append(False)
                continue
            valid_actions.append(True)
        valid_actions = [i for i in range(5) if valid_actions[i]]
        return valid_actions

    def observation_and_action_constrain_splitter(self, observation):
        action_mask = [1] if self._pick_action() else [0]
        for action in self.action_map.keys():
            nex_x, nex_y = self._movement_action(action)
            if nex_x < 0 or nex_x >= self.base_map.shape[1]:
                action_mask.append(False)
                continue
            if nex_y < 0 or nex_y >= self.base_map.shape[0]:
                action_mask.append(False)
                continue
            action_mask.append(1)
        return observation,  tf.convert_to_tensor(action_mask, dtype=np.int32)


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
        # build the observation generator
        if self.obs_mode == 1:
            obs_manager = ConvolutionalObservationManager(self.num_cells)
        else:
            obs_manager = ObservationManager()
        # build the reward function utility
        if self.reward_mode == 1:
            reward_manager = RewardManager()
        else:
            raise NotImplementedError
        # build enviroment
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
                int(products['picking_x'][pos]) - 1, int(products['picking_y'][pos]) - 1)
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

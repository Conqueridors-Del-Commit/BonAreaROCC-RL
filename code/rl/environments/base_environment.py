import gymnasium as gym
import pandas as pd


class BaseEnvironment(gym.Env):
    def __init__(self, env_config):
        self.internal_state = BaseInternalState(planogram_csv_path=env_config['planogram_csv'])

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        self.env.render()


class BaseInternalState:
    def __init__(self, planogram_csv_path: str):
        self.cells = self._load_cells(planogram_csv_path)
        self.picking_positions = self._load_picking_positions(planogram_csv_path)

    def _load_cells(self, planogram_csv_path: str):
        products = pd.read_csv(planogram_csv_path, delimiter=';')
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


if __name__ == "__main__":
    env = BaseEnvironment(env_config={'planogram_csv': 'data/data/planogram_table.csv'})
    for key, val in env.internal_state.picking_positions.items():
        print(key, val)

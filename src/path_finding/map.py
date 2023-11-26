import pandas as pd
import numpy as np

OBSTACLE = 1
CORRIDOR = 0


class Map:
    def __init__(self, map_file):
        self.map_file = map_file
        self.planogram_df = pd.read_csv(self.map_file, delimiter=';')
        self.map = self.load_map()
        self.map_width = self.map.shape[0]
        self.map_height = self.map.shape[1]

    def load_map(self):
        width = self.planogram_df['x'].max()
        height = self.planogram_df['y'].max()
        src_map = np.zeros(shape=(height, width))
        cell_type_dict = ["paso", "paso-entrada", "paso-salida"]
        for idx, row in self.planogram_df.iterrows():
            cell_item = row["description"]
            x = row["x"] - 1
            y = row["y"] - 1
            if cell_item in cell_type_dict:
                src_map[y, x] = CORRIDOR
            else:
                src_map[y, x] = OBSTACLE

        return src_map

    def get_item_position(self, item):
        item_row = self.planogram_df[self.planogram_df['description'] == item]
        return item_row['picking_y'].values[0] - 1, item_row['picking_x'].values[0] - 1

    def is_obstacle(self, x, y):
        return self.map[x][y] == 1

    def __str__(self):
        result = ""
        rows = self.map.shape[0]
        cols = self.map.shape[1]
        for i in range(rows):
            for j in range(cols):
                result += str(self.map[i][j]) + "\t"
            result += "\n"

        return result.rstrip("\n")

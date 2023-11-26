import pandas as pd


class PickingTimes:
    def __init__(self, picking_times_csv):
        self.picking_times_path = picking_times_csv
        self.picking_times = self.load_picking_times()

    def load_picking_times(self):
        picking_times_df = pd.read_csv(self.picking_times_path, delimiter=';')
        picking_times = {}
        for idx, row in picking_times_df.iterrows():
            picking_times[row["article_id"]] = [row["first_pick"], row["second_pick"], row["third_pick"],
                                                row["fourth_pick"], row["fifth_more_pick"]]
        return picking_times

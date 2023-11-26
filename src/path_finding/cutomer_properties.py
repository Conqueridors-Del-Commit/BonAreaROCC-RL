import pandas as pd


class CustomerProperties:
    def __init__(self, customer_file_csv, customer_id):
        self.customer_file_csv = customer_file_csv
        self.customer_id = customer_id
        self.step_seconds, self.picking_offset = self.load_customer_properties()

    def load_customer_properties(self):
        customer_properties_df = pd.read_csv(self.customer_file_csv, delimiter=';')
        customer_properties = customer_properties_df[customer_properties_df['customer_id'] == self.customer_id]
        return customer_properties["step_seconds"].values[0], customer_properties["picking_offset"].values[0]

import pandas as pd
from datetime import datetime


class Ticket:
    def __init__(self, ticket_file):
        self.ticket_file = ticket_file
        self.ticket_df = pd.read_csv(self.ticket_file, delimiter=';')
        self.ticket = self.load_ticket()
        self.enter_time = datetime.strptime(self.ticket_df["enter_date_time"].values[0], '%m/%d/%Y %H:%M')
        self.customer_id = self.ticket_df["customer_id"].values[0]
        self.ticket_id = self.ticket_df["ticket_id"].values[0]

    def load_ticket(self):
        ticket = {}
        for idx, row in self.ticket_df.iterrows():
            ticket[row["article_id"]] = row["quantity"]
        return ticket

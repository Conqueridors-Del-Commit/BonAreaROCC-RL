import pandas as pd


class Ticket:
    def __init__(self, ticket_file):
        self.ticket_file = ticket_file
        self.ticket = self.load_ticket()

    def load_ticket(self):
        ticket_df = pd.read_csv(self.ticket_file, delimiter=';')
        ticket = {}
        for idx, row in ticket_df.iterrows():
            ticket[row["article_id"]] = row["quantity"]
        return ticket

from src.path_finding.map import Map
from src.path_finding.ticket import Ticket
from src.path_finding.cutomer_properties import CustomerProperties
from src.path_finding.picking_times import PickingTimes
from datetime import timedelta
import csv


def apply_action(current_pos, action):
    x = current_pos[0]
    y = current_pos[1]
    if action == 'N':
        return x, y - 1
    elif action == 'S':
        return x, y + 1
    elif action == 'E':
        return x + 1, y
    elif action == 'W':
        return x - 1, y
    else:
        raise Exception("Invalid action")


class CsvCreator:
    def __init__(self, result_path, solution, map, ticket, customer_properties, picking_times):
        self.path = result_path
        self.solution = solution
        self.map = map
        self.ticket = ticket
        self.ticket_positions = self._get_item_positions(ticket)
        self.customer_properties = customer_properties
        self.picking_times = picking_times

    def create_csv(self):
        init_time = self.ticket.enter_time
        current_time = init_time

        with open(self.path, 'w+', newline='') as csvfile:
            fieldnames = ['customer_id', 'ticket_id', 'x', 'y', 'picking', 'x_y_date_time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
            writer.writeheader()
            new_x, new_y = 29, 20
            for i in range(self.customer_properties.step_seconds):
                writer.writerow(
                    {'customer_id': self.ticket.customer_id, 'ticket_id': self.ticket.ticket_id, 'x': new_x, 'y': new_y,
                     'picking': 0, 'x_y_date_time': current_time.strftime('%Y-%m-%d %H:%M:%S')})
                current_time = current_time + timedelta(seconds=1)

            for action in self.solution:
                new_x, new_y = apply_action((new_x, new_y), action)
                # If current position has item to pick add picking rows
                if (new_y - 1, new_x - 1) in self.ticket_positions:
                    picking_time_customer = self.customer_properties.picking_offset
                    article = self.ticket_positions[(new_y - 1, new_x - 1)]
                    times_to_apply = self.ticket.ticket[article]
                    for i in range(times_to_apply):
                        index = min(i, len(self.picking_times.picking_times[article]) - 1)
                        time_to_pick = self.picking_times.picking_times[article][index]
                        total_time_to_pick = time_to_pick + picking_time_customer
                        for i in range(total_time_to_pick):
                            writer.writerow(
                                {'customer_id': self.ticket.customer_id, 'ticket_id': self.ticket.ticket_id,
                                 'x': new_x, 'y': new_y,
                                 'picking': 1, 'x_y_date_time': current_time.strftime('%Y-%m-%d %H:%M:%S')})
                            current_time = current_time + timedelta(seconds=1)

                for i in range(self.customer_properties.step_seconds):
                    writer.writerow(
                        {'customer_id': self.ticket.customer_id, 'ticket_id': self.ticket.ticket_id, 'x': new_x,
                         'y': new_y,
                         'picking': 0, 'x_y_date_time': current_time.strftime('%Y-%m-%d %H:%M:%S')})
                    current_time = current_time + timedelta(seconds=1)

    def _get_item_positions(self, ticket):
        item_positions = {}
        for elem in ticket.ticket.keys():
            x, y = self.map.get_item_position(elem)
            item_positions[(int(x), int(y))] = elem
        return item_positions


class CsvCreatorBuilder:

    def __init__(self, result_path, solution, ticket_id):
        self.result_path = result_path
        self.solution = solution
        self.ticket_id = ticket_id

    def build(self):
        supermarket_map = Map(map_file='data/data/planogram_table.csv')
        ticket = Ticket(ticket_file=f'data/data/tickets/{self.ticket_id}.csv')
        customer_properties = CustomerProperties(customer_file_csv='data/data/hackathon_customers_properties.csv',
                                                 customer_id=ticket.customer_id)
        picking_times = PickingTimes(picking_times_csv='data/data/hackathon_article_picking_time.csv')
        return CsvCreator(result_path=self.result_path, solution=self.solution, map=supermarket_map, ticket=ticket,
                          customer_properties=customer_properties, picking_times=picking_times)

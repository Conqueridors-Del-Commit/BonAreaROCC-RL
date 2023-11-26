from src.path_finding.problem import Level1MiniProblemBuilder, Level1ProblemBuilder
from src.path_finding.search import aStarSearch
from src.path_finding.csv_creator import CsvCreatorBuilder
import pandas as pd
import time
import math
import util


def find_closest_item(problem, initial_position, visited_positions):
    min_dist = math.inf
    closest_item = None
    for item in problem.ticket_positions:
        if item not in visited_positions:
            distance = util.manhattan_distance(initial_position, item)
            if distance < min_dist:
                min_dist = distance
                closest_item = item
    return closest_item


def find_path(ticket_csv_path):
    before = time.time()
    init_pos = (19, 28)
    big_problem = Level1ProblemBuilder(ticket_file_path=ticket_csv_path).build()
    visited_positions = []
    final_solution = []
    for i in range(len(big_problem.ticket_positions)):
        final_pos = find_closest_item(big_problem, init_pos, visited_positions)
        visited_positions.append(final_pos)
        problem = Level1MiniProblemBuilder(init_pos, final_pos).build()
        init_pos = final_pos
        solution = aStarSearch(problem)
        final_solution += solution
    print(final_solution)
    print("Time: ", time.time() - before)
    ticket_id = big_problem.ticket.ticket_id
    creator = CsvCreatorBuilder(result_path=f'data/results/{ticket_id}_result.csv', solution=final_solution).build()
    creator.create_csv()


def find_all_tickets_paths():
    all_tickets = pd.read_csv('data/data/hackathon_tickets.csv', delimiter=';')
    ticket_ids = all_tickets['ticket_id'].unique()
    for ticket_id in ticket_ids:
        find_path(f'{ticket_id}.csv')


if __name__ == "__main__":
    find_all_tickets_paths()

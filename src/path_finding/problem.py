from src.path_finding.map import Map
from src.path_finding.ticket import Ticket
from src.path_finding.node import Node


class SearchProblem:
    """
    This class outlines the structure of a search problem
    """

    def get_start_state(self):
        """
        Returns the start state for the search problem.
        """
        raise NotImplementedError

    def is_goal_state(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        raise NotImplementedError

    def get_successors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        raise NotImplementedError

    def get_cost_of_actions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        raise NotImplementedError


class Directions:
    """
    Class to store the directions
    """
    NORTH = (-1, 0)
    SOUTH = (1, 0)
    EAST = (0, 1)
    WEST = (0, -1)
    DO_NOTHING = (0, 0)


class DummyProblem(SearchProblem):
    def __init__(self):
        self.grid = [['A', 'X', 'X'], ['A', 'A', 'A'], ['X', 'X', 'A']]
        self.start_state = (0, 0)
        self.goal_state = (2, 2)
        self.actions = {
            'N': Directions.NORTH,
            'S': Directions.SOUTH,
            'E': Directions.EAST,
            'W': Directions.WEST
        }

    def get_start_state(self):
        return self.start_state

    def is_goal_state(self, state):
        return state == self.goal_state

    def get_successors(self, state):
        successors = []
        for action in self.actions.keys():
            x, y = state
            dx, dy = self.actions[action]
            new_x = x + dx
            new_y = y + dy
            if 0 <= new_x < len(self.grid) and 0 <= new_y < len(self.grid[0]):
                if not self._is_obstacle(new_x, new_y):
                    successors.append(((new_x, new_y), action, 1))

        return successors

    def _is_obstacle(self, x, y):
        return self.grid[x][y] == 'X'

    def get_cost_of_actions(self, actions):
        return len(actions)


class Level1Problem:
    def __init__(self, map, ticket):
        # Map with corridors and walls
        self.map = map
        # ticket
        self.ticket = ticket
        # List of items to be picked in the ticket
        self.ticket_positions = self._get_item_positions(ticket)
        # Start state
        self.start_pos = (19, 28)
        # Actions dictionary
        self.actions = {
            'N': Directions.NORTH,
            'S': Directions.SOUTH,
            'E': Directions.EAST,
            'W': Directions.WEST,
        }

    def _get_item_positions(self, ticket):
        item_positions = []
        for elem in ticket.ticket.keys():
            x, y = self.map.get_item_position(elem)
            item_positions.append((int(x), int(y)))
        return item_positions

    def get_start_state(self):
        return self.start_pos, tuple(pos for pos in self.ticket_positions[:7])

    def is_goal_state(self, state: Node):
        position, remaining_items = state
        return len(remaining_items) == 0

    def get_successors(self, state):
        successors = []
        for action in self.actions.keys():
            x, y = state[0]
            dx, dy = self.actions[action]
            new_x = x + dx
            new_y = y + dy
            if 0 <= new_x < self.map.map_width and 0 <= new_y < self.map.map_height:
                if not self.map.is_obstacle(new_x, new_y):
                    positions = state[1]
                    items = tuple(pos for pos in positions if pos != (new_x, new_y))
                    successors.append((((new_x, new_y), items), action, 1))

        return successors


class Level1ProblemBuilder:
    def build(self):
        supermarket_map = Map(map_file='data/data/planogram_table.csv')
        ticket = Ticket(ticket_file='data/data/test_ticket.csv')
        return Level1Problem(map=supermarket_map, ticket=ticket)


class Level1MiniProblem:
    def __init__(self, map, start_pos, final_pos):
        # Map with corridors and walls
        self.map = map
        # Start state
        self.start_pos = start_pos
        # Final state
        self.final_pos = final_pos
        # Actions dictionary
        self.actions = {
            'N': Directions.NORTH,
            'S': Directions.SOUTH,
            'E': Directions.EAST,
            'W': Directions.WEST,
        }

    def get_start_state(self):
        return self.start_pos

    def is_goal_state(self, state: Node):
        return self.final_pos == state

    def get_successors(self, state):
        successors = []
        for action in self.actions.keys():
            x, y = state
            dx, dy = self.actions[action]
            new_x = x + dx
            new_y = y + dy
            if 0 <= new_x < self.map.map_width and 0 <= new_y < self.map.map_height:
                if not self.map.is_obstacle(new_x, new_y):
                    successors.append(((new_x, new_y), action, 1))

        return successors


class Level1MiniProblemBuilder:
    def __init__(self, initial_pos, final_pos):
        self.initial_pos = initial_pos
        self.final_pos = final_pos

    def build(self):
        supermarket_map = Map(map_file='data/data/planogram_table.csv')
        return Level1MiniProblem(map=supermarket_map, start_pos=self.initial_pos,
                                 final_pos=self.final_pos)


if __name__ == "__main__":
    problem = Level1ProblemBuilder().build()
    print(problem.ticket_positions)

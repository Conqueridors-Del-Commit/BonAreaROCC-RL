class SearchProblem:
    """
    This class outlines the structure of a search problem
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        raise NotImplementedError

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        raise NotImplementedError

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        raise NotImplementedError

    def getCostOfActions(self, actions):
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

    def getStartState(self):
        return self.start_state

    def isGoalState(self, state):
        return state == self.goal_state

    def getSuccessors(self, state):
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

    def getCostOfActions(self, actions):
        return len(actions)
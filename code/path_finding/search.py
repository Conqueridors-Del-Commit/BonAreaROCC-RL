import util
import node


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    # Función para calcular el coste de un nodo
    def priorityFunction(node):
        return node.get_cost() + heuristic(node.get_state(), problem)

    fringe = util.NodePriorityQueueWithFunction(priorityFunction=priorityFunction)
    fringe.push(node.Node(problem.getStartState()))
    expanded = set()

    while True:
        if fringe.isEmpty():
            raise Exception("Empty fringe")
        actual_node = fringe.pop()
        if problem.isGoalState(actual_node.get_state()):
            return actual_node.total_path()
        expanded.add(actual_node.get_state())
        for succ, act, cost in problem.getSuccessors(actual_node.get_state()):
            succesor_node = node.Node(succ, actual_node, act, cost + actual_node.get_cost())
            if succesor_node.get_state() not in expanded:
                fringe.update(succesor_node)

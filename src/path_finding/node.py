class Node:
    def __init__(self, _state, _parent=None, _action=None, _cost=0):
        self.state = _state
        self.parent = _parent
        self.action = _action
        self.cost = _cost

    def total_path(self):
        if not self.parent:
            return []
        else:
            return self.parent.total_path() + [self.action]

    def get_state(self):
        return self.state

    def get_cost(self):
        return self.cost

    def __str__(self):
        return f"Node[{self.state},{self.action},{self.cost}]"

    def __repr__(self):
        return self.__str__()

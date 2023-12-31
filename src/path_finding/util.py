import heapq
import pandas as pd


class PriorityQueue:
    """
      Implements a priority queue data structure. Each inserted item
      has a priority associated with it and the client is usually interested
      in quick retrieval of the lowest-priority item in the queue. This
      data structure allows O(1) access to the lowest-priority item.
    """

    def __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)


class NodePriorityQueue(PriorityQueue):

    def update(self, item, priority):
        for index, (p, c, i) in enumerate(self.heap):
            if i.get_state() == item.get_state():
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)


class PriorityQueueWithFunction(PriorityQueue):
    """
    Implements a priority queue with the same push/pop signature of the
    Queue and the Stack classes. This is designed for drop-in replacement for
    those two classes. The caller has to provide a priority function, which
    extracts each item's priority.
    """

    def __init__(self, priorityFunction):
        "priorityFunction (item) -> priority"
        self.priorityFunction = priorityFunction  # store the priority function
        PriorityQueue.__init__(self)  # super-class initializer

    def push(self, item):
        "Adds an item to the queue with priority from the priority function"
        PriorityQueue.push(self, item, self.priorityFunction(item))


class NodePriorityQueueWithFunction(PriorityQueueWithFunction):
    def update(self, item):
        for index, (p, c, i) in enumerate(self.heap):
            if i.get_state() == item.get_state():
                if p <= self.priorityFunction(item):
                    break
                del self.heap[index]
                self.heap.append((self.priorityFunction(item), c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item)


def manhattan_distance(state, goal_state):
    """
    Manhattan distance heuristic
    """
    x1, y1 = state
    x2, y2 = goal_state
    return abs(x1 - x2) + abs(y1 - y2)


def separate_tickets_into_different_csv_files():
    """
    Separate tickets into different csv files
    """
    all_tickets = pd.read_csv('data/data/hackathon_tickets.csv', delimiter=';')
    ticket_ids = all_tickets['ticket_id'].unique()
    for ticket_id in ticket_ids:
        ticket = all_tickets[all_tickets['ticket_id'] == ticket_id]
        ticket = ticket.set_index('enter_date_time')
        ticket.index = pd.to_datetime(ticket.index)
        ticket.to_csv(f'data/data/tickets/{ticket_id}.csv', index=True, sep=';')


def concatenate_tickets():
    # Concatenate all tickets located in data/results into one csv file
    all_tickets = pd.read_csv('data/data/hackathon_tickets.csv', delimiter=';')
    ticket_ids = all_tickets['ticket_id'].unique()
    concatenated_tickets = pd.DataFrame()
    for ticket_id in ticket_ids:
        ticket = pd.read_csv(f'data/results/{ticket_id}_result.csv', delimiter=';')
        concatenated_tickets = pd.concat([concatenated_tickets, ticket])

    concatenated_tickets.to_csv('data/results/all_tickets.csv', index=False, sep=';')


if __name__ == "__main__":
    #separate_tickets_into_different_csv_files()
    concatenate_tickets()

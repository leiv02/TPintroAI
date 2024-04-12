class Node:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.g = 0  # cost from start node to current node
        self.h = 0  # heuristic estimate from current node to goal node
        self.f = 0  # total estimated cost of the cheapest path from start to goal through current node

heuristic_values = {'A': 2, 'C': 2, 'D': 1, 'B': 5, 'Goal': 0}

costs = {
    ('Start', 'A'): 2,
    ('Start', 'B'): 3,
    ('Start', 'D'): 5,
    ('B', 'D'): 4,
    ('D', 'Goal'): 5,
    ('D', 'C'): 1,
    ('C', 'Goal'): 2,
    ('A', 'C'): 4
}

def successor(node):
    successors = []
    for key in costs:
        if key[0] == node.name:
            successors.append(key[1])
    return successors

def heuristic(state, goal_state):
    return heuristic_values[state]

def astar(start_node, goal_node):
    fringe = [start_node]
    closed = []

    while fringe:
        current_node = min(fringe, key=lambda x: x.f)
        fringe.remove(current_node)
        closed.append(current_node)

        if current_node.name == goal_node.name:
            return construct_path(current_node)

        successors = successor(current_node)
        for child_state in successors:
            child_node = Node(child_state, current_node)
            child_node.g = current_node.g + costs[(current_node.name, child_state)]
            child_node.h = heuristic(child_node.name, goal_node.name)
            child_node.f = child_node.g + child_node.h

            existing_node_fringe = next((node for node in fringe if node.name == child_node.name), None)
            existing_node_closed = next((node for node in closed if node.name == child_node.name), None)

            if existing_node_fringe and existing_node_fringe.f < child_node.f:
                continue
            if existing_node_closed and existing_node_closed.f < child_node.f:
                continue

            fringe.append(child_node)

    return None  # no path found

def construct_path(node):
    path = []
    while node:
        path.append(node.name)
        node = node.parent
    return path[::-1]  # reverse the path to get from start to goal

# Example usage:
start_node = Node('Start')
goal_node = Node('Goal')

solution_path = astar(start_node, goal_node)
if solution_path:
    print("Path found:", solution_path)
else:
    print("No path found.")

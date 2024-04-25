from collections import deque

graph = {
    'Start': {'B': 3, 'D': 5, 'A': 2},
    'A': {'C': 4},
    'B': {'D': 4},
    'C': {'Goal': 2},
    'D': {'Goal': 5, 'C': 1},
    'Goal': {}
}



def bfs(graph, start):
    visited = set()
    queue = deque([start])
    result = []
    
    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            result.append(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    queue.append(neighbor)
    return result

def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    
    result = [start]
    
    for neighbor in graph[start]:
        if neighbor not in visited:
            result.extend(dfs(graph, neighbor, visited))
    return result

# Example usage
bfs_result = bfs(graph, 'Start')
dfs_result = dfs(graph, 'Start')

print("BFS Visit Order:", bfs_result)
print("DFS Visit Order:", dfs_result)

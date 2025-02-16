import numpy as np
import heapq


def heuristic(a, b):
    # Chebyshev distance for 3D
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    dz = abs(a[2] - b[2])
    return max(dx, dy, dz)

def get_neighbors(point, grid):
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue  # Skip the current point itself
                neighbor = (point[0] + dx, point[1] + dy, point[2] + dz)
                # Check if the neighbor is within bounds and not an obstacle
                if (0 <= neighbor[0] < grid.shape[0] and
                    0 <= neighbor[1] < grid.shape[1] and
                    0 <= neighbor[2] < grid.shape[2] and
                    not grid[neighbor]):
                    neighbors.append(np.array(neighbor))
    return neighbors

def a_star(start, goal, grid):
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}
    
    while frontier:
        current_priority, current = heapq.heappop(frontier)
        
        if current == goal:
            break  # Reached the goal
        
        current_np = np.array(current)
        for next_pos in get_neighbors(current_np, grid):
            next_pos_tuple = tuple(next_pos)
            new_cost = cost_so_far[current] + 1
            if (next_pos_tuple not in cost_so_far or 
                new_cost < cost_so_far.get(next_pos_tuple, float('inf'))):
                cost_so_far[next_pos_tuple] = new_cost
                priority = new_cost + heuristic(next_pos, goal)
                heapq.heappush(frontier, (priority, next_pos_tuple))
                came_from[next_pos_tuple] = current
    
    return came_from


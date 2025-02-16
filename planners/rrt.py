import numpy as np
from collections import defaultdict

class RRTPlanner:
    def __init__(self, voxel_grid, start, goal, step_size=1.0, max_iterations=1000):
        self.voxel_grid = voxel_grid
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.vertices = {tuple(start): None}  # key: vertex, value: parent
        self.dimensions = voxel_grid.shape

    def is_valid_point(self, point):
        # Check if point is within bounds and collision-free
        return (all(0 <= p < d for p, d in zip(point, self.dimensions)) and 
                not self.voxel_grid[tuple(map(int, point))])

    def is_valid_edge(self, p1, p2):
        # Check if the edge between p1 and p2 is collision-free
        vec = p2 - p1
        distance = np.linalg.norm(vec)
        if distance == 0:
            return True
        
        # Check points along the edge
        num_points = int(distance / (self.step_size * 0.5))
        for i in range(1, num_points):
            point = p1 + (vec * i / num_points)
            if not self.is_valid_point(point):
                return False
        return True

    def sample_random_point(self):
        return np.random.uniform(0, np.array(self.dimensions) - 1)

    def find_nearest_vertex(self, point):
        vertices = np.array(list(self.vertices.keys()))
        distances = np.linalg.norm(vertices - point, axis=1)
        return tuple(vertices[np.argmin(distances)])

    def extend_tree(self, nearest, random_point):
        direction = random_point - np.array(nearest)
        norm = np.linalg.norm(direction)
        if norm == 0:
            return None
        
        new_point = np.array(nearest) + direction/norm * self.step_size
        if self.is_valid_edge(np.array(nearest), new_point):
            return tuple(new_point)
        return None

    def plan(self):
        for _ in range(self.max_iterations):
            if np.random.random() < 0.1:  # 10% chance to sample goal
                random_point = self.goal
            else:
                random_point = self.sample_random_point()

            nearest = self.find_nearest_vertex(random_point)
            new_point = self.extend_tree(nearest, random_point)

            if new_point is not None and new_point not in self.vertices:
                self.vertices[new_point] = nearest

                # Check if we can connect to goal
                if (np.linalg.norm(np.array(new_point) - self.goal) < self.step_size and 
                    self.is_valid_edge(np.array(new_point), self.goal)):
                    self.vertices[tuple(self.goal)] = new_point
                    return self.extract_path(), dict(self.vertices)  # Return both path and vertices

        return None, None  # Return None for both if no path found

    def extract_path(self):
        path = []
        current = tuple(self.goal)
        while current is not None:
            path.append(current)
            current = self.vertices[current]
        return path[::-1]  # Reverse path to get start-to-goal

import numpy as np
from collections import defaultdict
from enum import Enum

class NodeType(Enum):
    FREE = "free"      # All surrounding voxels are free
    EDGE = "edge"      # At least one surrounding voxel is occupied
    INSIDE = "inside"  # All surrounding voxels are occupied

class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.env_vector = None
        self.node_type = None

class RRTPlanner:
    def __init__(self, voxel_grid, start, goal, step_size=1.0, max_iterations=1000):
        self.voxel_grid = voxel_grid
        self.dimensions = voxel_grid.shape
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.step_size = step_size
        self.max_iterations = max_iterations

        # Initialize start node with environment information
        start_node = Node(tuple(start))
        start_node.env_vector = self._get_environment_vector(start)
        start_node.node_type = self._classify_node(start_node.env_vector)
        self.vertices = {tuple(start): start_node}
        
        

    def _get_environment_vector(self, point) -> np.ndarray:
        """Get 7-dimensional environment vector with offset 2"""
        directions = [
            (0, 0, 2),   # up
            (0, 0, -2),  # down
            (-2, 0, 0),  # left
            (2, 0, 0),   # right
            (0, 2, 0),   # front
            (0, -2, 0),  # back
            (0, 0, 0)    # current
        ]
        
        env_vector = np.zeros(7)
        point_int = np.round(point).astype(int)
        
        for i, offset in enumerate(directions):
            check_point = tuple(map(sum, zip(point_int, offset)))
            if all(0 <= p < d for p, d in zip(check_point, self.dimensions)):
                env_vector[i] = float(self.voxel_grid[check_point])
            else:
                env_vector[i] = 1.0  # Treat out-of-bounds as occupied
                
        return env_vector

    def _classify_node(self, env_vector) -> NodeType:
        """Classify node based on environment vector"""
        # Exclude current position (last element) from classification
        surrounding_vector = env_vector[:-1]
        
        if np.all(surrounding_vector == 0):
            return NodeType.FREE
        elif np.all(surrounding_vector == 1):
            return NodeType.INSIDE
        else:
            return NodeType.EDGE

    def is_valid_point(self, point):
        """Check if point is within bounds and collision-free, only checking current position"""
        try:
            # Convert point to integers for grid checking
            point_int = tuple(map(int, np.round(point)))
            
            # Check bounds
            if not all(0 <= p < d for p, d in zip(point_int, self.dimensions)):
                return False
            
            # Get environment vector and classify node
            env_vector = self._get_environment_vector(point)
            node_type = self._classify_node(env_vector)
            
            # Inside points are considered invalid
            if node_type == NodeType.INSIDE:
                return False
            
            # For Edge and Free points, only check the current position
            # This is the last element in the environment vector
            return not bool(env_vector[-1])  # False if current position is free, True if occupied

        except:
            return False

    def is_valid_edge(self, p1, p2):
        """Check if the edge between p1 and p2 is collision-free and return nearest valid point"""
        vec = p2 - p1
        distance = np.linalg.norm(vec)
        if distance == 0:
            return True, p2
        
        # Increase the number of points to check along the edge
        num_points = max(int(distance), 10)  # At least 10 points
        last_valid_point = p1
        last_valid_edge_point = None
        
        # Check points along the edge
        for i in range(num_points + 1):
            point = p1 + (vec * i / num_points)
            if not self.is_valid_point(point):
                # If we hit an invalid point, return the last valid edge-type point
                # or the last valid point if no edge-type point was found
                return False, (last_valid_edge_point if last_valid_edge_point is not None else last_valid_point)
            
            # Check if this valid point is an edge-type point
            env_vector = self._get_environment_vector(point)
            node_type = self._classify_node(env_vector)
            if node_type == NodeType.EDGE:
                last_valid_edge_point = point.copy()
            
            last_valid_point = point.copy()
            
        return True, p2

    def sample_random_point(self):
        """Sample random point with special handling for inside nodes"""
        random_point = np.random.uniform(0, np.array(self.dimensions) - 1)
        
        # Get environment info for the random point
        env_vector = self._get_environment_vector(random_point)
        node_type = self._classify_node(env_vector)
        
        # If it's an inside point, find the closest edge-type point towards goal
        if node_type == NodeType.INSIDE:
            return self._find_nearest_edge_point(random_point)
        
        return random_point

    def _find_nearest_edge_point(self, start_point):
        """Find the nearest edge-type point in the direction of the goal"""
        direction = self.goal - start_point
        direction = direction / np.linalg.norm(direction)
        
        current_point = start_point.copy()
        max_steps = int(max(self.dimensions))  # Maximum steps to search
        
        for step in range(1, max_steps):
            # Take steps towards goal
            test_point = start_point + direction * step
            test_point_int = np.round(test_point).astype(int)
            
            # Check bounds
            if not all(0 <= p < d for p, d in zip(test_point_int, self.dimensions)):
                break
                
            # Get environment info
            env_vector = self._get_environment_vector(test_point)
            node_type = self._classify_node(env_vector)
            
            # If we found an edge point, return it
            if node_type == NodeType.EDGE:
                return test_point
            
            # If we hit a free point, return the last edge point we found
            if node_type == NodeType.FREE:
                return current_point
                
            current_point = test_point
            
        # If no edge point found, return the original point
        return start_point

    def _check_direction_compatibility(self, env_vector, direction):
        """Check if direction matches environment vector"""
        # Get primary movement direction components
        dx = np.sign(direction[0])
        dy = np.sign(direction[1])
        dz = np.sign(direction[2])
        
        # Check corresponding environment vector elements
        checks = []
        if dx < 0:
            checks.append((2, env_vector[2]))  # left
        elif dx > 0:
            checks.append((3, env_vector[3]))  # right
            
        if dy < 0:
            checks.append((5, env_vector[5]))  # back
        elif dy > 0:
            checks.append((4, env_vector[4]))  # front
            
        if dz < 0:
            checks.append((1, env_vector[1]))  # down
        elif dz > 0:
            checks.append((0, env_vector[0]))  # up
            
        # Return True if all relevant directions are free (0)
        return all(value == 0 for _, value in checks)

    def find_nearest_vertex(self, point):
        """Find nearest vertex with compatible environment vector"""
        vertices = np.array([node.position for node in self.vertices.values()])
        distances = np.linalg.norm(vertices - point, axis=1)
        
        # Get indices of vertices sorted by distance
        sorted_indices = np.argsort(distances)
        
        # Check vertices in order of increasing distance
        for idx in sorted_indices:
            vertex_pos = tuple(vertices[idx])
            vertex_node = self.vertices[vertex_pos]
            
            # Calculate direction from vertex to target point
            direction = point - vertices[idx]
            if np.linalg.norm(direction) == 0:
                continue
                
            direction = direction / np.linalg.norm(direction)
            
            # Check if vertex's environment allows movement in this direction
            if self._check_direction_compatibility(vertex_node.env_vector, direction):
                return vertex_pos
            
        # If no compatible vertex found, return the nearest one
        return tuple(vertices[sorted_indices[0]])

    def extend_tree(self, nearest, random_point):
        """Modified extend_tree to use full step or nearest edge point"""
        direction = random_point - np.array(nearest)
        norm = np.linalg.norm(direction)
        if norm == 0:
            return None
        
        # First check if we can go directly to random_point
        is_valid, valid_point = self.is_valid_edge(np.array(nearest), random_point)
        
        if is_valid:
            # Use random_point directly if path is valid
            new_node = Node(tuple(random_point))
            new_node.env_vector = self._get_environment_vector(random_point)
            new_node.node_type = self._classify_node(new_node.env_vector)
        else:
            # Use the valid_point (which should be edge-type or closest to edge)
            new_node = Node(tuple(valid_point))
            new_node.env_vector = self._get_environment_vector(valid_point)
            new_node.node_type = self._classify_node(new_node.env_vector)
            
            # If we got an inside point, try to find nearest edge point
            if new_node.node_type == NodeType.INSIDE:
                edge_point = self._find_nearest_edge_point(valid_point)
                if not np.array_equal(edge_point, valid_point):
                    new_node = Node(tuple(edge_point))
                    new_node.env_vector = self._get_environment_vector(edge_point)
                    new_node.node_type = self._classify_node(edge_point.env_vector)
        
        new_node.parent = nearest
        return new_node

    def _get_biased_point(self, from_point, current_point, goal):
        """Calculate new point based on movement direction and goal direction"""
        dir_from_point = np.array(current_point) - np.array(from_point)
        dir_to_goal = np.array(goal) - np.array(current_point)
        
        # Normalize directions
        if np.linalg.norm(dir_from_point) != 0:
            dir_from_point = dir_from_point / np.linalg.norm(dir_from_point)
        if np.linalg.norm(dir_to_goal) != 0:
            dir_to_goal = dir_to_goal / np.linalg.norm(dir_to_goal)
        
        # Combine directions
        combined_dir = (dir_from_point + dir_to_goal) / 2
        if np.linalg.norm(combined_dir) != 0:
            combined_dir = combined_dir / np.linalg.norm(combined_dir)
            
        return tuple(np.array(current_point) + combined_dir * self.step_size)

    def plan(self):
        for _ in range(self.max_iterations):
            # if np.random.random() < 0.1:
            #     random_point = self.goal
            # else:
            #     random_point = self.sample_random_point()

            random_point = self.sample_random_point()

            nearest = self.find_nearest_vertex(random_point)
            new_node = self.extend_tree(nearest, random_point)

            if new_node is not None and new_node.position not in self.vertices:
                # Add the new node to vertices
                self.vertices[new_node.position] = new_node

                # Try to connect to goal without adding intermediate nodes
                # if np.linalg.norm(np.array(new_node.position) - self.goal) < self.step_size:
                is_valid, _ = self.is_valid_edge(np.array(new_node.position), self.goal)
                if is_valid:
                    # Add goal node directly
                    goal_node = Node(tuple(self.goal))
                    goal_node.parent = new_node.position
                    goal_node.env_vector = self._get_environment_vector(self.goal)
                    goal_node.node_type = self._classify_node(goal_node.env_vector)
                    self.vertices[tuple(self.goal)] = goal_node
                    return self.extract_path(), self.get_vertex_data()

                # Process free-type nodes after goal connection attempt
                if new_node.node_type == NodeType.FREE:
                    # Calculate biased point
                    biased_point = self._get_biased_point(
                        nearest,
                        new_node.position,
                        self.goal
                    )
                    # Try to extend from current position using biased point
                    edge_node = self.extend_tree(new_node.position, biased_point)
                    
                    # Add edge-type node if found
                    if (edge_node is not None and 
                        (edge_node.node_type == NodeType.EDGE or edge_node.node_type == NodeType.FREE)and 
                        edge_node.position not in self.vertices):
                        edge_node.parent = new_node.position
                        self.vertices[edge_node.position] = edge_node
                        
                        # Try to connect new edge node to goal
                        if np.linalg.norm(np.array(edge_node.position) - self.goal) < self.step_size:
                            is_valid, _ = self.is_valid_edge(np.array(edge_node.position), self.goal)
                            if is_valid:
                                goal_node = Node(tuple(self.goal))
                                goal_node.parent = edge_node.position
                                goal_node.env_vector = self._get_environment_vector(self.goal)
                                goal_node.node_type = self._classify_node(goal_node.env_vector)
                                self.vertices[tuple(self.goal)] = goal_node
                                return self.extract_path(), self.get_vertex_data()

        return None, None

    def get_vertex_data(self):
        """Convert vertex data to serializable format with environment information"""
        return {
            pos: {
                'position': node.position,
                'parent': node.parent,
                'env_vector': node.env_vector.tolist(),
                'node_type': node.node_type.value
            }
            for pos, node in self.vertices.items()
        }

    def extract_path(self):
        if tuple(self.goal) not in self.vertices:
            return None
        
        path = []
        current = tuple(self.goal)
        while current is not None:
            path.append(current)
            current = self.vertices[current].parent
        return path[::-1]

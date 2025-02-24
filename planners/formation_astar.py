import numpy as np
import heapq
from typing import List, Tuple, Dict, Set
from itertools import product

class FormationAstar:
    def __init__(self, grid, formation_shape="square", formation_size=2):
        """
        Initialize formation A* planner
        Args:
            grid: 3D numpy array representing the environment
            formation_shape: "square", "diamond", "line", or "custom"
            formation_size: Size of the formation (distance between drones)
        """
        self.grid = grid
        self.formation_shape = formation_shape
        self.formation_size = formation_size
        self.formation_offsets = self._get_formation_offsets()
        
    def _get_formation_offsets(self) -> List[Tuple[int, int, int]]:
        """Define relative positions of drones in formation"""
        if self.formation_shape == "square":
            return [
                (0, 0, 0),  # Leader
                (self.formation_size, 0, 0),  # Right
                (0, self.formation_size, 0),  # Back
                (self.formation_size, self.formation_size, 0)  # Diagonal
            ]
        elif self.formation_shape == "diamond":
            return [
                (0, 0, 0),  # Leader
                (self.formation_size, 0, 0),  # Right
                (0, self.formation_size, 0),  # Back
                (-self.formation_size, 0, 0)  # Left
            ]
        elif self.formation_shape == "line":
            return [
                (0, 0, 0),  # Leader
                (self.formation_size, 0, 0),  # Right
                (2*self.formation_size, 0, 0),  # Far right
                (3*self.formation_size, 0, 0)  # Furthest right
            ]
        else:  # Custom formation
            return [
                (0, 0, 0),  # Leader
                (self.formation_size, self.formation_size, 0),
                (-self.formation_size, self.formation_size, 0),
                (0, 2*self.formation_size, 0)
            ]

    def _check_formation_collision(self, leader_pos: Tuple[int, int, int]) -> bool:
        """Check if formation at given leader position collides with obstacles"""
        for offset in self.formation_offsets:
            pos = tuple(map(sum, zip(leader_pos, offset)))
            # Check bounds
            if not (0 <= pos[0] < self.grid.shape[0] and 
                   0 <= pos[1] < self.grid.shape[1] and 
                   0 <= pos[2] < self.grid.shape[2]):
                return True
            # Check collision
            if self.grid[pos]:
                return True
        return False

    def _get_formation_neighbors(self, leader_pos: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """Get valid neighboring positions for the formation"""
        neighbors = []
        # Define possible movements for the leader
        moves = list(product([-1, 0, 1], repeat=3))
        moves.remove((0, 0, 0))  # Remove staying in place
        
        for move in moves:
            new_pos = tuple(map(sum, zip(leader_pos, move)))
            if not self._check_formation_collision(new_pos):
                neighbors.append(new_pos)
        return neighbors

    def _heuristic(self, a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
        """Calculate heuristic distance between two points"""
        return max(abs(a[i] - b[i]) for i in range(3))

    def plan_formation_path(self, start: Tuple[int, int, int], 
                          goal: Tuple[int, int, int]) -> Tuple[List[Tuple[int, int, int]], List[List[Tuple[int, int, int]]]]:
        """
        Plan path for drone formation
        Returns:
            Tuple of (leader_path, all_drone_paths)
        """
        if self._check_formation_collision(start) or self._check_formation_collision(goal):
            return None, None

        # Initialize A* algorithm
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        while frontier:
            _, current = heapq.heappop(frontier)

            if current == goal:
                break

            for next_pos in self._get_formation_neighbors(current):
                new_cost = cost_so_far[current] + 1

                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self._heuristic(next_pos, goal)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current

        # Reconstruct leader path
        if goal not in came_from:
            return None, None

        leader_path = []
        current = goal
        while current is not None:
            leader_path.append(current)
            current = came_from[current]
        leader_path.reverse()

        # Generate paths for all drones
        all_drone_paths = []
        for offset in self.formation_offsets:
            drone_path = [(tuple(map(sum, zip(pos, offset)))) for pos in leader_path]
            all_drone_paths.append(drone_path)

        return leader_path, all_drone_paths

def plan_formation_astar(grid, start, goal, formation_shape="square", formation_size=2):
    """Wrapper function for formation A* planning"""
    planner = FormationAstar(grid, formation_shape, formation_size)
    return planner.plan_formation_path(start, goal)

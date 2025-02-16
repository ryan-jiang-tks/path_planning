from enum import Enum
from planners.astar import a_star
from planners.rrt import RRTPlanner
from planners.base_pso import PSO

class PlannerType(Enum):
    ASTAR = "astar"
    RRT = "rrt"
    PSO = "pso"

class PathPlanner:
    def __init__(self, voxel_grid):
        self.voxel_grid = voxel_grid
        self.planners = {
            PlannerType.ASTAR: self._astar_plan,
            PlannerType.RRT: self._rrt_plan,
            PlannerType.PSO: self._pso_plan
        }

    def _astar_plan(self, start, goal, **kwargs):
        path_dict = a_star(start, goal, self.voxel_grid)
        if not path_dict:
            return None
        
        # Convert A* dictionary to path list
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = path_dict.get(current, None)
        return path[::-1]  # Reverse to get start-to-goal order

    def _rrt_plan(self, start, goal, **kwargs):
        step_size = kwargs.get('step_size', 1.0)
        max_iterations = kwargs.get('max_iterations', 1000)
        
        planner = RRTPlanner(self.voxel_grid, start, goal, 
                           step_size=step_size,
                           max_iterations=max_iterations)
        return planner.plan()  # Now returns (path, vertices) tuple

    def _pso_objective_function(self, positions):
        """Objective function for PSO path finding - minimize path length while avoiding obstacles"""
        # Convert positions to waypoints
        num_waypoints = len(positions) // 3
        waypoints = [tuple(map(int, positions[i:i+3])) for i in range(0, len(positions), 3)]
        
        # Add start and goal to create complete path
        path = [self.start] + waypoints + [self.goal]
        
        # Check if any waypoint is invalid
        for point in waypoints:
            x, y, z = point
            if (not (0 <= x < self.voxel_grid.shape[0] and 
                    0 <= y < self.voxel_grid.shape[1] and 
                    0 <= z < self.voxel_grid.shape[2]) or 
                self.voxel_grid[x, y, z]):
                return float('inf')

        # Calculate total path length
        total_length = 0
        for i in range(len(path) - 1):
            # Check if path segment intersects with obstacles
            if not self._is_valid_segment(path[i], path[i+1]):
                return float('inf')
            
            # Add segment length to total
            total_length += sum((a - b) ** 2 for a, b in zip(path[i], path[i+1])) ** 0.5

        return total_length

    def _is_valid_segment(self, p1, p2):
        """Check if path segment between two points is collision-free"""
        steps = int(max(abs(a - b) for a, b in zip(p1, p2)) * 2)  # Double the max difference
        if steps == 0:
            return True

        for i in range(steps + 1):
            t = i / steps
            point = tuple(int(a + (b - a) * t) for a, b in zip(p1, p2))
            if self.voxel_grid[point]:
                return False
        return True

    def _pso_plan(self, start, goal, **kwargs):
        self.start = start  # Store start for objective function
        self.goal = goal    # Store goal for objective function
        num_waypoints = kwargs.get('num_waypoints', 20)  # Number of intermediate waypoints
        num_particles = kwargs.get('num_particles', 30)
        iterations = kwargs.get('iterations', 100)
        initial_positions = kwargs.get('initial_positions', None)
        
        pso = PSO(
            objective_function=self._pso_objective_function,
            dimensions=num_waypoints * 3,  # 3 coordinates per waypoint
            num_particles=num_particles,
            iterations=iterations,
            initial_positions=initial_positions
        )
        
        best_position, best_value = pso.optimize()
        optimization_history = pso.get_optimization_history()
        
        if best_value == float('inf'):
            return None, None

        # Convert best position to path
        waypoints = [tuple(map(int, best_position[i:i+3])) 
                    for i in range(0, len(best_position), 3)]
        final_path = [start] + waypoints + [goal]
        
        # Convert optimization history to waypoint history
        waypoint_history = []
        for pos, val in optimization_history:
            iter_waypoints = [tuple(map(int, pos[i:i+3])) 
                            for i in range(0, len(pos), 3)]
            waypoint_history.append((iter_waypoints, val))
            
        return final_path, waypoint_history

    def plan_path(self, start, goal, planner_type=PlannerType.ASTAR, **kwargs):
        """
        Unified interface for path planning
        
        Args:
            start: Start position tuple (x,y,z)
            goal: Goal position tuple (x,y,z)
            planner_type: PlannerType enum value
            **kwargs: Additional parameters for specific planners
                     RRT: step_size, max_iterations
                     PSO: num_particles, iterations, num_waypoints
        """
        if planner_type not in self.planners:
            raise ValueError(f"Unsupported planner type: {planner_type}")
            
        return self.planners[planner_type](start, goal, **kwargs)

import numpy as np
import random
from map_generation.voxel_map_generator import create_test_environment
from planners.path_planner import PathPlanner, PlannerType
from visualizers.path_visualizer import (
    visualize_environment,
    visualize_astar_path,
    visualize_rrt_path,
    visualize_pso_path
)

if __name__ == "__main__":
    # Create environment using the new generator
    size = 30
    voxel_grid = create_test_environment(
        size=size,
        environment_type="cylinder",
        num_obstacles=5,
        dilation_size=1
    )

    # Example usage
    start = (0, 0, 5)
    goal = (size-1, size-1, 15)

    # Check if start or goal is blocked
    if voxel_grid[start]:
        raise ValueError("Start position is blocked.")
    if voxel_grid[goal]:
        raise ValueError("Goal position is blocked.")
    
    # Visualize the environment (optional)
    # visualize_environment(voxel_grid)

    # Initialize unified path planner
    planner = PathPlanner(voxel_grid)

    # Get paths using different planners
    astar_path = planner.plan_path(start, goal, PlannerType.ASTAR)
    # visualize_astar_path(voxel_grid, astar_path)

    # Get RRT path
    rrt_result = planner.plan_path(start, goal, PlannerType.RRT, 
                                 step_size=1.0, max_iterations=1000)
    # if rrt_result and rrt_result[0]:
    #     visualize_rrt_path(voxel_grid, *rrt_result)

    # Initialize PSO with RRT path
    initial_positions = []
    if rrt_result and rrt_result[0]:
        rrt_path, _ = rrt_result
        num_waypoints = 10
        if len(rrt_path) > 2:
            indices = np.linspace(1, len(rrt_path)-2, num_waypoints, dtype=int)
            waypoints = [rrt_path[i] for i in indices]
            rrt_position = []
            for point in waypoints:
                rrt_position.extend(point)
            
            num_particles = 5
            for i in range(num_particles):
                variation = [p + random.uniform(0, 0.01) for p in rrt_position]
                initial_positions.append(variation)

    # Run PSO
    pso_result = planner.plan_path(start, goal, PlannerType.PSO,
                                 num_waypoints=10,
                                 num_particles=30,
                                 iterations=100,
                                 initial_positions=initial_positions)

    # Visualize PSO results
    visualize_pso_path(voxel_grid, pso_result, start, goal)
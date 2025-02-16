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
from benchmark import PathPlanningBenchmark

def run_path_planning_demo(size=30, visualize=True):
    """Run the path planning demonstration"""
    voxel_grid = create_test_environment(
        size=size,
        environment_type="cylinder",
        num_obstacles=5,
        dilation_size=1
    )

    start = (0, 0, 5)
    goal = (size-1, size-1, 15)

    if voxel_grid[start] or voxel_grid[goal]:
        raise ValueError("Start or goal position is blocked.")

    planner = PathPlanner(voxel_grid)
    
    # Run and visualize different planners
    if visualize:
        visualize_environment(voxel_grid)
    
    # A* planning
    astar_path = planner.plan_path(start, goal, PlannerType.ASTAR)
    if visualize:
        visualize_astar_path(voxel_grid, astar_path)

    # RRT planning
    rrt_result = planner.plan_path(start, goal, PlannerType.RRT, 
                                 step_size=1.0, max_iterations=1000)
    if visualize and rrt_result and rrt_result[0]:
        visualize_rrt_path(voxel_grid, *rrt_result)

    # PSO planning with RRT initialization
    initial_positions = create_pso_initial_positions(rrt_result)
    pso_result = planner.plan_path(start, goal, PlannerType.PSO,
                                 num_waypoints=10,
                                 num_particles=30,
                                 iterations=100,
                                 initial_positions=initial_positions)
    
    if visualize:
        visualize_pso_path(voxel_grid, pso_result, start, goal)

def create_pso_initial_positions(rrt_result, num_waypoints=10, num_particles=5):
    """Helper function to create PSO initial positions from RRT result"""
    initial_positions = []
    if rrt_result and rrt_result[0]:
        rrt_path, _ = rrt_result
        if len(rrt_path) > 2:
            indices = np.linspace(1, len(rrt_path)-2, num_waypoints, dtype=int)
            waypoints = [rrt_path[i] for i in indices]
            rrt_position = []
            for point in waypoints:
                rrt_position.extend(point)
            
            for _ in range(num_particles):
                variation = [p + random.uniform(0, 0.01) for p in rrt_position]
                initial_positions.append(variation)
    
    return initial_positions

def run_benchmark_evaluation():
    """Run comprehensive benchmark evaluation"""
    benchmark = PathPlanningBenchmark(size=30, num_tests=5)
    
    # Configure environments and planners to test
    environment_types = ["cylinder", "maze", "indoor"]
    planner_configs = [
        (PlannerType.ASTAR, {}),
        (PlannerType.RRT, {'step_size': 1.0, 'max_iterations': 1000}),
        (PlannerType.PSO, {'num_waypoints': 10, 'num_particles': 30, 'iterations': 100})
    ]

    # Run benchmark
    benchmark.run_benchmark(environment_types, planner_configs)
    
    # Print results
    benchmark.print_summary()

if __name__ == "__main__":
    # Choose what to run
    RUN_DEMO = True
    RUN_BENCHMARK = False
    
    if RUN_DEMO:
        print("\n=== Running Path Planning Demo ===")
        run_path_planning_demo(visualize=True)
    
    if RUN_BENCHMARK:
        print("\n=== Running Benchmark Evaluation ===")
        run_benchmark_evaluation()
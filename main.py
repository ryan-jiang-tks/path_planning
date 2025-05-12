import numpy as np
import random
from map_generation.voxel_map_generator import create_test_environment
from planners.path_planner import PathPlanner, PlannerType
from visualizers.path_visualizer import (
    visualize_environment,
    visualize_astar_path,
    visualize_rrt_path,
    visualize_pso_path,
    visualize_dqn_path,
    visualize_formation_path
)
from benchmark import PathPlanningBenchmark
from planners.formation_astar import plan_formation_astar

def run_path_planning_demo(size=30, visualize=True, planners_config=None):
    """
    Run the path planning demonstration
    Args:
        size: Environment size
        visualize: Whether to visualize results
        planners_config: List of tuples (PlannerType, dict of parameters)
    """
    original_grid, dilated_grid = create_test_environment(
        size=size,
        environment_type="cylinder",
        num_obstacles=5,
        dilation_size=3
    )

    start = (0, 0, 5)
    goal = (size-1, size-1, 5)

    if dilated_grid[start] or dilated_grid[goal]:
        raise ValueError("Start or goal position is blocked.")

    planner = PathPlanner(dilated_grid)  # Use dilated grid for planning
    
    if visualize:
        visualize_environment(original_grid)  # Show original grid
        visualize_environment(dilated_grid)  # Show dilated_grid

    # Default planner if none specified
    if planners_config is None:
        planners_config = [
            (PlannerType.MODIFIEDRRT, {'step_size': 1.0, 'max_iterations': 1000})
        ]
    
    # Run each specified planner
    for planner_type, params in planners_config:
        print(f"\nRunning {planner_type.value} planner...")
        result = planner.plan_path(start, goal, planner_type, **params)
        
        if result and result[0]:
            if planner_type in [PlannerType.RRT, PlannerType.MODIFIEDRRT]:
                visualize_rrt_path(original_grid, *result)  # Use original grid for visualization
            elif planner_type == PlannerType.PSO:
                visualize_pso_path(original_grid, result, start, goal)  # Use original grid for visualization
            elif planner_type == PlannerType.ASTAR:
                visualize_astar_path(original_grid, result)  # Use original grid for visualization
            print(f"Path found with {len(result[0])} points")
        else:
            print("No path found")

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
    benchmark = PathPlanningBenchmark(size=30, num_tests=10)
    
    # Configure environments and planners to test
    environment_types = ["cylinder", "indoor", "outdoor"]
    planner_configs = [
        (PlannerType.ASTAR, {}),
        (PlannerType.RRT, {'step_size': 1.0, 'max_iterations': 1000}),
        (PlannerType.MODIFIEDRRT, {'step_size': 1.0, 'max_iterations': 1000}),
        (PlannerType.PSO, {'num_waypoints': 10, 'num_particles': 30, 'iterations': 100}),
    ]

    # Run benchmark
    benchmark.run_benchmark(environment_types, planner_configs)
    
    # Print results
    benchmark.print_summary()

def run_formation_demo(size=30):
    """Run formation path planning demo"""
    original_grid, dilated_grid = create_test_environment(
        size=size,
        environment_type="outdoor",
        num_obstacles=5,
        dilation_size=1
    )

    start = (5, 5, 5)
    goal = (size-10, size-10, 5)

    # Try different formations
    formations = ["square", "diamond", "line"]
    for formation_shape in formations:
        leader_path, all_drone_paths = plan_formation_astar(
            dilated_grid, start, goal,  # Use dilated grid for planning
            formation_shape=formation_shape,
            formation_size=2
        )
        
        if leader_path:
            print(f"\nFound path for {formation_shape} formation")
            visualize_formation_path(original_grid, all_drone_paths, start, goal)  # Use original grid for visualization
        else:
            print(f"\nNo path found for {formation_shape} formation")

if __name__ == "__main__":
    # Choose what to run
    RUN_DEMO = 1
    RUN_FORMATION = 0
    RUN_BENCHMARK = 0
    
    # Select which planners to run with their parameters
    PLANNERS_TO_RUN = [
        (PlannerType.MODIFIEDRRT, {'step_size': 1.0, 'max_iterations': 1000}),
        # (PlannerType.ASTAR, {}),
        # (PlannerType.RRT, {'step_size': 1.0, 'max_iterations': 1000}),
        # (PlannerType.PSO, {
        #     'num_waypoints': 10,
        #     'num_particles': 30,
        #     'iterations': 100
        # }),
    ]

    # Run demo with specified planners
    if RUN_DEMO:
        print("\n=== Running Path Planning Demo ===")
        run_path_planning_demo(visualize=True, planners_config=PLANNERS_TO_RUN)
    
    if RUN_BENCHMARK:
        print("\n=== Running Benchmark Evaluation ===")
        run_benchmark_evaluation()


    if RUN_FORMATION:
        print("\n=== Running Formation Path Planning Demo ===")
        run_formation_demo()
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from planners.path_planner import PathPlanner, PlannerType
from map_generation.voxel_map_generator import create_test_environment


@dataclass
class BenchmarkResult:
    success: bool
    execution_time: float
    path_length: float
    path_smoothness: float
    memory_usage: int
    num_waypoints: int

class PathPlanningBenchmark:
    def __init__(self, size=30, num_tests=10):
        self.size = size
        self.num_tests = num_tests
        self.results: Dict[str, List[BenchmarkResult]] = {}

    def calculate_path_length(self, path):
        if not path:
            return float('inf')
        path = np.array(path)
        diffs = np.diff(path, axis=0)
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        return np.sum(distances)

    def calculate_path_smoothness(self, path):
        if not path or len(path) < 3:
            return 0.0
        path = np.array(path)
        vectors = np.diff(path, axis=0)
        angles = np.arccos(np.sum(vectors[:-1] * vectors[1:], axis=1) / 
                         (np.linalg.norm(vectors[:-1], axis=1) * np.linalg.norm(vectors[1:], axis=1)))
        return np.mean(angles)

    def run_single_test(self, environment_type: str, planner_type: PlannerType, 
                       start: Tuple[int, int, int], goal: Tuple[int, int, int], **kwargs) -> BenchmarkResult:
        # Create environment
        voxel_grid = create_test_environment(
            size=self.size,
            environment_type=environment_type,
            num_obstacles=5,
            dilation_size=1
        )

        # Initialize planner
        planner = PathPlanner(voxel_grid)

        # Measure execution time
        start_time = time.time()
        result = planner.plan_path(start, goal, planner_type, **kwargs)
        execution_time = time.time() - start_time

        # Extract path from result (handle different return types)
        if planner_type == PlannerType.PSO:
            path = result[0] if result and result[0] else None
        elif planner_type == PlannerType.RRT:
            path = result[0] if result and result[0] else None
        else:
            path = result

        # Calculate metrics
        return BenchmarkResult(
            success=path is not None,
            execution_time=execution_time,
            path_length=self.calculate_path_length(path) if path else float('inf'),
            path_smoothness=self.calculate_path_smoothness(path) if path else 0.0,
            memory_usage=len(path) if path else 0,
            num_waypoints=len(path) if path else 0
        )

    def run_benchmark(self, environment_types=None, planner_configs=None):
        if environment_types is None:
            environment_types = ["random", "cylinder", "maze", "indoor", "outdoor"]
        
        if planner_configs is None:
            planner_configs = [
                (PlannerType.ASTAR, {}),
                (PlannerType.RRT, {'step_size': 1.0, 'max_iterations': 1000}),
                (PlannerType.PSO, {'num_waypoints': 10, 'num_particles': 30, 'iterations': 100})
            ]

        self.results = {}

        for env_type in environment_types:
            for planner_type, config in planner_configs:
                key = f"{env_type}_{planner_type.value}"
                self.results[key] = []

                for _ in range(self.num_tests):
                    # Generate random start and goal positions
                    start = (0, 0, 5)  # Fixed start for consistency
                    goal = (self.size-1, self.size-1, 15)  # Fixed goal for consistency

                    result = self.run_single_test(env_type, planner_type, start, goal, **config)
                    self.results[key].append(result)

    def print_summary(self):
        print("\nBenchmark Summary:")
        print("-" * 80)
        print(f"{'Environment-Planner':<25} {'Success %':<10} {'Avg Time(s)':<12} "
              f"{'Avg Length':<12} {'Avg Smooth':<12} {'Avg Points':<12}")
        print("-" * 80)

        for key, results in self.results.items():
            success_rate = sum(r.success for r in results) / len(results) * 100
            avg_time = np.mean([r.execution_time for r in results])
            avg_length = np.mean([r.path_length for r in results if r.success])
            avg_smoothness = np.mean([r.path_smoothness for r in results if r.success])
            avg_points = np.mean([r.num_waypoints for r in results if r.success])

            print(f"{key:<25} {success_rate:>9.1f}% {avg_time:>11.3f} "
                  f"{avg_length:>11.2f} {avg_smoothness:>11.2f} {avg_points:>11.1f}")

def run_benchmark_example():
    benchmark = PathPlanningBenchmark(size=30, num_tests=5)
    
    # Configure specific environments and planners to test
    environment_types = ["cylinder", "maze"]
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
    run_benchmark_example()

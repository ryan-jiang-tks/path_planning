import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

from map_generation.voxel_map_generator import create_test_environment
from planners.path_planner import PathPlanner, PlannerType

if __name__ == "__main__":

    # Create environment using the new generator
    size = 30
    voxel_grid = create_test_environment(
        size=size,
        environment_type="cylinder",  # Change to outdoor environment
        num_obstacles=5,
        dilation_size=1  # Reduced dilation size for better detail
    )

    # Example usage
    start = (0, 0, 5)
    goal = (size-1, size-1, 15)

    # Check if start or goal is blocked
    if voxel_grid[start]:
        raise ValueError("Start position is blocked.")
    if voxel_grid[goal]:
        raise ValueError("Goal position is blocked.")
    
    # # Visualize the voxel grid
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.voxels(voxel_grid, facecolors='blue', edgecolor='k', alpha=0.3)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.title('3D Environment')
    # plt.show()

    

    # Initialize unified path planner
    planner = PathPlanner(voxel_grid)

    # Get paths using different planners
    astar_path = planner.plan_path(start, goal, PlannerType.ASTAR)

    # Get RRT path first to initialize PSO
    rrt_result = planner.plan_path(start, goal, PlannerType.RRT, 
                                 step_size=1.0, max_iterations=1000)
    if rrt_result and rrt_result[0]:rrt_path, rrt_vertices = rrt_result

# to set a inital position for pso(when the circumstances are not good for pso to find a path)
    initial_positions = []
    if rrt_result and rrt_result[0]:
        rrt_path, _ = rrt_result
        # Convert RRT path to PSO particle positions
        num_waypoints = 10
        if len(rrt_path) > 2:  # If we have waypoints besides start/goal
            # Sample points from RRT path
            indices = np.linspace(1, len(rrt_path)-2, num_waypoints, dtype=int)
            waypoints = [rrt_path[i] for i in indices]
            
            # Convert to flat array format for PSO
            rrt_position = []
            for point in waypoints:
                rrt_position.extend(point)
            
            # Create variations of RRT path for multiple particles
            num_particles = 5
            for i in range(num_particles):
                # Add random variations to RRT path
                variation = [p + random.uniform(0, 0.01) for p in rrt_position]
                initial_positions.append(variation)




    # Run PSO with initial positions(optional)
    pso_result = planner.plan_path(start, goal, PlannerType.PSO,
                                 num_waypoints=10,
                                 num_particles=30,
                                 iterations=100,
                                 initial_positions=initial_positions,

                                 )

    # Visualize A* path
    # if astar_path:
    #     path_grid = np.zeros_like(voxel_grid)
    #     for point in astar_path:
    #         path_grid[point] = True
            
    #     fig = plt.figure(figsize=(10, 8))
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.voxels(voxel_grid, facecolors='blue', edgecolor='k', alpha=0.3)
    #     ax.voxels(path_grid, facecolors='red', edgecolor='k', alpha=0.7)
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('Z')
    #     plt.title('3D A* Pathfinding')
    #     plt.show()

    # Visualize RRT path with multiple views
    # if rrt_path and rrt_vertices:
    #     # Create figure with multiple subplots
    #     fig = plt.figure(figsize=(15, 10))
        
    #     # Main view with full tree
    #     ax1 = fig.add_subplot(111, projection='3d')
    #     ax1.voxels(voxel_grid, facecolors='lightblue', edgecolor='blue', alpha=0.1)
    #     # Plot full RRT tree
    #     for vertex, parent in rrt_vertices.items():
    #         if parent is not None:
    #             ax1.plot([parent[0], vertex[0]], 
    #                     [parent[1], vertex[1]], 
    #                     [parent[2], vertex[2]], 'g-', alpha=0.3)
    #     # Plot final path
    #     path = np.array(rrt_path)
    #     ax1.plot(path[:, 0], path[:, 1], path[:, 2], 'r-', linewidth=2, label='Path')
    #     ax1.set_title('RRT Tree and Path')
    #     ax1.view_init(elev=30, azim=45)
        

        # Set labels for all subplots
        # for ax in [ax1]:
        #     ax.set_xlabel('X')
        #     ax.set_ylabel('Y')
        #     ax.set_zlabel('Z')
        #     ax.legend()
        
        # plt.tight_layout()
        # plt.show()
        
        # # Print path statistics
        # print(f"RRT path length: {len(rrt_path)} points")
        # print(f"RRT tree size: {len(rrt_vertices)} vertices")
        # print(f"Path start: {rrt_path[0]}")
        # print(f"Path end: {rrt_path[-1]}")

    # Visualize PSO results
    # Visualize A* path
    # if astar_path:
    #     path_grid = np.zeros_like(voxel_grid)
    #     for point in astar_path:
    #         path_grid[point] = True
            
    #     fig = plt.figure(figsize=(10, 8))
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.voxels(voxel_grid, facecolors='blue', edgecolor='k', alpha=0.3)
    #     ax.voxels(path_grid, facecolors='red', edgecolor='k', alpha=0.7)
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('Z')
    #     plt.title('3D A* Pathfinding')
    #     plt.show()

    # Visualize RRT path with multiple views
    # if rrt_path and rrt_vertices:
    #     # Create figure with multiple subplots
    #     fig = plt.figure(figsize=(15, 10))
        
    #     # Main view with full tree
    #     ax1 = fig.add_subplot(111, projection='3d')
    #     ax1.voxels(voxel_grid, facecolors='lightblue', edgecolor='blue', alpha=0.1)
    #     # Plot full RRT tree
    #     for vertex, parent in rrt_vertices.items():
    #         if parent is not None:
    #             ax1.plot([parent[0], vertex[0]], 
    #                     [parent[1], vertex[1]], 
    #                     [parent[2], vertex[2]], 'g-', alpha=0.3)
    #     # Plot final path
    #     path = np.array(rrt_path)
    #     ax1.plot(path[:, 0], path[:, 1], path[:, 2], 'r-', linewidth=2, label='Path')
    #     ax1.set_title('RRT Tree and Path')
    #     ax1.view_init(elev=30, azim=45)
        

        # Set labels for all subplots
        # for ax in [ax1]:
        #     ax.set_xlabel('X')
        #     ax.set_ylabel('Y')
        #     ax.set_zlabel('Z')
        #     ax.legend()
        
        # plt.tight_layout()
        # plt.show()
        
        # # Print path statistics
        # print(f"RRT path length: {len(rrt_path)} points")
        # print(f"RRT tree size: {len(rrt_vertices)} vertices")
        # print(f"Path start: {rrt_path[0]}")
        # print(f"Path end: {rrt_path[-1]}")

    if pso_result and pso_result[0]:  # Check both the result and the path
        final_path, waypoint_history = pso_result  # Unpack the result tuple
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot obstacles
        ax.voxels(voxel_grid, facecolors='blue', edgecolor='k', alpha=0.1)
        
        # Plot final path
        path = np.array(final_path)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], 'r-', linewidth=2, label='Final Path')
        
        if waypoint_history:  # Check if we have optimization history
            # Plot waypoint evolution (last 10 iterations)
            num_hist = min(10, len(waypoint_history))
            cmap = plt.cm.viridis(np.linspace(0, 1, num_hist))
            for i, ((waypoints, value), color) in enumerate(zip(waypoint_history[-num_hist:], cmap)):
                waypoints_with_endpoints = [start] + waypoints + [goal]
                points = np.array(waypoints_with_endpoints)
                ax.plot(points[:, 0], points[:, 1], points[:, 2], '--', 
                       color=color, alpha=0.3, 
                       label=f'Iteration -{len(waypoint_history)-i}')
        
            # Print statistics
            print(f"Final path length: {len(final_path)} points")
            print(f"Total iterations: {len(waypoint_history)}")
            print(f"Final path cost: {waypoint_history[-1][1]:.2f}")
        
        # Plot final waypoints
        ax.scatter(path[1:-1, 0], path[1:-1, 1], path[1:-1, 2], 
                  color='green', s=100, label='Final Waypoints')
        
        # Plot start and goal
        ax.scatter(*start, color='blue', s=100, label='Start')
        ax.scatter(*goal, color='red', s=100, label='Goal')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.title('PSO Path Planning Evolution')
        plt.show()
    else:
        print("PSO failed to find a valid path")
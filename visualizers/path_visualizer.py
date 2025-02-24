import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_environment(voxel_grid):
    """Visualize the 3D environment"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(voxel_grid, facecolors='blue', edgecolor='k', alpha=0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D Environment')
    plt.show()

def visualize_astar_path(voxel_grid, path):
    """Visualize A* path in the environment"""
    if not path:
        print("No A* path to visualize")
        return
        
    path_grid = np.zeros_like(voxel_grid)
    for point in path:
        path_grid[point] = True
        
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(voxel_grid, facecolors='blue', edgecolor='k', alpha=0.3)
    ax.voxels(path_grid, facecolors='red', edgecolor='k', alpha=0.7)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D A* Pathfinding')
    plt.show()

def visualize_rrt_path(voxel_grid, rrt_path, rrt_vertices):
    """Visualize RRT path and tree"""
    if not rrt_path or not rrt_vertices:
        print("No RRT path to visualize")
        return
        
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.voxels(voxel_grid, facecolors='blue', edgecolor='k', alpha=0.1)
    
    # Plot full RRT tree with different colors based on node type if available
    for vertex_pos, vertex_data in rrt_vertices.items():
        # Handle both normal RRT and modified RRT vertex data
        if isinstance(vertex_data, dict):  # Modified RRT case
            parent_pos = vertex_data.get('parent')
            node_type = vertex_data.get('node_type', 'unknown')
            # Set color based on node type
            if node_type == 'edge':
                color = 'r'
            elif node_type == 'free':
                color = 'g'
            else:
                color = 'y'
        else:  # Normal RRT case
            parent_pos = vertex_data
            color = 'g'

        if parent_pos is not None:
            ax1.plot([parent_pos[0], vertex_pos[0]], 
                    [parent_pos[1], vertex_pos[1]], 
                    [parent_pos[2], vertex_pos[2]], 
                    color=color, alpha=0.3, linewidth=1)
    
    # Plot final path
    path = np.array(rrt_path)
    ax1.plot(path[:, 0], path[:, 1], path[:, 2], 
            'b-', linewidth=2, label='Path')
    
    # Plot start and goal points
    ax1.scatter(path[0,0], path[0,1], path[0,2], 
               color='g', s=100, label='Start')
    ax1.scatter(path[-1,0], path[-1,1], path[-1,2], 
               color='r', s=100, label='Goal')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    ax1.set_title('RRT Tree and Path')
    ax1.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.show()
    
    # Print path statistics
    print(f"Path length: {len(rrt_path)} points")
    print(f"Tree size: {len(rrt_vertices)} vertices")

def visualize_pso_path(voxel_grid, pso_result, start, goal):
    """Visualize PSO path and optimization history"""
    if not pso_result or not pso_result[0]:
        print("No PSO path to visualize")
        return
        
    final_path, waypoint_history = pso_result
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot obstacles
    ax.voxels(voxel_grid, facecolors='blue', edgecolor='k', alpha=0.1)
    
    # Plot final path
    path = np.array(final_path)
    ax.plot(path[:, 0], path[:, 1], path[:, 2], 'r-', linewidth=2, label='Final Path')
    
    if waypoint_history:
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

def visualize_dqn_path(voxel_grid, path, start, goal):
    """Visualize DQN path in the environment"""
    if not path:
        print("No DQN path to visualize")
        return
        
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot obstacles
    ax.voxels(voxel_grid, facecolors='blue', edgecolor='k', alpha=0.1)
    
    # Convert path to numpy array for plotting
    path = np.array(path)
    
    # Plot path
    ax.plot(path[:, 0], path[:, 1], path[:, 2], 
            'r-', linewidth=2, label='DQN Path')
    
    # Plot waypoints
    ax.scatter(path[1:-1, 0], path[1:-1, 1], path[1:-1, 2], 
              color='green', s=100, label='Waypoints')
    
    # Plot start and goal
    ax.scatter(*start, color='blue', s=100, label='Start')
    ax.scatter(*goal, color='red', s=100, label='Goal')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title('DQN Path Planning Result')
    plt.show()
    
    # Print path statistics
    print(f"Path length: {len(path)} points")
    print(f"Path start: {path[0]}")
    print(f"Path end: {path[-1]}")

def visualize_formation_path(voxel_grid, all_drone_paths, start, goal):
    """Visualize paths for drone formation"""
    if not all_drone_paths or not all_drone_paths[0]:
        print("No formation path to visualize")
        return

    # Create figure
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot obstacles
    ax.voxels(voxel_grid, facecolors='gray', edgecolor='k', alpha=0.1)
    
    # Colors for different drones
    colors = ['r', 'b', 'g', 'y']
    
    # Plot paths for each drone
    for i, path in enumerate(all_drone_paths):
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], 
                f'{colors[i]}-', linewidth=2, label=f'Drone {i+1}')
        
        # Plot formation connections at regular intervals
        step = max(1, len(path) // 10)  # Show formation at 10 points along path
        for j in range(0, len(path), step):
            for k in range(i+1, len(all_drone_paths)):
                other_path = np.array(all_drone_paths[k])
                ax.plot([path[j, 0], other_path[j, 0]], 
                       [path[j, 1], other_path[j, 1]], 
                       [path[j, 2], other_path[j, 2]], 
                       'k--', alpha=0.3)
    
    # Plot start and goal positions for all drones
    for i, path in enumerate(all_drone_paths):
        ax.scatter(path[0][0], path[0][1], path[0][2], 
                  c=colors[i], marker='o', s=100, label=f'Start {i+1}')
        ax.scatter(path[-1][0], path[-1][1], path[-1][2], 
                  c=colors[i], marker='s', s=100, label=f'Goal {i+1}')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title('Formation Path Planning')
    
    # Add formation shape indicator
    # plt.figtext(0.02, 0.02, f'Formation: {formation_shape}', fontsize=10)
    
    plt.show()

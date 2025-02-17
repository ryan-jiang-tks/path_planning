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
    
    # Plot full RRT tree
    for vertex, parent in rrt_vertices.items():
        if parent is not None:
            ax1.plot([parent[0], vertex[0]], 
                    [parent[1], vertex[1]], 
                    [parent[2], vertex[2]], 'g-', alpha=0.3)
    
    # Plot final path
    path = np.array(rrt_path)
    ax1.plot(path[:, 0], path[:, 1], path[:, 2], 'r-', linewidth=5, label='Path')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    ax1.set_title('RRT Tree and Path')
    ax1.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.show()
    
    # Print path statistics
    print(f"RRT path length: {len(rrt_path)} points")
    print(f"RRT tree size: {len(rrt_vertices)} vertices")
    print(f"Path start: {rrt_path[0]}")
    print(f"Path end: {rrt_path[-1]}")

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

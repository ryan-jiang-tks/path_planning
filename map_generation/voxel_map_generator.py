import numpy as np
from map_generation.voxel_dilator import dilate_voxel_grid

class VoxelMapGenerator:
    def __init__(self, size):
        self.size = size

    def create_empty_grid(self):
        return np.zeros((self.size, self.size, self.size), dtype=bool)

    def add_random_obstacles(self, grid, num_obstacles, seed=None):
        if seed is not None:
            np.random.seed(seed)
        coords = np.random.randint(0, self.size, (3, num_obstacles))
        grid[coords[0], coords[1], coords[2]] = True
        return grid

    def add_wall(self, grid, start, end):
        """Add a wall between two points"""
        x1, y1, z1 = start
        x2, y2, z2 = end
        grid[x1:x2+1, y1:y2+1, z1:z2+1] = True
        return grid

    def add_room(self, grid, corner1, corner2):
        """Add a room with walls"""
        x1, y1, z1 = corner1
        x2, y2, z2 = corner2
        # Create walls
        grid[x1:x2+1, y1, z1:z2+1] = True  # Front wall
        grid[x1:x2+1, y2, z1:z2+1] = True  # Back wall
        grid[x1, y1:y2+1, z1:z2+1] = True  # Left wall
        grid[x2, y1:y2+1, z1:z2+1] = True  # Right wall
        return grid

    def generate_indoor_environment(self):
        grid = self.create_empty_grid()
        
        # Add rooms
        rooms = [
            ((5, 5, 0), (12, 12, 5)),
            ((15, 5, 0), (22, 12, 5)),
            ((5, 15, 0), (12, 22, 5)),
            ((15, 15, 0), (22, 22, 5))
        ]
        
        for corner1, corner2 in rooms:
            grid = self.add_room(grid, corner1, corner2)
            
        return grid

    def process_environment(self, grid, dilation_size=1):
        """Process the environment with dilation and final checks"""
        # Dilate the obstacles
        processed_grid = dilate_voxel_grid(grid, dilation_size)
        
        # Ensure start and goal positions are free
        processed_grid[0, 0, 0] = False  # Clear start position
        processed_grid[-1, -1, -1] = False  # Clear goal position
        
        return processed_grid

    def add_mountain(self, grid, center, height, radius):
        """Add a mountain with gaussian shape"""
        x_c, y_c = center
        for x in range(max(0, x_c - radius), min(self.size, x_c + radius)):
            for y in range(max(0, y_c - radius), min(self.size, y_c + radius)):
                # Calculate height using gaussian function
                dist = np.sqrt((x - x_c)**2 + (y - y_c)**2)
                if dist < radius:
                    h = int(height * np.exp(-(dist**2)/(2*(radius/2)**2)))
                    grid[x, y, :h] = True
        return grid

    def add_tree(self, grid, position, height=5):
        """Add a simple tree with trunk and foliage"""
        x, y = position
        if x < 0 or x >= self.size-2 or y < 0 or y >= self.size-2:
            return grid
        
        # Add trunk
        trunk_height = height - 2
        grid[x:x+1, y:y+1, :trunk_height] = True
        
        # Add foliage (pyramid shape)
        for h in range(2):
            size = 2 - h
            grid[x-size:x+size+1, y-size:y+size+1, trunk_height+h] = True
        
        return grid

    def add_house(self, grid, position, size=4):
        """Add a simple house with walls and roof"""
        x, y = position
        if x < 0 or x >= self.size-size or y < 0 or y >= self.size-size:
            return grid
        
        # Walls
        wall_height = 4
        grid[x:x+size, y, :wall_height] = True  # Front wall
        grid[x:x+size, y+size-1, :wall_height] = True  # Back wall
        grid[x, y:y+size, :wall_height] = True  # Left wall
        grid[x+size-1, y:y+size, :wall_height] = True  # Right wall
        
        # Roof (pyramid shape)
        for h in range(2):
            roof_size = size - h
            start = h // 2
            grid[x+start:x+size-start, y+start:y+size-start, wall_height+h] = True
        
        return grid

    def generate_outdoor_environment(self, num_mountains=2, num_trees=2, num_houses=2):
        """Generate an outdoor environment with mountains, trees, and houses"""
        grid = self.create_empty_grid()
        
        # Add ground level
        grid[:, :, 0] = True
        
        # Add mountains
        for _ in range(num_mountains):
            center = (np.random.randint(5, self.size-5), np.random.randint(5, self.size-5))
            height = np.random.randint(self.size-10, self.size-5)
            radius = np.random.randint(3, 5)
            grid = self.add_mountain(grid, center, height, radius)
        
        # Add trees
        for _ in range(num_trees):
            position = (np.random.randint(3, self.size-3), np.random.randint(3, self.size-3))
            height = np.random.randint(4, self.size/3)
            grid = self.add_tree(grid, position, height)
        
        # Add houses
        for _ in range(num_houses):
            position = (np.random.randint(0, self.size-6), np.random.randint(0, self.size-6))
            grid = self.add_house(grid, position)
        
        return grid

    def add_cylinder(self, grid, center, radius, height):
        """Add a cylinder to the grid"""
        x_c, y_c = center
        for x in range(max(0, x_c - radius), min(self.size, x_c + radius + 1)):
            for y in range(max(0, y_c - radius), min(self.size, y_c + radius + 1)):
                # Check if point is within circle
                if (x - x_c) ** 2 + (y - y_c) ** 2 <= radius ** 2:
                    grid[x, y, :height] = True
        return grid

    def generate_cylinder_environment(self, num_cylinders=5, min_radius=5, max_radius=5, 
                                   min_height=5, max_height=15):
        """Generate an environment with multiple cylinders"""
        grid = self.create_empty_grid()
        
        # Add ground level
        grid[:, :, 0] = True

        min_height=self.size/2
        max_height=self.size-3
        
        # Add random cylinders
        for _ in range(num_cylinders):
            radius = np.random.randint(min_radius, max_radius + 1)
            height = np.random.randint(min_height, max_height + 1)
            center = (
                np.random.randint(radius + 1, self.size - radius - 1),
                np.random.randint(radius + 1, self.size - radius - 1)
            )
            grid = self.add_cylinder(grid, center, radius, height)
        
        return grid

def create_test_environment(size=30, environment_type="random", num_obstacles=5, dilation_size=2):
    """
    Create a test environment with specified parameters
    
    Args:
        size (int): Size of the cubic grid
        environment_type (str): "random", "indoor", "outdoor", or "cylinder"
        num_obstacles (int): Number of random obstacles or cylinders
        dilation_size (int): Size of obstacle dilation
        
    Returns:
        tuple: (original_grid, dilated_grid)
    """
    generator = VoxelMapGenerator(size)
    
    if environment_type == "random":
        grid = generator.create_empty_grid()
        grid = generator.add_random_obstacles(grid, num_obstacles, seed=44)
    elif environment_type == "indoor":
        grid = generator.generate_indoor_environment()
    elif environment_type == "outdoor":
        grid = generator.generate_outdoor_environment()
    elif environment_type == "cylinder":
        grid = generator.generate_cylinder_environment(num_obstacles)
    else:
        raise ValueError("Invalid environment type")
    
    return grid, generator.process_environment(grid, dilation_size)

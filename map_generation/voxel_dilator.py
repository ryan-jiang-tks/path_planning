
import numpy as np


def dilate_voxel_grid(voxel_grid, dilation_size=1):
    """
    Dilates the voxel grid by expanding occupied voxels into their neighbors.

    Args:
        voxel_grid (np.ndarray): A 3D boolean array where `True` represents occupied voxels.
        dilation_size (int): The number of layers to dilate the occupied voxels.

    Returns:
        np.ndarray: The dilated voxel grid.
    """
    # Create a copy of the original grid to avoid modifying it in place
    dilated_grid = voxel_grid.copy()

    # Define the neighborhood offsets for 3D dilation
    neighborhood_offsets = [
        (dx, dy, dz)
        for dx in range(-dilation_size, dilation_size + 1)
        for dy in range(-dilation_size, dilation_size + 1)
        for dz in range(-dilation_size, dilation_size + 1)
        if not (dx == 0 and dy == 0 and dz == 0)  # Exclude the center voxel
    ]

    # Iterate over the grid and dilate occupied voxels
    for x in range(voxel_grid.shape[0]):
        for y in range(voxel_grid.shape[1]):
            for z in range(voxel_grid.shape[2]):
                if voxel_grid[x, y, z]:  # If the current voxel is occupied
                    for dx, dy, dz in neighborhood_offsets:
                        nx, ny, nz = x + dx, y + dy, z + dz
                        # Check if the neighbor is within bounds
                        if (
                            0 <= nx < voxel_grid.shape[0]
                            and 0 <= ny < voxel_grid.shape[1]
                            and 0 <= nz < voxel_grid.shape[2]
                        ):
                            dilated_grid[nx, ny, nz] = True  # Mark the neighbor as occupied

    return dilated_grid
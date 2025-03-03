import numpy as np
from helpers import check_data

def extract_hypercubes(loadpath, nbytes, file_shape, cube_shape, method, num_cubes):
    """
    Partition a 3D array into non-overlapping hypercubes and subsample them.

    Parameters:
        data (np.ndarray): 3D data array with shape (nx, ny, nz)
        cube_shape (tuple): Dimensions of each hypercube (hx, hy, hz)
        method (str): Subsampling method ('random' or 'maxent')
        num_cubes (int or None): If specified and less than total cubes, selects that many cubes.

    Returns:
        cubes (np.ndarray): Array of selected hypercubes with shape (N, hx, hy, hz)
        indices (list): List of indices (i,j,k) corresponding to each cube
    """
    nx, ny, nz = file_shape
    hx, hy, hz = cube_shape

    check_data(loadpath, nx, ny, nz, nbytes)
    data_memmap = np.memmap(loadpath, dtype=np.float32, mode='r', shape=(nz, ny, nx)) # NOTE: data is stored [z, y, x]

    # Compute how many cubes fit in each dimension.
    num_x = nx // hx
    num_y = ny // hy
    num_z = nz // hz

    indices = []
    for ix in range(num_x):
        for iy in range(num_y):
            for iz in range(num_z):
                indices.append((ix, iy, iz))

    # If the user has specified to select fewer cubes, apply selection.
    if num_cubes is not None and num_cubes < len(indices):
        if method == 'uniform': 
            sel_indices = np.arange(len(indices), num_cubes)
        if method == 'random':
            sel_indices = np.random.choice(len(indices), num_cubes, replace=False)
        elif method == 'maxent':
            raise ValueError("maxent not yet implemented")
            # For maximum entropy selection, you could compute the entropy of each cube.
            # Here is a simple example using histogram-based entropy.
            def cube_entropy(cube):
                hist, _ = np.histogram(cube, bins=10, density=True)
                hist = hist + 1e-12  # avoid log(0)
                return -np.sum(hist * np.log(hist))
            entropies = np.array([cube_entropy(cube) for cube in cubes])
            # Select cubes with the highest entropy values.
            sel_indices = np.argsort(entropies)[-num_cubes:]
            cubes = cubes[sel_indices]
        else:
            raise ValueError("Unknown subsampling method: choose 'random' or 'maxent'.")

    cubes = []
    for i  in range(len(sel_indices)):
        ix, iy, iz = indices[sel_indices[i]]
        x0, y0, z0 = ix * hx, iy * hy, iz * hz
        cube = data_memmap[z0:z0+hz, y0:y0+hy, x0:x0+hx]
        cubes.append(cube.copy().transpose(2, 1, 0)) # transposing data to be [x, y, z]
    cubes = np.array(cubes)
    
    data_memmap._mmap.close()
    del data_memmap, cube

    return cubes

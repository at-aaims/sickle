import numpy as np

def extract_hypercubes(data, cube_shape=(32, 32, 32), method='random', num_cubes=None):
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
    nx, ny, nz = data.shape
    hx, hy, hz = cube_shape

    # Compute how many cubes fit in each dimension.
    num_x = nx // hx
    num_y = ny // hy
    num_z = nz // hz

    cubes = []
    indices = []
    for ix in range(num_x):
        for iy in range(num_y):
            for iz in range(num_z):
                x0, y0, z0 = ix * hx, iy * hy, iz * hz
                cube = data[x0:x0+hx, y0:y0+hy, z0:z0+hz]
                cubes.append(cube)
                indices.append((ix, iy, iz))
    cubes = np.array(cubes)

    # If the user has specified to select fewer cubes, apply selection.
    if num_cubes is not None and num_cubes < len(cubes):
        if method == 'random':
            sel_indices = np.random.choice(len(cubes), num_cubes, replace=False)
            cubes = cubes[sel_indices]
        elif method == 'maxent':
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
    return cubes, indices

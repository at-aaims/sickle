import numpy as np
from helpers import check_data
#from .maxent_hypercubes import maxent_hypercubes
from .maxent_sequential import maxent_hypercubes


def extract_hypercube_IDs(loadpaths, nbytes, file_shape, cube_shape, method, num_cubes):
    """ Partition a 3D array into non-overlapping hypercubes. """

    nx, ny, nz = file_shape
    hx, hy, hz = cube_shape

    for loadpath in loadpaths:
        check_data(loadpath, nx, ny, nz, nbytes)

    # Compute how many cubes fit in each dimension.
    num_x = (nx-2) // hx
    num_y = ny // hy
    num_z = nz // hz

    indices = []
    for ix in range(num_x):
        for iy in range(num_y):
            for iz in range(num_z):
                indices.append((ix, iy, iz))

    # If the user has specified to select fewer cubes, apply selection.
    if num_cubes is not None and num_cubes <= len(indices):
        if method == 'uniform': 
            sel_indices = np.arange(len(indices), num_cubes)
        elif method == 'random':
            sel_indices = np.random.choice(len(indices), num_cubes, replace=False)
        elif method == 'maxent':
            sampled_subcubes = maxent_hypercubes(loadpaths, nx, ny, nz, hx, hy, hz, 
                                                n_clusters=10, n_cubes=num_cubes, 
                                                # note: keep 256 < batch_size < 1024 per core
                                                # main issue: number of clusters
                                                batch_size=1024, n_init=20, max_iter=10, n_iters=10)
            # Convert 3D to 1D
            sel_indices = np.array([ix + num_x * (iy + num_y * iz) for (ix, iy, iz) in sampled_subcubes])
        else:
            raise ValueError("Unknown subsampling method.")

    hypercubeIDs = [indices[i] for i in sel_indices]

    return hypercubeIDs

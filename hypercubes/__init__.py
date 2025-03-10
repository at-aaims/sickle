import numpy as np
from helpers import check_data
#from .maxent_hypercubes import maxent_hypercubes
from .maxent_sequential import maxent_hypercubes

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
            # raise ValueError("maxent not yet implemented")
            sampled_subcubes = maxent_hypercubes(loadpath, nx, ny, nz, hx, hy, hz, 
                                                n_clusters=10, n_cubes=num_cubes, 
                                                # note: keep 256 < batch_size < 1024 per core
                                                # main issue: number of clusters
                                                #batch_size=int(4096/size), n_init=20, max_iter=10, n_iters=10)
                                                batch_size=1024, n_init=20, max_iter=10, n_iters=10)
            # Convert 3D to 1D
            sel_indices = np.array([ix + num_x * (iy + num_y * iz) for (ix, iy, iz) in sampled_subcubes])
            #print("**", sampled_subcubes)
        else:
            raise ValueError("Unknown subsampling method: choose 'random' or 'maxent'.")

    cubes = []
    for i  in range(len(sel_indices)):
        ix, iy, iz = indices[sel_indices[i]]
        #print("***", ix, iy, iz)
        x0, y0, z0 = ix * hx, iy * hy, iz * hz
        cube = data_memmap[z0:z0+hz, y0:y0+hy, x0:x0+hx]
        cubes.append(cube.copy().transpose(2, 1, 0)) # transposing data to be [x, y, z]
    cubes = np.array(cubes)
    
    data_memmap._mmap.close()
    del data_memmap, cube

    return cubes

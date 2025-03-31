import os
import numpy as np
from helpers import check_data
from . import get_hypercube_extractor

class HypercubeHandler:
    def __init__(self, method, dims_full, dims_sl, nbytes, num_hypercubes, **selector_kwargs):
        """
        Initializes the hypercube handler.
        
        Parameters:
          method (str): The hypercube selection method ('uniform', 'random', 'maxent').
          dims_full (tuple): The full dimensions of the data (nx, ny, nz).
          dims_sl (tuple): The subcube dimensions (nxsl, nysl, nzsl).
          nbytes (int): Number of bytes per data point.
          num_hypercubes (int): Number of hypercubes to extract.
          selector_kwargs: Additional keyword arguments for the extractor.
        """
        self.dims_full = dims_full
        self.dims_sl = dims_sl
        self.nbytes = nbytes
        self.num_hypercubes = num_hypercubes
        
        # Create an extractor function using the provided method.
        self.extractor = get_hypercube_extractor(method, **selector_kwargs)
    
    def extract_ids(self, loadpaths):
        """
        Extracts the hypercube IDs given a list of file paths.
        
        Parameters:
          loadpaths (list): List of file paths for the variables at a given timestep.
          
        Returns:
          List of selected hypercube indices.
        """
        return self.extractor(loadpaths, self.nbytes, self.dims_full, self.dims_sl, self.num_hypercubes)
    
    def load_hypercubes(self, var, ts, hypercube_ids, base_path):
        """
        Loads hypercube data for a given variable at a specific timestep.
        
        Parameters:
          var (str): Variable name.
          ts (float): Timestep value.
          hypercube_ids (list): List of hypercube indices (tuples).
          base_path (str): Base directory for the data files.
          
        Returns:
          A flattened NumPy array containing the hypercube data.
        """
        file_path = os.path.join(base_path, f'{var}_{ts:0.6f}')
        # Verify that the data in the file is valid.
        check_data(file_path, *self.dims_full, self.nbytes)
        
        # Note: data is stored as [z, y, x] so we pass dims_full accordingly.
        data_memmap = np.memmap(file_path, dtype=np.float32, mode='r',
                                shape=(self.dims_full[2], self.dims_full[1], self.dims_full[0]))
        
        cubes = []
        for ix, iy, iz in hypercube_ids:
            x0 = ix * self.dims_sl[0]
            y0 = iy * self.dims_sl[1]
            z0 = iz * self.dims_sl[2]
            cube = data_memmap[z0:z0+self.dims_sl[2],
                               y0:y0+self.dims_sl[1],
                               x0:x0+self.dims_sl[0]]
            # Transpose to bring data to [x, y, z] order, then copy the cube.
            cubes.append(cube.copy().transpose(2, 1, 0))
        
        cubes = np.array(cubes)
        # Clean up the memmap.
        data_memmap._mmap.close()
        return cubes.reshape(-1)

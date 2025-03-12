import os
import numpy as np
import glob
from dataloaders import DataLoader


class GESTDataLoader(DataLoader):

    def __init__(self, args):
        super().__init__(args.path, dims=args.dims)
        self.args = args
        self.path = args.path
        self.grid_size = (args.nx, args.ny, args.nz)
        self.subcube_size = (args.nxsl, args.nysl, args.nzsl)

    #def __init__(self, base_path, grid_size=(2048, 2048, 2048), chunk_size=(1024, 1024, 1024), subcube_size=(32, 32, 32), subcube_origin=(512, 512, 512), verbose=False):
    #    self.base_path = base_path
    #    self.grid_size = grid_size
    #    self.chunk_size = chunk_size
    #    self.subcube_size = subcube_size
    #    self.subcube_origin = subcube_origin
    #    self.verbose = verbose

    def _get_filenames(self, variable):
        """Retrieve sorted filenames for a given variable."""
        var_path = os.path.join(self.path, variable)
        file_pattern = os.path.join(var_path, 'cube_*')
        files = sorted(glob.glob(file_pattern), key=lambda x: int(x.split('.')[-1]))
        if self.verbose:
            print(f"Found {len(files)} files for variable: {variable}")
        return files

    def _read_binary_cube(self, filename, has_vector=False):
        """Read a binary cube file using memory mapping for large files and extract a subcube."""
        dtype = np.float32  # Assuming 32-bit floating point precision
        shape = (3, *self.grid_size) if has_vector else self.grid_size  # Use full grid size

        if self.verbose:
            print(f"Memory-mapping file: {filename} with expected shape: {shape}")

        data = np.memmap(filename, dtype=dtype, mode='r', shape=shape, order='F')

        # Define subcube extraction range
        sx, sy, sz = self.subcube_size
        ox, oy, oz = (self.grid_size[0] // 2 - sx // 2, 
                      self.grid_size[1] // 2 - sy // 2, 
                      self.grid_size[2] // 2 - sz // 2)  # Centered subcube

        if self.verbose:
            print(f"Extracting sub-region: ({ox}:{ox+sx}, {oy}:{oy+sy}, {oz}:{oz+sz})")

        # Extract the subcube
        subcube = (data[:, ox:ox+sx, oy:oy+sy, oz:oz+sz] if has_vector else 
                   data[ox:ox+sx, oy:oy+sy, oz:oz+sz])

        if self.verbose:
            print(f"Extracted sub-region shape: {subcube.shape}")

        return subcube


    def load_xyz(self):
        """Generate 1D x, y, z coordinate arrays instead of full 3D grids."""
        if self.verbose:
            print("Generating coordinate arrays...")
        x = np.linspace(0, 1, self.grid_size[0])
        y = np.linspace(0, 1, self.grid_size[1])
        z = np.linspace(0, 1, self.grid_size[2])
        if self.verbose:
            print("Coordinate arrays generated successfully.")
        return x, y, z  # Returning 1D coordinate arrays instead of full grids

    #def load_multiple_timesteps(self, write_interval, num_timesteps, target, cv, file_filter='*_*'):
    def load_multiple_timesteps(self, variables=['velocity', 'pressure', 'enstrophy', 'dissipation']):
        """Load multiple variables from different timesteps, extracting only the subcube region."""
        variables=['velocity', 'pressure', 'enstrophy', 'dissipation']
        data_dict = {}
        for var in variables:
            files = self._get_filenames(var)
            if self.verbose:
                print(f"Loading {var} from {len(files)} files...")
            
            all_subcubes = []
            for f in files:
                if self.verbose:
                    print(f"Processing file: {f}")
                subcube = self._read_binary_cube(f, has_vector=(var == 'velocity'))
                all_subcubes.append(subcube)
            
            data_dict[var] = np.stack(all_subcubes, axis=0)  # Stack along the time dimension
            if self.verbose:
                print(f"Finished loading {var}, final shape: {data_dict[var].shape}")
        
        return data_dict

DataLoader = GESTDataLoader

if __name__ == "__main__":
    dataset_path = "/lustre/orion/tur120/world-shared/daludot/phy_cube_data"
    loader = GESTDataLoader(dataset_path, verbose=True)
    x, y, z = loader.load_xyz()
    print(f"Loaded coordinate arrays: x({len(x)}), y({len(y)}), z({len(z)})")
    
    data = loader.load_multiple_timesteps()
    for key, value in data.items():
        print(f"{key}: shape {value.shape}")

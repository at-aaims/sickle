import os
import numpy as np
import glob

class GESTDataLoader:
    def __init__(self, base_path, grid_size=(2048, 2048, 2048), chunk_size=(1024, 1024, 1024), verbose=False):
        self.base_path = base_path
        self.grid_size = grid_size
        self.chunk_size = chunk_size
        self.verbose = verbose

    def _get_filenames(self, variable):
        """Retrieve sorted filenames for a given variable."""
        var_path = os.path.join(self.base_path, variable)
        file_pattern = os.path.join(var_path, 'cube_*')
        files = sorted(glob.glob(file_pattern), key=lambda x: int(x.split('.')[-1]))
        if self.verbose:
            print(f"Found {len(files)} files for variable: {variable}")
        return files

    def _read_binary_cube(self, filename, has_vector=False, start_indices=(0, 0, 0), sub_size=(32, 32, 32)):
        """Read a binary cube file and extract a subset region."""
        dtype = np.float32
        nx, ny, nz = self.chunk_size
        nxsl, nysl, nzsl = sub_size
        sx, sy, sz = start_indices  # Starting indices for slicing
        
        if has_vector:
            shape = (3, nz, ny, nx)  # Vector field
        else:
            shape = (nz, ny, nx)  # Scalar field

        if self.verbose:
            print(f"Memory-mapping file: {filename}, extracting region: ({sx}:{sx+nxsl}, {sy}:{sy+nysl}, {sz}:{sz+nzsl})")

        # Memory-map the file but do not load it fully into memory
        data = np.memmap(filename, dtype=dtype, mode='r', shape=shape, order='F')

        # Extract the required sub-region
        if has_vector:
            data = data[:, sz:sz+nzsl, sy:sy+nysl, sx:sx+nxsl]  # (3, sub_nz, sub_ny, sub_nx)
        else:
            data = data[sz:sz+nzsl, sy:sy+nysl, sx:sx+nxsl]  # (sub_nz, sub_ny, sub_nx)

        if self.verbose:
            print(f"Extracted sub-region shape: {data.shape}")

        return data

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

    def load_multiple_timesteps(self, variables=['velocity', 'pressure', 'enstrophy', 'dissipation'], start_indices=(0, 0, 0), sub_size=(32, 32, 32)):
        """Load a small subset of the data for multiple variables."""
        data_dict = {}
        for var in variables:
            files = self._get_filenames(var)
            if self.verbose:
                print(f"Loading {var} from {len(files)} files...")
            
            all_chunks = []
            for f in files:
                if self.verbose:
                    print(f"Processing file: {f}")
                chunk = self._read_binary_cube(f, has_vector=(var == 'velocity'), start_indices=start_indices, sub_size=sub_size)
                all_chunks.append(chunk)
            
            data_dict[var] = np.concatenate(all_chunks, axis=1)  # Merge along y-axis
            if self.verbose:
                print(f"Finished loading {var}, final shape: {data_dict[var].shape}")
        
        return data_dict

if __name__ == "__main__":
    dataset_path = "/lustre/orion/tur120/world-shared/daludot/phy_cube_data"
    loader = GESTDataLoader(dataset_path, verbose=True)

    # Define a random start index for the sub-cube
    start_indices = (512, 512, 512)  # Choose a meaningful location inside the full 2048³ dataset
    sub_size = (32, 32, 32)  # Extracting a small cube

    x, y, z = loader.load_xyz()
    print(f"Loaded coordinate arrays: x({len(x)}), y({len(y)}), z({len(z)})")

    # Load a small sub-region of the dataset
    data = loader.load_multiple_timesteps(start_indices=start_indices, sub_size=sub_size)

    for key, value in data.items():
        print(f"{key}: shape {value.shape}")

import os
import numpy as np
import glob
import time
from dataloaders import DataLoader


class GESTDataLoader(DataLoader):

    def __init__(self, args):
        super().__init__(args.path, dims=args.dims)
        self.args = args
        self.path = args.path
        self.grid_size = (args.nx, args.ny, args.nz)
        self.subcube_size = (args.nxsl, args.nysl, args.nzsl)
        self.subcube_origin = (0, 0, 0)  # Default subcube origin
        self.varmap = dict(enstrophy='enst', velocity='uvw', pressure='p', dissipation='diss')

    def _get_filenames(self, variable, verbose=True):
        """Retrieve sorted filenames for a given variable."""
        var_prefix = self.varmap.get(variable, variable)
        var_path = os.path.join(self.path, variable)
        file_pattern = os.path.join(var_path, f'cube_{var_prefix}.*')
        files = sorted(glob.glob(file_pattern), key=lambda x: int(x.split('.')[-1]))
        if self.verbose:
            print(f"Found {len(files)} files for variable: {variable}")
        return files

    def _extract_times(self, file_names):
        """Extract available timesteps from file names."""
        return sorted(set(int(f.split('.')[-1]) for f in file_names))

    def _read_binary_cube(self, filename, has_vector=False):
        """Read a binary cube file using memory mapping for large files and extract a subcube."""
        dtype = np.float32  # Assuming 32-bit floating point precision
        shape = (3, *self.grid_size) if has_vector else self.grid_size  # Use full grid size

        if self.verbose:
            print(f"Memory-mapping file: {filename} with expected shape: {shape}")

        start_time = time.time()
        data = np.memmap(filename, dtype=dtype, mode='r', shape=shape, order='F')
        print(f"Loaded file {filename}")

        # Extract subcube region
        x0, y0, z0 = self.subcube_origin
        x1, y1, z1 = x0 + self.subcube_size[0], y0 + self.subcube_size[1], z0 + self.subcube_size[2]
        subcube = data[:, x0:x1, y0:y1, z0:z1] if has_vector else data[x0:x1, y0:y1, z0:z1]
        
        if self.verbose:
            print(f"Extracted sub-region shape: {subcube.shape}")

        return subcube.reshape(3, -1).T if has_vector else subcube.reshape(-1)

    def load_xyz(self):
        """Generate 1D x, y, z coordinate arrays instead of full 3D grids."""
        if self.verbose:
            print("Generating coordinate arrays...")
        x = np.linspace(0, 1, self.grid_size[0])
        y = np.linspace(0, 1, self.grid_size[1])
        z = np.linspace(0, 1, self.grid_size[2])
        self.num_pts = np.prod(self.subcube_size)
        if self.verbose:
            print(f"Coordinate arrays generated successfully. num_pts: {self.num_pts}")
        return x, y, z

    def load_multiple_timesteps(self, write_interval, num_timesteps, target, cv, file_filter='*_*'):
        """Load multiple variables from different timesteps, extracting only the subcube region."""
        x_labels = self.args.input_vars
        y_labels = self.args.output_vars
        cv_labels = self.args.cluster_var

        # Extract timesteps from available files
        file_names = self._get_filenames(x_labels[0])  # Assuming all vars share timesteps
        t_labels = self._extract_times(file_names)
        print('Available timesteps (t_labels):', t_labels)

        # Ensure num_timesteps does not exceed available timesteps
        num_timesteps = min(num_timesteps, len(t_labels))

        num_points = self.num_pts  # Adjusted to match the subcube size
        X = np.zeros((num_timesteps, num_points, len(x_labels) + 2))  # Extra space for velocity components
        Y = np.zeros((num_timesteps, num_points, len(y_labels)))
        cv_arr = np.zeros((num_timesteps, num_points, len(cv_labels)))

        for j, ts in enumerate(t_labels[:num_timesteps]):
            for i, var in enumerate(cv_labels):
                var_prefix = self.varmap.get(var, var)
                start_time = time.time()
                cv_arr[j, :, i] = self._read_binary_cube(f"{self.path}/{var}/cube_{var_prefix}.{ts}", has_vector=False)

            for i, var in enumerate(x_labels):
                var_prefix = self.varmap.get(var, var)
                start_time = time.time()
                subcube = self._read_binary_cube(f"{self.path}/{var}/cube_{var_prefix}.{ts}", has_vector=(var == 'velocity'))
                
                if var == 'velocity':
                    X[j, :, i:i+3] = subcube  # Spread velocity components across multiple channels
                else:
                    X[j, :, i] = subcube

            for i, var in enumerate(y_labels):
                var_prefix = self.varmap.get(var, var)
                start_time = time.time()
                Y[j, :, i] = self._read_binary_cube(f"{self.path}/{var}/cube_{var_prefix}.{ts}", has_vector=False)

        return X, Y, cv_arr

DataLoader = GESTDataLoader

if __name__ == "__main__":
    dataset_path = "/lustre/orion/tur120/world-shared/daludot/phy_cube_data"
    loader = GESTDataLoader(dataset_path, verbose=True)
    x, y, z = loader.load_xyz()
    print(f"Loaded coordinate arrays: x({len(x)}), y({len(y)}), z({len(z)})")

    data = loader.load_multiple_timesteps(write_interval=1, num_timesteps=8, target=None, cv=None)
    for key, value in data.items():
        print(f"{key}: shape {value.shape}")

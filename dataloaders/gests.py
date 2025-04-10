import os
import numpy as np
import glob
import time

from dataloaders import DataLoader
from helpers import check_data

# Import your hypercube handler if hypercube extraction is enabled.
# For example, if you implemented it in hypercubes/hypercube_manager.py:
try:
    from hypercubes.hypercube_manager import HypercubeHandler
except ImportError:
    HypercubeHandler = None


class GESTSDataLoader(DataLoader):

    def __init__(self, args, extractor=None):
        super().__init__(args.path, dims=args.dims)
        self.args = args
        self.path = args.path
        self.grid_size = (args.nx, args.ny, args.nz)
        self.subcube_size = (args.nxsl, args.nysl, args.nzsl)
        self.subcube_origin = (0, 0, 0)  # Default subcube origin
        self.varmap = dict(enstrophy='enst', velocity='uvw', pressure='p', dissipation='diss')
        self.verbose = getattr(args, 'verbose', False)
        nskips = (self.args.nxskip, self.args.nyskip, self.args.nzskip)

        # Use the extractor function if provided.
        if extractor:
            self.hypercube_handler = HypercubeHandler(
                method=args.hypercubes,
                dims_full=self.grid_size,
                dims_sl=self.subcube_size,
                nbytes=args.nbytes,
                num_hypercubes=args.num_hypercubes,
                num_clusters=self.args.num_clusters,
                nskips=nskips,
                use_parallel=True
            )
        else:
            self.hypercube_handler = None

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

    def _read_binary_cube(self, filename, cubeid=0, has_vector=False):
        """
        Read a binary cube file using memmapping, extracting the subcube corresponding to the provided cubeid.
        
        Parameters:
          filename (str): Path to the binary cube file.
          cubeid (int): Index (0 to 7) of the cube within the file. For 2048^3 datasets, this is always 0.
          has_vector (bool): Flag indicating if the data is a vector (velocity) which requires special handling.
        Returns:
          subcube: A flattened array containing the extracted subcube data.
        """
        dtype = np.float32  # 32-bit floating point precision assumed

        # Determine the base shape of the full cube stored in the file.
        # For the 8192^3 case, "grid_size" is the full shape for the file (e.g. (8192, 8192, 8192) 
        # or the equivalent regional shape in your loader), but note that each file is divided into 8 cubes along X.
        base_shape = self.grid_size
        
        # Determine number of cubes in this file (should be 8 for 8192³ datasets, 1 for 2048³).
        num_cubes_in_file = 8 if base_shape[0] > self.subcube_size[0] else 1

        # For files with 8 cubes, assume they are arranged as 8 X-neighbors.
        # Compute the size of one cube along X.
        cube_x_size = base_shape[0] // num_cubes_in_file
        cube_shape = (cube_x_size, base_shape[1], base_shape[2])

        # Calculate the number of bytes per cube.
        itemsize = np.dtype(dtype).itemsize
        bytes_per_cube = np.prod(cube_shape) * itemsize
        if has_vector:
            # For velocity, there are 3 components stored one after the other.
            bytes_per_cube *= 3

        # Calculate file offset based on cubeid.
        cube_offset = cubeid * bytes_per_cube

        if self.verbose:
            print(f"Reading file {filename} cubeid {cubeid} with offset {cube_offset} bytes, expected shape {cube_shape}, vector: {has_vector}")

        # Define shape for memmap. For velocity, include channel dimension.
        if has_vector:
            shape = (3, *cube_shape)  # (3, X, Z, Y) because data is stored in X-Z-Y order
        else:
            shape = cube_shape  # (X, Z, Y)

        # Create memmap with the specified offset. Note order='F' used by default.
        data = np.memmap(filename, dtype=dtype, mode='r', offset=cube_offset, shape=shape, order='F')

        # Rearranging data from stored X-Z-Y order to the expected X-Y-Z order.
        if not has_vector:
            # For scalar data, simply swap the last two axes.
            # For a shape (X, Z, Y), transposing axes (0, 2, 1) yields (X, Y, Z).
            data = np.transpose(data, (0, 2, 1))
        else:
            # For velocity, data shape is (3, X, Z, Y). Process each component separately.
            for comp in range(3):
                data[comp, :, :, :] = np.transpose(data[comp, :, :, :], (0, 2, 1))

        # Compute subcube origin based on fileid and cubeid.
        # This should yield the starting indices (x0, y0, z0) for the subcube within the cube.
        x0, y0, z0 = self.compute_subcube_origin(filename, cubeid)
        x1 = x0 + self.subcube_size[0]
        y1 = y0 + self.subcube_size[1]
        z1 = z0 + self.subcube_size[2]

        if self.verbose:
            print(f"Extracting subcube from indices x:{x0}-{x1}, y:{y0}-{y1}, z:{z0}-{z1}")

        # Extract the subcube region.
        if has_vector:
            # For vector data, extract and then flatten such that channels become features.
            subcube = data[:, x0:x1, y0:y1, z0:z1]
            subcube = subcube.reshape(3, -1).T
        else:
            subcube = data[x0:x1, y0:y1, z0:z1]
            subcube = subcube.reshape(-1)

        if self.verbose:
            print(f"Extracted subcube shape: {subcube.shape}")
        
        return subcube


    # --- New helper methods for hypercube extraction ---
    def _get_hypercube_IDs(self, variable, ts):
        """
        Compute hypercube IDs for a given variable at timestep ts.
        We use one variable’s file as the basis.
        """
        var_prefix = self.varmap.get(variable, variable)
        file_path = os.path.join(self.path, variable, f"cube_{var_prefix}.{ts}")
        # We pass a list containing this file path to the hypercube handler.
        return self.hypercube_handler.extract_ids([file_path])

    def _load_and_process_hypercubes(self, variable, ts, hypercubeIDs):
        var_prefix = self.varmap.get(variable, variable)
        # Convert ts to int so that a float like 0.0 becomes 0
        file_path = os.path.join(self.path, variable, f"cube_{var_prefix}.{int(ts)}")
        
        # Determine channels if needed and run check_data, etc.
        channels = 3 if variable == 'velocity' else 1
        check_data(file_path, self.args.nx, self.args.ny, self.args.nz, self.args.nbytes, channels=channels)
        
        return self.hypercube_handler.load_hypercubes(file_path, hypercubeIDs, has_vector=channels==3)

    def load_xyz(self):
        """Generate 1D x, y, z coordinate arrays instead of full 3D grids."""
        if self.verbose:
            print("Generating coordinate arrays...")
        x = np.linspace(0, 1, self.subcube_size[0])
        y = np.linspace(0, 1, self.subcube_size[1])
        z = np.linspace(0, 1, self.subcube_size[2])
        self.num_pts = np.prod(self.subcube_size)
        if self.verbose:
            print(f"Coordinate arrays generated successfully. num_pts: {self.num_pts}")
        return x, y, z

    def load_multiple_timesteps(self, write_interval, num_timesteps, target, cv, file_filter='*_*'):
        """Load multiple variables from different timesteps, extracting only the subcube region.
           If hypercube extraction is enabled, use that instead of reading the full subcube.
        """
        x_labels = self.args.input_vars
        y_labels = self.args.output_vars
        cv_labels = self.args.cluster_var

        # Extract timesteps from available files (assuming all variables share timesteps)
        file_names = self._get_filenames(x_labels[0])
        t_labels = self._extract_times(file_names)
        print('Available timesteps (t_labels):', t_labels)

        if self.args.timesteps:
            desired_timesteps = sorted(self.args.timesteps)
            # Only keep timesteps that are both desired and present in t_labels.
            t_labels = [int(ts) for ts in desired_timesteps if ts in t_labels]
            print('Filtered timesteps to load:', t_labels)

        num_timesteps = len(t_labels)

        # When extracting hypercubes, each loaded file returns an array of shape:
        num_points = self.args.num_hypercubes * self.num_pts  # number of hypercubes per file
        # Adjust the channel dimensions as needed (here we assume extra space for velocity channels)
        X = np.zeros((num_timesteps, num_points, len(x_labels) + 2))
        Y = np.zeros((num_timesteps, num_points, len(y_labels)))
        cv_arr = np.zeros((num_timesteps, num_points, len(cv_labels)))

        for j, ts in enumerate(t_labels[:num_timesteps]):
            if self.hypercube_handler is not None:
                # --- Use hypercube extraction ---
                # Compute hypercube IDs using one of the variables (for example, the first cv variable)
                hypercubeIDs = self._get_hypercube_IDs(cv_labels[0], ts)
                print(f"timestep {ts} hypercubeIDs {hypercubeIDs}")
                for i, var in enumerate(cv_labels):
                    cv_arr[j, :, i] = self._load_and_process_hypercubes(var, ts, hypercubeIDs)

                dest_col = 0
                for var in x_labels:
                    subcube = self._load_and_process_hypercubes(var, ts, hypercubeIDs)
                    if var == 'velocity':
                        # If velocity has multiple channels, spread them across consecutive channels.
                        #subcube = subcube.reshape(-1, 3)
                        X[j, :, dest_col:dest_col+3] = subcube
                        dest_col += 3
                    else:
                        X[j, :, dest_col] = subcube
                        dest_col += 1

                for i, var in enumerate(y_labels):
                    Y[j, :, i] = self._load_and_process_hypercubes(var, ts, hypercubeIDs)

            else:
                # --- Fallback to full subcube extraction ---
                for i, var in enumerate(cv_labels):
                    var_prefix = self.varmap.get(var, var)
                    cv_arr[j, :, i] = self._read_binary_cube(os.path.join(self.path, var, f"cube_{var_prefix}.{ts}"),
                                                              has_vector=False)
                for var in x_labels:
                    var_prefix = self.varmap.get(var, var)
                    subcube = self._read_binary_cube(os.path.join(self.path, var, f"cube_{var_prefix}.{ts}"),
                                                     has_vector=(var == 'velocity'))

                    dest_col = 0
                    if var == 'velocity':
                        X[j, :, dest_col:dest_col+3] = subcube
                        dest_col += 3
                    else:
                        X[j, :, dest_col] = subcube
                        dest_col += 1

                for i, var in enumerate(y_labels):
                    var_prefix = self.varmap.get(var, var)
                    Y[j, :, i] = self._read_binary_cube(os.path.join(self.path, var, f"cube_{var_prefix}.{ts}"),
                                                       has_vector=False)

        return X, Y, cv_arr


# For backward compatibility.
DataLoader = GESTSDataLoader

if __name__ == "__main__":
    # Example usage for testing:
    dataset_path = "/path/to/phy_cube_data/2048"
    # Make sure your args object includes the necessary fields, for example:
    class DummyArgs:
        def __init__(self):
            self.path = dataset_path
            self.dims = 3
            self.nx = 128
            self.ny = 128
            self.nz = 128
            self.nxsl = 32
            self.nysl = 32
            self.nzsl = 32
            self.input_vars = ['velocity', 'enstrophy']
            self.output_vars = ['pressure']
            self.cluster_var = ['dissipation']
            self.hypercubes = True
            self.nbytes = 4
            self.num_hypercubes = 16
            self.subsample_method = 'random'
            self.verbose = True

    args = DummyArgs()
    loader = GESTSDataLoader(args)
    x, y, z = loader.load_xyz()
    print(f"Loaded coordinate arrays: x({len(x)}), y({len(y)}), z({len(z)})")

    X, Y, cv_arr = loader.load_multiple_timesteps(write_interval=1, num_timesteps=8, target=None, cv=None)
    print(f"X shape: {X.shape}, Y shape: {Y.shape}, cv_arr shape: {cv_arr.shape}")

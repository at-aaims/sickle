import os
import glob
import numpy as np
import re
from dataloaders import DataLoader
from helpers import get_1Dgrid, get_data_memmap
from hypercubes import extract_hypercubes

class DataLoaderSSTBinary(DataLoader):

    def __init__(self, args):
        super().__init__(args.path, dims=args.dims)
        self.args = args
        self.path = args.path

    def _extract_times(self, file_names):
        pattern = r'_([0-9]+\.[0-9]+)$'
        time_pattern = re.compile(pattern)
        unique_times = set()
        for name in file_names:
            match = time_pattern.search(name)
            if match:
                unique_times.add(float(match.group(1)))
        return sorted(unique_times)

    def load_xyz(self):
        x = get_1Dgrid(self.args.Lh, self.args.nx-2, self.args.nxoffset, self.args.nxsl, self.args.nxskip)
        if self.args.gravity == 'y':
            y = get_1Dgrid(self.args.Lv, self.args.ny, self.args.nyoffset, self.args.nysl, self.args.nyskip)
            z = get_1Dgrid(self.args.Lh, self.args.nz, self.args.nzoffset, self.args.nzsl, self.args.nzskip)
        elif self.args.gravity == 'z':
            y = get_1Dgrid(self.args.Lh, self.args.ny, self.args.nyoffset, self.args.nysl, self.args.nyskip)
            z = get_1Dgrid(self.args.Lv, self.args.nz, self.args.nzoffset, self.args.nzsl, self.args.nzskip)
        else:
            raise Exception("Gravity should be defined")
        self.nx, self.ny, self.nz = x.shape[0], y.shape[0], z.shape[0]
        self.num_pts = self.nx * self.ny * self.nz
        print('num_pts:', self.num_pts)
        return x, y, z

    def _load_and_process(self, var, ts):
        """
        Loads a variable file at a given timestep and reshapes it.
        When hypercube extraction is enabled (i.e. args.hypercubes is not None),
        it uses the hypercube dimensions and, if necessary, partitions the full dataset
        into hypercubes. Otherwise, it extracts a single subcube from the full dataset.
        """
        file_path = os.path.join(self.path, f'{var}_{ts:0.6f}')
        print(f'Loading file: {file_path}')

        # If hypercube extraction is enabled, use cube dimensions; else use full dataset dims.
        dims = (self.args.nxsl, self.args.nysl, self.args.nzsl)

        # Load the data using the chosen dimensions.
        box = get_data_memmap(
            file_path,
            dims[0], dims[1], dims[2],
            self.args.nxsl, self.args.nysl, self.args.nzsl,
            self.args.nxoffset, self.args.nyoffset, self.args.nzoffset,
            self.args.nxskip, self.args.nyskip, self.args.nzskip,
            self.args.nbytes
        )
        full_data = box.reshape(dims)

        if self.args.hypercubes is not None:
            # If the file size exactly matches one hypercube, simply wrap it.
            if full_data.size == (self.args.nsxl * self.args.nsyl * self.args.nzsl):
                return full_data.reshape(1, -1)
            else:
                # Otherwise, assume the file holds a full dataset and partition it.
                hypercubes, _ = extract_hypercubes(
                    full_data,
                    cube_shape=(self.args.nsxl, self.args.nsyl, self.args.nzsl),
                    method=self.args.hypercubes,   # "random" or "maxent"
                    num_cubes=self.args.num_hypercubes
                )
                return hypercubes.reshape(hypercubes.shape[0], -1)
        else:
            # When not using hypercube extraction, extract a subcube based on offsets.
            subcube = full_data[
                self.args.nxoffset:self.args.nxoffset+self.args.nxsl,
                self.args.nyoffset:self.args.nysl,
                self.args.nzoffset:self.args.nzoffset+self.args.nzsl
            ]
            return subcube.reshape(-1)

    def load_multiple_timesteps(self, write_interval, num_timesteps, target, cv, file_filter='*_*'):
        self.path = os.path.dirname(self.path)
        file_names = glob.glob(os.path.join(self.path, file_filter))
        file_names = [os.path.basename(f) for f in file_names]
        print('Files:', sorted(file_names))

        x_labels = self.args.input_vars  # e.g., ['u', 'v', 'w', 'r']
        y_labels = self.args.output_vars  # e.g., ['p']
        cv_labels = self.args.cluster_var   # e.g., ['pv']
        t_labels = self._extract_times(file_names)
        print('Available timesteps (t_labels):', t_labels)

        if self.args.timesteps:
            desired_timesteps = sorted(self.args.timesteps)
            tolerance = 1e-6
            filtered_t_labels = []
            for dt in desired_timesteps:
                matches = [ts for ts in t_labels if abs(ts - dt) < tolerance]
                if matches:
                    filtered_t_labels.append(matches[0])
                else:
                    print(f"Warning: Timestep {dt} not found in the data.")
            t_labels = filtered_t_labels
            print('Filtered timesteps to load:', t_labels)

        num_timesteps = len(t_labels)
        hypercube_enabled = getattr(self.args, 'subsample_hypercube', False)
        if hypercube_enabled:
            # When extracting hypercubes, each loaded file returns an array of shape:
            # (num_hypercubes, flattened_cube_size)
            flattened_cube_size = self.args.nsxl * self.args.nsyl * self.args.nzsl
            num_points = self.args.num_hypercubes  # number of hypercubes per file
            X = np.zeros((num_timesteps, num_points, flattened_cube_size, len(x_labels)))
            Y = np.zeros((num_timesteps, num_points, flattened_cube_size, len(y_labels)))
            cv_arr = np.zeros((num_timesteps, num_points, flattened_cube_size))
        else:
            num_points = self.num_pts
            X = np.zeros((num_timesteps, num_points, len(x_labels)))
            Y = np.zeros((num_timesteps, num_points, len(y_labels)))
            cv_arr = np.zeros((num_timesteps, num_points))

        print("Loading NN input vars...")
        for i, var in enumerate(x_labels):
            for j, ts in enumerate(t_labels):
                data = self._load_and_process(var, ts)
                if hypercube_enabled:
                    # data shape: (num_hypercubes, flattened_cube_size)
                    X[j, :, :, i] = data
                else:
                    # data shape: (flattened_cube_size,)
                    X[j, :, i] = data

        print("Loading NN output vars...")
        for i, var in enumerate(y_labels):
            for j, ts in enumerate(t_labels):
                data = self._load_and_process(var, ts)
                if hypercube_enabled:
                    Y[j, :, :, i] = data
                else:
                    Y[j, :, i] = data

        print("Loading cluster vars...")
        for i, var in enumerate(cv_labels):
            for j, ts in enumerate(t_labels):
                data = self._load_and_process(var, ts)
                if hypercube_enabled:
                    cv_arr[j, :, :] = data
                else:
                    cv_arr[j, :] = data

        return X, Y, cv_arr

DataLoader = DataLoaderSSTBinary

if __name__ == "__main__":
    # Simple test code for DataLoaderSSTBinary.
    class DummyArgs:
        def __init__(self):
            self.path = "path/to/sst/binary/data"
            self.dims = 3
            self.Lh = 10
            self.Lv = 5
            self.nx = 100
            self.ny = 50
            self.nz = 20
            self.nxoffset = 0
            self.nyoffset = 0
            self.nzoffset = 0
            self.nxsl = 1
            self.nysl = 1
            self.nzsl = 1
            self.nxskip = 1
            self.nyskip = 1
            self.nzskip = 1
            self.nbytes = 4
            self.input_vars = ['u', 'v', 'w', 'r']
            self.output_vars = ['p']
            self.cluster_var = ['pv']
            self.timesteps = None
            # Flag to enable hypercube extraction and its parameters:
            self.subsample_hypercube = True
            self.nsxl = 32
            self.nsyl = 32
            self.nzsl = 32
            self.subsample_method = 'random'
            self.num_hypercubes = 16  # for example

    args = DummyArgs()
    dl = DataLoaderSSTBinary(args)
    # The following call would load data and, if enabled, extract hypercubes.
    X, Y, cv_arr = dl.load_multiple_timesteps(write_interval=1, num_timesteps=1, target=None, cv=None)

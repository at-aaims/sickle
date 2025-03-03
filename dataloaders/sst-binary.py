import os
import glob
import re
import numpy as np
from dataloaders import DataLoader
from helpers import get_1Dgrid, get_data_memmap

class DataLoaderSSTBinary(DataLoader):

    def __init__(self, args):
        # Note: Here we pass args.path and args.dims to the parent.
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

    def _load_and_reshape(self, var, ts):
        """
        Load data for a given variable and timestep, and reshape it.
        """
        file_path = os.path.join(self.path, f'{var}_{ts:0.6f}')
        print(f'Loading file: {file_path}')
        box = get_data_memmap(
            file_path,
            self.args.nx, self.args.ny, self.args.nz,
            self.args.nxsl, self.args.nysl, self.args.nzsl,
            self.args.nxoffset, self.args.nyoffset, self.args.nzoffset,
            self.args.nxskip, self.args.nyskip, self.args.nzskip,
            self.args.nbytes
        )
        return box.reshape(-1)

    def _load_vars(self, labels, t_labels, target, assign_func):
        """
        Load a set of variables and assign them into the target array using
        a provided assignment function.
        """
        for i, var in enumerate(labels):
            for j, ts in enumerate(t_labels):
                data = self._load_and_reshape(var, ts)
                assign_func(target, j, i, data)

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
        num_pts = self.num_pts
        X = np.zeros((num_timesteps, num_pts, len(x_labels)))
        Y = np.zeros((num_timesteps, num_pts, len(y_labels)))
        cv_arr = np.zeros((num_timesteps, num_pts))

        print("Loading NN input vars...")
        self._load_vars(
            x_labels, t_labels, X,
            lambda arr, j, i, data: arr.__setitem__((j, slice(None), i), data)
        )

        print("Loading NN output vars...")
        self._load_vars(
            y_labels, t_labels, Y,
            lambda arr, j, i, data: arr.__setitem__((j, slice(None), i), data)
        )

        print("Loading cluster vars...")
        # Note: cv_arr is 2D per timestep so we don't index the third dimension.
        for i, var in enumerate(cv_labels):
            for j, ts in enumerate(t_labels):
                data = self._load_and_reshape(var, ts)
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
            self.gravity = 'y'
            self.input_vars = ['u', 'v', 'w', 'r']
            self.output_vars = ['p']
            self.cluster_var = ['pv']
            self.timesteps = None

    args = DummyArgs()
    dl = DataLoaderSSTBinary(args)
    try:
        x, y, z = dl.load_xyz()
        print("Loaded XYZ:", x.shape, y.shape, z.shape)
        X, Y, cv = dl.load_multiple_timesteps(write_interval=100, num_timesteps=5, target='p', cv=['pv'])
        print("Data shapes:", X.shape, Y.shape, cv.shape)
    except Exception as e:
        print("Error during SSTBinary DataLoader test:", e)

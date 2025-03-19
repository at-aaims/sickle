import os
import glob
import numpy as np
import re
from dataloaders import DataLoader
from helpers import get_1Dgrid, check_data


class DataLoaderSSTBinary(DataLoader):

    def __init__(self, args, extractor=None):
        super().__init__(args.path, args.dims, args.verbose)
        self.args = args
        self.path = args.path
        if extractor:
            self.extract_hypercube_IDs = extractor

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

    def _get_hypercube_IDs(self, cv_vars, ts):
        """
        Identifies hypercube IDs to sample using cluster vars.
        """
        dims_full = (self.args.nx, self.args.ny, self.args.nz)
        dims_sl = (self.args.nxsl, self.args.nysl, self.args.nzsl)

        # file_path = os.path.join(self.path, f'{var}_{ts:0.6f}')
        file_paths = [os.path.join(self.path, f'{v}_{ts:0.6f}') for v in cv_vars]
        print(f'Finding hypercubes for timestep {ts:0.6f} using vars: {cv_vars}')

        hypercubeIDs = self.extract_hypercube_IDs(file_paths, self.args.nbytes, dims_full, dims_sl, \
                                                  self.args.num_hypercubes)

        return hypercubeIDs#, cv_arr
    
    def _load_and_process_hypercubes(self, var, ts, hypercubeIDs):
        """
        Loads a variable file at a given timestep and reshapes it.
        It uses certain hypercube based on thier IDs, identified 
        using hypercubes sampling.
        """
        file_path = os.path.join(self.path, f'{var}_{ts:0.6f}')
        print(f'Loading file using hypercube IDs: {file_path}')
        check_data(file_path, self.args.nx, self.args.ny, self.args.nz, self.args.nbytes)
        data_memmap = np.memmap(file_path, dtype=np.float32, mode='r', shape=(self.args.nz, self.args.ny, self.args.nx)) # NOTE: data is stored [z, y, x]
        
        cubes = []
        for ix, iy, iz in hypercubeIDs:
            x0, y0, z0 = ix * self.args.nxsl, iy * self.args.nysl, iz * self.args.nzsl
            cube = data_memmap[z0:z0+self.args.nzsl, y0:y0+self.args.nysl, x0:x0+self.args.nxsl]
            cubes.append(cube.copy().transpose(2, 1, 0)) # transposing data to be [x, y, z]
        cubes = np.array(cubes)
        data_memmap._mmap.close()
        del data_memmap, cube

        return cubes.reshape(-1)

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
        hypercube_enabled = self.args.hypercubes

        # When extracting hypercubes, each loaded file returns an array of shape:
        num_points = self.args.num_hypercubes * self.num_pts  # number of hypercubes per file
        X = np.zeros((num_timesteps, num_points, len(x_labels)))
        Y = np.zeros((num_timesteps, num_points, len(y_labels)))
        cv_arr = np.zeros((num_timesteps, num_points, len(cv_labels)))

        for j, ts in enumerate(t_labels):
            """
            TODO:
            This for loop is embarrassingly parallel.
            Same as subsampling for loop.
            """
            hypercubeIDs = self._get_hypercube_IDs(cv_labels, ts)
            for i, var in enumerate(cv_labels):
                cv_arr[j, :, i] = self._load_and_process_hypercubes(var, ts, hypercubeIDs)
            for i, var in enumerate(x_labels):
                X[j, :, i] = self._load_and_process_hypercubes(var, ts, hypercubeIDs)
            for i, var in enumerate(y_labels):
                Y[j, :, i] = self._load_and_process_hypercubes(var, ts, hypercubeIDs)
            
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
            self.nxsl = 32
            self.nysl = 32
            self.nzsl = 32
            self.subsample_method = 'random'
            self.num_hypercubes = 16  # for example

    args = DummyArgs()
    dl = DataLoaderSSTBinary(args)
    # The following call would load data and, if enabled, extract hypercubes.
    X, Y, cv_arr = dl.load_multiple_timesteps(write_interval=1, num_timesteps=1, target=None, cv=None)

"""Usage example: python dataloader.py --path $HOME/foam/cylinder"""
import glob
import numpy as np
import os
import pandas as pd
import re

from mpi4py import MPI

from constants import FieldPredictionType
from helpers import get_1Dgrid, get_data_memmap
# We do this so that users who are not using OpenFOAM need not install fluidfoam
try: 
    from fluidfoam import readscalar, readvector, readforce
except:
    print("WARNING: fluidfoam not able to be loaded")


class DataLoader():

    def __init__(self, path, dims=2, verbose=False):
        self.path = path
        self.verbose = verbose
        self.dims = dims

    def to_csv(self, Y, X, time, columns):
        """Output CSV file named by timestamp, e.g. 1000.csv"""
        df = pd.DataFrame(np.concatenate((Y, X), axis=1), columns=columns)
        df.to_csv(str(time) + '.csv', index=False)


class DataLoaderOF(DataLoader):

    def load_forces(self, write_interval=100):
        forces = readforce(self.path, time_name='0', name='forces')
        # Drag force is composed of both a viscous and pressure components
        time = forces[:, 0]
        drag = forces[:, 1] + forces[:, 2]
        return time[::write_interval], drag[::write_interval]

    def load_xyz(self):
        x, y, z = readvector(self.path, '0', 'C.gz')
        x = np.expand_dims(x, axis=1)
        y = np.expand_dims(y, axis=1)
        z = np.expand_dims(z, axis=1)
        return x, y, z

    def load_multiple_timesteps(self, write_interval, num_timesteps, target, cv):

        p = readscalar(self.path, str(write_interval), 'p.gz')
        num_pts = p.shape[0]
        
        print(num_pts, num_timesteps)

        p = np.zeros((num_timesteps, num_pts))
        u = np.zeros((num_timesteps, num_pts))
        v = np.zeros((num_timesteps, num_pts))
        w = np.zeros((num_timesteps, num_pts))
        wz = np.zeros((num_timesteps, num_pts))

        for i, ts in enumerate(range(write_interval, write_interval*num_timesteps+1, write_interval)):
            print(i, ts)
            p[i, :] = readscalar(self.path, str(ts), 'p.gz')
            u[i, :], v[i, :], w[i, :] = readvector(self.path, str(ts), 'U.gz')
            _, _, wz[i, :] = readvector(self.path, str(ts), 'vorticity.gz')
    
        params = {'p': p, 'wz': wz, 'pwz': np.stack((p, wz), axis=1)}

        if target == 'drag':
            params['drag'] = self.load_forces()[1].reshape(-1, 1)

        if self.dims == 2:
            X = np.stack((u, v), axis=-1)
        elif self.dims == 3:
            X = np.stack((u, v, w), axis=-1)
        else:
            raise ValueError("dims must be either 2 or 3")
        Y = params[target]
        cv = params[cv[0]]

        return X, Y, cv


class DataLoaderCSV(DataLoader):
    
    def __init__(self, path, verbose=False, prefix='cylinder_t', zwidth=4):
        super().__init__(path, verbose)
        self.prefix = prefix
        self.zwidth = zwidth
    
    def load_xyz(self):
        dfpath = os.path.join(self.path, f'{self.prefix}{str(1).zfill(self.zwidth)}.csv')
        data = pd.read_csv(dfpath)
        x = data["X"].to_numpy()
        y = data["Y"].to_numpy()
        self.num_points = len(x)
        return x, y

    def load_multiple_timesteps(self, write_interval, num_timesteps, target, cv):
        num_pts = self.num_points
        x_labels = ["dudx", "dudy", "dvdx", "dvdy", "vortZ"]
        Y = np.zeros((num_timesteps, num_pts))
        X = np.zeros((num_timesteps, num_pts, len(x_labels)))
        cv = np.zeros((num_timesteps, num_pts))
        
        for i, ts in enumerate(range(write_interval, write_interval*num_timesteps+1, write_interval)):
            dfpath = os.path.join(self.path,f'cylinder_t{str(i+1).zfill(self.zwidth)}.csv')
            data = pd.read_csv(dfpath)
            tke_val = abs(data["TKE"].to_numpy())
            tke_0 = np.where(tke_val <= 1.0e-9)[0]
            tke_val[tke_0] = 1.0e-8
            Y[i, :] = np.log(tke_val)
            X[i, :] = data[x_labels].to_numpy()
            cv[i,:] = data["vortZ"].to_numpy()

        return X, Y, cv


class DataLoaderNPZ(DataLoader):

    def __init__(self, path, verbose=False, prefix='cylinder_t', zwidth=4):
        super().__init__(path, verbose)

    def _extract_times(self, file_names):
        time_pattern = re.compile(r'_t(\d+\.\d+)\.npz')
        unique_times = set()
        for name in file_names:
            match = time_pattern.search(name)
            if match:
                unique_times.add(float(match.group(1)))

        return sorted(unique_times)

    def load_xyz(self):
        data = np.load(self.path, allow_pickle=True)
        x, y, z = data['x'], data['y'], data['z']
        self.nx, self.ny, self.nz = x.shape[0], y.shape[0], z.shape[0]
        self.nskip = data['args'].item().nxskip
        self.num_pts = self.nx * self.ny * self.nz
        print('num_pts:', self.num_pts)
        return x, y, z
    
    def load_multiple_timesteps(self, write_interval, num_timesteps, target,\
                                cv, file_filter='*t*', label='PVsample'):

        self.path = os.path.dirname(self.path)
        file_names = glob.glob(os.path.join(self.path, file_filter))
        file_names = [os.path.basename(f) for f in file_names] 
        print('files:', sorted(file_names))
        x_labels = ['u', 'v', 'w', 'r']
        t_labels = self._extract_times(file_names)
        print('t_labels:', t_labels)
        
        num_timesteps = len(t_labels)
        num_pts = self.num_pts
        Y = np.zeros((num_timesteps, num_pts))
        X = np.zeros((num_timesteps, num_pts, len(x_labels)))
        cv = np.zeros((num_timesteps, num_pts))

        # Read neural net inputs / observations
        for i, var in enumerate(x_labels):
            for j, ts in enumerate(t_labels):
                path = os.path.join(self.path, f'{var}_{label}-nx{self.nx}ny{self.ny}nz{self.nz}_nskip{self.nskip}_t{ts:0.6f}.npz')
                data = np.load(path)
                box = data["datacube"]
                X[j, :, i] = box.reshape(-1)

        # Read neural net outputs / targets
        y_labels = ['p'] # pressure
        for j, ts in enumerate(t_labels):
            path = os.path.join(self.path, f'{y_labels[0]}_{label}-nx{self.nx}ny{self.ny}nz{self.nz}_nskip{self.nskip}_t{ts:0.6f}.npz')
            data = np.load(path)
            box = data["datacube"]
            Y[j, :] = box.reshape(-1)

        # Read cluster variable for MaxEnt analysis
        cv_labels = ['pv'] # potential vorticity
        for j, ts in enumerate(t_labels):
            path = os.path.join(self.path, f'{cv_labels[0]}_{label}-nx{self.nx}ny{self.ny}nz{self.nz}_nskip{self.nskip}_t{ts:0.6f}.npz')
            data = np.load(path)
            box = data["datacube"]
            cv[j, :] = box.reshape(-1)

        return X, Y, cv

class DataLoaderSSTBinary(DataLoader):

    def __init__(self, args):
        super().__init__(args)
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

    def load_multiple_timesteps(self, write_interval, num_timesteps, target,\
                                cv, file_filter='*_*'):

        self.path = os.path.dirname(self.path)
        file_names = glob.glob(os.path.join(self.path, file_filter))
        file_names = [os.path.basename(f) for f in file_names] 
        print('Files:', sorted(file_names))

        x_labels = self.args.input_vars # ['u', 'v', 'w', 'r']
        y_labels = self.args.output_vars # ['p'] # pressure
        cv_labels = self.args.cluster_var # ['pv'] # potential vorticity
        t_labels = self._extract_times(file_names)
        print('Available timesteps (t_labels):', t_labels)
        
        if self.args.timesteps:
            desired_timesteps = sorted(self.args.timesteps)
            # Use a tolerance to handle floating point precision
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
        cv = np.zeros((num_timesteps, num_pts)) #, len(cv_labels)))

        # Read neural net inputs / observations
        print("Loading NN input vars...")
        for i, var in enumerate(x_labels):
            for j, ts in enumerate(t_labels):
                path = os.path.join(self.path, f'{var}_{ts:0.6f}')
                print(f'Loading file: {path}')
                box = get_data_memmap( path, self.args.nx, self.args.ny, self.args.nz, 
                                                  self.args.nxsl, self.args.nysl, self.args.nzsl, 
                                                  self.args.nxoffset, self.args.nyoffset, self.args.nzoffset, 
                                                  self.args.nxskip, self.args.nyskip, self.args.nzskip, self.args.nbytes)
                X[j, :, i] = box.reshape(-1)

        # Read neural net outputs / targets
        print("Loading NN output vars...")
        for i, var in enumerate(y_labels):
            for j, ts in enumerate(t_labels):
                path = os.path.join(self.path, f'{var}_{ts:0.6f}')
                print(f'Loading file: {path}')
                box = get_data_memmap( path, self.args.nx, self.args.ny, self.args.nz, 
                                                  self.args.nxsl, self.args.nysl, self.args.nzsl, 
                                                  self.args.nxoffset, self.args.nyoffset, self.args.nzoffset, 
                                                  self.args.nxskip, self.args.nyskip, self.args.nzskip, self.args.nbytes)
                # Y[j, :] = box.reshape(-1)
                Y[j, :, i] = box.reshape(-1)

        # Read cluster variable for MaxEnt analysis
        print("Loading cluster vars...")
        for i, var in enumerate(cv_labels):
            for j, ts in enumerate(t_labels):
                path = os.path.join(self.path, f'{var}_{ts:0.6f}')
                print(f'Loading file: {path}')
                box = get_data_memmap( path, self.args.nx, self.args.ny, self.args.nz, 
                                                  self.args.nxsl, self.args.nysl, self.args.nzsl, 
                                                  self.args.nxoffset, self.args.nyoffset, self.args.nzoffset, 
                                                  self.args.nxskip, self.args.nyskip, self.args.nzskip, self.args.nbytes)
                cv[j, :] = box.reshape(-1)
                # cv[j, :, i] = box.reshape(-1)

        return X, Y, cv


def create_sequences_from_csv(path, sequence_length):
    """Read the CSV files and create sequences"""
    files = sorted([f for f in os.listdir(path) if f.endswith('.csv')])
    sequences = []
    labels = []

    for i in range(sequence_length, len(files) + 1):
        sequence = []
        label_seq = []
        for j in range(i-sequence_length, i):
            file = files[j]
            df = pd.read_csv(os.path.join(path, file))
            label_seq.append(df.iloc[:, 0].values)  # assume the first column is the target
            sequence.append(df.iloc[:, 1:].values)  # rest of the columns are features

        sequences.append(np.array(sequence))
        labels.append(np.array(label_seq))

    return np.array(sequences), np.array(labels)


def create_sequences(X, Y, window_size=3, overlap=2, field_prediction_type=FieldPredictionType.GLOBAL):
    """ Create time sequences of a given window size from the input arrays X and Y with specified overlap """
    nt, nsamples, nvars = X.shape
    stride = window_size - overlap
    assert stride > 0, f"window_size ({window_size}) must be > overlap ({overlap})"
    num_sequences = (nt - window_size) // stride + 1

    X_sequences = np.zeros((num_sequences, window_size, nsamples * nvars))
    if field_prediction_type == FieldPredictionType.GLOBAL:
        Y_sequences = np.zeros((num_sequences, window_size))
    else:
        Y_sequences = np.zeros((num_sequences, window_size, nsamples))

    for i in range(num_sequences):
        start_index = i * stride
        X_sequences[i] = X[start_index:start_index + window_size].reshape(window_size, nsamples * nvars)
        if field_prediction_type == FieldPredictionType.GLOBAL:
            Y_sequences[i] = Y[start_index:start_index + window_size].flatten()
        else:
            Y_sequences[i] = Y[start_index:start_index + window_size].reshape(window_size, -1)

    return X_sequences, Y_sequences


if __name__ == "__main__":

    from args import args
    
    print(args.path)
    print(args.num_timesteps)
    
    #dl = DataLoaderOF(args.path, dims=args.dims)
    dl = DataLoaderNPZ(args.path)
    
    #X, Y = dl.read_solution('1000')
    #print(X.shape, Y.shape)

    #X, Y = create_sequences(*dl.load_multiple_timesteps(100, 100))
    #print(X.shape, Y.shape)

    x, y, z = dl.load_xyz()
    print(x.shape, y.shape, z.shape)
    X, Y, cv = dl.load_multiple_timesteps(args.write_interval, args.num_timesteps, target=args.target, cv=args.cluster_var)
    print(X.shape, Y.shape, cv.shape)

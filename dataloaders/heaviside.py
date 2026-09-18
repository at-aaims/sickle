import numpy as np
from dataloaders import DataLoader


class DataLoaderHeaviside(DataLoader):
    """Synthetic sine + Heaviside-sine test field for MaxEnt subsampling.

    eta(r) = sin(2*pi*r) + 1[r >= r0] * A1 * sin(2*pi*k1*r),  r_j = (j+0.5)/N

    A localized high-frequency "switch" region turns on at r0, giving an
    exact, known structural event ({r >= r0}) to check whether MaxEnt
    subsampling preferentially resolves it relative to random sampling.
    Ported from jb_maxent's `core_k2_field` (which itself started as a
    direct pull of this repo's subsampling/maxent.py).

    There is no real time evolution: the field is deterministic given
    (N, A1, k1, r0), so every requested timestep is an identical copy of
    the same field. Timesteps only give independent random draws through
    the subsampler.
    """

    def __init__(self, args, **kwargs):
        super().__init__(args.path, args.dims, args.verbose)
        self.N = getattr(args, 'N', 4096)
        self.A1 = getattr(args, 'A1', 0.8)
        self.k1 = getattr(args, 'k1', 16.0)
        self.r0 = getattr(args, 'r0', 0.75)

    def _field(self):
        r = (np.arange(self.N, dtype=np.float64) + 0.5) / self.N
        event = r >= self.r0
        eta = np.sin(2.0 * np.pi * r) + event * self.A1 * np.sin(2.0 * np.pi * self.k1 * r)
        return {'r': r, 'eta': eta, 'event': event.astype(np.float64)}

    def load_xyz(self):
        r = self._field()['r']
        self.num_pts = self.N
        x = np.expand_dims(r, axis=1)
        y = np.zeros_like(x)
        z = np.zeros_like(x)
        return x, y, z

    def load_multiple_timesteps(self, write_interval, num_timesteps, target, cv):
        params = self._field()
        num_pts = self.N

        X = np.tile(params['r'].reshape(1, num_pts, 1), (num_timesteps, 1, 1))
        Y = np.tile(params[target].reshape(1, num_pts, 1), (num_timesteps, 1, 1))
        cv_arr = np.tile(params[cv[0]].reshape(1, num_pts, 1), (num_timesteps, 1, 1))

        return X, Y, cv_arr


DataLoader = DataLoaderHeaviside

if __name__ == "__main__":
    # Simple test code for DataLoaderHeaviside.
    class DummyArgs:
        def __init__(self):
            self.path = './data'
            self.dims = 2
            self.verbose = True
            self.N = 4096
            self.A1 = 0.8
            self.k1 = 16.0
            self.r0 = 0.75

    args = DummyArgs()
    dl = DataLoaderHeaviside(args)
    x, y, z = dl.load_xyz()
    print("XYZ shapes:", x.shape, y.shape, z.shape)
    X, Y, cv = dl.load_multiple_timesteps(write_interval=1, num_timesteps=5, target='eta', cv=['eta'])
    print("Data shapes:", X.shape, Y.shape, cv.shape)

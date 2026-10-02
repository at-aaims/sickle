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

    Time evolution (all off by default, which gives the static field, every
    timestep identical): with t = k*step_dt for timestep k,

    eta(r,t) = sin(2*pi*(r - wave_speed*t))
               + 1[r >= r0(t)] * A(t) * sin(2*pi*k1*(r - hf_wave_speed*t))

    - wave_speed, hf_wave_speed: advection speed of the base wave and of the
      high-frequency wave (periodic domain).
    - front_speed: speed of the switch front, r0(t) = r0 + front_speed*t,
      reflected back and forth between 0 and 1. event(t) is the exact
      indicator for that timestep.
    - amp_eps, amp_tau: bursty amplitude A(t) = A1*max(0, 1 + amp_eps*s(t)),
      where s(t) is a unit-variance Ornstein-Uhlenbeck process with
      correlation time amp_tau.
    - field_seed: seeds s(t), so the field is reproducible even with `noseed`.
    """

    def __init__(self, args, **kwargs):
        super().__init__(args.path, args.dims, args.verbose)
        self.N = getattr(args, 'N', 4096)
        self.A1 = getattr(args, 'A1', 0.8)
        self.k1 = getattr(args, 'k1', 16.0)
        self.r0 = getattr(args, 'r0', 0.75)
        self.wave_speed = getattr(args, 'wave_speed', 0.0)
        self.hf_wave_speed = getattr(args, 'hf_wave_speed', 0.0)
        self.front_speed = getattr(args, 'front_speed', 0.0)
        self.amp_eps = getattr(args, 'amp_eps', 0.0)
        self.amp_tau = getattr(args, 'amp_tau', 1.0)
        self.step_dt = getattr(args, 'step_dt', 1.0)
        self.field_seed = getattr(args, 'field_seed', 0)

    def _amplitude_modulation(self, num_timesteps):
        """Unit-variance OU samples s(t_k), k = 0..num_timesteps-1 (AR(1) form)."""
        s = np.zeros(num_timesteps)
        if self.amp_eps == 0.0 or self.amp_tau <= 0.0:
            return s
        rng = np.random.default_rng(self.field_seed)
        a = np.exp(-self.step_dt / self.amp_tau)
        s[0] = rng.standard_normal()
        for k in range(1, num_timesteps):
            s[k] = a * s[k - 1] + np.sqrt(1.0 - a * a) * rng.standard_normal()
        return s

    def _field(self, t=0.0, s=0.0):
        r = (np.arange(self.N, dtype=np.float64) + 0.5) / self.N
        u = (self.r0 + self.front_speed * t) % 2.0
        r0_t = u if u <= 1.0 else 2.0 - u
        event = r >= r0_t
        amp = self.A1 * max(0.0, 1.0 + self.amp_eps * s)
        eta = (np.sin(2.0 * np.pi * (r - self.wave_speed * t))
               + event * amp * np.sin(2.0 * np.pi * self.k1 * (r - self.hf_wave_speed * t)))
        return {'r': r, 'eta': eta, 'event': event.astype(np.float64)}

    def load_xyz(self):
        r = self._field()['r']
        self.num_pts = self.N
        x = np.expand_dims(r, axis=1)
        y = np.zeros_like(x)
        z = np.zeros_like(x)
        return x, y, z

    def load_multiple_timesteps(self, write_interval, num_timesteps, target, cv):
        num_pts = self.N
        mod = self._amplitude_modulation(num_timesteps)
        fields = [self._field(k * self.step_dt, mod[k]) for k in range(num_timesteps)]

        X = np.stack([f['r'] for f in fields]).reshape(num_timesteps, num_pts, 1)
        Y = np.stack([f[target] for f in fields]).reshape(num_timesteps, num_pts, 1)
        cv_arr = np.stack([f[cv[0]] for f in fields]).reshape(num_timesteps, num_pts, 1)

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

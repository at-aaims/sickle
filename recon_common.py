"""Shared pieces for the Heaviside sparse-to-dense reconstruction study.

Field generation, MaxEnt sampling probabilities, mask sampling, baselines and
metrics used by train_recon.py, eval_recon.py and predict_plot.py.
"""
import contextlib
import io
import sys
from types import SimpleNamespace

import numpy as np
import torch

# sickle's args.py parses sys.argv at import time (pulled in via plotting.py
# by subsampling.maxent), so import with a clean argv and silence its print.
_argv = sys.argv
sys.argv = [_argv[0]]
with contextlib.redirect_stdout(io.StringIO()):
    from dataloaders.heaviside import DataLoaderHeaviside
    from subsampling.maxent import MaxentSubsampler
sys.argv = _argv

N = 4096

# Time-evolution settings from config/synthetic/*.yaml.
FIELD_DEFAULTS = dict(N=N, A1=0.8, k1=16.0, step_dt=0.05, wave_speed=0.2,
                      hf_wave_speed=0.5, front_speed=0.05, amp_eps=0.4, amp_tau=0.3)

# The loader's field_seed only drives the amplitude process, so each "seed"
# here also gets its own switch location r0; otherwise every seed would share
# one front trajectory and the held-out test seed would not be held out.
FIELDS = {
    0: dict(field_seed=0, r0=0.35),
    1: dict(field_seed=1, r0=0.50),
    2: dict(field_seed=2, r0=0.65),
    3: dict(field_seed=3, r0=0.80),
    4: dict(field_seed=4, r0=0.575),  # test only
}
TRAIN_FIELDS = [0, 1, 2, 3]
TEST_FIELD = 4
NUM_TIMESTEPS = 200
VAL_FRAC = 0.2  # last 20% of each training field's timesteps
K1_BAND = (8, 24)  # FFT bins (cycles per unit length) for the spectral error


def make_field(field_id, num_timesteps=NUM_TIMESTEPS, static=False):
    """Return r (N,), eta (T, N) and event (T, N) for one field configuration."""
    cfg = dict(FIELD_DEFAULTS, **FIELDS[field_id], path='.', dims=1, verbose=False)
    if static:
        cfg.update(wave_speed=0.0, hf_wave_speed=0.0, front_speed=0.0, amp_eps=0.0)
    dl = DataLoaderHeaviside(SimpleNamespace(**cfg))
    mod = dl._amplitude_modulation(num_timesteps)
    fields = [dl._field(k * dl.step_dt, mod[k]) for k in range(num_timesteps)]
    r = fields[0]['r']
    eta = np.stack([f['eta'] for f in fields])
    event = np.stack([f['event'] for f in fields])
    return r, eta, event


def maxent_probs(eta, num_clusters=10):
    """Per-snapshot MaxEnt sampling probabilities, shape (T, N).

    Uses sickle's MaxentSubsampler (k-means on eta, KL in-strength per
    cluster) exactly as subsample.py does with cluster_var [eta]; only the
    final draw is left out so fresh masks can be drawn from the same
    probabilities every epoch.
    """
    args = SimpleNamespace(cluster_var=['eta'], num_clusters=num_clusters, plot=False)
    cv = eta[..., None]
    sub = MaxentSubsampler(cv, args, coords=(None, None, None), cv=cv)
    probs = np.zeros_like(eta)
    for t in range(eta.shape[0]):
        data = cv[t]
        labels, clusters = sub.perform_kmeans(data, num_clusters)
        with contextlib.redirect_stdout(io.StringIO()):
            in_strengths = sub.compute_entropy(clusters, timestep=t)
        p = in_strengths[labels]
        probs[t] = p / p.sum()
    return probs


def sample_masks(num_samples, probs=None, batch=None, n=N, generator=None, device='cpu'):
    """Draw masks of shape (B, n), sampling without replacement.

    num_samples: int or (B,) tensor of per-row sample counts M.
    probs: None for uniform random sampling, else (B, n) weights; weighted
    draws use Gumbel top-k, equivalent to np.random.choice(replace=False, p=...).
    """
    if probs is None:
        keys = torch.rand(batch, n, generator=generator, device=device)
    else:
        u = torch.rand(probs.shape, generator=generator, device=device).clamp_min(1e-20)
        keys = torch.log(probs.clamp_min(1e-30)) - torch.log(-torch.log(u))
    ranks = keys.argsort(dim=1, descending=True).argsort(dim=1)
    m = torch.as_tensor(num_samples, device=device)
    if m.ndim == 0:
        m = m.expand(keys.shape[0])
    return (ranks < m[:, None]).float()


def make_input(eta, mask, r):
    """Stack the U-Net input channels: sparse eta, mask, r. Shapes (B, n) -> (B, 3, n)."""
    return torch.stack([eta * mask, mask, r.expand_as(eta)], dim=1)


# ---------------------------------------------------------------- baselines

def interp_linear(r, eta, mask):
    idx = np.flatnonzero(mask)
    return np.interp(r, r[idx], eta[idx])


def interp_cubic(r, eta, mask):
    from scipy.interpolate import CubicSpline
    idx = np.flatnonzero(mask)
    # Hold the end values outside the sampled range, like np.interp; the
    # spline's polynomial extrapolation diverges across large end gaps.
    return CubicSpline(r[idx], eta[idx])(np.clip(r, r[idx[0]], r[idx[-1]]))


def fourier_mlp_fit(r, eta, masks, steps=3000, num_features=128, sigma=4.0,
                    hidden=128, lr=1e-3, seed=0, device='cuda'):
    """Fit one Fourier-feature MLP (r -> eta) per snapshot, all in parallel.

    eta, masks: (B, N) arrays. Returns predictions on the full grid, (B, N).
    sigma is the std of the random frequencies, in cycles per unit length;
    4 was best of {1, 2, 3, 4, 8, 16, 32} on training field 1 at M = 64 and 256.
    Each snapshot gets its own weights; batched with bmm.
    """
    g = torch.Generator(device='cpu').manual_seed(seed)
    B = eta.shape[0]
    rt = torch.tensor(r, dtype=torch.float32, device=device)
    freqs = (torch.randn(num_features, generator=g) * sigma).to(device)
    feats = torch.cat([torch.sin(2 * np.pi * rt[:, None] * freqs),
                       torch.cos(2 * np.pi * rt[:, None] * freqs)], dim=1)  # (N, 2F)

    def init(fan_in, fan_out):
        w = torch.randn(B, fan_in, fan_out, generator=g) * np.sqrt(2.0 / fan_in)
        return w.to(device).requires_grad_(), torch.zeros(B, 1, fan_out, device=device, requires_grad=True)

    W1, b1 = init(2 * num_features, hidden)
    W2, b2 = init(hidden, hidden)
    W3, b3 = init(hidden, 1)
    params = [W1, b1, W2, b2, W3, b3]

    def forward(x):  # x: (B, P, 2F)
        h = torch.relu(torch.bmm(x, W1) + b1)
        h = torch.relu(torch.bmm(h, W2) + b2)
        return (torch.bmm(h, W3) + b3)[..., 0]

    et = torch.tensor(eta, dtype=torch.float32, device=device)
    mt = torch.tensor(masks, dtype=torch.float32, device=device)
    x_all = feats.expand(B, -1, -1)
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        pred = forward(x_all)
        loss = (((pred - et) ** 2) * mt).sum() / mt.sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return forward(x_all).cpu().numpy()


# ------------------------------------------------------------------ metrics

def metrics(pred, true, event):
    """Per-snapshot error sums and counts; all inputs (B, N). Returns dict of (B,) arrays.

    Kept as sums so pool_metrics() can pool over snapshots. Per-snapshot ratios
    are unstable here: when the front sits near r = 1 the burst region is a
    few points wide (or empty), and when A(t) hits 0 there is no energy in the
    k1 band to normalize by.
    """
    err2 = (pred - true) ** 2
    smooth = event < 0.5
    burst = ~smooth
    lo, hi = K1_BAND
    Fp = np.abs(np.fft.rfft(pred, axis=1))[:, lo:hi + 1]
    Ft = np.abs(np.fft.rfft(true, axis=1))[:, lo:hi + 1]
    return {
        'sse': err2.sum(1), 'n': np.full(len(err2), err2.shape[1]),
        'sse_smooth': (err2 * smooth).sum(1), 'n_smooth': smooth.sum(1),
        'sse_burst': (err2 * burst).sum(1), 'n_burst': burst.sum(1),
        'spec_num': ((Fp - Ft) ** 2).sum(1), 'spec_den': (Ft ** 2).sum(1),
    }


def pool_metrics(rows):
    """Pool metric sums over snapshots (list of dicts from metrics()).

    MSEs are total squared error over total points in each region; the
    spectral error is the relative L2 error of |FFT| in the k1 band, pooled.
    """
    s = {k: float(np.sum([r[k] for r in rows])) for k in
         ['sse', 'n', 'sse_smooth', 'n_smooth', 'sse_burst', 'n_burst', 'spec_num', 'spec_den']}
    return {
        'mse': s['sse'] / s['n'],
        'mse_smooth': s['sse_smooth'] / s['n_smooth'],
        'mse_burst': s['sse_burst'] / s['n_burst'],
        'spec_err': np.sqrt(s['spec_num'] / s['spec_den']),
    }

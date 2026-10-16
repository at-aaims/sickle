"""Train a small 1D U-Net on randomized Heaviside fields for the browser demo.

Unlike train_recon.py, which sees five fixed field configurations, every
training example here draws its own burst wavenumber k, burst amplitude A,
front position r0, background wave, sample count M and noise level, so the
model has to reconstruct a family of signals rather than one template.

The network predicts a residual on top of linear interpolation of the samples
and is small enough (channels 16/32/64/128, N = 1024) to run in plain
JavaScript. Weights are exported as float16 for demo/heaviside_demo/index.html.

    python demo/heaviside_demo/train_demo.py            # train + export
    python demo/heaviside_demo/train_demo.py --export   # re-export only
"""
import argparse
import base64
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from archs.unet1d import UNet1D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N = 1024
CHANNELS = (16, 32, 64, 128)

# Training ranges. The demo sliders go past these so out-of-range behavior is visible.
RANGES = dict(
    k=(2.0, 32.0),        # burst wavenumber (cycles per unit length)
    A=(0.0, 1.5),         # burst amplitude
    r0=(0.0, 1.0),        # front position
    k0=(0.5, 3.0),        # background wavenumber
    A0=(0.0, 1.5),        # background amplitude
    M=(8, 256),           # number of samples, drawn log-uniformly
    noise=(0.0, 0.05),    # measurement noise std (zero for half the batch)
)


def make_fields(B, gen, device):
    """Random fields, shape (B, N), plus a dict of their parameters."""
    def U(lo, hi):
        return lo + (hi - lo) * torch.rand(B, 1, generator=gen, device=device)
    r = torch.arange(N, device=device, dtype=torch.float32)[None] / N
    p = {name: U(*RANGES[name]) for name in ('k', 'A', 'r0', 'k0', 'A0')}
    phi0, phi1 = U(0, 1), U(0, 1)
    eta = (p['A0'] * torch.sin(2 * math.pi * (p['k0'] * r + phi0))
           + (r >= p['r0']) * p['A'] * torch.sin(2 * math.pi * (p['k'] * r + phi1)))
    return eta, p


def make_masks(B, gen, device):
    """Random masks with M log-uniform in RANGES['M'].

    Half are uniform random positions, half are a jittered regular grid, so the
    model does not rely on one sampling pattern.
    """
    lo, hi = RANGES['M']
    M = torch.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * torch.rand(B, generator=gen, device=device))
    M = M.round().long().clamp(lo, hi)
    keys = torch.rand(B, N, generator=gen, device=device)
    ranks = keys.argsort(dim=1).argsort(dim=1)
    rand_mask = ranks < M[:, None]
    # jittered grid: one point per cell of width N / M
    cell = N / M.float()
    j = torch.arange(hi, device=device)[None].float()
    pos = ((j + torch.rand(B, hi, generator=gen, device=device)) * cell[:, None]).long().clamp(max=N - 1)
    valid = j < M[:, None]
    grid_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    grid_mask.scatter_(1, torch.where(valid, pos, torch.zeros_like(pos)), valid)
    # scatter of invalid entries wrote False at index 0; restore any real sample there
    grid_mask[:, 0] |= (valid & (pos == 0)).any(dim=1)
    use_grid = torch.rand(B, 1, generator=gen, device=device) < 0.5
    return torch.where(use_grid, grid_mask, rand_mask)


def linear_interp(vals, mask):
    """Batched linear interpolation of vals at mask==True, held constant past the ends."""
    B, n = vals.shape
    idx = torch.arange(n, device=vals.device).expand(B, n)
    prev = torch.where(mask, idx, torch.full_like(idx, -1)).cummax(dim=1).values
    nxt = torch.where(mask, idx, torch.full_like(idx, n)).flip(1).cummin(dim=1).values.flip(1)
    first = torch.where(mask, idx, torch.full_like(idx, n)).min(dim=1, keepdim=True).values
    last = prev[:, -1:]
    prev = torch.where(prev < 0, first.expand(B, n), prev)
    nxt = torch.where(nxt >= n, last.expand(B, n), nxt)
    vp, vn = vals.gather(1, prev), vals.gather(1, nxt)
    span = (nxt - prev).clamp(min=1).float()
    w = ((idx - prev).float() / span).clamp(0, 1)
    return vp + w * (vn - vp)


def make_input(obs, mask):
    """Channels: linear interpolation, mask, observed values (zero off-mask)."""
    m = mask.float()
    interp = linear_interp(obs, mask)
    return torch.stack([interp, m, obs * m], dim=1), interp


class DemoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet1D(in_channels=3, out_channels=1, channels=CHANNELS, kernel_size=5)

    def forward(self, x):
        return x[:, 0] + self.unet(x)[:, 0]


def batch(B, gen, device):
    eta, _ = make_fields(B, gen, device)
    mask = make_masks(B, gen, device)
    sigma = RANGES['noise'][1] * torch.rand(B, 1, generator=gen, device=device)
    sigma = sigma * (torch.rand(B, 1, generator=gen, device=device) < 0.5)
    obs = eta + sigma * torch.randn(B, N, generator=gen, device=device)
    x, interp = make_input(obs, mask)
    return x, eta, interp


def train(args, device):
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    model = DemoNet().to(device)
    print(f'params: {sum(p.numel() for p in model.parameters()):,}')
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    val_gen = torch.Generator(device=device).manual_seed(1234)
    xv, ev, iv = batch(1024, val_gen, device)
    hist = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        x, eta, _ = batch(args.batch, gen, device)
        loss = ((model(x) - eta) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                val = ((model(xv) - ev) ** 2).mean().item()
                base = ((iv - ev) ** 2).mean().item()
            model.train()
            hist.append(dict(step=step, train=loss.item(), val=val, val_linear=base))
            print(f'step {step:6d}  train {loss.item():.3e}  val {val:.3e}  (linear {base:.3e})  {time.time() - t0:.0f}s', flush=True)
    os.makedirs(args.out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out, 'model.pth'))
    with open(os.path.join(args.out, 'history.json'), 'w') as f:
        json.dump(dict(args=vars(args), ranges=RANGES, history=hist), f, indent=1)
    return model


def export(model, out_js, device):
    """Write weights (float16, base64) and a reference test case for the JS port."""
    model = model.to(device).eval()
    sd = model.unet.state_dict()
    names = list(sd.keys())
    arrays = [sd[k].detach().cpu().numpy().astype(np.float32) for k in names]
    blob = np.concatenate([a.ravel() for a in arrays]).astype(np.float16)
    # Test case: run the model on float16-rounded weights so JS should match to float32 noise.
    model16 = DemoNet().to(device).eval()
    model16.unet.load_state_dict({k: torch.from_numpy(a.astype(np.float16).astype(np.float32)) for k, a in zip(names, arrays)})
    gen = torch.Generator(device=device).manual_seed(7)
    x, eta, _ = batch(1, gen, device)
    with torch.no_grad():
        y = model16(x)
    payload = dict(
        N=N, channels=CHANNELS, ranges=RANGES,
        tensors=[dict(name=k, shape=list(a.shape)) for k, a in zip(names, arrays)],
        weights=base64.b64encode(blob.tobytes()).decode('ascii'),
        test=dict(x=x[0].cpu().numpy().round(6).tolist(), y=y[0].cpu().numpy().round(6).tolist()),
    )
    with open(out_js, 'w') as f:
        f.write('window.DEMO_MODEL = ')
        json.dump(payload, f, separators=(',', ':'))
        f.write(';\n')
    print(f'wrote {out_js} ({os.path.getsize(out_js) / 1e6:.2f} MB, {blob.size:,} weights)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='models/unet1d/demo_randomized')
    ap.add_argument('--export', action='store_true', help='skip training, export the saved model')
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.export:
        model = DemoNet()
        model.load_state_dict(torch.load(os.path.join(args.out, 'model.pth'), map_location='cpu'))
    else:
        model = train(args, device)
    export(model, os.path.join(HERE, 'model_weights.js'), device)

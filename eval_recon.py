"""Evaluate reconstructions on the held-out test field.

For each test-mask type (random, maxent) and sample budget M, draws one mask
per test snapshot (shared by every method) and runs linear / cubic
interpolation, a per-snapshot Fourier-feature MLP, and every trained U-Net
found under models/unet1d/*_<field tag>. Writes per-snapshot metrics,
a summary table and all predictions to studies/heaviside_recon/<field tag>/.

    python eval_recon.py            # time-evolving field
    python eval_recon.py --static   # milestone 1
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

import recon_common as rc
from archs.unet1d import build_model

M_VALUES = [32, 64, 128, 256, 512]
TEST_MASKS = ['random', 'maxent']


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--static', action='store_true')
    p.add_argument('--stride', type=int, default=1, help='use every stride-th test timestep')
    p.add_argument('--save_stride', type=int, default=5, help='save predictions for every save_stride-th evaluated snapshot')
    p.add_argument('--mlp_steps', type=int, default=3000)
    p.add_argument('--table_m', type=int, default=256)
    return p.parse_args()


def load_unets(tag, device):
    models = {}
    for d in sorted(glob.glob(os.path.join('models', 'unet1d', f'*_{tag}'))):
        cfg = json.load(open(os.path.join(d, 'config.json')))
        model = build_model((3, rc.N)).to(device)
        model.load_state_dict(torch.load(os.path.join(d, 'model.pth'), map_location=device))
        model.eval()
        models[f"unet_{cfg['masks']}"] = model
    return models


def main():
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tag = 'static' if args.static else 'evolving'
    out = os.path.join('studies', 'heaviside_recon', tag)
    os.makedirs(out, exist_ok=True)

    r, eta, event = rc.make_field(rc.TEST_FIELD, static=args.static)
    ts = np.arange(0, rc.NUM_TIMESTEPS, args.stride)
    eta, event = eta[ts], event[ts]
    probs = np.load(os.path.join('data', f'recon_maxent_probs_{tag}.npz'))[f'field{rc.TEST_FIELD}'][ts]
    unets = load_unets(tag, device)
    print(f'test field {rc.TEST_FIELD}: {len(ts)} snapshots; U-Nets: {list(unets)}', flush=True)

    rt = torch.tensor(r, dtype=torch.float32, device=device)
    et = torch.tensor(eta, dtype=torch.float32, device=device)
    keep = np.arange(0, len(ts), args.save_stride)
    rows, preds = [], {'r': r, 'eta': eta[keep], 'event': event[keep], 'timesteps': ts[keep]}
    for mi, test_masks in enumerate(TEST_MASKS):
        for M in M_VALUES:
            g = torch.Generator(device=device).manual_seed(1000 * mi + M)
            p = torch.tensor(probs, dtype=torch.float32, device=device) if test_masks == 'maxent' else None
            mask_t = rc.sample_masks(M, p, batch=len(ts), generator=g, device=device)
            mask = mask_t.cpu().numpy()
            key = f'{test_masks}_M{M}'
            preds[f'mask_{key}'] = mask[keep].astype(bool)

            out_m = {
                'linear': np.stack([rc.interp_linear(r, eta[b], mask[b]) for b in range(len(ts))]),
                'cubic': np.stack([rc.interp_cubic(r, eta[b], mask[b]) for b in range(len(ts))]),
                'fourier_mlp': rc.fourier_mlp_fit(r, eta, mask, steps=args.mlp_steps, device=device),
            }
            with torch.no_grad():
                x = rc.make_input(et, mask_t, rt)
                for name, model in unets.items():
                    out_m[name] = model(x)[:, 0].cpu().numpy()

            for method, pred in out_m.items():
                preds[f'pred_{method}_{key}'] = pred[keep].astype(np.float32)
                met = rc.metrics(pred, eta, event)
                for b in range(len(ts)):
                    rows.append(dict(method=method, test_masks=test_masks, M=M, timestep=int(ts[b]),
                                     **{k: float(v[b]) for k, v in met.items()}))
            summary = '  '.join(f"{k}={rc.pool_metrics([x for x in rows if x['method'] == k and x['M'] == M and x['test_masks'] == test_masks])['mse']:.2e}"
                                for k in out_m)
            print(f'{key:12s} {summary}', flush=True)

    np.savez_compressed(os.path.join(out, 'predictions.npz'), **preds)
    with open(os.path.join(out, 'metrics.json'), 'w') as f:
        json.dump(rows, f)
    write_table(rows, out, args.table_m)


METHOD_LABELS = {
    'linear': 'Linear interp',
    'cubic': 'Cubic interp',
    'fourier_mlp': 'Fourier MLP',
    'unet_random': 'U-Net (trained on random masks)',
    'unet_maxent': 'U-Net (trained on MaxEnt masks)',
}
METRIC_COLS = [('mse', 'MSE overall'), ('mse_smooth', 'MSE r < r0'),
               ('mse_burst', 'MSE r >= r0'), ('spec_err', 'Spectral error near k1')]


def summarize(rows, n_boot=1000, seed=0):
    """Pooled metrics per (method, test_masks, M), with a 95% bootstrap
    interval from resampling snapshots (the same resamples for every group)."""
    groups = {}
    for x in rows:
        groups.setdefault((x['method'], x['test_masks'], x['M']), []).append(x)
    n = len(next(iter(groups.values())))
    boots = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    summ = {}
    for key, xs in groups.items():
        xs = sorted(xs, key=lambda x: x['timestep'])
        point = rc.pool_metrics(xs)
        samples = [rc.pool_metrics([xs[i] for i in b]) for b in boots]
        summ[key] = {k: (point[k], *np.percentile([s[k] for s in samples], [2.5, 97.5]))
                     for k, _ in METRIC_COLS}
    return summ


def fmt_ci(v, lo, hi):
    """1.22e-03 [1.10, 1.35]: value with its 95% interval at the value's exponent."""
    e = int(np.floor(np.log10(v)))
    return f'{v / 10**e:.2f}e{e:+03d} [{lo / 10**e:.2f}, {hi / 10**e:.2f}]'


def write_table(rows, out, table_m):
    summ = summarize(rows)
    methods = [m for m in METHOD_LABELS if any(k[0] == m for k in summ)]
    lines = ['| Method | Test masks | ' + ' | '.join(c for _, c in METRIC_COLS) + ' |',
             '| --- | --- | ' + ' | '.join('---' for _ in METRIC_COLS) + ' |']
    for method in methods:
        for tm in TEST_MASKS:
            s = summ[(method, tm, table_m)]
            lines.append(f'| {METHOD_LABELS[method]} | {"MaxEnt" if tm == "maxent" else "Random"} | '
                         + ' | '.join(fmt_ci(*s[k]) for k, _ in METRIC_COLS) + ' |')
    table = '\n'.join(lines)
    with open(os.path.join(out, f'table_M{table_m}.md'), 'w') as f:
        f.write(table + '\n')
    with open(os.path.join(out, 'summary.json'), 'w') as f:
        json.dump([dict(method=k[0], test_masks=k[1], M=k[2], **{m: v[0] for m, v in s.items()},
                        **{m + '_lo': v[1] for m, v in s.items()}, **{m + '_hi': v[2] for m, v in s.items()})
                   for k, s in summ.items()], f, indent=1)
    print(f'\nM = {table_m} (pooled over snapshots; 95% bootstrap interval in brackets)\n{table}')


if __name__ == '__main__':
    main()

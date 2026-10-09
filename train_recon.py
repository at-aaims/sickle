"""Train a 1D U-Net to reconstruct the Heaviside field from sparse samples.

Each training step draws fresh masks (random or MaxEnt-weighted) with a
per-snapshot sample count M drawn log-uniformly from [m_min, m_max], so one
model covers the whole sample-budget sweep.

    python train_recon.py --masks random
    python train_recon.py --masks maxent
    python train_recon.py --masks random --static   # milestone 1
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

import recon_common as rc
from archs.unet1d import build_model


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--masks', choices=['random', 'maxent'], default='random')
    p.add_argument('--static', action='store_true', help='static field (all speeds and amp_eps 0)')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--m_min', type=int, default=32)
    p.add_argument('--m_max', type=int, default=512)
    p.add_argument('--patience', type=int, default=30)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default=None, help='output dir (default models/unet1d/<tag>)')
    return p.parse_args()


def field_tag(static):
    return 'static' if static else 'evolving'


def load_split(static, device):
    """Train / val snapshots (eta, event) as tensors, plus r and MaxEnt probs."""
    probs_path = os.path.join('data', f'recon_maxent_probs_{field_tag(static)}.npz')
    cache = dict(np.load(probs_path)) if os.path.exists(probs_path) else {}
    n_val = int(rc.NUM_TIMESTEPS * rc.VAL_FRAC)
    parts = {'train': [], 'val': []}
    for f in rc.TRAIN_FIELDS + [rc.TEST_FIELD]:
        r, eta, event = rc.make_field(f, static=static)
        key = f'field{f}'
        if key not in cache:
            print(f'computing MaxEnt probabilities for field {f}', flush=True)
            cache[key] = rc.maxent_probs(eta)
        if f == rc.TEST_FIELD:
            continue
        probs = cache[key]
        parts['train'].append((eta[:-n_val], event[:-n_val], probs[:-n_val]))
        parts['val'].append((eta[-n_val:], event[-n_val:], probs[-n_val:]))
    os.makedirs('data', exist_ok=True)
    np.savez(probs_path, **cache)

    def cat(items):
        return [torch.tensor(np.concatenate(x), dtype=torch.float32, device=device) for x in zip(*items)]

    rt = torch.tensor(r, dtype=torch.float32, device=device)
    return rt, cat(parts['train']), cat(parts['val'])


def sample_m(n, m_min, m_max, gen, device):
    u = torch.rand(n, generator=gen, device=device)
    return torch.exp(math.log(m_min) + u * (math.log(m_max) - math.log(m_min))).round().long()


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tag = f'{args.masks}_{field_tag(args.static)}'
    out = args.out or os.path.join('models', 'unet1d', tag)
    os.makedirs(out, exist_ok=True)

    r, (eta_tr, ev_tr, p_tr), (eta_va, ev_va, p_va) = load_split(args.static, device)
    print(f'train {tuple(eta_tr.shape)}  val {tuple(eta_va.shape)}  masks={args.masks}', flush=True)

    # Fixed validation masks so the val loss is comparable across epochs.
    gval = torch.Generator(device=device).manual_seed(1234)
    m_va = sample_m(len(eta_va), args.m_min, args.m_max, gval, device)
    mask_va = rc.sample_masks(m_va, p_va if args.masks == 'maxent' else None,
                              batch=len(eta_va), generator=gval, device=device)
    x_va = rc.make_input(eta_va, mask_va, r)

    model = build_model((3, rc.N)).to(device)
    print(f'parameters: {sum(p.numel() for p in model.parameters()):,}', flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.MSELoss()
    gtr = torch.Generator(device=device).manual_seed(args.seed)

    history, best, bad = [], float('inf'), 0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(eta_tr), generator=gtr, device=device)
        tr_loss = 0.0
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            eta = eta_tr[idx]
            m = sample_m(len(idx), args.m_min, args.m_max, gtr, device)
            mask = rc.sample_masks(m, p_tr[idx] if args.masks == 'maxent' else None,
                                   batch=len(idx), generator=gtr, device=device)
            pred = model(rc.make_input(eta, mask, r))[:, 0]
            loss = crit(pred, eta)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(idx)
        sched.step()
        tr_loss /= len(eta_tr)

        model.eval()
        with torch.no_grad():
            va_loss = crit(model(x_va)[:, 0], eta_va).item()
        history.append({'epoch': epoch, 'train': tr_loss, 'val': va_loss})
        if va_loss < best:
            best, bad = va_loss, 0
            torch.save(model.state_dict(), os.path.join(out, 'model.pth'))
        else:
            bad += 1
        if epoch % 10 == 0 or bad == 0 and epoch > args.epochs - 10:
            print(f'epoch {epoch:4d}  train {tr_loss:.3e}  val {va_loss:.3e}  best {best:.3e}  '
                  f'{time.time() - t0:.0f}s', flush=True)
        if bad >= args.patience:
            print(f'early stop at epoch {epoch}', flush=True)
            break

    # No normalization is applied (eta is O(1)); recorded so eval can check.
    meta = dict(vars(args), tag=tag, best_val=best, epochs_run=len(history),
                normalization=None, in_channels=['eta*mask', 'mask', 'r'],
                fields=rc.FIELDS, field_defaults=rc.FIELD_DEFAULTS,
                train_fields=rc.TRAIN_FIELDS, test_field=rc.TEST_FIELD,
                num_timesteps=rc.NUM_TIMESTEPS, val_frac=rc.VAL_FRAC)
    with open(os.path.join(out, 'config.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(out, 'history.json'), 'w') as f:
        json.dump(history, f)
    print(f'saved {out}/model.pth  best val {best:.3e}', flush=True)


if __name__ == '__main__':
    main()

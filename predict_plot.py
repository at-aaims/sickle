"""Figures for the Heaviside reconstruction study (reads eval_recon.py output).

    python predict_plot.py [--static] [--timestep 60] [--M 64 256]

Writes to plots/heaviside_recon/<field tag>/:
  recon_t<T>_M<M>.png   truth, samples and U-Net prediction, full domain + burst zoom,
                        random vs MaxEnt test masks side by side
  sweep_mse_burst.png   MSE on r >= r0 against M, per method and mask type
  training_curves.png   train / val loss of each U-Net
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Categorical slots from the dataviz reference palette, in fixed order.
SERIES = {
    'unet_random': ('#2a78d6', 'U-Net (trained on random masks)'),
    'unet_maxent': ('#eb6834', 'U-Net (trained on MaxEnt masks)'),
    'cubic': ('#1baf7a', 'Cubic interp'),
    'linear': ('#eda100', 'Linear interp'),
    'fourier_mlp': ('#e87ba4', 'Fourier MLP'),
}
INK, MUTED, TRUTH, GRID = '#0b0b0b', '#52514e', '#9a9893', '#e4e2dd'
MASK_LABEL = {'random': 'Random test masks', 'maxent': 'MaxEnt test masks'}

plt.rcParams.update({
    'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
    'savefig.dpi': 160, 'savefig.bbox': 'tight',
})


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--static', action='store_true')
    p.add_argument('--timestep', type=int, default=None, help='test timestep (default: one with a mid-size burst)')
    p.add_argument('--M', type=int, nargs='+', default=[64, 256])
    p.add_argument('--model', default='unet_random', help='which U-Net to draw in the reconstruction figure')
    return p.parse_args()


def front_location(r, event_row):
    on = np.flatnonzero(event_row > 0.5)
    return r[on[0]] if len(on) else None


def plot_reconstruction(P, b, M, model, path):
    r, eta, event = P['r'], P['eta'][b], P['event'][b]
    r0 = front_location(r, event)
    color, label = SERIES[model]
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.8), sharey='row',
                             gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.35, 'wspace': 0.06})
    for col, masks in enumerate(['random', 'maxent']):
        mask = P[f'mask_{masks}_M{M}'][b]
        pred = P[f'pred_{model}_{masks}_M{M}'][b]
        burst = event > 0.5
        mse_b = np.mean((pred[burst] - eta[burst]) ** 2) if burst.any() else np.nan
        for row in range(2):
            ax = axes[row, col]
            ax.plot(r, eta, color=TRUTH, lw=1.0, label='Ground truth', zorder=1)
            ax.plot(r, pred, color=color, lw=1.6, label=f'{label} prediction', zorder=2)
            ax.plot(r[mask], eta[mask], 'o', ms=4, mfc=INK, mec='white', mew=0.6,
                    label=f'Samples (M = {M})', zorder=3)
            if r0 is not None:
                ax.axvline(r0, color=MUTED, ls='--', lw=1.0, zorder=0)
            if row == 0:
                ax.set_title(f'{MASK_LABEL[masks]}:  MSE on r ≥ r0 = {mse_b:.2e}', fontsize=10, color=INK, loc='left')
                ax.set_xlim(0, 1)
                if r0 is not None:
                    ax.text(r0, 0.04, ' r0(t)', transform=ax.get_xaxis_transform(), color=MUTED, fontsize=9)
            else:
                lo = max(0.0, (r0 if r0 is not None else 0.9) - 0.1)
                ax.set_xlim(lo, min(1.0, lo + 0.25))
                ax.set_xlabel('r')
                ax.set_title('Zoom on the burst region', fontsize=9, color=MUTED, loc='left')
        axes[0, col].tick_params(labelbottom=True)
    axes[0, 0].set_ylabel('eta')
    axes[1, 0].set_ylabel('eta')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.12, 0.965), ncol=3, fontsize=9)
    fig.suptitle(f'Reconstruction of the held-out test field, timestep {int(P["timesteps"][b])}, M = {M}',
                 x=0.125, y=1.0, ha='left', fontsize=12, color=INK)
    fig.savefig(path)
    plt.close(fig)


def plot_sweep(summary, path, n_snap, metric='mse_burst', ylabel='MSE on r ≥ r0 (burst region)'):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, gridspec_kw={'wspace': 0.06})
    methods = [m for m in SERIES if any(s['method'] == m for s in summary)]
    for ax, masks in zip(axes, ['random', 'maxent']):
        for m in methods:
            rows = sorted((s for s in summary if s['method'] == m and s['test_masks'] == masks), key=lambda s: s['M'])
            Ms = [s['M'] for s in rows]
            ax.fill_between(Ms, [s[metric + '_lo'] for s in rows], [s[metric + '_hi'] for s in rows],
                            color=SERIES[m][0], alpha=0.15, lw=0)
            ax.plot(Ms, [s[metric] for s in rows], '-o', color=SERIES[m][0], lw=2, ms=5, label=SERIES[m][1])
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xticks(Ms, [str(x) for x in Ms])
        ax.minorticks_off()
        ax.set_xlabel('Samples per snapshot, M (of N = 4096)')
        ax.set_title(MASK_LABEL[masks], loc='left', fontsize=10, color=INK)
    axes[0].set_ylabel(ylabel)
    axes[1].legend(loc='upper right', fontsize=9)
    fig.suptitle(f'Reconstruction error against sample budget, held-out test field (pooled over {n_snap} snapshots, 95% bootstrap band)',
                 x=0.125, ha='left', fontsize=12, color=INK)
    fig.savefig(path)
    plt.close(fig)


def plot_training(tag, path):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for m in ['unet_random', 'unet_maxent']:
        hist_path = os.path.join('models', 'unet1d', f'{m[5:]}_{tag}', 'history.json')
        if not os.path.exists(hist_path):
            continue
        h = json.load(open(hist_path))
        ep = [x['epoch'] for x in h]
        ax.plot(ep, [x['val'] for x in h], color=SERIES[m][0], lw=2, label=f'{SERIES[m][1]}, val')
        ax.plot(ep, [x['train'] for x in h], color=SERIES[m][0], lw=1, alpha=0.5, label=f'{SERIES[m][1]}, train')
    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    ax.set_title(f'U-Net training curves ({tag} field)', loc='left', fontsize=11, color=INK)
    ax.legend(fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def main():
    args = get_args()
    tag = 'static' if args.static else 'evolving'
    src = os.path.join('studies', 'heaviside_recon', tag)
    out = os.path.join('plots', 'heaviside_recon', tag)
    os.makedirs(out, exist_ok=True)
    P = dict(np.load(os.path.join(src, 'predictions.npz')))
    summary = json.load(open(os.path.join(src, 'summary.json')))

    ts = P['timesteps']
    if args.timestep is None:
        frac = P['event'].mean(1)
        b = int(np.argmin(np.abs(frac - 0.3)))  # a snapshot with a mid-size burst
    else:
        b = int(np.argmin(np.abs(ts - args.timestep)))
    model = args.model if f'pred_{args.model}_random_M{args.M[0]}' in P else 'unet_random'
    for M in args.M:
        path = os.path.join(out, f'recon_t{int(ts[b]):03d}_M{M}.png')
        plot_reconstruction(P, b, M, model, path)
        print('wrote', path)
    rows = json.load(open(os.path.join(src, 'metrics.json')))
    n_snap = len({x['timestep'] for x in rows})
    plot_sweep(summary, os.path.join(out, 'sweep_mse_burst.png'), n_snap)
    plot_sweep(summary, os.path.join(out, 'sweep_mse.png'), n_snap, metric='mse', ylabel='MSE over all points')
    plot_training(tag, os.path.join(out, 'training_curves.png'))
    print('wrote sweep and training figures to', out)


if __name__ == '__main__':
    main()

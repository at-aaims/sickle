# Heaviside field reconstruction: results (Claude's run)

Run 2026-10-09, following the training plan
(https://claude.ai/code/artifact/5443796e-8b05-4fee-8ea7-04d360de16bb).
All numbers are on the held-out test field (field 4, 200 snapshots), pooled over
snapshots, with 95% bootstrap intervals (resampling snapshots) in brackets.

## Headline findings

1. **The U-Net beats every baseline for M <= 256, by a wide margin when samples
   are scarce.** At M = 64 its overall MSE is 7.3e-3 against 4.0e-2 for the best
   baseline (linear interpolation), about 5x lower; in the burst region
   (r >= r0) it is 2.6e-2 against 1.6e-1 to 2.4e-1, 6x to 9x lower.
   At M = 256 the gap narrows to about 1.4x overall (1.0e-3 vs 1.45e-3 for cubic).
2. **At M = 512 the U-Net loses** (6.8e-4 vs 4.1e-4 for linear). It has an
   error floor around 6e-4: it does not reproduce the sampled values exactly, so
   when samples are dense, plain interpolation wins. Training was still slowly
   improving at epoch 200 (see `training_curves.png`).
3. **MaxEnt and random sampling are indistinguishable**, for every method and
   every M; all differences sit inside the bootstrap intervals. The U-Nets
   trained on random and on MaxEnt masks are also indistinguishable.
   This is expected, because on this field sickle's MaxEnt sampler reduces to
   uniform sampling (next section). It is a property of the sampler on 1D
   data, not a result about MaxEnt in general.

## Why MaxEnt is uniform here

`subsampling/maxent.py` runs k-means on `eta` (`cluster_var: [eta]`), builds a
50-bin histogram per cluster and scores clusters by summed pairwise KL
divergence. With one scalar variable, k-means splits `eta` into
non-overlapping value ranges, so cluster histograms share no bins and every
pairwise KL is set by the 1e-10 floor (about 21). In-strengths come out within
+/-2% of each other; per-point probabilities have max/min of about 1.00 on the
evolving field (1.07 on the static field). MaxEnt puts about 65% of its
probability mass on the burst when the burst covers about 65% of the domain.

Any fix would change the sampler (for example a second cluster variable such as
a local gradient, a different divergence, or restoring a rescaling of the
in-strengths), so it is out of scope for this run.

## Results, time-evolving field

M = 64:

| Method | Test masks | MSE overall | MSE r < r0 | MSE r >= r0 | Spectral error near k1 |
| --- | --- | --- | --- | --- | --- |
| Linear interp | Random | 3.95e-02 [3.41, 4.46] | 1.72e-03 [1.23, 2.25] | 2.03e-01 [1.77, 2.30] | 5.22e-01 [4.98, 5.45] |
| Linear interp | MaxEnt | 4.00e-02 [3.39, 4.61] | 1.35e-03 [1.02, 1.70] | 2.08e-01 [1.80, 2.38] | 5.27e-01 [5.00, 5.54] |
| Cubic interp | Random | 5.01e-02 [3.67, 6.88] | 6.41e-03 [4.04, 9.34] | 2.40e-01 [1.69, 3.35] | 4.77e-01 [4.21, 5.42] |
| Cubic interp | MaxEnt | 5.34e-02 [3.76, 7.41] | 3.76e-03 [2.43, 5.30] | 2.69e-01 [1.90, 3.80] | 5.03e-01 [4.43, 5.70] |
| Fourier MLP | Random | 5.00e-02 [4.38, 5.66] | 2.38e-02 [2.06, 2.74] | 1.64e-01 [1.37, 1.94] | 4.70e-01 [4.42, 4.99] |
| Fourier MLP | MaxEnt | 4.80e-02 [4.23, 5.44] | 2.01e-02 [1.70, 2.36] | 1.70e-01 [1.42, 1.98] | 4.79e-01 [4.46, 5.13] |
| U-Net (trained on random masks) | Random | 7.32e-03 [6.40, 8.28] | 3.03e-03 [2.48, 3.62] | 2.60e-02 [2.13, 3.09] | 1.83e-01 [1.67, 2.00] |
| U-Net (trained on random masks) | MaxEnt | 7.92e-03 [6.82, 9.02] | 2.66e-03 [2.21, 3.13] | 3.08e-02 [2.52, 3.62] | 1.96e-01 [1.78, 2.12] |
| U-Net (trained on MaxEnt masks) | Random | 6.88e-03 [6.01, 7.79] | 2.84e-03 [2.33, 3.46] | 2.44e-02 [1.98, 2.91] | 1.79e-01 [1.62, 1.96] |
| U-Net (trained on MaxEnt masks) | MaxEnt | 7.35e-03 [6.29, 8.44] | 2.46e-03 [2.06, 2.89] | 2.86e-02 [2.34, 3.40] | 1.84e-01 [1.67, 2.00] |

M = 256 (the plan's table):

| Method | Test masks | MSE overall | MSE r < r0 | MSE r >= r0 | Spectral error near k1 |
| --- | --- | --- | --- | --- | --- |
| Linear interp | Random | 2.08e-03 [1.67, 2.53] | 3.22e-04 [2.29, 4.21] | 9.74e-03 [7.46, 12.16] | 9.82e-02 [8.66, 11.07] |
| Linear interp | MaxEnt | 1.76e-03 [1.52, 1.99] | 2.64e-04 [1.91, 3.50] | 8.24e-03 [6.99, 9.68] | 8.77e-02 [8.08, 9.48] |
| Cubic interp | Random | 1.45e-03 [1.09, 1.83] | 5.99e-04 [3.66, 8.95] | 5.14e-03 [3.58, 6.78] | 5.95e-02 [4.85, 7.00] |
| Cubic interp | MaxEnt | 1.53e-03 [0.84, 2.75] | 4.50e-04 [2.97, 6.29] | 6.20e-03 [2.67, 12.92] | 7.53e-02 [3.74, 12.41] |
| Fourier MLP | Random | 1.57e-03 [1.19, 2.03] | 7.19e-04 [5.77, 8.73] | 5.25e-03 [3.30, 7.62] | 6.16e-02 [5.06, 7.35] |
| Fourier MLP | MaxEnt | 2.41e-03 [1.11, 4.74] | 6.35e-04 [5.24, 7.70] | 1.01e-02 [0.31, 2.30] | 9.13e-02 [4.62, 14.39] |
| U-Net (trained on random masks) | Random | 1.02e-03 [0.85, 1.21] | 5.66e-04 [4.75, 6.68] | 2.99e-03 [2.12, 4.10] | 4.31e-02 [3.74, 4.96] |
| U-Net (trained on random masks) | MaxEnt | 9.96e-04 [8.57, 11.43] | 5.21e-04 [4.35, 6.23] | 3.06e-03 [2.30, 3.95] | 3.99e-02 [3.49, 4.56] |
| U-Net (trained on MaxEnt masks) | Random | 1.00e-03 [0.84, 1.21] | 5.48e-04 [4.61, 6.43] | 2.97e-03 [2.09, 4.17] | 4.31e-02 [3.67, 4.99] |
| U-Net (trained on MaxEnt masks) | MaxEnt | 9.82e-04 [8.42, 11.25] | 5.17e-04 [4.30, 6.16] | 3.00e-03 [2.30, 3.79] | 4.03e-02 [3.58, 4.51] |

Intervals are written at the value's exponent: `9.96e-04 [8.57, 11.43]` means
[8.57e-4, 1.143e-3]. All M values are in `evolving/summary.json`.

## Milestones

- [x] M1: static field, random masks. The U-Net beats cubic interpolation at every
      M up to 256 (overall MSE 1.4e-3 vs 3.2e-3 at M = 256; 5.7e-3 vs 9.6e-2 at M = 64).
- [x] M2: time-evolving field, reported on the held-out field.
- [x] M3: MaxEnt masks and the random-vs-MaxEnt comparison (no difference; see above).
- [x] M4: M sweep, tables and figures. The optional animation was not made.

## Setup, and where it departs from the plan

- **Fields.** Time evolution from `config/synthetic/*.yaml` (step_dt 0.05,
  wave_speed 0.2, hf_wave_speed 0.5, front_speed 0.05, amp_eps 0.4, amp_tau 0.3),
  200 timesteps. *Departure:* the loader's `field_seed` only drives the
  amplitude process, so with seeds alone every field shares one front
  trajectory and the "held-out" seed would not be held out. Each field
  therefore also gets its own r0: fields 0 to 3 use 0.35, 0.50, 0.65 and 0.80
  for training; test field 4 uses 0.575.
- **Splits.** Train: fields 0 to 3, first 160 timesteps. Val: their last 40.
  Test: all 200 timesteps of field 4.
- **Masks.** Fresh masks every step. MaxEnt probabilities come from sickle's own
  `MaxentSubsampler.perform_kmeans` / `compute_entropy`, cached per snapshot in
  `data/recon_maxent_probs_*.npz`; masks are drawn without replacement with
  Gumbel top-k, which is equivalent to the sampler's
  `np.random.choice(replace=False, p=...)`. Test masks are shared by all methods.
- **Sample budget.** *Departure:* one model covers all budgets, with M drawn
  log-uniformly from [32, 512] per training snapshot. M = 32 was added to the
  sweep; with k1 = 16 the burst wavelength is 1/16, so M = 256 already gives
  about 16 samples per wavelength, and the hard regime starts below M = 128.
- **U-Net** (`archs/unet1d.py`). Input channels: eta * mask, mask, r. Levels
  4096 / 2048 / 1024 / 512 with 32 / 64 / 128 / 256 channels and a 256-point
  bottleneck, kernel size 5, GroupNorm + GELU, 2.84M parameters. Adam at 1e-3 with
  cosine decay, 200 epochs, batch 32, early-stopping patience 30 (never
  triggered on the evolving field). Plain MSE loss; no burst weighting.
  No normalization (eta is O(1)). About 6.5 minutes per model on an RTX A6000.
- **Baselines.** Linear (`np.interp`); cubic spline with the end values held
  constant outside the sampled range (unclamped extrapolation blew up to
  MSE ~1e2 at low M); per-snapshot Fourier-feature MLP (128 random frequencies,
  sigma = 4 cycles, chosen from {1, 2, 3, 4, 8, 16, 32} on training field 1;
  2 x 128 ReLU, 3000 Adam steps).
- **Metrics.** MSEs pooled over snapshots (total squared error / total points
  in each region), because per-snapshot ratios are unstable when the front
  sits near r = 1 and the burst region is a few points wide. Spectral error:
  relative L2 error of |FFT| over bins 8 to 24, pooled the same way (per
  snapshot it divides by ~0 whenever the amplitude A(t) hits 0).

## Next steps worth trying

- Enforce data consistency (copy the samples into the output, or predict a
  residual on top of linear interpolation). That should remove the M = 512
  floor.
- Train longer; the validation loss had not flattened at 200 epochs.
- Fix the 1D degeneracy in MaxEnt before drawing any conclusion about MaxEnt
  vs random on this field.

## Reproduce

```bash
source ~/turbulence/.venv-sickle/bin/activate    # torch 2.14 + CUDA, numpy, scipy, sklearn, mpi4py
export MPLBACKEND=Agg
python train_recon.py --masks random --static    # M1
python eval_recon.py --static && python predict_plot.py --static
python train_recon.py --masks random
python train_recon.py --masks maxent
python eval_recon.py && python predict_plot.py   # about 10 minutes, mostly the MLP fits
```

Files: `recon_common.py`, `archs/unet1d.py`, `train_recon.py`, `eval_recon.py`,
`predict_plot.py`; checkpoints and configs in `models/unet1d/`; metrics,
tables and predictions in `studies/heaviside_recon/`; figures in
`plots/heaviside_recon/`. `subsample.py` now also saves `indices` in its npz.

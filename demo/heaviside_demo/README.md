# Heaviside reconstruction demo

An interactive page: ground truth, samples and a U-Net reconstruction, with sliders for
the burst wavenumber k, amplitude A, front position r0, background wave, number of
samples M, sampling pattern (uniform random, MaxEnt, jittered grid, regular grid,
random with a gap) and noise. The U-Net runs in the browser (plain JS, in a
Web Worker); no server is needed. Open `heaviside_demo.html`.

Unlike `train_recon.py`, which trains on five fixed field configurations, every training
example here draws its own parameters, so the model reconstructs a family of signals.
Training ranges: k 2 to 32, A 0 to 1.5, r0 anywhere, k0 0.5 to 3, A0 0 to 1.5, M 8 to 256
(log-uniform, uniform-random or jittered-grid positions), noise sigma up to 0.05 on half
the fields. The page shows the training ranges on its sliders and flags settings outside them.

Model: `archs/unet1d.py` with channels 16/32/64/128, N = 1024, 711k parameters. Input
channels are the linear interpolation of the samples, the mask and the sample values; the
output is a correction added to the linear interpolation. 30k AdamW steps at batch 128,
about 20 minutes on an RTX A6000. Final validation MSE 4.2e-2 vs 1.5e-1 for linear
interpolation, averaged over the training ranges.

Gain over linear interpolation (MSE ratio, mean over 20 random masks) on k-sweeps with
A = 0.8, r0 = 0.5:

| k | M = 16 | M = 32 | M = 64 | M = 128 | M = 256 |
| --- | --- | --- | --- | --- | --- |
| 8 | 2.3x | 8.3x | 7.4x | 2.7x | 1.0x |
| 16 | 3.0x | 14.5x | 32.1x | 14.8x | 3.3x |
| 32 | 1.3x | 6.1x | 38.1x | 48.6x | 16.3x |
| 48 (outside training) | 1.0x | 0.7x | 0.6x | 0.7x | 0.8x |

At k = 48 the model does worse than linear interpolation: it has not learned to extrapolate
in wavenumber.

## Files

- `train_demo.py`: trains the model (`models/unet1d/demo_randomized/`, not committed) and
  exports `model_weights.js` (float16 weights plus a reference input/output pair).
- `unet.js`: the JS forward pass.
- `maxent.js`: JS port of sickle's MaxEnt sampler (`subsampling/maxent.py`), run on the
  ground-truth field. sklearn's k-means++ is replaced by Lloyd's algorithm from quantile
  centers; on test fields the per-point probabilities match sickle's within an L1 distance
  of 0.015. As in the study, they come out close to uniform (max/min about 1.02 to 1.07),
  so MaxEnt masks look like random masks to the model. The page plots the probabilities.
- `check_js.js`: compares `unet.js` against the PyTorch reference (`node check_js.js`).
- `index.html`: page template; `build_page.py` inlines the weights, `unet.js` and `maxent.js` into
  `heaviside_demo.html`.

```bash
source ~/turbulence/.venv-sickle/bin/activate
python demo/heaviside_demo/train_demo.py     # train + export (or --export to re-export)
node demo/heaviside_demo/check_js.js
python demo/heaviside_demo/build_page.py
```

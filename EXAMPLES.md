# EXAMPLES

This document provides detailed examples for using SICKLE in various scenarios, including local testing, Frontier runs, parallel execution, and training.

## Table of Contents

- [Subsampling Examples](#subsampling-examples)
  - [Local Testing with Random and Maxent Methods](#local-testing-with-random-and-maxent-methods)
  - [Frontier Testing](#frontier-testing)
- [Training Examples](#training-examples)
- [Parallel Execution Examples](#parallel-execution-examples)
- [Visualizing Subsampled Hypercubes and Points](#visualizing-subsampled-hypercubes-and-points)
- [Flow Over Cylinder Example](#flow-over-cylinder-example)
- [Comparing Methods](#comparing-methods)

## Subsampling Examples

### Local Testing with Random and Maxent Methods

- **Random Subsampling on a Laptop:**

  ```bash
  python subsample.py config/laptop/default.yaml -m random --target drag -ns 540
  ```

- **Maxent Subsampling on a Laptop:**

  ```bash
  python subsample.py config/laptop/default.yaml -m maxent --target drag -ns 540 -cv p
  ```

### Frontier Testing

- **Maxent Subsampling on Frontier:**

  Use the YAML configuration file to provide defaults, with optional command-line switches:
  
  ```bash
  python subsample.py config/OF/default.yaml --plot
  ```

- **Full Subsampling on Frontier:**

  ```bash
  python subsample.py config/OF/default.yaml -m full --plot -ns 100
  ```

## Training Examples

- **Basic Training Using a YAML File:**

  ```bash
  python -u train.py config/OF/default.yaml
  ```

- **Overriding Specific Parameters (e.g., Epochs):**

  ```bash
  python -u train.py config/OF/default.yaml --epochs 3000
  ```

## Parallel Execution Examples

### Using MPI for Parallel Subsampling

- **OpenFOAM Dataset with Random and Maxent Methods:**

  ```bash
  srun -n 4 python -u subsample-mpi.py config/OF/default.yaml -m random --target drag -ns 540
  srun -n 4 python -u subsample-mpi.py config/OF/default.yaml -m maxent --target drag -ns 540
  ```

- **Taylor Green Dataset Example (with Specific Timesteps):**

  ```bash
  OPENBLAS_NUM_THREADS=4 srun -n 4 python -u subsample-mpi.py config/OF/default.yaml -m maxent --plot -ns 100 --input_vars u v w r --output_vars p --cluster_var pv --nx 514 --ny 512 --nz 256 --gravity z --nxsl 128 --nysl 128 --nzsl 64
  ```

## Visualizing Subsampled Hypercubes and Points

For 3D `dtype: sst-binary` configs (e.g. `config/SST/P1/test.yaml`), pass `--plot` to get
diagnostic PNGs per timestep in `<plot_dir>` (default `./plots`). This works for any
`num_hypercubes`, including `num_hypercubes: 1`:

```bash
python subsample.py config/SST/P1/test.yaml --plot
```

- `kmeans_<ts>.png` -- every candidate point, across all hypercubes, at its true position
  in the full domain, colored by k-means cluster
- `subsample_plot_t<ts>.png` -- just the maxent-selected points, same coloring
- `contour_<ts>_h<i>.png` -- filled contour of the cluster field within hypercube `i`
  (one file per hypercube)
- `histogram_<ts>.png` -- histogram of cluster-label counts
- `prob_dists_<ts>.png` -- probability distribution of the cluster variable: full dataset
  vs. random sampling vs. maxent sampling (shows maxent's bias toward the tails)
- `adj_matrix_<ts>.png` -- KL-divergence matrix between cluster distributions

**Alternative: a single combined plot of every hypercube's global position and its
sampled points**, via `scripts/visualize_subsample.py`. This reads the
`hypercube_ids_<ts>.npz` / `indices_<ts>.npy` side-files `subsample.py` already writes to
`output_dir`, so it requires the config's `timesteps:` list to be set explicitly (as
`test.yaml` does) and that timestep to have already been subsampled:

```bash
python subsample.py config/SST/P1/test.yaml
python scripts/visualize_subsample.py config/SST/P1/test.yaml --timestep 28.04
```

Output: `<plot_dir>/global_hypercubes_t<ts>.png` -- a wireframe box per hypercube plus its
sampled points, in one 3D view.

## Flow Over Cylinder Example

- **Subsampling with the Flow Over Cylinder Case:**

  ```bash
  python subsample.py config/OF/default.yaml -m maxent --target drag -ns 1080 -nc 20 -cv wz
  python subsample.py config/OF/default.yaml -m uips --target drag -ns 1080 -nc 20 --plot
  ```

## Comparing Methods

- **Comparing Subsampling Distributions:**

  Generate a histogram to compare different subsampling methods:
  
  ```bash
  python compare_methods.py config/OF/default.yaml --target drag -ns 1080 -nc 20 -cv wz
  ```

- **Additional Comparison Example:**

  ```bash
  python compare_methods.py config/OF/default.yaml --plot -ns 100
  ```

---

*Note:* Replace configuration file paths (e.g., `config/OF/default.yaml` or `config/laptop/default.yaml`) with the actual paths used in your environment.

#!/usr/bin/env python3
"""
Visualize where sickle's selected hypercubes and subsampled points actually
sit within the full simulation domain, for a single timestep.

This does NOT use plotting.py's plot_kmeans_3d / plot_samples / plot_contour_box_3d
path (triggered by `--plot` inside subsampling/maxent.py) -- that code assumes
exactly one hypercube (it colors points using the local coordinate grid of a
single hypercube) and throws a shape-mismatch error as soon as
`num_hypercubes > 1`. This script instead rebuilds true global positions from
the side-files subsample.py/dataloaders already write to --output_dir:

  - hypercube_ids_<ts>.npz  (written by dataloaders/sst-binary.py)
      hypercube_ids: (num_hypercubes, 3) array of (ix, iy, iz) block
      coordinates -- the hypercube's corner is at
      (ix*nxsl, iy*nysl, iz*nzsl) in full-grid index space.

  - indices_<ts>.npy        (written by subsample.py)
      flat indices, in [0, num_hypercubes*nxsl*nysl*nzsl), selecting which
      points (out of ALL hypercubes concatenated) were subsampled.

Both files are only written when the config sets `timesteps:` explicitly.

Usage:
    python scripts/visualize_subsample.py config/SST/P1/test.yaml --timestep 28.04
"""
import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("config", help="YAML config that was passed to subsample.py")
    cli.add_argument("--timestep", type=float, required=True, help="timestep to visualize, e.g. 28.04")
    cli.add_argument("--out", default=None, help="output PNG path (default: <plot_dir>/global_hypercubes_t<ts>.png)")
    cli_args = cli.parse_args()

    # Reuse the project's own config/CLI parser (args.py) exactly the way
    # subsample.py does, so nx/ny/nz/nxsl/nysl/nzsl/output_dir/plot_dir all
    # come from the same YAML the actual run used.
    sys.argv = [sys.argv[0], cli_args.config]
    from args import args

    ts = cli_args.timestep
    hc_path = os.path.join(args.output_dir, f"hypercube_ids_{ts:0.6f}.npz")
    idx_path = os.path.join(args.output_dir, f"indices_{ts:0.6f}.npy")

    if not os.path.exists(hc_path):
        sys.exit(f"Missing {hc_path}\n"
                  f"(hypercube_ids_<ts>.npz is written by the dataloader during subsample.py -- "
                  f"rerun it against this config first)")
    if not os.path.exists(idx_path):
        sys.exit(f"Missing {idx_path}\n"
                  f"(indices_<ts>.npy is only written when the config's `timesteps:` list is set explicitly "
                  f"and includes {ts})")

    hypercube_ids = np.load(hc_path)["hypercube_ids"]  # (num_hypercubes, 3) -> (ix, iy, iz)
    indices = np.load(idx_path)                         # (num_samples_total,)

    nxsl, nysl, nzsl = args.nxsl, args.nysl, args.nzsl
    num_pts = nxsl * nysl * nzsl
    nx_full, ny_full, nz_full = args.nx - 2, args.ny, args.nz  # args.nx carries a +2 padding convention

    hcube_of_point, local_flat = np.divmod(indices, num_pts)
    lx, ly, lz = np.unravel_index(local_flat, (nxsl, nysl, nzsl))
    ix = hypercube_ids[hcube_of_point, 0]
    iy = hypercube_ids[hcube_of_point, 1]
    iz = hypercube_ids[hcube_of_point, 2]
    gx = ix * nxsl + lx
    gy = iy * nysl + ly
    gz = iz * nzsl + lz

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("tab10")

    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for h, (bix, biy, biz) in enumerate(hypercube_ids):
        x0, x1 = bix * nxsl, (bix + 1) * nxsl
        y0, y1 = biy * nysl, (biy + 1) * nysl
        z0, z1 = biz * nzsl, (biz + 1) * nzsl
        corners = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                             [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
        color = cmap(h % 10)
        for a, b in edges:
            ax.plot(*zip(corners[a], corners[b]), color=color, linewidth=1.3)

    ax.scatter(gx, gy, gz, c=hcube_of_point, cmap="tab10", s=10, alpha=0.85, depthshade=False)

    ax.set_xlim(0, nx_full)
    ax.set_ylim(0, ny_full)
    ax.set_zlim(0, nz_full)
    ax.set_xlabel("x (grid idx)")
    ax.set_ylabel("y (grid idx)")
    ax.set_zlabel("z (grid idx)")
    ax.set_title(f"{len(hypercube_ids)} hypercubes ({nxsl}x{nysl}x{nzsl}) + "
                 f"{len(indices)} subsampled points, t={ts}")

    out = cli_args.out or os.path.join(args.plot_dir, f"global_hypercubes_t{ts:.6f}.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

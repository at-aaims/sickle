#!/usr/bin/env python3
"""
Turn a small, already-transferred raw sample of an SST-TG dataset (see
../SST_DATA_SAMPLE.md) into a single training-ready .npz.

Reads a strided sub-cube (nxsl x nysl x nzsl, skipping nxskip/nyskip/nzskip
points) out of each requested {var}_{timestep:.6f} binary file via memmap
(helpers.get_data_memmap), for each timestep and each of the four variables
[u, v, w, r], and stacks them into one array.

Example:
    python scripts/prepare_sst_sample.py \
        --path "$SAMPLE_DIR" \
        --timesteps 0.01 0.02 \
        --nx 514 --ny 512 --nz 256 \
        --nxsl 128 --nysl 128 --nzsl 128 \
        --nxskip 4 --nyskip 4 --nzskip 2 \
        --out "$PROJWORK/stf249/sst_turbulence/processed/P1F4R3200_sample.npz"
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers import get_1Dgrid, get_data_memmap  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path", required=True, help="directory containing the raw {var}_{timestep} binary files")
    p.add_argument("--out", required=True, help="output .npz path")
    p.add_argument("--timesteps", type=float, nargs="+", required=True,
                    help="timesteps to load, e.g. 0.01 0.02 (matched against {var}_{ts:.6f} filenames)")
    p.add_argument("--vars", nargs="+", default=["u", "v", "w", "r"],
                    help="variable name prefixes, in the channel order they'll be stacked (default: u v w r)")
    p.add_argument("--nx", type=int, required=True, help="full grid points in x (as stored in the raw file)")
    p.add_argument("--ny", type=int, required=True, help="full grid points in y (as stored in the raw file)")
    p.add_argument("--nz", type=int, required=True, help="full grid points in z (as stored in the raw file)")
    p.add_argument("--nxsl", type=int, default=128, help="sub-cube points in x")
    p.add_argument("--nysl", type=int, default=128, help="sub-cube points in y")
    p.add_argument("--nzsl", type=int, default=128, help="sub-cube points in z")
    p.add_argument("--nxskip", type=int, default=1, help="stride in x (1 = full resolution, no downsampling)")
    p.add_argument("--nyskip", type=int, default=1, help="stride in y")
    p.add_argument("--nzskip", type=int, default=1, help="stride in z")
    p.add_argument("--nxoffset", type=int, default=0, help="sub-cube corner offset in x")
    p.add_argument("--nyoffset", type=int, default=0, help="sub-cube corner offset in y")
    p.add_argument("--nzoffset", type=int, default=0, help="sub-cube corner offset in z")
    p.add_argument("--nbytes", type=int, default=4, help="bytes per value (4 = float32, matches the published format)")
    p.add_argument("--gravity", choices=["y", "z"], default="z", help="stratification axis, for grid coordinate generation")
    p.add_argument("--Lh", type=float, default=1.0, help="horizontal domain length, for grid coordinates")
    p.add_argument("--Lv", type=float, default=0.5, help="vertical domain length, for grid coordinates")
    return p.parse_args()


def main():
    args = parse_args()

    if args.gravity == "z":
        x = get_1Dgrid(args.Lh, args.nx, args.nxoffset, args.nxsl, args.nxskip)
        y = get_1Dgrid(args.Lh, args.ny, args.nyoffset, args.nysl, args.nyskip)
        z = get_1Dgrid(args.Lv, args.nz, args.nzoffset, args.nzsl, args.nzskip)
    else:
        x = get_1Dgrid(args.Lh, args.nx, args.nxoffset, args.nxsl, args.nxskip)
        y = get_1Dgrid(args.Lv, args.ny, args.nyoffset, args.nysl, args.nyskip)
        z = get_1Dgrid(args.Lh, args.nz, args.nzoffset, args.nzsl, args.nzskip)

    n_v = len(args.vars)
    loaded_rows = []      # list of (nxsl, nysl, nzsl, n_v) arrays, one per successfully loaded timestep
    loaded_timesteps = []  # kept strictly in lockstep with loaded_rows

    for ts in args.timesteps:
        row = np.zeros((args.nxsl, args.nysl, args.nzsl, n_v), dtype=np.float32)
        ok = True
        for vi, var in enumerate(args.vars):
            fname = f"{var}_{ts:0.6f}"
            fpath = os.path.join(args.path, fname)
            if not os.path.exists(fpath):
                print(f"[skip] missing file for timestep {ts}: {fpath}")
                ok = False
                break
            cube = get_data_memmap(
                fpath, args.nx, args.ny, args.nz,
                args.nxsl, args.nysl, args.nzsl,
                args.nxoffset, args.nyoffset, args.nzoffset,
                args.nxskip, args.nyskip, args.nzskip,
                args.nbytes,
            )
            row[:, :, :, vi] = cube
            print(f"[loaded] {fname}: sub-cube shape {cube.shape}")
        if ok:
            loaded_rows.append(row)
            loaded_timesteps.append(ts)

    if not loaded_rows:
        raise SystemExit("No timesteps loaded (all were missing/incomplete) -- nothing to save.")

    X = np.stack(loaded_rows, axis=0)  # [len(loaded_timesteps), nxsl, nysl, nzsl, n_v], never misaligned with timesteps

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out,
        X=X,
        x=x, y=y, z=z,
        timesteps=np.array(loaded_timesteps),
        variables=np.array(args.vars),
    )
    print(f"[saved] {args.out}  X.shape={X.shape}  channels={args.vars}")
    if len(loaded_timesteps) != len(args.timesteps):
        print(f"[warning] requested {len(args.timesteps)} timesteps, only {len(loaded_timesteps)} loaded successfully")


if __name__ == "__main__":
    main()

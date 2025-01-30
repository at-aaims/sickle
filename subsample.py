import numpy as np
import os
import sys

from args import args 
from helpers import check_and_create_dirs, load_data
from algorithms import create_maxent_subsampler, subsample_random, subsample_uips, build_pdf
from constants import FieldPredictionType
from plotting import plot_samples, plot2d_contour, plot_corner


def extract_yz_plane(X, timestep, feature_index, x_index, nx=128, ny=64, nz=128):
    """Extracts the y-z plane for a given x-index at a specific timestep and feature."""
    data_slice = X[timestep, :, feature_index]
    data_3d = data_slice.reshape(nx, ny, nz)
    return data_3d[x_index, :, :]


def subsample_data(X, Y, x, y, z, subsample_fn, args):
    num_timesteps = X.shape[0] // args.window * args.window + 1
    print(f"num_timesteps: {num_timesteps}")

    Xout = np.zeros((num_timesteps, args.num_samples, X.shape[2]))

    if args.field_prediction_type == FieldPredictionType.GLOBAL: # global quantity prediction
        Yout = np.zeros((num_timesteps, 1, Y.shape[2]))
    elif args.field_prediction_type == FieldPredictionType.LOCAL:  # local field prediction
        if args.method == "full": 
            raise Exception("For baseline full field input, prediction cannot be subsampled. Change `args.target`.")
        Yout = np.zeros((num_timesteps, args.num_samples, Y.shape[2]))
    elif args.field_prediction_type == FieldPredictionType.FULL:  # full field prediction
        Yout = np.zeros((num_timesteps, Y.shape[1], Y.shape[2]))
    else:
        raise Exception("Enter a valid `args.target`.")

    for timestep in range(0, num_timesteps - args.window, args.window):
        indices = subsample_fn(X, args.num_samples, timestep)

        if args.plot and args.method != "full":
            plot_samples(indices, x, y, z, timestep, args)

        for sub_timestep in range(args.window):
            ts = timestep + sub_timestep
            Xout[ts, :, :] = X[ts, indices, :]

            if args.field_prediction_type == FieldPredictionType.GLOBAL:
                subsampled_Y = Y[ts, :]
            elif args.field_prediction_type == FieldPredictionType.FULL:
                subsampled_Y = Y[ts, :, :]
            else:
                subsampled_Y = Y[ts, indices, :]
            Yout[ts, :] = subsampled_Y

            if args.plot: # plot 2D slice
                if args.method == "full":
                    yz_plane = extract_yz_plane(Xout, timestep, 3, 0, nx=args.nxsl, ny=args.nysl, nz=args.nzsl)
                    plot2d_contour(yz_plane, y, z, ts)
    
    return Xout, Yout

if __name__ == "__main__":

    # Ensure required directories exist
    check_and_create_dirs(args.output_dir)
    check_and_create_dirs(args.plot_dir)

    # Load the data
    X, Y, cv, x, y, z = load_data(args.path, args)
    num_timesteps = X.shape[0] // args.window * args.window + 1
    print(f"X: {X.shape}; Y: {Y.shape}; cv: {cv.shape}; x: {x.shape}; y: {y.shape}; z: {z.shape}; num_timesteps: {num_timesteps}")

    if args.method == "full": 
        args.num_samples = X.shape[1]

    if args.method == "maxent":
        # Define the subsampling function for maximum entropy
        subsample_fn = create_maxent_subsampler(cv, args)
    elif args.method == "full":
        # No subsampling, use all indices
        subsample_fn = lambda X, n, t: np.arange(X.shape[1])
    elif args.method == "uips":
        # Phase-space sampling
        if args.plot: 
            X_flat = X.reshape(-1, X.shape[-1])
            plot_corner(X_flat)

        def subsample_fn(X, n, t):
            X_local = X[t]
            hist, bin_edges = build_pdf(X_local, nbins=args.bins)
            return subsample_uips(X_local[None, ...], n, hist, bin_edges)

    else: 
        # Random subsampling
        subsample_fn = lambda X, n, t: subsample_random(X, n, t)

    # Perform subsampling
    Xout, Yout = subsample_data(X, Y, x, y, z, subsample_fn, args)
    print(f"Xout: {Xout.shape}; Yout: {Yout.shape}")

    # Reshape Xout and Yout to 1D or 3D based on args.method and args.field_prediction_type
    if args.method == "full":
        Xout = Xout.reshape(num_timesteps, len(x), len(y), len(z), Xout.shape[2])
    if args.field_prediction_type == FieldPredictionType.FULL:
        Yout = Yout.reshape(num_timesteps, len(x), len(y), len(z), Yout.shape[2])

    # Save output
    fileprefix = f"nxsl{args.nxsl}-nysl{args.nysl}-nzsl{args.nzsl}-ns{args.num_samples}-window{args.window}_method-{args.method}"
    outfilename = f"subsampled_{fileprefix}.npz"
    outfile = os.path.join(args.output_dir, outfilename)
    np.savez(outfile, X=Xout, Y=Yout, x=x, y=y, z=z)

    print(f'Subsampled data saved to {outfile}')

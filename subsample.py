import numpy as np
import os
import sys

from args import args 
from helpers import check_and_create_dirs, load_data
from algorithms import create_maxent_subsampler, subsample_random
from constants import FieldPredictionType
from plotting import plot_samples

def subsample_data(X, Y, x, y, z, subsample_fn, args):
    num_timesteps = X.shape[0] // args.window * args.window + 1

    if args.method == "full": # baseline case with full field input
        Xout = np.zeros((num_timesteps, X.shape[1], X.shape[2]))
    else:  # all other subsampling cases
        Xout = np.zeros((num_timesteps, args.num_samples, X.shape[2]))

    if args.field_prediction_type == FieldPredictionType.GLOBAL: # global quantity prediction
        Yout = np.zeros((num_timesteps, 1, Y.shape[2]))
    elif args.field_prediction_type == FieldPredictionType.LOCAL:  # local field prediction
        if args.method == "full": raise Exception("For baseline full field input, prediction cannot be subsampled. Change `args.target`.")
        Yout = np.zeros((num_timesteps, args.num_samples, Y.shape[2]))
    elif args.field_prediction_type == FieldPredictionType.FULL:  # full field prediction
        Yout = np.zeros((num_timesteps, Y.shape[1], Y.shape[2]))
    else:
        raise Exception("Enter a valid `args.target`.")

    for timestep in range(0, num_timesteps - args.window, args.window):
        if args.method != "full": # no need to subsample for baseline case with full field input
            indices = subsample_fn(X, args.num_samples, timestep)
        for sub_timestep in range(args.window):
            ts = timestep + sub_timestep
            if args.method == "full":
                Xout[ts, :] = X[ts, :, :]
            else:
                Xout[ts, :, :] = X[ts, indices, :]
              
            if args.field_prediction_type == FieldPredictionType.GLOBAL:
                subsampled_Y = Y[ts, :]
            elif args.field_prediction_type == FieldPredictionType.LOCAL:
                subsampled_Y = Y[ts, indices, :]
            elif args.field_prediction_type == FieldPredictionType.FULL:
                subsampled_Y = Y[ts, :, :]
            else:
                raise Exception("Enter a valid `args.target`.")
            Yout[ts, :] = subsampled_Y

            if args.plot and args.method != "full":
                plot_samples(indices, x, y, z, args)
    
    # reshape Xout and Yout to 1D or 3D based on args.method and args.field_prediction_type
    if args.method == "full":
        Xout = Xout.reshape(num_timesteps, len(x), len(y), len(z), Xout.shape[2])
    if args.field_prediction_type == FieldPredictionType.FULL:
        Yout = Yout.reshape(num_timesteps, len(x), len(y), len(z), Yout.shape[2])

    return Xout, Yout

if __name__ == "__main__":

    # Ensure required directories exist
    check_and_create_dirs(args.output_dir)
    check_and_create_dirs(args.plot_dir)

    # Load the data
    X, Y, cv, x, y, z = load_data(args.path, args)

    if args.method == "maxent":
        # Define the subsampling function for maximum entropy
        subsample_fn = create_maxent_subsampler(cv, args)
    else:
        subsample_fn = lambda X, n, t: subsample_random(X, n, t)

    # Perform subsampling
    Xout, Yout = subsample_data(X, Y, x, y, z, subsample_fn, args)
    print(f"Xout: {Xout.shape}; Yout: {Yout.shape}")

    # Save output
    fileprefix = f"nxsl{args.nxsl}-nysl{args.nysl}-nzsl{args.nzsl}-ns{args.num_samples}-window{args.window}"
    outfilename = f"subsampled_{fileprefix}.npz"
    outfile = os.path.join(args.output_dir, outfilename)
    np.savez(outfile, X=Xout, Y=Yout, x=x, y=y, z=z)

    print(f'Subsampled data saved to {outfile}')

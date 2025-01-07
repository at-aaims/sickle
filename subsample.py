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

    Xout = np.zeros((num_timesteps, args.num_samples, X.shape[2]))

    if args.field_prediction_type == FieldPredictionType.GLOBAL: # global quantity prediction
        Yout = np.zeros((num_timesteps, 1))
    else: # local field prediction
        Yout = np.zeros((num_timesteps, args.num_samples))
    
    for timestep in range(0, num_timesteps - args.window, args.window):
        indices = subsample_fn(X, args.num_samples, timestep)
        for sub_timestep in range(args.window):
            ts = timestep + sub_timestep
            Xout[ts, :, :] = X[ts, indices, :]
            if args.field_prediction_type == FieldPredictionType.GLOBAL:
                subsampled_Y = Y[ts]
            else:
                subsampled_Y = Y[ts, indices]
            Yout[ts, :] = subsampled_Y

            if args.plot:
                plot_samples(indices, x, y, z, args)

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

    # Save output
    fileprefix = f"nxsl{args.nxsl}-nysl{args.nysl}-nzsl{args.nzsl}-ns{args.num_samples}-window{args.window}"
    outfilename = f"subsampled_{fileprefix}.npz"
    outfile = os.path.join(args.output_dir, outfilename)
    np.savez(outfile, X=Xout, Y=Yout, x=x, y=y, z=z)

    print(f'Subsampled data saved to {outfile}')

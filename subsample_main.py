import numpy as np
from subsampling_utils import *
from constants import FPT_GLOBAL

def subsample_data(X, Y, x, y, z, subsample_fn, args):
    num_timesteps = X.shape[0] // args.window * args.window + 1

    Xout = np.zeros((num_timesteps, args.num_samples, X.shape[2]))

    if args.field_prediction_type == FPT_GLOBAL: # global quantity prediction
        Yout = np.zeros((num_timesteps, 1))
    else: # local field prediction
        Yout = np.zeros((num_timesteps, args.num_samples))
    
    for timestep in range(0, num_timesteps - args.window, args.window):
        #indices = subsample_fn(X, args.num_samples)
        indices = subsample_fn(X, args.num_samples, timestep)
        for sub_timestep in range(args.window):
            ts = timestep + sub_timestep
            Xout[ts, :, :] = X[ts, indices, :]
            subsampled_Y = Y[ts] if args.field_prediction_type == FPT_GLOBAL else Y[ts, indices]
            Yout[ts, :] = subsampled_Y

            if args.plot:
                plot_samples(indices, x, y, z, args)

    return Xout, Yout

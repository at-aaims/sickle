import dataloader
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import plotting
import numpy as np 
import os

from args import args
from constants import *
from helpers import load

if not args.noseed:
  random_seed = 2024
  np.random.seed(random_seed)

def subsample_random(X, num_samples, random_seed=[0]):
    random_seed[0] += 1
    if not args.noseed:
        print('random seed: ', random_seed[0])
        np.random.seed(random_seed[0])
    return np.random.choice(X.shape[1], num_samples, replace=False)


if not os.path.exists(SNPDIR): os.makedirs(SNPDIR)

dfpath = os.path.join(SNPDIR, DRAWFN)
if not os.path.exists(dfpath): print("can't find snapshot... running dataloader")

if args.snapshot and os.path.exists(dfpath):
    data = load(dfpath)
    X, Y, cv, x, y, z = data['X'], data['Y'], data['cv'], data['x'], data['y'], data['z']

else:
    if args.dtype == "csv":
        dl = dataloader.DataLoaderCSV(args.path, dims=args.dims)
    elif args.dtype == "npz":
        dl = dataloader.DataLoaderNPZ(args.path)
    elif args.dtype == "sst-binary":
        dl = dataloader.DataLoaderSSTBinary(args)
    else:
        dl = dataloader.DataLoaderOF(args.path, dims=args.dims)
    x, y, z = dl.load_xyz()
    X, Y, cv = dl.load_multiple_timesteps(args.write_interval, args.num_timesteps, \
                                          target=args.target, cv=args.cluster_var)
    print(f'Shape of data: NN Input/observation - {X.shape}, NN Output/targets - {Y.shape}, Timesteps - {args.num_timesteps}, spatial x - {x.shape}, spatial y - {y.shape}, spatial z - {z.shape}')

    np.savez(dfpath, X=X, Y=Y, cv=cv, x=x, y=y, z=z)
    print(f"output file {dfpath}")

print(f'Data shape: Spatial x - {x.shape}, spatial y - {y.shape}, NN Input/observation - {X.shape}, NN Output/targets - {Y.shape}, Clustering/maxEnt variable - {cv.shape}')

num_timesteps = X.shape[0] // args.window * args.window + 1
print('num_timesteps:', X.shape[0])

Xout = np.zeros((num_timesteps, args.num_samples, X.shape[2]))

if args.field_prediction_type == FPT_GLOBAL: # global quantity prediction
    Yout = np.zeros((num_timesteps, 1))
else: # local field prediction
    Yout = np.zeros((num_timesteps, args.num_samples))

for timestep in range(0, num_timesteps - args.window, args.window):

    print(f"\nTIMESTEP: {timestep}-{timestep + args.window}\n")

    indices = subsample_random(X, args.num_samples)
    print(f'Random sample indices: {indices}')

    ts = timestep 
    for sub_timestep in range(args.window):
        if args.verbose: print(f"timestep: {ts}")

        # Find the indices of the original dataset, data, that have optimal clusters
        print(ts, len(indices), X.shape, X[ts, indices].shape)
        subsampled_X = X[ts, indices, :]
        subsampled_Y = Y[ts] if args.field_prediction_type == FPT_GLOBAL else Y[ts, indices]

        if args.verbose: print(subsampled_X.shape, subsampled_Y.shape)

        Xout[ts, :, :] = subsampled_X
        try:
            Yout[ts, :] = subsampled_Y
        except Exception as e:
            raise Exception("Try removing ./snapshots/raw_data.npz and re-running" + str())

    if args.plot:

        plt.clf()
        plt.rcParams.update({'font.size': 10})
        if args.dims == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = plt.subplot(111, projection='3d')
            ax.view_init(elev=20., azim=-35)
            if args.dtype == 'npz' or args.dtype == 'sst-binary':
                x_indices, y_indices, z_indices = np.unravel_index(indices, (x.shape[0], y.shape[0], z.shape[0]))
                ax.scatter(x[x_indices], z[z_indices], y[y_indices], c='k', s=2, alpha=0.5)
            else:
                ax.scatter(x[indices], y[indices], z[indices], c='k', s=2, vmin=-0.5, vmax=0.5, alpha=0.5)
            # Plot contour box
            plotting.plot_contour_box(ax, x, y, z, cv[timestep,:])
        else:
            plt.figure(figsize=(9, 2))
            plt.scatter(x[indices], y[indices], c='k', s=2, vmin=-0.5, vmax=0.5, alpha=0.5)
            plt.xlim([-25, 65])
            plt.ylim([-10, 10])
            plt.axis('equal')
        plt.savefig(os.path.join(PLTDIR, f'frame_{ts:04d}_random.png'), dpi=100, bbox_inches='tight')

    ts += 1


print(f'Shape of sampled data: NN Input/observation - {Xout.shape}, NN Output/targets - {Yout.shape}')
outfile = os.path.join(SNPDIR, 'subsampled.npz')

if args.dtype == 'npz' or args.dtype == 'sst-binary': 
    x_indices, y_indices, z_indices = np.unravel_index(indices, (x.shape[0], y.shape[0], z.shape[0]))
    arrays = { 'X': Xout, 'Y': Yout, 'x': x[x_indices], 'y': y[y_indices], 'z': z[z_indices], 'target': args.target }
else:
    arrays = { 'X': Xout, 'Y': Yout, 'x': x[indices], 'y': y[indices], 'target': args.target }

np.savez(outfile, **arrays)
if args.subsample != "proportional": print('min number of samples over all timesteps:', mins)
print(f'output {outfile}')

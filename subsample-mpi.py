import numpy as np
import os

from algorithms import create_maxent_subsampler, subsample_random
from args import args
from helpers import check_and_create_dirs, load_data
from mpi4py import MPI


# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Rank of the current process
size = comm.Get_size()  # Total number of processes

if rank == 0:
    # Ensure output directory
    check_and_create_dirs(args.output_dir)
    check_and_create_dirs(args.plot_dir)

    # Load data
    X, Y, cv, x, y, z = load_data(args.path, args)

    # Save data for broadcasting to all ranks
    fileprefix = f"nxsl{args.nxsl}-nysl{args.nysl}-nzsl{args.nzsl}"
    data_bcast = {"X": X, "Y": Y, "cv": cv, "x": x, "y": y, "z": z, "fileprefix": fileprefix}

else:
    data_bcast = None

# Broadcast data and extract vars
data_bcast = comm.bcast(data_bcast, root=0)

x, y, z, X, Y, cv, fileprefix = data_bcast['x'], data_bcast['y'], data_bcast['z'], data_bcast['X'], data_bcast['Y'], data_bcast['cv'], data_bcast['fileprefix']

comm.Barrier()  # Synchronize all processes

# Divide timesteps among processes
local_timesteps = np.array_split(range(X.shape[0]), size)[rank]

# Define subsample function based on method
def get_subsample_fn():
    if args.method == "maxent":
        subsample_fn = create_maxent_subsampler(cv, args)
    elif args.method == "random":
        subsample_fn = subsample_random
    else:
        raise ValueError(f"Unsupported sampling method: {args.method}")
    return subsample_fn

subsample_fn = get_subsample_fn()

# Process local timesteps
local_results = []
for timestep in local_timesteps:
    if args.method == "maxent":
        indices = subsample_fn(X, args.num_samples, timestep)
    elif args.method == "random":
        indices = subsample_fn(X, args.num_samples, timestep)

    local_results.append((timestep, indices))

# Gather results from all processes
all_results = comm.gather(local_results, root=0)

# Root process saves the results
if rank == 0:
    # Flatten results
    all_results = [item for sublist in all_results for item in sublist]
    outfile = os.path.join(args.output_dir, 'subsampled_results.npz')
    #np.savez(outfile, results=all_results) # this will crash on some datasets
    np.savez(outfile, results=np.array(all_results, dtype=object))
    print(f"Results saved to {outfile}")

MPI.Finalize()

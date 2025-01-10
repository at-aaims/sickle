import numpy as np
import os
from mpi4py import MPI
from algorithms import create_maxent_subsampler, subsample_random
from args import args
from dataloader_mpi import parallel_load_data
from helpers import check_and_create_dirs, load_data


def broadcast_large_array(data, comm, root=0):
    """Broadcast large arrays by chunking."""
    rank = comm.Get_rank()
    
    if rank == root:
        shape = data.shape
        dtype = data.dtype
    else:
        shape = None
        dtype = None
    
    # Broadcast metadata
    shape = comm.bcast(shape, root=root)
    dtype = comm.bcast(dtype, root=root)
    
    if rank != root:
        data = np.empty(shape, dtype=dtype)
    
    # Calculate safe chunk size
    MAX_ELEMENTS = 2**30  # Safely below INT_MAX
    total_elements = np.prod(shape)
    
    # Flatten array for chunking
    flat_data = data.ravel()
    
    for i in range(0, total_elements, MAX_ELEMENTS):
        end = min(i + MAX_ELEMENTS, total_elements)
        comm.Bcast([flat_data[i:end], MPI.DOUBLE], root=root)
    
    return data


# If uncommenting the following line, comment out the load_data() call below
#X, Y, cv, x, y, z = parallel_load_data(args.path, args)

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

else:
    X = None
    Y = None
    cv = None
    x = None
    y = None
    z = None
    fileprefix = None

# Broadcast large arrays with chunking
X = broadcast_large_array(X, comm) if rank == 0 else broadcast_large_array(None, comm)
Y = broadcast_large_array(Y, comm) if rank == 0 else broadcast_large_array(None, comm)
x = broadcast_large_array(x, comm) if rank == 0 else broadcast_large_array(None, comm)
y = broadcast_large_array(y, comm) if rank == 0 else broadcast_large_array(None, comm)
z = broadcast_large_array(z, comm) if rank == 0 else broadcast_large_array(None, comm)
cv = broadcast_large_array(cv, comm) if rank == 0 else broadcast_large_array(None, comm)

# Broadcast smaller arrays normally
#cv = comm.bcast(cv, root=0)
#x = comm.bcast(x, root=0)
#y = comm.bcast(y, root=0)
#z = comm.bcast(z, root=0)
fileprefix = comm.bcast(fileprefix, root=0)

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
    fileprefix = f"nxsl{args.nxsl}-nysl{args.nysl}-nzsl{args.nzsl}-ns{args.num_samples}-window{args.window}"
    outfilename = f"subsampled_{fileprefix}.npz"
    outfile = os.path.join(args.output_dir, outfilename)
    np.savez(outfile, results=np.array(all_results, dtype=object))
    print(f"Results saved to {outfile}")

MPI.Finalize()

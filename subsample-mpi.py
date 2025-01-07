import glob
import numpy as np
import os

from algorithms import create_maxent_subsampler, subsample_random
from args import args
from dataloader import DataLoaderSSTBinary
from helpers import check_and_create_dirs, get_data_memmap
from mpi4py import MPI


def parallel_load_data(path, args):
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    print(f"Rank {rank} started, total ranks = {size}")

    # Instantiate data loader
    dl = DataLoaderSSTBinary(args)

    # Load spatial grids (common to all ranks)
    x, y, z = dl.load_xyz()

    # Prepare file loading in parallel
    path = os.path.dirname(args.path)
    file_filter = '*_*'
    file_names = glob.glob(os.path.join(path, file_filter))
    file_names = [os.path.basename(f) for f in file_names]

    # Extract time labels (shared across ranks)
    t_labels = dl._extract_times(file_names)
    num_timesteps = len(t_labels)

    # Partition timesteps among MPI ranks
    local_timesteps = np.array_split(t_labels, size)[rank]

    # Prepare arrays
    num_pts = dl.num_pts
    X = np.zeros((len(local_timesteps), num_pts, len(args.input_vars)))
    Y = np.zeros((len(local_timesteps), num_pts))
    cv = np.zeros((len(local_timesteps), num_pts))

    def load_var(var_list, target_array, label):
        print(f"Rank {rank} loading {label} variables...")
        for i, var in enumerate(var_list):
            for j, ts in enumerate(local_timesteps):
                file_path = os.path.join(path, f'{var}_{ts:0.6f}')
                print(f'Rank {rank} loading file: {file_path}')
                box = get_data_memmap(
                    file_path, args.nx, args.ny, args.nz,
                    args.nxsl, args.nysl, args.nzsl,
                    args.nxoffset, args.nyoffset, args.nzoffset,
                    args.nxskip, args.nyskip, args.nzskip
                )

                # Reshape box to 1D
                reshaped_box = box.reshape(-1)

                # Dynamically handle 2D or 3D target arrays
                if len(target_array.shape) == 3:  # Multi-variable (3D)
                    target_array[j, :, i] = reshaped_box
                elif len(target_array.shape) == 2:  # Single-variable (2D)
                    target_array[j, :] = reshaped_box
                else:
                    raise ValueError(f"Unsupported target array shape: {target_array.shape}")

    # Load data in parallel
    load_var(args.input_vars, X, "input")
    load_var(args.output_vars, Y, "output")
    load_var(args.cluster_var, cv, "cluster")

    # Gather results from all ranks
    X_all = comm.gather(X, root=0)
    Y_all = comm.gather(Y, root=0)
    cv_all = comm.gather(cv, root=0)

    # Concatenate results on root
    if rank == 0:
        X_all = np.concatenate(X_all, axis=0)
        Y_all = np.concatenate(Y_all, axis=0)
        cv_all = np.concatenate(cv_all, axis=0)
        print("Data loading complete.")
        return X_all, Y_all, cv_all, x, y, z
    else:
        return None, None, None, None, None, None


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
        chunk = flat_data[i:end] if rank == root else None
        comm.Bcast([flat_data[i:end], MPI.DOUBLE], root=root)
    
    return data


X, Y, cv, x, y, z = parallel_load_data(args.path, args)

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Rank of the current process
size = comm.Get_size()  # Total number of processes

if rank == 0:
    # Ensure output directory
    check_and_create_dirs(args.output_dir)
    check_and_create_dirs(args.plot_dir)

    # Load data
    #print("loading data start...")
    #X, Y, cv, x, y, z = load_data(args.path, args)
    #X, Y, cv, x, y, z = parallel_load_data(args.path, args)
    #print("loading data done...")

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

# Broadcast smaller arrays normally
cv = comm.bcast(cv, root=0)
x = comm.bcast(x, root=0)
y = comm.bcast(y, root=0)
z = comm.bcast(z, root=0)
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

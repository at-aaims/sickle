import numpy as np
import os
from mpi4py import MPI
from algorithms import create_maxent_subsampler, subsample_random, build_pdf, subsample_uips
from args import args
from dataloaders import load_data #, parallel_load_data
from helpers import check_and_create_dirs
from plotting import plot_corner
from viz import save_vtu


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
#X, Y, cv, x, y, z = parallel_load_data(args)

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Rank of the current process
size = comm.Get_size()  # Total number of processes

print(f"Rank {rank} sees output_dir = {args.output_dir}", flush=True)

if rank == 0:
    print(f"*** TOTAL NUMBER OF PROCESSES: {size} *************")

    # Ensure output directory
    check_and_create_dirs(args.output_dir)
    check_and_create_dirs(args.plot_dir)

    # Load data
    X, Y, cv, x, y, z = load_data(args)

    # Save data for broadcasting to all ranks
    args_to_broadcast = args

else:
    X = None
    Y = None
    cv = None
    x = None
    y = None
    z = None
    args_to_broadcast = None

# Broadcast large arrays with chunking
X = broadcast_large_array(X, comm) if rank == 0 else broadcast_large_array(None, comm)
Y = broadcast_large_array(Y, comm) if rank == 0 else broadcast_large_array(None, comm)
x = broadcast_large_array(x, comm) if rank == 0 else broadcast_large_array(None, comm)
y = broadcast_large_array(y, comm) if rank == 0 else broadcast_large_array(None, comm)
z = broadcast_large_array(z, comm) if rank == 0 else broadcast_large_array(None, comm)
cv = broadcast_large_array(cv, comm) if rank == 0 else broadcast_large_array(None, comm)

# Broadcast smaller arrays normally
args = comm.bcast(args_to_broadcast, root=0)

comm.Barrier()  # Synchronize all processes

# Divide timesteps among processes
local_timesteps = np.array_split(range(X.shape[0]), size)[rank]

# Define subsample function based on method
def get_subsample_fn():
    if args.method == "maxent":
        subsample_fn = create_maxent_subsampler(cv, args)
    elif args.method == "random":
        subsample_fn = subsample_random
    elif args.method == "uips":
        # Phase-space sampling
        if args.plot:
            X_flat = X.reshape(-1, X.shape[-1])
            plot_corner(X_flat)

        def subsample_fn(X, n, t):
            X_local = X[t]
            hist, bin_edges = build_pdf(X_local, nbins=args.bins)
            return subsample_uips(X_local[None, ...], n, hist, bin_edges)
    elif args.method == "full":
        subsample_fn = lambda X, n, t: np.arange(X.shape[1])
    else:
        raise ValueError(f"Unsupported sampling method: {args.method}")
    return subsample_fn

subsample_fn = get_subsample_fn()

# Process local timesteps
local_results = []
for timestep in local_timesteps:
    print(f"[DEBUG] args.num_samples before calling subsample_fn: {args.num_samples}")
    indices = subsample_fn(X, args.num_samples, timestep)
    local_results.append((timestep, indices))

# Gather results from all processes
all_results = comm.gather(local_results, root=0)

# Root process aggregates results
if rank == 0:
    if not os.path.exists(args.output_dir):
        print(f"Output directory {args.output_dir} does not exist!", flush=True)
    else:
        print(f"Output directory {args.output_dir} exists with permissions:", flush=True)
        print(oct(os.stat(args.output_dir).st_mode))

    print("***** METHOD IS: ", args.method, flush=True)

    # Save output to file
    outfilename = f"subsampled_{args.fileprefix}.npz"
    outfile = os.path.join(args.output_dir, outfilename)
    np.savez(outfile, X=X, Y=Y, x=x, y=y, z=z)
    print(f'Subsampled data saved to {outfile}', flush=True)

    # Flatten results and sort by timestep
    all_results = [item for sublist in all_results for item in sublist]
    all_results.sort(key=lambda x: x[0])  # Sort by timestep

    # Extract only the indices in the correct order
    indices_list = [indices for _, indices in all_results]
    # Save indices for debugging - following may be not be working
    #np.savez(outfile, results=np.array(indices_list, dtype=object))
    #print(f"Results saved to {outfile}")

    # Define num_timesteps from the loaded data X
    num_timesteps = X.shape[0]

    # If sampling method is "full", reshape X to a 3D spatial grid per timestep.
    if args.method == "full":
        # Assuming X's last dimension is the channel (or feature) dimension.
        X = X.reshape(num_timesteps, len(x), len(y), len(z), X.shape[-1])

    # If field prediction type is FULL, reshape Y similarly.
    if args.field_prediction_type == FieldPredictionType.FULL:
        Y = Y.reshape(num_timesteps, len(x), len(y), len(z), Y.shape[-1])

    # Save to VTK unstructured format
    if args.viz:
        save_vtu(X, Y, x, y, z, indices_list, args.output_dir, args.fileprefix)

MPI.Finalize()

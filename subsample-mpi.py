import numpy as np
import os

from mpi4py import MPI
from args import args
from constants import FieldPredictionType
from dataloaders import load_data  # , parallel_load_data
from helpers import check_and_create_dirs
from subsampling import get_subsampler
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

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()   # Rank of the current process
size = comm.Get_size()   # Total number of processes

print(f"Rank {rank} sees output_dir = {args.output_dir}", flush=True)

if rank == 0:
    print(f"*** TOTAL NUMBER OF PROCESSES: {size} *************")
    # Ensure output directory
    check_and_create_dirs(args.output_dir)
    check_and_create_dirs(args.plot_dir)
    # Load data
    X, Y, cv, x, y, z = load_data(args)
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
args = comm.bcast(args_to_broadcast, root=0)

comm.Barrier()  # Synchronize all processes

# Divide timesteps among processes
local_timesteps = np.array_split(range(X.shape[0]), size)[rank]

# For maxent, pass cv; otherwise just use X.
if args.method == "maxent":
    subsampler = get_subsampler(X, args, cv=cv)
else:
    subsampler = get_subsampler(X, args)

# Each process computes indices for its assigned timesteps.
local_results = []
for timestep in local_timesteps:
    if args.method == "full":
        indices = np.arange(X.shape[1])
    else:
        indices = subsampler.sample(args.num_samples, timestep)
    local_results.append((timestep, indices))

# Gather results from all processes
all_results = comm.gather(local_results, root=0)

if rank == 0:
    # Root process aggregates results
    if not os.path.exists(args.output_dir):
        print(f"Output directory {args.output_dir} does not exist!", flush=True)
    else:
        print(f"Output directory {args.output_dir} exists with permissions:", flush=True)
        print(oct(os.stat(args.output_dir).st_mode))
    
    print("***** METHOD IS: ", args.method, flush=True)
    
    # Flatten and sort the results by timestep.
    all_results = [item for sublist in all_results for item in sublist]
    all_results.sort(key=lambda x: x[0])
    indices_list = [indices for _, indices in all_results]
    
    # Save to VTK unstructured format if visualization is enabled.
    if args.viz:
        save_vtu(X, Y, x, y, z, indices_list, args.output_dir, args.fileprefix)
    
    # Now, compute the subsampled outputs.
    num_timesteps = X.shape[0]

    # Preallocate output arrays based on X and Y shapes:
    if args.method == "full": 
        Xout = np.zeros((num_timesteps, X.shape[1], X.shape[2]))
    else:
        Xout = np.zeros((num_timesteps, args.num_samples, X.shape[2]))

    if args.field_prediction_type == FieldPredictionType.GLOBAL:
        Yout = np.zeros((num_timesteps, 1, Y.shape[2]))
    elif args.field_prediction_type == FieldPredictionType.FULL:
        Yout = np.zeros((num_timesteps, Y.shape[1], Y.shape[2]))
    else:  # LOCAL
        Yout = np.zeros((num_timesteps, args.num_samples, Y.shape[2]))
    
    # Iterate over each window of timesteps.
    # (This code assumes that timesteps were processed in windows of size args.window.)
    for (timestep, indices) in all_results:
        # For each window starting at timestep, fill in the outputs.
        for sub_timestep in range(args.window):
            ts = timestep + sub_timestep
            if ts >= num_timesteps:
                continue  # Skip if we exceed available timesteps.
            print(Xout.shape, X.shape)
            if args.method == "full":
                Xout[ts] = X[ts]
            else:
                Xout[ts, :] = X[ts, indices]

            if args.field_prediction_type == FieldPredictionType.GLOBAL:
                subsampled_Y = Y[ts, :]
            elif args.field_prediction_type == FieldPredictionType.FULL:
                subsampled_Y = Y[ts, :, :]
            else:
                subsampled_Y = Y[ts, indices, :]
            Yout[ts, :] = subsampled_Y
    
    # Reshape outputs to a spatial grid:
    num_timesteps *= args.num_hypercubes
    if args.method == "full":
        Xout = Xout.reshape(num_timesteps, len(x), len(y), len(z), Xout.shape[-1])
    if args.field_prediction_type == FieldPredictionType.FULL:
        Yout = Yout.reshape(num_timesteps, len(x), len(y), len(z), Yout.shape[-1])
    
    # Save output to file.
    outfilename = f"subsampled_{args.fileprefix}.npz"
    outfile = os.path.join(args.output_dir, outfilename)
    np.savez(outfile, X=Xout, Y=Yout, x=x, y=y, z=z)
    print(f'Subsampled data saved to {outfile}', flush=True)
    print(f"Output: X: {Xout.shape}; Y: {Yout.shape}; x: {x.shape}; y: {y.shape}; z: {z.shape}", flush=True)
    
MPI.Finalize()

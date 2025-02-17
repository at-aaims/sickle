import numpy as np
import pandas as pd
import importlib

# -------------------------------------------------------------------
# Parent Class
# -------------------------------------------------------------------

class DataLoader:
    def __init__(self, path, dims=2, verbose=False):
        self.path = path
        self.dims = dims
        self.verbose = verbose

    def to_csv(self, Y, X, time, columns):
        """Output CSV file named by timestamp, e.g. 1000.csv"""
        df = pd.DataFrame(np.concatenate((Y, X), axis=1), columns=columns)
        df.to_csv(str(time) + '.csv', index=False)

# -------------------------------------------------------------------
# Sequential Loader Factory Function
# -------------------------------------------------------------------

def load_data(args):
    """
    Factory function to create and use a dataloader instance based on args.dtype.
    
    Expects that the module in the dataloaders package is named exactly as args.dtype
    (for example, if args.dtype is "sst-binary", then the module is dataloaders/sst-binary.py)
    and that each module exposes its main dataloader class as "DataLoader".
    
    After instantiation, load_xyz() and load_multiple_timesteps() are called and their
    outputs are returned.
    
    Returns:
         X, Y, cv, x, y, z
    """
    module = importlib.import_module("dataloaders." + args.dtype)
    DataLoaderClass = module.DataLoader  # each module must expose its main class as DataLoader

    # For loaders (like sst-binary) that require the full args object
    if args.dtype == "sst-binary":
        dl = DataLoaderClass(args)
    else:
        dl = DataLoaderClass(args.path, dims=args.dims)

    x, y, z = dl.load_xyz()
    X, Y, cv = dl.load_multiple_timesteps(
        args.write_interval, args.num_timesteps, target=args.target, cv=args.cluster_var
    )
    return X, Y, cv, x, y, z

# -------------------------------------------------------------------
# Parallel Loader Factory Function (with subsampling)
# -------------------------------------------------------------------

def broadcast_large_array(data, comm, root=0):
    """
    Broadcast large arrays in chunks.
    
    This function first broadcasts the metadata (shape and dtype), then splits the array into
    chunks that are safely below the INT_MAX limit and broadcasts each chunk.
    """
    rank = comm.Get_rank()
    if rank == root:
        shape = data.shape
        dtype = data.dtype
    else:
        shape = None
        dtype = None
    shape = comm.bcast(shape, root=root)
    dtype = comm.bcast(dtype, root=root)
    if rank != root:
        data = np.empty(shape, dtype=dtype)

    MAX_ELEMENTS = 2**30  # safely below INT_MAX
    total_elements = np.prod(shape)
    flat_data = data.ravel()

    for i in range(0, total_elements, MAX_ELEMENTS):
        end = min(i + MAX_ELEMENTS, total_elements)
        comm.Bcast([flat_data[i:end], np.dtype(dtype).char], root=root)
    return data


def parallel_load_data(args, subsample=True):
    """
    Parallel version of load_data.
    
    On the root process (rank 0), the data are loaded by calling load_data().
    Then, all data arrays are broadcast (in chunks) to all MPI processes.
    
    Optionally, the function performs subsampling on the time dimension:
      - It divides timesteps among processes.
      - Each process applies a subsample function (selected based on args.method).
      - Results are gathered back at the root and saved.
    
    Returns:
         X, Y, cv, x, y, z, subsample_results (the latter is only valid on rank 0)
    """
    from mpi4py import MPI
    import os
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # On the root process, ensure output directories (if needed) and load data
    if rank == 0:
        # (Assuming you have a helper function to check/create directories.)
        from helpers import check_and_create_dirs
        check_and_create_dirs(args.output_dir)
        check_and_create_dirs(args.plot_dir)
        X, Y, cv, x, y, z = load_data(args)
        fileprefix = f"nxsl{args.nxsl}-nysl{args.nysl}-nzsl{args.nzsl}"
    else:
        X = Y = cv = x = y = z = None
        fileprefix = None

    # Broadcast the large arrays to all processes
    X  = broadcast_large_array(X,  comm, root=0)
    Y  = broadcast_large_array(Y,  comm, root=0)
    cv = broadcast_large_array(cv, comm, root=0)
    x  = broadcast_large_array(x,  comm, root=0)
    y  = broadcast_large_array(y,  comm, root=0)
    z  = broadcast_large_array(z,  comm, root=0)

    comm.Barrier()  # synchronize all processes

    if not subsample:
        return X, Y, cv, x, y, z, None

    # Divide timesteps among processes for subsampling
    local_timesteps = np.array_split(range(X.shape[0]), size)[rank]

    def get_subsample_fn():
        if args.method == "maxent":
            from algorithms import create_maxent_subsampler
            return create_maxent_subsampler(cv, args)
        elif args.method == "random":
            from algorithms import subsample_random
            return subsample_random
        elif args.method == "uips":
            # Example: define a placeholder subsample function for uips.
            def subsample_fn(X, n, t):
                # (Replace with actual uips subsampling implementation.)
                return np.arange(X[t].shape[0])
            return subsample_fn
        elif args.method == "full":
            return lambda X, n, t: np.arange(X.shape[1])
        else:
            raise ValueError(f"Unsupported sampling method: {args.method}")

    subsample_fn = get_subsample_fn()
    local_results = []
    for timestep in local_timesteps:
        indices = subsample_fn(X, args.num_samples, timestep)
        local_results.append((timestep, indices))

    # Gather subsampling results on the root process
    all_results = comm.gather(local_results, root=0)
    if rank == 0:
        # Flatten results
        all_results = [item for sublist in all_results for item in sublist]
        if args.method == "full":
            fileprefix = f"nxsl{args.nx}-nysl{args.ny}-nzsl{args.nz}-ns{args.num_samples}-window{args.window}"
        else:
            fileprefix = f"nxsl{args.nxsl}-nysl{args.nysl}-nzsl{args.nzsl}-ns{args.num_samples}-window{args.window}"
        outfilename = f"subsampled_{fileprefix}.npz"
        outfile = os.path.join(args.output_dir, outfilename)
        np.savez(outfile, results=np.array(all_results, dtype=object))
        print(f"Results saved to {outfile}")
    else:
        all_results = None

    return X, Y, cv, x, y, z, all_results

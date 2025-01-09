import glob
import numpy as np
import os

from dataloader import DataLoaderSSTBinary
from helpers import get_data_memmap
from mpi4py import MPI


def parallel_load_data(path, args):
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    print(f"Rank {rank}: Starting execution with {size} ranks")

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

    # Partition timesteps among MPI ranks
    local_timesteps = np.array_split(t_labels, size)[rank]

    # Prepare arrays (handle empty cases explicitly)
    num_pts = dl.num_pts
    if len(local_timesteps) == 0:
        # Handle empty partitions gracefully
        X = np.zeros((0, num_pts, len(args.input_vars)))
        Y = np.zeros((0, num_pts))
        cv = np.zeros((0, num_pts))
    else:
        X = np.zeros((len(local_timesteps), num_pts, len(args.input_vars)))
        Y = np.zeros((len(local_timesteps), num_pts))
        cv = np.zeros((len(local_timesteps), num_pts))

    # Function to load a single variable
    def load_var(var_list, target_array, label):
        print(f"Rank {rank}: Loading {label} variables...")
        for i, var in enumerate(var_list):
            for j, ts in enumerate(local_timesteps):
                file_path = os.path.join(path, f'{var}_{ts:0.6f}')
                print(f'Rank {rank}: Loading file {file_path}')
                box = get_data_memmap(
                    file_path, args.nx, args.ny, args.nz,
                    args.nxsl, args.nysl, args.nzsl,
                    args.nxoffset, args.nyoffset, args.nzoffset,
                    args.nxskip, args.nyskip, args.nzskip
                )

                # Reshape and assign based on dimensions
                reshaped_box = box.reshape(-1)
                if len(target_array.shape) == 3:
                    target_array[j, :, i] = reshaped_box
                elif len(target_array.shape) == 2:
                    target_array[j, :] = reshaped_box
                else:
                    raise ValueError(f"Unsupported target array shape: {target_array.shape}")

    # Load input, output, and cluster variables
    load_var(args.input_vars, X, "input")
    load_var(args.output_vars, Y, "output")
    load_var(args.cluster_var, cv, "cluster")

    # Debug shapes before gathering
    print(f"Rank {rank}: X shape = {X.shape}, Y shape = {Y.shape}, cv shape = {cv.shape}")

    # Prepare sizes for Gatherv
    local_X_size = X.size
    local_Y_size = Y.size
    local_cv_size = cv.size

    sizes_X = comm.gather(local_X_size, root=0)
    sizes_Y = comm.gather(local_Y_size, root=0)
    sizes_cv = comm.gather(local_cv_size, root=0)

    # Gather data using Gatherv
    if rank == 0:
        total_X = sum(sizes_X)
        total_Y = sum(sizes_Y)
        total_cv = sum(sizes_cv)
        recvbuf_X = np.empty(total_X, dtype=X.dtype)
        recvbuf_Y = np.empty(total_Y, dtype=Y.dtype)
        recvbuf_cv = np.empty(total_cv, dtype=cv.dtype)
    else:
        recvbuf_X = None
        recvbuf_Y = None
        recvbuf_cv = None

    comm.Gatherv(sendbuf=X.flatten(), recvbuf=(recvbuf_X, sizes_X), root=0)
    comm.Gatherv(sendbuf=Y.flatten(), recvbuf=(recvbuf_Y, sizes_Y), root=0)
    comm.Gatherv(sendbuf=cv.flatten(), recvbuf=(recvbuf_cv, sizes_cv), root=0)

    # Root processes the gathered data
    if rank == 0:
        # Reconstruct arrays from gathered buffers
        X_all = np.concatenate([recvbuf_X[i : i + sizes_X[r]] for r, i in enumerate(np.cumsum([0] + sizes_X[:-1]))], axis=0)
        Y_all = np.concatenate([recvbuf_Y[i : i + sizes_Y[r]] for r, i in enumerate(np.cumsum([0] + sizes_Y[:-1]))], axis=0)
        cv_all = np.concatenate([recvbuf_cv[i : i + sizes_cv[r]] for r, i in enumerate(np.cumsum([0] + sizes_cv[:-1]))], axis=0)

        print("Rank 0: Data loading complete.")
        return X_all, Y_all, cv_all, x, y, z
    else:
        return None, None, None, None, None, None

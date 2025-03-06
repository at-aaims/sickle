"""
Parallel maxEnt-based hypercube selection for SST data.
Sample run script:
srun -n 64 python -u mpi_hypercubes.py -ncl 20 -ncu 100
"""

from mpi4py import MPI
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from scipy.stats import entropy
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-p', '--path', type=str, default='/lustre/orion/proj-shared/gen150/dsml/data/P1F4R32_nx512ny512nz256_6vars/pv_28.040000', help='File path to data.')
parser.add_argument('-ncl', '--n_clusters', type=int, default=10, help='Number of clusters.')
parser.add_argument('-ncu', '--n_cubes', type=int, default=100, help='Number of hypercubes to extract.')
parser.add_argument("--nx", type=int, default=512+2, required=False, help="number of grid points in x dir for full data")
parser.add_argument("--ny", type=int, default=512, required=False, help="number of grid points in y dir for full data")
parser.add_argument("--nz", type=int, default=256, required=False, help="number of grid points in z dir for full data")
parser.add_argument("--nxsl", type=int, default=32, required=False, help="number of grid points in x dir for sampled data")
parser.add_argument("--nysl", type=int, default=32, required=False, help="number of grid points in y dir for sampled data")
parser.add_argument("--nzsl", type=int, default=32, required=False, help="number of grid points in z dir for sampled data")
args = parser.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if rank == 0:
    seed_value = np.random.randint(0, 1000000)
else:
    seed_value = None

# Broadcast the seed to all ranks so they use the same random state
seed_value = comm.bcast(seed_value, root=0)
np.random.seed(seed_value)  # Set the seed for NumPy

def load_data_mpi(comm, loadpath, nx, ny, nz, rank, size, split_axis=0):
    """
    Load a portion of the dataset for each MPI rank using memmap.
    Distribute data across ranks using MPI Scatter.
    Parameters:
    - loadpath: str, path to the binary file
    - nx, ny, nz: int, full data cube dimensions
    - rank: int, MPI rank of the process
    - size: int, total number of MPI processes

    Returns:
    - subcube: reshaped 3D NumPy array, portion of the dataset assigned to the process
    """

    if rank == 0:
        n_features = 1
        nskip = 1
        # Load full dataset on rank 0
        data_memmap = np.memmap(loadpath, dtype=np.float32, mode='r', shape=(nz, ny, nx))[::nskip,::nskip,:-2:nskip]
        # data_memmap = data_memmap.reshape(-1, n_features)  # Flatten for k-means

        # Create a coordinate grid that matches the shape of data_memmap.
        # Using np.indices yields an array of shape (3, nz_out, ny_out, nx_out).
        # Transpose to get shape (nz_out, ny_out, nx_out, 3) so that splitting along axis 0 (z-axis) works directly.
        coords = np.indices(data_memmap.shape).transpose(1, 2, 3, 0)
        
        # Split data along the Z-axis
        split_data = np.array_split(data_memmap, size, axis=split_axis)
        split_coords = np.array_split(coords, size, axis=split_axis)
    else:
        split_data = None  # Other ranks have no data yet
        split_coords = None
        n_features = None

    n_features = comm.bcast(n_features, root=0)

    # Scatter chunks to all ranks
    local_data = comm.scatter(split_data, root=0).reshape(-1, n_features)  # Flatten for k-means

    # Scatter the coordinate chunks; each process reshapes its local coordinates to (-1, 3).
    local_coords = comm.scatter(split_coords, root=0).reshape(-1, 3)

    return local_data, local_coords

def compute_local_histograms(local_data, local_labels, n_clusters, num_bins, bin_range):
    """
    Compute histogram counts for each cluster based on a feature.
    
    Parameters:
        local_data: np.ndarray, shape (N,)
            The feature values (e.g., intensity, or any scalar) for each data point.
        local_labels: np.ndarray, shape (N,)
            Cluster assignments (0 to K-1) for each data point.
        n_clusters: int
            Total number of clusters.
        num_bins: int
            # bins in the histogram.
        bin_range: tuple, (float, float)
            global bin range for histogram
           
    Returns:
       local_histograms: list of np.ndarray
           A list (length n_clusters) where each element is a 1D array of histogram counts.
    """
    local_histograms = [np.zeros(num_bins, dtype=np.int64) for _ in range(n_clusters)]
    for feature, label in zip(local_data, local_labels):
        counts, _ = np.histogram(feature, bins=num_bins, range=bin_range, density=False)
        local_histograms[label] += counts
    return local_histograms

def reduce_histograms(comm, local_histograms, n_clusters):
    """
    Reduce local histograms from all processes into global histograms.
    
    Parameters:
       comm: MPI communicator.
       local_histograms: list of np.ndarray for each cluster.
       n_clusters: int, number of clusters.
       
    Returns:
       global_histograms: list of np.ndarray for each cluster (on the root process).
    """
    global_histograms = []
    for k in range(n_clusters):
        global_hist = np.zeros_like(local_histograms[k])
        comm.Reduce(local_histograms[k], global_hist, op=MPI.SUM, root=0)
        global_histograms.append(global_hist)
    return global_histograms

def compute_cluster_probability_distributions(global_histograms):
    """
    Normalize each histogram to obtain a probability distribution.
    
    Parameters:
       global_histograms: list of np.ndarray
           Global histogram counts for each cluster.
       
    Returns:
       cluster_distributions: list of np.ndarray
           Normalized distributions (each sums to 1).
    """
    cluster_distributions = []
    for hist in global_histograms:
        total = hist.sum()
        if total > 0:
            distribution = hist / total
        else:
            distribution = hist  # remains zeros if no counts
        cluster_distributions.append(distribution + 1e-10) # to avoid 0 probability
    return cluster_distributions

def compute_node_strengths(cluster_distributions):
    """
    Compute node strengths for each cluster using KL divergence.
    
    For cluster i, the node strength is computed as the sum over j != i of:
         KL(P_i || P_j)
    where P_i is the probability distribution of cluster i.
    
    Parameters:
       cluster_distributions: list of np.ndarray
           The probability distributions for each cluster.
           
    Returns:
       node_strengths: np.ndarray of shape (K,)
    """
    K = len(cluster_distributions)
    node_strengths = np.zeros(K)
    for i in range(K):
        for j in range(K):
            if i != j:
                # scipy.stats.entropy computes KL(P || Q)
                kl_div = entropy(cluster_distributions[i], cluster_distributions[j])
                node_strengths[i] += kl_div
    return node_strengths

def extract_local_subcube_totals(coords, data_node_strength, subcube_size):
    """
    Partition the 3D domain into subcubes and sum node strengths in each subcube.
    Add node strength of each local data to corresponding subcube ID.
    
    Parameters:
      coords: np.ndarray, shape (N,3)
         3D spatial coordinates for each data point.
      data_node_strength: np.ndarray, shape (N,)
         The node strength for each data point.
      subcube_size: tuple (nxsl, nysl, nzsl)
         The dimensions of each subcube.
    
    Returns:
      subcube_totals: dict
         Mapping from subcube ID (a tuple of indices) to the summed node strength.
    """
    nxsl, nysl, nzsl = subcube_size
    subcube_totals = defaultdict(float)
    for (x, y, z), strength in zip(coords, data_node_strength):
        cube_id = (int(x // nxsl), int(y // nysl), int(z // nzsl))
        subcube_totals[cube_id] += strength
    return subcube_totals

def mpi_hypercubes(loadpath, nx, ny, nz, nxsl, nysl, nzsl, n_clusters, n_cubes, batch_size=10000, n_init=10, max_iter=100, n_iters=10):
    """
    Parallel maxEnt hypercube selection using k-means clustering using MPI4PY + MiniBatchKMeans.

    Parameters:
    - loadpath: str, path to the binary file
    - nx, ny, nz: int, full data cube dimensions
    - n_clusters: int, number of clusters
    - batch_size: int, mini-batch size for k-means
    - n_init: int, number of initialization runs
    - max_iter: int, maximum iterations for k-means
    - n_iters: int, # of syncs between ranks to converge on the centers

    Returns:
    - Cluster centers from root process.
    """

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # Process rank
    size = comm.Get_size()  # Total processes
    if rank == 0:
        print(f"Total processors: {size}")

    # Load dataset slice for each MPI rank
    local_data, local_coords = load_data_mpi(comm, loadpath, nx, ny, nz, rank, size)
    # print(f"local_data (rank {rank}): {local_data.shape}")
    # print(f"local_coords (rank {rank}): {local_coords.shape}")

    # Start timing for MPI K-Means
    mpi_start_time = MPI.Wtime()

    # Step 1: Initialize cluster centers on rank 0
    if rank == 0:
        kmeans_init = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, n_init=n_init, max_iter=max_iter, init="k-means++", random_state=seed_value)
        kmeans_init.fit(local_data)  # Initial training to get centers
        cluster_centers = kmeans_init.cluster_centers_
    else:
        cluster_centers = np.empty((n_clusters, local_data.shape[1]), dtype=np.float32)

    comm.Bcast(cluster_centers, root=0)  # Broadcast initial centers to all ranks

    # Step 2: Iteratively update centers with global synchronization
    for i in range(n_iters):
        # Use the precomputed centers to initialize MiniBatchKMeans properly
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, n_init=1, max_iter=max_iter, init=cluster_centers, random_state=seed_value)

        # Each rank updates centers using its local data
        kmeans.partial_fit(local_data)

        # Collect updated centers across all ranks
        local_centers = kmeans.cluster_centers_

        # Use Allreduce to sum the cluster centers across all ranks
        global_centers = np.zeros_like(local_centers)
        comm.Allreduce(local_centers, global_centers, op=MPI.SUM)

        # Normalize by number of processes
        cluster_centers = global_centers / size

        if rank == 0 and i % (n_iters // 5) == 0:  # Print progress every few iterations
            print(f"Iteration {i+1}/{n_iters} - Synchronizing cluster centers...")

    comm.Barrier()  # Ensure all ranks finish
    cluster_centers = cluster_centers[np.lexsort(tuple(cluster_centers[:, i] for i in range(cluster_centers.shape[1]-1, -1, -1)))]

    # Stop timing
    mpi_end_time = MPI.Wtime()
    mpi_time = mpi_end_time - mpi_start_time

    if rank == 0:
        print(f"Parallel (MPI) K-Means Time: {mpi_time:.4f} seconds")
        print(f"Centers: {cluster_centers}")

    # 1. Compute local data labels
    local_labels = pairwise_distances_argmin(local_data, cluster_centers)
    # print(f"local_labels (rank {rank}): {len(local_labels)}")

    # 2. Compute local histograms for each label
    bin_range = (comm.allreduce(np.min(local_data), op=MPI.MIN), comm.allreduce(np.max(local_data), op=MPI.MAX))
    # print(f"bin_range (rank {rank}): {bin_range}")
    num_bins = 10
    mpi_start_time = MPI.Wtime()
    local_histograms = compute_local_histograms(local_data, local_labels, n_clusters, num_bins, bin_range)
    mpi_end_time = MPI.Wtime()
    mpi_time = mpi_end_time - mpi_start_time
    if rank == 0:
        print(f"local_histograms Time: {mpi_time:.4f} seconds")
        print(f"local_histograms (rank {rank}): {len(local_histograms)}")

    # 3. Reduce local histograms to global histogram for all lables
    global_histograms = reduce_histograms(comm, local_histograms, n_clusters)
    # if rank == 0:
    #     print(f"global_histograms: {global_histograms}")

    # 4. On the root process, compute the probability distributions and node strength of each cluster.
    if rank == 0:
        # Compute probability from counts
        cluster_distributions = compute_cluster_probability_distributions(global_histograms)
        # Compute node strengths (using pairwise KL divergence)
        node_strengths = compute_node_strengths(cluster_distributions)
    else:
        cluster_distributions = None
        node_strengths = None
    
    # 5. Assign node strength to data points (based on cluster_label) 
    # Broadcast the computed node strengths to all processes.
    node_strengths = comm.bcast(node_strengths, root=0)
    if rank == 0:
        print(f"node_strengths: {node_strengths}")
    # For each data point, assign its cluster’s node strength.
    local_data_node_strength = np.array([node_strengths[label] for label in local_labels])

    # 6. Partition the 3D domain into subcubes and aggregate node strengths.
    subcube_size = (nxsl, nysl, nzsl)
    local_subcube_totals = extract_local_subcube_totals(local_coords, local_data_node_strength, subcube_size)
    
    # 7. Combine local subcube node strengths to global
    # Gather the local subcube totals to the root
    all_local_subcubes = comm.gather(local_subcube_totals, root=0)
    if rank == 0:
        # Merge dictionaries to obtain global subcube totals.
        global_subcube_totals = defaultdict(float)
        for local_dict in all_local_subcubes:
            for cube_id, total in local_dict.items():
                global_subcube_totals[cube_id] += total
        # print(f"global_subcube_totals: {global_subcube_totals}")
        
        # 8. Rank subcubes based on total node strength and sample.
        subcube_ids = list(global_subcube_totals.keys())
        strengths = np.array([global_subcube_totals[cube_id] for cube_id in subcube_ids])
        
        # If all subcubes have zero strength, sample uniformly.
        if strengths.sum() > 0:
            probabilities = strengths / strengths.sum()
        else:
            probabilities = None

        sampled_indices = np.random.choice(len(subcube_ids), size=n_cubes, replace=False, p=probabilities)
        sampled_subcubes = [subcube_ids[i] for i in sampled_indices]
        print("Sampled subcube IDs:", sampled_subcubes)
        return cluster_centers, sampled_subcubes
    
    return None, None

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    loadpath = args.path
    nx, ny, nz = (args.nx, args.ny, args.nz)
    nxsl, nysl, nzsl = (args.nxsl, args.nysl, args.nzsl)
    n_clusters = args.n_clusters
    n_cubes = args.n_cubes

    if rank == 0:
        try:
            num_x = nx // nxsl
            num_y = ny // nysl
            num_z = nz // nzsl
            total_n_cubes = num_x * num_y * num_z
            if n_cubes > total_n_cubes:
                raise Exception(f"Requested number of hypercubes ({n_cubes}) > total number of cubes ({total_n_cubes}).")
        except Exception as e:
            print(f"Rank 0 encountered an error: {e}", flush=True)
            comm.Abort(1)  # Ensure all ranks exit immediately

    cluster_centers, sampled_subcubes = mpi_hypercubes(loadpath, nx, ny, nz, nxsl, nysl, nzsl, 
                                                    n_clusters=n_clusters, n_cubes=n_cubes, 
                                                    batch_size=int(4096/size), n_init=20, max_iter=10, n_iters=10)

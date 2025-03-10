"""
Sequential maxEnt-based hypercube selection for SST data.
This version loads the full dataset, performs MiniBatchKMeans clustering,
computes node strengths for each cluster, partitions the domain into subcubes,
and selects hypercubes based on aggregated node strength—all without MPI.
"""

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from scipy.stats import entropy
from collections import defaultdict

def load_data_seq(loadpath, nx, ny, nz, nskip=1):
    """
    Load the full dataset using a memmap and create a coordinate grid.
    
    Parameters:
        loadpath : str
            Path to the binary file.
        nx, ny, nz : int
            Dimensions of the full data cube.
        nskip : int, optional
            Stride to use when reading the file.
    
    Returns:
        data : np.ndarray
            The loaded data as a memmap array.
        coords : np.ndarray
            Coordinate grid of shape (nz, ny, nx, 3).
    """
    # Load data as a memmap; note the slicing to mimic your original code.
    data = np.memmap(loadpath, dtype=np.float32, mode='r', shape=(nz, ny, nx))[::nskip, ::nskip, :-2:nskip]
    # Create a coordinate grid: shape will be (nz, ny, nx, 3)
    coords = np.indices(data.shape).transpose(1, 2, 3, 0)
    return data, coords

def sequential_kmeans(data, n_clusters, batch_size, n_init, max_iter, n_iters, seed_value=0):
    """
    Run MiniBatchKMeans sequentially on the entire dataset.
    
    Parameters:
        data : np.ndarray, shape (N, n_features)
            The data for clustering.
        n_clusters : int
            Number of clusters.
        batch_size : int
            Mini-batch size.
        n_init : int
            Number of initialization runs.
        max_iter : int
            Maximum iterations per run.
        n_iters : int
            Number of additional iterations with partial_fit.
        seed_value : int
            Random seed.
    
    Returns:
        cluster_centers : np.ndarray, shape (n_clusters, n_features)
            The final cluster centers.
    """
    np.random.seed(seed_value)
    # Initial training to get centers
    kmeans_init = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size,
                                  n_init=n_init, max_iter=max_iter, init="k-means++",
                                  random_state=seed_value)
    kmeans_init.fit(data)
    cluster_centers = kmeans_init.cluster_centers_
    
    # Iteratively update centers
    for i in range(n_iters):
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size,
                                 n_init=1, max_iter=max_iter, init=cluster_centers,
                                 random_state=seed_value)
        kmeans.partial_fit(data)
        cluster_centers = kmeans.cluster_centers_
    
    # Optionally, sort centers lexicographically
    cluster_centers = cluster_centers[np.lexsort(tuple(cluster_centers[:, i] for i in range(cluster_centers.shape[1]-1, -1, -1)))]
    return cluster_centers

def compute_local_histograms(data, labels, n_clusters, num_bins, bin_range):
    """
    Compute histogram counts for each cluster.
    
    Parameters:
        data : np.ndarray, shape (N, n_features)
            The feature values (flattened) for each data point.
        labels : np.ndarray, shape (N,)
            Cluster assignments for each data point.
        n_clusters : int
            Number of clusters.
        num_bins : int
            Number of histogram bins.
        bin_range : tuple
            The (min, max) range for the histogram.
    
    Returns:
        local_histograms : list of np.ndarray
            A list (length n_clusters) of histogram counts.
    """
    local_histograms = [np.zeros(num_bins, dtype=np.int64) for _ in range(n_clusters)]
    for feature, label in zip(data, labels):
        counts, _ = np.histogram(feature, bins=num_bins, range=bin_range, density=False)
        local_histograms[label] += counts
    return local_histograms

def compute_cluster_probability_distributions(global_histograms):
    """
    Normalize each histogram to a probability distribution.
    """
    cluster_distributions = []
    for hist in global_histograms:
        total = hist.sum()
        if total > 0:
            distribution = hist / total
        else:
            distribution = hist
        cluster_distributions.append(distribution + 1e-10)  # Avoid zero probabilities
    return cluster_distributions

def compute_node_strengths(cluster_distributions):
    """
    Compute node strengths for each cluster using KL divergence.
    """
    K = len(cluster_distributions)
    node_strengths = np.zeros(K)
    for i in range(K):
        for j in range(K):
            if i != j:
                kl_div = entropy(cluster_distributions[i], cluster_distributions[j])
                node_strengths[i] += kl_div
    return node_strengths

def extract_local_subcube_totals(coords, data_node_strength, subcube_size):
    """
    Partition the 3D domain into subcubes and sum node strengths.
    
    Parameters:
        coords : np.ndarray, shape (nz, ny, nx, 3)
            The coordinate grid.
        data_node_strength : np.ndarray, shape (N,)
            Node strength for each data point.
        subcube_size : tuple (nxsl, nysl, nzsl)
            Dimensions of each subcube.
    
    Returns:
        subcube_totals : dict
            Mapping from subcube ID to summed node strength.
    """
    nxsl, nysl, nzsl = subcube_size
    subcube_totals = defaultdict(float)
    # Flatten the coordinate grid to (N, 3)
    flat_coords = coords.reshape(-1, 3)
    for (x, y, z), strength in zip(flat_coords, data_node_strength):
        cube_id = (int(x // nxsl), int(y // nysl), int(z // nzsl))
        subcube_totals[cube_id] += strength
    return subcube_totals

def maxent_hypercubes(loadpath, nx, ny, nz, nxsl, nysl, nzsl, n_clusters, n_cubes,
                      batch_size=10000, n_init=10, max_iter=100, n_iters=10, seed_value=0):
    """
    Sequential maxEnt hypercube selection.
    
    Parameters:
        loadpath : str
            Path to the binary file.
        nx, ny, nz : int
            Dimensions of the full dataset.
        nxsl, nysl, nzsl : int
            Dimensions of the subcube (sampled region).
        n_clusters : int
            Number of clusters for k-means.
        n_cubes : int
            Number of hypercubes to select.
        batch_size, n_init, max_iter, n_iters : int
            Parameters for MiniBatchKMeans.
        seed_value : int
            Random seed.
    
    Returns:
        cluster_centers : np.ndarray
            Final cluster centers.
        sampled_subcubes : list
            List of selected hypercube IDs.
    """
    # 1. Load full dataset and coordinate grid
    data, coords = load_data_seq(loadpath, nx, ny, nz)
    
    # Flatten the data for clustering; assume 1 feature per point
    n_features = 1
    flat_data = data.reshape(-1, n_features)
    
    # 2. Run sequential MiniBatchKMeans clustering on the entire dataset
    cluster_centers = sequential_kmeans(flat_data, n_clusters, batch_size, n_init, max_iter, n_iters, seed_value)
    
    # 3. Compute cluster labels for each data point
    labels = pairwise_distances_argmin(flat_data, cluster_centers)
    
    # 4. Compute local histogram for each cluster based on data values
    bin_range = (np.min(flat_data), np.max(flat_data))
    num_bins = 10
    local_histograms = compute_local_histograms(flat_data, labels, n_clusters, num_bins, bin_range)
    
    # In a sequential setup, global histograms equal the local ones
    global_histograms = local_histograms
    
    # 5. Compute probability distributions and node strengths for clusters
    cluster_distributions = compute_cluster_probability_distributions(global_histograms)
    node_strengths = compute_node_strengths(cluster_distributions)
    
    # 6. Assign node strength to each data point based on its cluster label
    data_node_strength = np.array([node_strengths[label] for label in labels])
    
    # 7. Partition the 3D domain into subcubes and sum node strengths in each subcube
    subcube_totals = extract_local_subcube_totals(coords, data_node_strength, (nxsl, nysl, nzsl))
    
    # 8. Rank subcubes based on total node strength and sample n_cubes
    subcube_ids = list(subcube_totals.keys())
    strengths = np.array([subcube_totals[cube_id] for cube_id in subcube_ids])
    
    if strengths.sum() > 0:
        probabilities = strengths / strengths.sum()
    else:
        probabilities = None
    
    sampled_indices = np.random.choice(len(subcube_ids), size=n_cubes, replace=False, p=probabilities)
    sampled_subcubes = [subcube_ids[i] for i in sampled_indices]
    print("Sampled subcube IDs:", sampled_subcubes)
    
    #return cluster_centers, sampled_subcubes
    return sampled_subcubes

# Example usage:
if __name__ == '__main__':
    # Example parameters (adjust as needed)
    loadpath = 'path/to/your/data.bin'
    nx, ny, nz = 514, 512, 256  # Example full data dimensions
    nxsl, nysl, nzsl = 32, 32, 32  # Dimensions of each hypercube
    n_clusters = 10
    n_cubes = 5
    batch_size = 1024
    n_init = 10
    max_iter = 100
    n_iters = 10
    seed_value = 42

    centers, sampled_cubes = maxent_hypercubes(loadpath, nx, ny, nz, nxsl, nysl, nzsl,
                                               n_clusters, n_cubes, batch_size, n_init,
                                               max_iter, n_iters, seed_value)

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.stats

from sklearn.cluster import KMeans
from args import args
from plotting import plot_adjacency_matrix


def subsample_random(X, num_samples, timestep, seed=[0]):
    seed[0] += 1
    if not args.noseed:
        print('random seed: ', seed[0])
        np.random.seed(seed[0])
    return np.random.choice(X.shape[1], num_samples, replace=False)


def perform_kmeans(data, num_clusters):
    """ Performs k-means clustering on the data.  """
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(data)
    labels = kmeans.labels_
    y_pred = kmeans.predict(data)
    clusters = [data[np.argwhere(y_pred == i).flatten()] for i in range(args.num_clusters)]
    clusters = [cluster.flatten() for cluster in clusters]
    return labels, clusters


def create_maxent_subsampler(cv, args):
    """ Creates a closure for subsample_maxent that captures cv and args. """

    def subsample_maxent_closure(X, num_samples, timestep):
        """ Subsampling based on maximum entropy via proportional sampling of clusters. """

        # Extract data for the given timestep
        data = cv[timestep, :].reshape(-1, 1)

        # Perform k-means clustering
        num_clusters = args.num_clusters
        cluster_labels, clusters = perform_kmeans(data, num_clusters)

        # Compute in-strength values from probability distributions and adjacency matrix
        in_strengths = compute_entropy(clusters, timestep=timestep)

        # Probabilistically select samples from clusters based on in-strength values
        probs = np.zeros((data.shape[0]))
        for i in range(num_clusters):
            probs[cluster_labels == i] = in_strengths[i]

        # Normalize probabilities
        probs = (probs - np.min(probs)) / (np.max(probs) - np.min(probs))
        probs /= np.sum(probs)

        # Randomly sample indices based on probabilities
        indices = np.random.choice(data.shape[0], num_samples, replace=False, p=probs)

        return np.array(indices)

    return subsample_maxent_closure


def compute_entropy(clusters, timestep=0, num_bins=50):
    """
    Computes probability distributions, adjacency matrix, and graph metrics.

    Parameters:
    - clusters (list of arrays): Subsets of data grouped into clusters.
    - num_bins (int): Number of bins for histograms. Default is 50.
    - timestep (int): Current timestep for labeling output. Default is 0.

    Returns:
    - in_degree (list of floats): List of in-degree relative entropy strengths
    """
    # Initialize probability distributions and bin edges
    prob_dists = []
    bin_edges_list = []

    # Specify a consistent bin range and count
    bin_range = (np.min([np.min(cluster) for cluster in clusters]),
                 np.max([np.max(cluster) for cluster in clusters]))

    # Create probability distribution of entire plane
    #counts, bin_edges = np.histogram(data, bins=num_bins, range=bin_range, density=False)
    #global_prob_dist = counts / np.sum(counts)

    # Compute cluster-level distributions
    samples_per_cluster = []
    for cluster in clusters:
        counts, bin_edges = np.histogram(cluster, bins=num_bins, range=bin_range, density=False)
        samples_per_cluster.append(np.sum(counts))
        prob_dist = counts / np.sum(counts)
        prob_dists.append(prob_dist)
        bin_edges_list.append(bin_edges)

    # Initialize adjacency matrix
    n_dists = args.num_clusters if args else len(clusters)
    adj_matrix = np.zeros((n_dists, n_dists))

    # Compute relative entropy for adjacency matrix
    for i in range(n_dists):
        for j in range(n_dists):
            p = prob_dists[i] + 1e-10  # Avoid division by zero
            q = prob_dists[j] + 1e-10
            adj_matrix[i, j] = scipy.stats.entropy(p, q)

    # Compute total and standard deviation of entropy
    total_entropy = np.sum(adj_matrix)
    stdev_entropy = np.std(adj_matrix)

    print(f"total entropy: {total_entropy}, stdev: {stdev_entropy}", flush=True)

    # Display adjacency matrix
    df = pd.DataFrame(adj_matrix)
    pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))
    print(df, flush=True)

    # Plot adjacency matrix if required
    if args.plot:
        plot_adjacency_matrix(adj_matrix, n_dists, timestep)

    # Compute strengths
    in_strengths = np.sum(adj_matrix, axis=0)
    print("in-strengths:", in_strengths)
    out_strengths = np.sum(adj_matrix, axis=1)
    print("out-strengths:", out_strengths)

    return in_strengths


def subsample_maxent(X, cv, num_samples):
    """
    Subsampling based on maximum entropy using proportional sampling.
    """
    # Perform k-means clustering
    num_clusters = args.num_clusters
    data = cv[timestep, :].reshape(-1, 1)
    labels, clusters = perform_kmeans(data, num_clusters)

    in_strengths = compute_entropy(clusters)

    # Probabilistically select from clusters according to in-strength values
    probs = np.zeros((data.shape[0]))
    for i in range(args.num_clusters):
        probs[cluster_labels == i] = in_strengths[i]

    probs = (probs - np.min(probs)) / (np.max(probs) - np.min(probs))
    probs /= np.sum(probs)

    indices = np.random.choice(data.shape[0], args.num_samples, replace=False, p=probs)
    #indices2 = np.copy(indices)

    return np.array(indices)


def build_pdf(X, nbins=10):
    """
    Build a multi-dimensional histogram (PDF) from the entire dataset X.

    Parameters
    ----------
    X : np.ndarray
        Shape (num_timesteps, samples, num_vars).
    nbins : int or list
        Number of bins for each dimension, or a list with number of bins per dimension.

    Returns
    -------
    hist : np.ndarray
        The counts in each bin (multi-dimensional).
    bin_edges : list of np.ndarray
        The edges for each dimension.
    """
    # Flatten over time and sample index -> shape (num_timesteps*samples, num_vars)
    X_flat = X.reshape(-1, X.shape[-1])

    # For simplicity, we set the same number of bins for each feature dimension
    # If you want different bins per dimension, pass a list/tuple to `bins=...`
    hist, bin_edges = np.histogramdd(X_flat, bins=nbins)

    return hist, bin_edges


def subsample_uips(X, n, hist, bin_edges):
    """
    Uniform in phase space: 
    1) Identify all bins that have nonzero counts. 
    2) Pick 'n' bins uniformly at random among the nonzero ones.
    3) From each chosen bin, pick a random data point from X that lies in that bin.
    
    Parameters
    ----------
    X : np.ndarray
        Shape (num_timesteps, samples, num_vars). 
        In practice, you'll likely pass X[t,...] for a single timestep, or the entire data.
    n : int
        Number of points to subsample.
    hist : np.ndarray
        Histogram from build_phase_space_pdf.
    bin_edges : list of np.ndarray
        The bin edges for each dimension (output of np.histogramdd).

    Returns
    -------
    indices : array-like
        Indices (1D) in X[t,...] that correspond to the chosen subsampled points.
    """
    # Flatten to shape (N, num_vars) to quickly find which bin each point belongs to
    X_flat = X.reshape(-1, X.shape[-1])
    N = X_flat.shape[0]

    # (A) Identify bin index for each point in X
    #     np.digitize returns the bin index along each dimension.
    #     We then combine them into a single "bin ID".
    bin_indices_per_dim = []
    for dim in range(X_flat.shape[1]):
        # digitize returns indices in [1..len(bin_edges[dim])], we shift to [0..]
        # (also be mindful of points on bin_edges boundaries)
        bi = np.digitize(X_flat[:, dim], bin_edges[dim]) - 1
        # Clip out-of-bounds if needed (i.e. points exactly at the max can be len(bin_edges[dim]) - 1)
        bi = np.clip(bi, 0, len(bin_edges[dim]) - 2)
        bin_indices_per_dim.append(bi)
    bin_indices_per_dim = np.array(bin_indices_per_dim).T  # shape (N, num_vars)

    # Convert multi-dim bin indices into a “single integer” bin ID.
    # One approach is to unravel them with np.ravel_multi_index, but we need the shape from hist.
    bin_id = np.ravel_multi_index(tuple(bin_indices_per_dim.T), dims=hist.shape)

    # (B) Identify which bins are nonzero
    nonzero_bin_ids = np.where(hist.ravel() > 0)[0]

    # (C) We pick n bins uniformly at random among these nonzero bins
    chosen_bins = np.random.choice(nonzero_bin_ids, size=n, replace=True)

    # (D) For each chosen bin, pick a random data point that lies in that bin
    chosen_indices = []
    for cb in chosen_bins:
        # all points that lie in bin cb
        candidate_points = np.where(bin_id == cb)[0]
        if len(candidate_points) == 0:
            # fallback if the bin is truly empty, though we said it's nonzero
            # you could skip or pick a fallback:
            continue
        # pick one from candidate_points randomly
        chosen_pt = np.random.choice(candidate_points)
        chosen_indices.append(chosen_pt)

    # chosen_indices are indices in the flattened array
    # If you want them as indices in the original shape (time, sample), 
    # you’d have to invert that. 
    # But typically you'd do this step once globally (not per timestep).
    # For demonstration, we’ll just return the flattened indices for now.
    chosen_indices = np.array(chosen_indices, dtype=int)

    return chosen_indices

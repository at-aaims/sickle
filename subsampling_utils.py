import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.stats

from sklearn.cluster import KMeans
from args import args


def check_and_create_dirs(directory):
    """ Checks if a directory exists, and creates it if it doesn't.  """
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_data(path, args):
    """ Loads data based on the specified type in args.  """
    from dataloader import DataLoaderCSV, DataLoaderNPZ, DataLoaderSSTBinary, DataLoaderOF

    if args.dtype == "csv":
        dl = DataLoaderCSV(args.path, dims=args.dims)
    elif args.dtype == "npz":
        dl = DataLoaderNPZ(args.path)
    elif args.dtype == "sst-binary":
        dl = DataLoaderSSTBinary(args)
    else:
        dl = DataLoaderOF(args.path, dims=args.dims)

    x, y, z = dl.load_xyz()
    X, Y, cv = dl.load_multiple_timesteps(
        args.write_interval, args.num_timesteps, target=args.target, cv=args.cluster_var
    )
    return X, Y, cv, x, y, z


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

    print("*** counts: ", samples_per_cluster)

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

    print(f"total entropy: {total_entropy}, stdev: {stdev_entropy}")

    # Display adjacency matrix
    df = pd.DataFrame(adj_matrix)
    pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))
    print(df)

    # Plot adjacency matrix if required
    if args.plot:
        plt.clf()
        plt.rcParams.update({'font.size': 18})
        plt.figure(figsize=(12, 10), facecolor='1')
        ticks = np.arange(n_dists)
        plt.xticks(ticks)
        plt.yticks(ticks)
        plt.xlabel('Cluster number')
        plt.ylabel('Cluster number')
        plt.imshow(adj_matrix, cmap='inferno')
        cbar = plt.colorbar(); cbar.set_label(r'relative entropy, $D$')
        plt.axis('equal')
        plt.savefig(os.path.join(plot_dir, f'adj_matrix_{timestep:04d}.png'), dpi=100)

    # Create graph from adjacency matrix
    #graph = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph())

    # Compute strengths
    in_strengths = np.sum(adj_matrix, axis=0)
    out_strengths = np.sum(adj_matrix, axis=1)
    print("in-strengths:", in_strengths)
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


def plot_samples(indices, x, y, z, args):
    """
    Plots subsampled data based on input dimensions and settings.
    """
    plt.clf()
    plt.rcParams.update({'font.size': 10})

    if args.dims == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = plt.subplot(111, projection='3d')
        ax.view_init(elev=20., azim=-35)

        if args.dtype in ['npz', 'sst-binary']:
            x_indices, y_indices, z_indices = np.unravel_index(
                indices, (x.shape[0], y.shape[0], z.shape[0])
            )
            ax.scatter(
                x[x_indices], z[z_indices], y[y_indices], c='k', s=2, alpha=0.5
            )
        else:
            ax.scatter(x[indices], y[indices], z[indices], c='k', s=2, alpha=0.5)

    else:
        plt.figure(figsize=(9, 2))
        plt.scatter(x[indices], y[indices], c='k', s=2, alpha=0.5)
        plt.xlim([-25, 65])
        plt.ylim([-10, 10])
        plt.axis('equal')

    plt.savefig(os.path.join(args.plot_dir, 'subsample_plot.png'), dpi=100, bbox_inches='tight')

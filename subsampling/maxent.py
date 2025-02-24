# subsampling/maxent.py
import numpy as np
from sklearn.cluster import KMeans
import scipy.stats
import pandas as pd
from .base import Subsampler
from plotting import plot_adjacency_matrix

class MaxentSubsampler(Subsampler):
    def __init__(self, data, args, cv=None):
        super().__init__(data, args)
        if cv is None:
            raise ValueError("MaxentSubsampler requires a cv array")
        self.cv = cv

    def sample(self, num_samples, timestep):
        # Use cv for clustering instead of self.data.
        data = self.cv[timestep, :].reshape(-1, 1)
        num_clusters = self.args.num_clusters
        
        # Perform k-means clustering.
        labels, clusters = self.perform_kmeans(data, num_clusters)
        
        # Compute in-strength values (entropy) from the clusters.
        in_strengths = self.compute_entropy(clusters, timestep=timestep)
        
        # Create probabilities based on the in-strength values.
        probs = np.zeros(data.shape[0])
        for i in range(num_clusters):
            probs[labels == i] = in_strengths[i]
        
        # Normalize probabilities.
        probs = (probs - np.min(probs)) / (np.max(probs) - np.min(probs))
        probs /= np.sum(probs)
        
        # Randomly sample indices based on the computed probabilities.
        indices = np.random.choice(data.shape[0], num_samples, replace=False, p=probs)
        return np.array(indices)

    def perform_kmeans(self, data, num_clusters):
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        kmeans.fit(data)
        labels = kmeans.labels_
        clusters = [data[labels == i].flatten() for i in range(num_clusters)]
        return labels, clusters

    def compute_entropy(self, clusters, timestep=0, num_bins=50):
        prob_dists = []
        bin_edges_list = []
        bin_range = (min([np.min(cluster) for cluster in clusters]),
                     max([np.max(cluster) for cluster in clusters]))
        for cluster in clusters:
            counts, bin_edges = np.histogram(cluster, bins=num_bins, range=bin_range, density=False)
            prob_dist = counts / np.sum(counts)
            prob_dists.append(prob_dist)
            bin_edges_list.append(bin_edges)
        n_clusters = len(clusters)
        adj_matrix = np.zeros((n_clusters, n_clusters))
        for i in range(n_clusters):
            for j in range(n_clusters):
                p = prob_dists[i] + 1e-10  # avoid division by zero
                q = prob_dists[j] + 1e-10
                adj_matrix[i, j] = scipy.stats.entropy(p, q)
        in_strengths = np.sum(adj_matrix, axis=0)
        if self.args.plot:
            df = pd.DataFrame(adj_matrix)
            print(df)
            plot_adjacency_matrix(adj_matrix, n_clusters, timestep)
        return in_strengths

# subsampling/maxent.py

import numpy as np
from sklearn.cluster import KMeans
import scipy.stats
import pandas as pd
from .base import Subsampler
# Assume that plotting utilities have been moved to an appropriate module.
from plotting import plot_adjacency_matrix

class MaxentSubsampler(Subsampler):
    def sample(self, num_samples, timestep):
        # Extract data for the current timestep (assuming self.data is time-indexed).
        data = self.data[timestep].reshape(-1, 1)
        num_clusters = self.args.num_clusters
        
        # Perform k-means clustering.
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        kmeans.fit(data)
        cluster_labels = kmeans.labels_
        clusters = [data[cluster_labels == i].flatten() for i in range(num_clusters)]
        
        # Compute in-strengths (entropy) from clusters.
        in_strengths = self.compute_entropy(clusters, timestep)
        
        # Create probabilities proportional to in-strengths.
        probs = np.zeros(data.shape[0])
        for i in range(num_clusters):
            probs[cluster_labels == i] = in_strengths[i]
        probs = (probs - probs.min()) / (probs.max() - probs.min())
        probs /= np.sum(probs)
        
        # Randomly sample indices based on the computed probabilities.
        indices = np.random.choice(data.shape[0], num_samples, replace=False, p=probs)
        return indices

    def compute_entropy(self, clusters, timestep, num_bins=50):
        # Compute probability distributions for each cluster.
        prob_dists = []
        bin_edges_list = []
        bin_range = (min(cluster.min() for cluster in clusters),
                     max(cluster.max() for cluster in clusters))
        
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

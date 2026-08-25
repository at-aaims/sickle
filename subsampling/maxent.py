import os
import numpy as np
from sklearn.cluster import KMeans
import scipy.stats
import pandas as pd
from .base import Subsampler
from plotting import plot_adjacency_matrix, plot_kmeans_3d, plot_prob_dists, \
                     plot_cluster_histogram, plot_contour_box_3d, plot_samples, \
                     plot_points_3d

class MaxentSubsampler(Subsampler):
    def __init__(self, data, args, **kwargs):
        super().__init__(data, args)

        # Extract coords from kwargs
        coords = kwargs.get('coords')
        if coords is None:
            raise ValueError("MaxentSubsampler requires coords")
        self.coords = coords

        # Extract cv from kwargs
        cv = kwargs.get('cv')
        if cv is None:
            raise ValueError("MaxentSubsampler requires a cv array")
        self.cv = cv

    def sample(self, num_samples, timestep):
        # Use cv for clustering instead of self.data.
        num_cvs = len(self.args.cluster_var)
        data = self.cv[timestep, :].reshape(-1, num_cvs)
        num_clusters = self.args.num_clusters
        
        # Perform k-means clustering.
        labels, clusters = self.perform_kmeans(data, num_clusters)
        
        # Compute in-strength values (entropy) from the clusters.
        in_strengths = self.compute_entropy(clusters, timestep=timestep)
        
        # Create probabilities based on the in-strength values.
        probs = np.zeros(data.shape[0])
        for i in range(num_clusters):
            probs[labels == i] = in_strengths[i]
        
        # Normalize into a probability distribution. in_strengths (KL-divergence
        # based) are already non-negative, so summing suffices; min-max scaling
        # here would floor the lowest in-strength cluster to exactly zero
        # probability, permanently excluding it from ever being sampled.
        probs /= np.sum(probs)
        
        # Randomly sample indices based on the computed probabilities.
        indices = np.random.choice(data.shape[0], num_samples, replace=False, p=probs)

        # Generate additional plots if enabled.
        if self.args.plot:
            self.generate_plots(data, labels, clusters, indices, timestep, num_samples)

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
        df = pd.DataFrame(adj_matrix)
        print(df)

        if self.args.plot:
            plot_adjacency_matrix(adj_matrix, n_clusters, timestep)
        return in_strengths

    def _global_point_coords(self, hypercube_ids):
        """
        Reconstruct grid-index (x, y, z) coordinates for every point in the
        concatenated multi-hypercube data array, in the same
        hypercube-major / row-major-local order used when the dataloader
        built X/Y/cv (see hypercubes/hypercube_manager.py:load_hypercubes).

        hypercube_ids: (num_hypercubes, 3) array of (ix, iy, iz) block
        coordinates -- hypercube h's corner sits at
        (ix*nxsl, iy*nysl, iz*nzsl) in full-grid index space.
        """
        nxsl, nysl, nzsl = self.args.nxsl, self.args.nysl, self.args.nzsl
        lx, ly, lz = np.meshgrid(np.arange(nxsl), np.arange(nysl), np.arange(nzsl), indexing='ij')
        lx, ly, lz = lx.ravel(), ly.ravel(), lz.ravel()

        gx, gy, gz = [], [], []
        for (ix, iy, iz) in hypercube_ids:
            gx.append(ix * nxsl + lx)
            gy.append(iy * nysl + ly)
            gz.append(iz * nzsl + lz)
        return np.concatenate(gx), np.concatenate(gy), np.concatenate(gz)

    def generate_plots(self, data, labels, clusters, maxent_indices, timestep, num_samples):
        """
        Generate plots for KMeans clustering, cluster histogram, and
        probability distributions comparing the full dataset, random sampling,
        and MaxEnt sampling.

        Parameters:
          data           : The cv data at the given timestep.
          labels         : KMeans cluster labels.
          clusters       : List of cluster arrays.
          maxent_indices : Indices chosen via MaxEnt sampling.
          timestep       : Current timestep (for filenames).
          num_samples    : Number of samples requested.
        """
        # self.coords is the local (nxsl, nysl, nzsl) axis grid of a single
        # hypercube; labels/maxent_indices span ALL num_hypercubes cubes
        # concatenated together, so plot_kmeans_3d/plot_samples (which
        # rebuild a meshgrid from those axes) only work when there's exactly
        # one hypercube. When the dataloader recorded each hypercube's true
        # global block position, use that instead so every hypercube's
        # points land at their real location.
        hypercube_ids_by_ts = getattr(self.args, 'hypercube_ids_per_timestep', None)
        hypercube_ids = hypercube_ids_by_ts[timestep] if hypercube_ids_by_ts is not None else None

        if hypercube_ids is not None:
            gx, gy, gz = self._global_point_coords(hypercube_ids)

            plot_points_3d(gx, gy, gz, labels, self.args.plot_dir,
                            filename=f'kmeans_{timestep:04d}.png',
                            title=f'KMeans clustering of {self.args.cluster_var}')

            plot_points_3d(gx[maxent_indices], gy[maxent_indices], gz[maxent_indices],
                            labels[maxent_indices], self.args.plot_dir,
                            filename=f'subsample_plot_t{timestep:04d}.png',
                            title='MaxEnt subsampled points')

            # A single filled-contour box only makes sense over one
            # contiguous grid, so draw one per hypercube instead of trying
            # to force disjoint boxes into a single regular grid.
            nxsl, nysl, nzsl = self.args.nxsl, self.args.nysl, self.args.nzsl
            num_pts = nxsl * nysl * nzsl
            for h, (ix, iy, iz) in enumerate(hypercube_ids):
                x_h = ix * nxsl + np.arange(nxsl)
                y_h = iy * nysl + np.arange(nysl)
                z_h = iz * nzsl + np.arange(nzsl)
                contour_data_h = self.cv[timestep, h * num_pts:(h + 1) * num_pts, 0]
                plot_contour_box_3d(x_h, y_h, z_h, contour_data_h, timestep, suffix=f'_h{h}')
        else:
            x, y, z = self.coords

            # Plot the KMeans 3D scatter
            plot_kmeans_3d(x, y, z, labels, timestep, self.args.plot_dir, self.args.cluster_var)

            # Plot the 3D contour box. Only the first cluster_var component
            # is shown -- a filled contour is inherently a single scalar
            # field, so this doesn't attempt to reshape all cluster_var
            # channels together (that previously raised a reshape error
            # whenever cluster_var had more than one entry).
            contour_data = self.cv[timestep, :, 0]
            plot_contour_box_3d(x, y, z, contour_data, timestep)

            # Plot samples
            plot_samples(maxent_indices, labels[maxent_indices], x, y, z, timestep)

        # Plot the cluster histogram.
        plot_cluster_histogram(labels, self.args.num_clusters, timestep, self.args.plot_dir)

        # Compute probability distributions for comparison:
        num_bins = 50
        bin_range = (min([np.min(cluster) for cluster in clusters]),
                     max([np.max(cluster) for cluster in clusters]))
        # Global probability distribution from the entire cv data at this timestep.
        global_counts, bin_edges = np.histogram(data, bins=num_bins, range=bin_range, density=False)
        global_prob_dist = global_counts / np.sum(global_counts)

        # Random sampling: select random indices and compute histogram.
        rand_indices = np.random.choice(data.shape[0], num_samples, replace=False)
        rand_counts, _ = np.histogram(data[rand_indices], bins=num_bins, range=bin_range, density=False)
        random_prob_dist = rand_counts / np.sum(rand_counts)

        # MaxEnt sampling probability distribution from the selected indices.
        maxent_counts, _ = np.histogram(data[maxent_indices], bins=num_bins, range=bin_range, density=False)
        maxent_prob_dist = maxent_counts / np.sum(maxent_counts)

        # Plot the probability distributions.
        plot_prob_dists(bin_edges, global_prob_dist, random_prob_dist, maxent_prob_dist,
                        timestep, self.args.cluster_var)

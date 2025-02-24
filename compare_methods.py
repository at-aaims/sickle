import numpy as np
import matplotlib.pyplot as plt
import os

from args import args
from subsampling import get_subsampler
from dataloaders import load_data
from scipy.stats import gaussian_kde


if __name__ == "__main__":

    # Load the data
    X, Y, cv, x, y, z = load_data(args)
    num_timesteps = X.shape[0] // args.window * args.window + 1
    print(f"X: {X.shape}; Y: {Y.shape}; cv: {cv.shape}; x: {x.shape}; y: {y.shape}; z: {z.shape}; num_timesteps: {num_timesteps}")

    ts = 4 # timestep

    histograms = {}  # Store histogram data for plotting
    for method in ["full", "random", "uips", "maxent"]:
        # Get the appropriate subsample function
        subsample_fn = get_subsample_fn(method, cv, args)
        
        # Apply the subsampling function to get indices
        indices = subsampler.sample(X, args.num_samples, ts)
        
        # Extract the subsampled cluster variable
        subsampled_cv = cv[ts, indices]
        histograms[method] = subsampled_cv

    print("Data Lengths:")
    for method, subsampled_cv in histograms.items():
        print(f"{method}: {len(subsampled_cv)}")

    # Use a predefined colormap
    colormap = plt.cm.tab10  # You can use 'tab10', 'viridis', 'plasma', etc.
    colors = [colormap(i) for i in range(len(histograms))]
    print(colors)

    # Plot the histograms
    plt.figure(figsize=(6, 4))
    for i, (method, subsampled_cv) in enumerate(histograms.items()):
        color = colors[i]

        xmin, xmax = np.min(subsampled_cv), np.max(subsampled_cv)
        bins = np.linspace(xmin, xmax, args.bins)

        plt.hist(subsampled_cv, bins=bins, alpha=0.6, label=method, density=True, color=color)

        kde = gaussian_kde(subsampled_cv)
        kde_values = kde(bins)
        #kde_values_normalized = kde_values / np.max(kde_values)
        plt.plot(bins, kde_values, label=method, color=color)

        values, bin_edges = np.histogram(subsampled_cv, bins=bins, density=True)
        area = np.sum(values * np.diff(bin_edges))
        print(f"Total Area: {area}")

    fs = 10

    xlabel = {'p': 'Pressure', 'pv': 'Potential Vorticity'}

    # Set plot title and labels
    #plt.title("Histogram of Potential Vorticity")
    plt.title("P1F4R32", fontsize=fs)
    #plt.title("OF2DCyl")
    #plt.xlabel("Potential Vorticity", fontsize=fs)
    plt.xlabel(xlabel[args.cluster_var[0]], fontsize=fs)
    #plt.xlabel("Vorticity")
    plt.ylabel("Probability Density", fontsize=fs)
    #plt.yscale('log')

    # Set tick label size
    plt.tick_params(axis='both', labelsize=fs)

    ax = plt.gca()
    ax.tick_params(axis='x', labelsize=fs)
    ax.tick_params(axis='y', labelsize=fs)

    # Add legend
    plt.legend(loc='upper right', ncol=2, handlelength=1.0, fontsize=fs)

    # Set x-axis limits
    plt.xlim(xmin, xmax)

    # Save and show the plot
    plt.tight_layout()
    plt.savefig(os.path.join(args.plot_dir, f"subsampling_methods_histograms_t{ts}_ns{args.num_samples}.png"))
    plt.close()

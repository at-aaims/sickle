import numpy as np
import matplotlib.pyplot as plt
import os

from args import args
from helpers import load_data
from algorithms import create_maxent_subsampler, subsample_random, subsample_uips, build_pdf
from scipy.stats import gaussian_kde


def get_subsample_fn(method, cv, args):
    if method == "maxent":
        return create_maxent_subsampler(cv, args)
    elif method == "random":
        return subsample_random
    elif method == "uips":
        def subsample_fn(X, n, t):
            X_local = X[t]
            hist, bin_edges = build_pdf(X_local, nbins=args.bins)
            return subsample_uips(X_local[None, ...], n, hist, bin_edges)
        return subsample_fn
    elif method == "full":
        return lambda X, n, t: np.arange(X.shape[1])
    else:
        raise ValueError(f"Unsupported sampling method: {method}")

if __name__ == "__main__":

    # Load the data
    X, Y, cv, x, y, z = load_data(args.path, args)
    num_timesteps = X.shape[0] // args.window * args.window + 1
    print(f"X: {X.shape}; Y: {Y.shape}; cv: {cv.shape}; x: {x.shape}; y: {y.shape}; z: {z.shape}; num_timesteps: {num_timesteps}")

    ts = 0 # timestep

    histograms = {}  # Store histogram data for plotting
    for method in ["full", "random", "uips", "maxent"]:
        # Get the appropriate subsample function
        subsample_fn = get_subsample_fn(method, cv, args)
        
        # Apply the subsampling function to get indices
        indices = subsample_fn(X, args.num_samples, ts)
        
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
    xmin, xmax = -5, 5
    bins = np.linspace(xmin, xmax, args.bins)

    # Plot the histograms
    plt.figure(figsize=(6, 4))
    for i, (method, subsampled_cv) in enumerate(histograms.items()):
        color = colors[i]
        #plt.hist(subsampled_cv, bins=args.bins, alpha=0.6, label=method, density=True, color=color)
        plt.hist(subsampled_cv, bins=bins, alpha=0.6, label=method, density=True, color=color)

        kde = gaussian_kde(subsampled_cv)
        plt.plot(bins, kde(bins), label=method, color=color)

        values, bin_edges = np.histogram(subsampled_cv, bins=bins, density=True)
        area = np.sum(values * np.diff(bin_edges))
        print(f"Total Area: {area}")

    # Set plot title and labels
    #plt.title("Histogram of Potential Vorticity")
    plt.title("P1F4R32", fontsize=12)
    #plt.title("OF2DCyl")
    plt.xlabel("Potential Vorticity", fontsize=10)
    #plt.xlabel("Vorticity")
    plt.ylabel("Density", fontsize=10)
    #plt.yscale('log')
    #plt.legend()
    plt.legend(loc='upper right', ncol=2, handlelength=1.0, fontsize=8)


    # Set x-axis limits
    plt.xlim(xmin, xmax)

    # Save and show the plot
    plt.tight_layout()
    plt.savefig(os.path.join(args.plot_dir, "subsampling_methods_histograms.png"))
    plt.close()

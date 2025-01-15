import numpy as np
import matplotlib.pyplot as plt
from args import args
from helpers import load_data
from algorithms import create_maxent_subsampler, subsample_random, subsample_uips, build_pdf

def get_subsample_fn(method, cv, args):
    if method == "maxent":
        return create_maxent_subsampler(cv, args)
    elif method == "random":
        return subsample_random
    elif method == "uips":
        def subsample_fn(X, n, t):
            X_local = X[t]
            hist, bin_edges = build_pdf(X_local, nbins=20)
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

    ts = 0  # Use timestep 0 for this example

    histograms = {}  # Store histogram data for plotting
    for method in ["maxent", "random", "uips", "full"]:
        # Get the appropriate subsample function
        subsample_fn = get_subsample_fn(method, cv, args)
        
        # Apply the subsampling function to get indices
        indices = subsample_fn(X, args.num_samples, ts)
        
        # Extract the subsampled cluster variable
        subsampled_cv = cv[ts, indices]
        histograms[method] = subsampled_cv  # Store subsampled data

    # Plot the histograms
    plt.figure(figsize=(6, 4))
    for method, subsampled_cv in histograms.items():
        plt.hist(subsampled_cv, bins=30, alpha=0.6, label=method, density=True)

    # Set plot title and labels
    plt.title("Comparison of Subsampling Methods (Cluster Variable)")
    plt.xlabel("Cluster Variable (cv)")
    plt.ylabel("Density")
    plt.legend()

    # Set x-axis limits
    plt.xlim(-5, 5)

    # Save and show the plot
    plt.tight_layout()
    plt.savefig("subsampling_methods_histograms.png")
    plt.close()

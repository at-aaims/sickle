import numpy as np
import os
import pyvista as pv
import sys

from args import args 
from algorithms import create_maxent_subsampler, subsample_random, subsample_uips, build_pdf
from constants import FieldPredictionType
from dataloaders import load_data
from helpers import check_and_create_dirs
from plotting import plot_samples, plot2d_contour, plot_corner


def save_vtu(Xout, x, y, z, indices, output_dir, fileprefix):
    """
    Saves all timesteps in VTU (Unstructured Grid) format for time-series visualization in ParaView.

    Xout: shape (num_timesteps, num_samples, num_features)  # Features = (u, v, w, rho)
    x, y, z: shape (32,) - Grid coordinates (1D arrays)
    indices: shape (num_timesteps, num_samples) - Indices of subsampled points in a flattened (32,32,32) grid
    """
    num_timesteps = Xout.shape[0]  # Number of time steps
    num_samples = Xout.shape[1]  # Number of subsampled points per timestep

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Convert x, y, z into a full (32, 32, 32) 3D grid
    X_full, Y_full, Z_full = np.meshgrid(x, y, z, indexing="ij")  # Shape: (32, 32, 32)
    
    # Step 2: Flatten to (32768, 3) so that we can index it correctly
    grid_points = np.column_stack([X_full.ravel(), Y_full.ravel(), Z_full.ravel()])  # Shape: (32768, 3)

    # Store filenames for the PVD time series file
    vtu_filenames = []

    for timestep in range(num_timesteps - 1):
        # Step 3: Extract subsampled spatial coordinates using indices
        print(timestep, indices[timestep])
        subsampled_points = grid_points[indices[timestep]]  # Shape: (num_samples, 3)

        # Define connectivity: each point is a separate vertex cell
        cells = np.hstack([np.ones((num_samples, 1), dtype=int), np.arange(num_samples).reshape(-1, 1)]).flatten()

        # Create UnstructuredGrid with proper connectivity
        point_cloud = pv.UnstructuredGrid(cells, np.full(num_samples, pv.CellType.VERTEX), subsampled_points)

        # Store velocity (u, v, w) and density (rho) as point data
        feature_names = ["u", "v", "w", "rho"]  # Rename features for clarity in ParaView
        for i, name in enumerate(feature_names):
            point_cloud.point_data[name] = Xout[timestep, :, i]

        # Set default color variable in ParaView (optional)
        point_cloud.active_scalars_name = "rho"  # Default to coloring by density

        # Save each timestep as a VTU file
        vtu_filename = os.path.join(output_dir, f"subsampled_{fileprefix}_t{timestep}.vtu")
        point_cloud.save(vtu_filename)
        vtu_filenames.append((timestep, vtu_filename))

    # Generate a PVD file for time animation
    pvd_filename = os.path.join(output_dir, f"subsampled_{fileprefix}.pvd")
    with open(pvd_filename, "w") as pvd_file:
        pvd_file.write('<VTKFile type="Collection" version="1.0">\n')
        pvd_file.write('  <Collection>\n')
        for timestep, vtu_file in vtu_filenames:
            pvd_file.write(f'    <DataSet timestep="{timestep}" file="{os.path.basename(vtu_file)}"/>\n')
        pvd_file.write('  </Collection>\n')
        pvd_file.write('</VTKFile>\n')

    print(f'Saved all timesteps in {pvd_filename}')


def extract_yz_plane(X, timestep, feature_index, x_index, nx=128, ny=64, nz=128):
    """Extracts the y-z plane for a given x-index at a specific timestep and feature."""
    data_slice = X[timestep, :, feature_index]
    data_3d = data_slice.reshape(nx, ny, nz)
    return data_3d[x_index, :, :]


def subsample_data(X, Y, x, y, z, subsample_fn, args):
    num_timesteps = X.shape[0] // args.window * args.window + 1
    print(f"num_timesteps: {num_timesteps}")

    Xout = np.zeros((num_timesteps, args.num_samples, X.shape[2]))

    if args.field_prediction_type == FieldPredictionType.GLOBAL:
        Yout = np.zeros((num_timesteps, 1, Y.shape[2]))
    elif args.field_prediction_type == FieldPredictionType.LOCAL:
        if args.method == "full":
            raise Exception("For baseline full field input, prediction cannot be subsampled. Change `args.target`.")
        Yout = np.zeros((num_timesteps, args.num_samples, Y.shape[2]))
    elif args.field_prediction_type == FieldPredictionType.FULL:
        Yout = np.zeros((num_timesteps, Y.shape[1], Y.shape[2]))
    else:
        raise Exception("Enter a valid `args.target`.")

    subsampled_indices_list = []  # Store subsampled indices for later use

    for timestep in range(0, num_timesteps - args.window, args.window):
        indices = subsample_fn(X, args.num_samples, timestep)
        subsampled_indices_list.append(indices)

        if args.plot and args.method != "full":
            plot_samples(indices, x, y, z, timestep, args)

        for sub_timestep in range(args.window):
            ts = timestep + sub_timestep
            Xout[ts, :, :] = X[ts, indices, :]

            if args.field_prediction_type == FieldPredictionType.GLOBAL:
                subsampled_Y = Y[ts, :]
            elif args.field_prediction_type == FieldPredictionType.FULL:
                subsampled_Y = Y[ts, :, :]
            else:
                subsampled_Y = Y[ts, indices, :]
            Yout[ts, :] = subsampled_Y

            if args.plot:
                if args.method == "full":
                    yz_plane = extract_yz_plane(Xout, timestep, 3, 0, nx=args.nxsl, ny=args.nysl, nz=args.nzsl)
                    plot2d_contour(yz_plane, y, z, ts)

    return Xout, Yout, np.array(subsampled_indices_list)


if __name__ == "__main__":

    # Ensure required directories exist
    check_and_create_dirs(args.output_dir)
    check_and_create_dirs(args.plot_dir)

    # Load the data
    X, Y, cv, x, y, z = load_data(args)
    num_timesteps = X.shape[0] // args.window * args.window + 1
    print(f"X: {X.shape}; Y: {Y.shape}; cv: {cv.shape}; x: {x.shape}; y: {y.shape}; z: {z.shape}; num_timesteps: {num_timesteps}")

    if args.method == "full": 
        args.num_samples = X.shape[1]

    if args.method == "maxent":
        # Define the subsampling function for maximum entropy
        subsample_fn = create_maxent_subsampler(cv, args)
    elif args.method == "full":
        # No subsampling, use all indices
        subsample_fn = lambda X, n, t: np.arange(X.shape[1])
    elif args.method == "uips":
        # Phase-space sampling
        if args.plot: 
            X_flat = X.reshape(-1, X.shape[-1])
            plot_corner(X_flat)

        def subsample_fn(X, n, t):
            X_local = X[t]
            hist, bin_edges = build_pdf(X_local, nbins=args.bins)
            return subsample_uips(X_local[None, ...], n, hist, bin_edges)

    else: 
        # Random subsampling
        subsample_fn = lambda X, n, t: subsample_random(X, n, t)

    # Perform subsampling
    Xout, Yout, indices_list = subsample_data(X, Y, x, y, z, subsample_fn, args)
    print(f"Xout: {Xout.shape}; Yout: {Yout.shape}")

    # Reshape Xout and Yout to 1D or 3D based on args.method and args.field_prediction_type
    if args.method == "full":
        Xout = Xout.reshape(num_timesteps, len(x), len(y), len(z), Xout.shape[2])
    if args.field_prediction_type == FieldPredictionType.FULL:
        Yout = Yout.reshape(num_timesteps, len(x), len(y), len(z), Yout.shape[2])

    # Save output
    fileprefix = f"nxsl{args.nxsl}-nysl{args.nysl}-nzsl{args.nzsl}-ns{args.num_samples}-window{args.window}_method-{args.method}"
    outfilename = f"subsampled_{fileprefix}.npz"
    outfile = os.path.join(args.output_dir, outfilename)
    np.savez(outfile, X=Xout, Y=Yout, x=x, y=y, z=z)
    print(f'Subsampled data saved to {outfile}')

    # Save to VTK
    save_vtu(Xout, x, y, z, indices_list, args.output_dir, fileprefix)

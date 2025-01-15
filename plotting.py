import corner
import matplotlib.pyplot as plt
import numpy as np
import os
from args import args


def plot_samples(indices, x, y, z, ts, args):
    """ Plot subsampled data based on input dimensions and settings. """
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

    fn = f'subsample_plot_t{ts:04d}.png'
    print(f'Creating {fn}')
    plt.savefig(os.path.join(args.plot_dir, fn), dpi=100, bbox_inches='tight')
    plt.close()


def plot2d_contour(data, y, z, timestep):
    """ Plots a 2D plane from 1D data, y, and z coordinates. """
    # Reshape the 1D data to match the grid defined by y and z
    data_2d = data.reshape(len(y), len(z))

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.contourf(z, y, data_2d, cmap="viridis")
    plt.colorbar(label="Value")
    plt.xlabel("z")
    plt.ylabel("y")
    plt.title("yz plane")
    plt.savefig(os.path.join(args.plot_dir, f'plot2d_{timestep:04d}.png'), dpi=100)
    plt.close()


def plot_kmeans(x, y, labels):
    plt.figure(figsize=(9, 2))
    plt.scatter(x, y, c=labels, marker='.', cmap='tab10')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'KMeans clustering of {args.cluster_var}')
    plt.colorbar()
    plt.savefig(os.path.join(args.plot_dir, f'kmeans_{timestep:04d}.png'), dpi=100)
    plt.close()


def plot_adjacency_matrix(adj_matrix, n_dists, timestep):
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
    plt.savefig(os.path.join(args.plot_dir, f'adj_matrix_{timestep:04d}.png'), dpi=100)
    plt.close()


def plot_histograms(X_train, X_test, Y_train, Y_test):
    bins = 50
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].hist(X_train[:,0], bins=bins, alpha=0.8, edgecolor='lightblue', color='lightblue')
    ax[0].hist(X_test[:,0], bins=bins, alpha=0.5, edgecolor='red', color='red')
    ax[0].set_title('Histogram of X[0]')
    ax[1].hist(X_train[:,1], bins=bins, alpha=0.8, edgecolor='lightblue', color='lightblue')
    ax[1].hist(X_test[:,1], bins=bins, alpha=0.5, edgecolor='red', color='red')
    ax[1].set_title('Histogram of X[1]')
    ax[2].hist(Y_train, bins=bins, alpha=0.8, edgecolor='lightblue', color='lightblue')
    ax[2].hist(Y_test, bins=bins, alpha=0.5, edgecolor='red', color='red')
    ax[2].set_title('Histogram of Y')
    plt.tight_layout()
    plt.savefig(os.path.join(args.plot_dir, f'histogram_train_test.png'), dpi=100)
    plt.close()


def plot_learning_curve(train_loss_history, val_loss_history):
    plt.figure(figsize=(10,5))
    plt.rcParams.update({'font.size': 18})
    plt.title('Learning curve')
    plt.plot(train_loss_history, label='training')
    plt.plot(val_loss_history, label='validation',alpha=0.5)
    plt.yscale('log')
    plt.xlabel('Epoch'); plt.ylabel(r'Loss ($mse$)')
    plt.legend(frameon=False);
    plt.savefig(os.path.join(args.plot_dir, f'ML_loss-curves.png'), dpi=100)
    plt.close()


def plot_ML_outputs(Y_test_ML, Y_test):
    print(f"Plotting data: {Y_test_ML.shape}, {Y_test.shape}")
    nvar = Y_test.shape[1] # num variables in Y
    plt.clf()
    plt.rcParams.update({'font.size': 15})

    # Calculate the number of rows and columns for the subplots
    ncols = 2  # Set number of columns (you can adjust this as needed)
    nrows = int(np.ceil(nvar / ncols))  # Calculate number of rows needed
    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 6 * nrows / 2), sharex=True, facecolor="1")
    axs = axs.ravel()  # Flatten the axs array for easy indexing
    # Plot the variables
    for i in range(nvar):
        axs[i].scatter(Y_test[:, i], Y_test_ML[:, i], s=20)
        # Calculate the min and max for the current variable
        min_val = min(Y_test[:, i].min(), Y_test_ML[:, i].min())
        max_val = max(Y_test[:, i].max(), Y_test_ML[:, i].max())
        # Plot the y = x line based on the min and max values
        axs[i].plot([min_val, max_val], [min_val, max_val], '--', color='k')

    # Set labels on the appropriate subplots
    for i in range(nrows):
        axs[i * ncols].set_ylabel('Predicted')
    for i in range(min(ncols, nvar)):
        axs[-ncols + i].set_xlabel('True')

    # Hide any unused subplots
    for i in range(nvar, nrows * ncols):
        axs[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(args.plot_dir, f'ML_output.png'), dpi=100)
    plt.close()


def plot_contour_box(ax, x, y, z, data):
    # Plot contour box
    xx, yy, zz = np.meshgrid(x,y,z, indexing='ij')
    datacube = data.reshape(xx.shape)
    clevels = np.linspace(0.001*datacube.min(), 0.001*datacube.max(), 101)
    kw = {
        'vmin': clevels.min(),
        'vmax': clevels.max(),
        'levels': clevels,
        'cmap': 'RdBu_r',
        'extend': 'both',
        'alpha': 0.5
    }
    # Plot contour surfaces
    A = ax.contourf(
        xx[:, -1, :], zz[:, -1, :], datacube[:, -1, :],
        zdir='z', offset=yy.max(), **kw
    )
    B = ax.contourf(
        xx[:, :, 0], datacube[:, :, 0], yy[:, :, 0],
        zdir='y', offset=0, **kw
    )
    C = ax.contourf(
        datacube[-1, :, :], zz[-1, :, :], yy[-1, :, :],
        zdir='x', offset=xx.max(), **kw
    )
    # Set limits of the plot from coord limits
    xmin, xmax = xx.min(), xx.max()
    ymin, ymax = yy.min(), yy.max()
    zmin, zmax = zz.min(), zz.max()
    ax.set(xlim=[xmin, xmax], zlim=[ymin, ymax], ylim=[zmin, zmax])
    # Plot edges
    edges_kw = dict(color='0.5', linewidth=0.5, zorder=1e3)
    ax.plot([xmax, xmax], [zmin, zmax], ymin, **edges_kw)
    ax.plot([xmax, xmax], [zmin, zmax], ymax, **edges_kw)
    ax.plot([xmin, xmax], [zmin, zmin], ymin, **edges_kw)
    ax.plot([xmin, xmax], [zmin, zmin], ymax, **edges_kw)
    ax.plot([xmax, xmax], [zmin, zmin], [ymin, ymax], **edges_kw)
    # Set labels and zticks
    ax.set(
        xlabel='X',
        ylabel='Z',
        zlabel='Y',
    )
    # Set zoom and angle view
    ax.view_init(20, -45)
    aspectratio = int(len(x)/len(y))
    ax.set_box_aspect([aspectratio,aspectratio,1], zoom=1)
    ax.grid(False);
    # ax.axis("off");


def plot_corner(X):
    """
    X: (N, num_vars)
    Creates a corner plot with histograms on the diagonal and
    scatter/density plots on the off-diagonal.
    """
    figure = corner.corner(X,
                           bins=30,
                           quantiles=[0.16, 0.5, 0.84],
                           show_titles=True,
                           title_fmt=".2f",
                           labels=[f"var_{i}" for i in range(X.shape[1])])
    plt.savefig(os.path.join(args.plot_dir, f'uips_pdf.png'), dpi=100)
    plt.close()

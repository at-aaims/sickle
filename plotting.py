import numpy as np

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







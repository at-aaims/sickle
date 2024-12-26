import numpy as np
import os
import time
from constants import *

from matplotlib import pyplot as plt

def compute_euclidean_distance(x, y):
    return np.sqrt(x**2 + y**2)

def scale(func, x):
    """convert data to 2D scale and reshape back to 3D"""
    return func(x.reshape(-1, x.shape[-1])).reshape(x.shape)

def scale_probabilities(probs, a=0.01, b=0.99):
    """
    Scale a list of probabilities linearly from range [a, b].
    
    Args:
    - probs: List of probabilities
    - a, b: Range for scaling (default is [0.01, 0.99])

    Returns:
    - Scaled list of probabilities
    """
    A, B = min(probs), max(probs)
    scaled_probs = [(x - A) * (b - a) / (B - A) + a for x in probs]
    return np.array(scaled_probs)

def print_stats(label, X, Y):

    stats = lambda x : f"min: {np.amin(x):.04f}, mean: {np.mean(x):.04f}, max: {np.amax(x):.04f}"

    print(label)
    print(X.shape)
    print('X[0]:', stats(X[:, 0]))
    print('X[1]:', stats(X[:, 1]))
    print('Y:', stats(Y[:]))

def verbose_io(func):
    def wrapper(*args, **kwargs):
        print(f"{func.__name__} {args[0]}")
        return func(*args, **kwargs)
    return wrapper

@verbose_io
def load(*args, **kwargs):
    return np.load(*args, **kwargs)

@verbose_io
def savez(*args, **kwargs):
    np.save(*args, **kwargs)

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
    plt.show()

def plot_learning_curve(train_loss_history, val_loss_history):
    plt.figure(figsize=(10,5))
    plt.rcParams.update({'font.size': 18})
    plt.title('Learning curve')
    plt.plot(train_loss_history, label='training')
    plt.plot(val_loss_history, label='validation',alpha=0.5)
    plt.yscale('log')
    plt.xlabel('Epoch'); plt.ylabel(r'Loss ($mse$)')
    plt.legend(frameon=False);
    # plt.savefig(os.path.join(PLTDIR, f'ML_loss-curves.png'), dpi=100)

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
    # plt.savefig(os.path.join(PLTDIR, f'ML_output.png'), dpi=100)

# # Function to compute grid coordinates for subdomain/box
def get_1Dgrid(Lh, nx, nxoffset, nxsl, nxskip):
    '''
      Lh: Length of grid dimension
      nx: # of points in original grid
      nxoffset: corner of the original gridfrom which the subdomain grid to be created
      nxsl: # of points of subdomain
      nxskip: # points to be skipped from original domain to create subdomain - subsampling
    '''
    dx = Lh/nx
    xin = 0 + (dx*nxoffset)
    xfi = xin + dx*nxsl*nxskip
    x = np.linspace(xin, xfi, nxsl)
    return x

def get_data_memmap(loadpath, nx, ny, nz, nxsl, nysl, nzsl, nxoffset, nyoffset, nzoffset, nxskip, nyskip, nzskip):
    # Check data
    check_data(loadpath, nx, ny, nz, nbyte=4)
    # Memory-map the binary file
    t = time.time()
    data_memmap = np.memmap(loadpath, dtype=np.float32, mode='r', shape=(nz, ny, nx)) # NOTE: data is stored [z, y, x]
    elpsdt = time.time() - t
    # print(f'Time elapsed for memmap: {int(elpsdt/60)} min {elpsdt%60:.4f} sec')
    # Extract the sub-cube
    t = time.time()
    sub_cube = data_memmap[ nzoffset:nzoffset+(nzsl*nzskip):nzskip, # start from `nzoffset` location and get `nzsl` points, but skip every `nzskip` point
                          nyoffset:nyoffset+(nysl*nyskip):nyskip, 
                          nxoffset:nxoffset+(nxsl*nxskip):nxskip] 
    elpsdt = time.time() - t
    # print(f'Time elapsed for slice: {int(elpsdt/60)} min {elpsdt%60:.4f} sec')
    # Copy the sub-cube to a new array to avoid memory-mapping issues when processing
    t = time.time()
    datacube = sub_cube.copy().transpose(2, 1, 0) # transposing data to be [x, y, z]
    elpsdt = time.time() - t
    # print(f'Time elapsed for copying data: {int(elpsdt/60)} min {elpsdt%60:.4f} sec')
    data_memmap._mmap.close()
    del data_memmap, sub_cube
    # Print the shape of the sub-cube
    # print(f'Shape of the sub-cube: {datacube.shape}')
    return datacube

# ## Check data
def check_data(loadpath, nx, ny, nz, nbyte):
  # print('Checking data file...')
  # read in test binary and check number of samples
  binary = open(loadpath, 'rb')
  binary.seek(0,2) ## seeks to the end of the file (needed for getting number of bytes)
  num_bytes = binary.tell() ## how many bytes are in this file is stored as num_bytes
  
  if int(num_bytes/nbyte)==nx*ny*nz:
      num_samp = nx*ny*nz
      # print(f'Number of samples counted == actual. Check complete.')
  else:
      print(f'Number of bytes in file =\t{num_bytes:,}')
      print(f'Number of counted samples =\t{int(num_bytes/nbyte):,}')
      print(f'Number of actual samples =\t{nx*ny*nz:,}')
      raise Exception(f'Number of samples counted != actual')
  binary.close()


import numpy as np
import os
from subsample_main import subsample_data
from subsampling_utils import load_data, create_maxent_subsampler, check_and_create_dirs
from args import args

# Ensure required directories exist
check_and_create_dirs(args.output_dir)

# Load the data
X, Y, cv, x, y, z = load_data(args.path, args)

# Define the subsampling function for maximum entropy
#subsample_fn = lambda X, n: subsample_maxent(X, n)
#subsample_fn = create_maxent_subsampler(cv)
subsample_fn = create_maxent_subsampler(cv, args)

# Perform subsampling
Xout, Yout = subsample_data(X, Y, x, y, z, subsample_fn, args)

# Save output
outfile = os.path.join(args.output_dir, 'subsampled_maxent.npz')
np.savez(outfile, X=Xout, Y=Yout, x=x, y=y, z=z)

print(f'Subsampled data saved to {outfile}')

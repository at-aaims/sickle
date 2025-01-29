import argparse
import os
import yaml

from constants import FieldPredictionType

# Function to load configuration from a YAML file
def load_config(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as yaml_file:
            return yaml.safe_load(yaml_file)
    return {}

# Add the YAML file argument
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("config_file", nargs="?", default=None, help="Path to the YAML configuration file. If not provided, default parameters will be used.")
archs = ['fcn', 'fcn_sst', 'lstm', 'transformer', 'MLP_transformer', 'CNN_transformer']
choices = ['maxent', 'random', 'full', 'uips']
parser.add_argument('-m', '--method', choices=choices, default='maxent', help='subsample method')
parser.add_argument('--arch', type=str, default='fcn_sst', choices=archs, help='Type of neural network architecture')
parser.add_argument('--bins', type=int, default=100, help='Number of bins to represent PDFs')
parser.add_argument('-b', '--batch', type=int, default=32, help='batch size')
choices = ['p', 'pv', 'wz', 'pwz', 'r', 'u', 'v', 'w']
parser.add_argument('-cv', '--cluster_var', nargs="+", type=str, default='pv', choices=choices, help='cluster variable')
parser.add_argument('--cutoff', type=float, default=0.5, help='optimal data cutoff factor, e.g., 0.1 keep top ten percent')
parser.add_argument('--dims', type=int, default=2, choices=[2, 3], help='dataset dimensionality, 2 or 3 dimensions')
parser.add_argument('-e', '--epochs', type=int, default=5, help='number of epochs')
parser.add_argument('--hybrid', type=float, default=1, help='hybrid maxent+random sampling approach')
parser.add_argument('--nbytes', type=int, default=4, help='how many bytes used for each number')
parser.add_argument('-nn', '--knn', type=int, default=0, help='use knn to include neighbars')
parser.add_argument('-nc', '--num_clusters', type=int, default=10, help='number of clusters')
parser.add_argument('-ns', '--num_samples', type=int, default=100, help='number of subsamples')
parser.add_argument('-o', '--output', action='store_true', default=False, help='output optional files')
parser.add_argument('--num_timesteps', type=int, default=100, help='OpenFOAM number of timestamps')
parser.add_argument('--path', type=str, default='./data', help='path to data')
parser.add_argument('--patience', type=int, default=5, help='number epochs for early stopping')
parser.add_argument('--plot', action='store_true', default=False, help='show plots')
parser.add_argument('--noseed', action='store_true', default=False, help='don\'t use random number seed')
choices = ['StandardScaler', 'MinMaxScaler', 'PowerTransformer', 'None']
parser.add_argument('--xscaler', type=str, default='MinMaxScaler', choices=choices, help='scaler function')
parser.add_argument('--yscaler', type=str, default='StandardScaler', choices=choices, help='scaler function')
parser.add_argument('--yscalefactor', type=float, default=3, help='scalefactor to divide target by before training')
parser.add_argument('--dtype', type=str, default='openfoam', choices=['openfoam', 'csv', 'npz', 'sst-binary'], help='data type')
parser.add_argument('--test_frac', type=float, default=0.1, help='fraction of data to hold out for testing')
parser.add_argument('--target', type=str, default='wz', choices=['drag', 'p', 'p_full', 'wz', 'tke'], help='training target')
parser.add_argument('--timesteps', nargs='+', type=float, default=None, help='Specific timesteps to load (e.g., --timesteps 28.04 29.24)')
parser.add_argument('-s', '--snapshot', action='store_true', default=False, help='load snapshots/raw_data.npz instead of running dataloader')
parser.add_argument('--sequence', action='store_true', default=False, help='aggregate individual time-steps into a sequence')
parser.add_argument('-v', '--verbose', action='store_true', default=False, help='verbose output')
parser.add_argument('--overlap', type=int, default=1, help='number of time steps to overlap windows')
parser.add_argument('--window', type=int, default=2, help='time window sequence size')
parser.add_argument('--write_interval', type=int, default=100, help='OpenFOAM write interval')
# SST data args
parser.add_argument("--nx", type=int, default=512+2, required=False, help="number of grid points in x dir for full data")
parser.add_argument("--ny", type=int, default=512, required=False, help="number of grid points in y dir for full data")
parser.add_argument("--nz", type=int, default=256, required=False, help="number of grid points in z dir for full data")
parser.add_argument("--nxsl", type=int, default=128, required=False, help="number of grid points in x dir for sampled data")
parser.add_argument("--nysl", type=int, default=128, required=False, help="number of grid points in y dir for sampled data")
parser.add_argument("--nzsl", type=int, default=128, required=False, help="number of grid points in z dir for sampled data")
parser.add_argument("--nxskip", type=int, default=1, required=False, help="subsampling rate in x dir (for full resolution, use value 1)")
parser.add_argument("--nyskip", type=int, default=1, required=False, help="subsampling rate in y dir (for full resolution, use value 1)")
parser.add_argument("--nzskip", type=int, default=1, required=False, help="subsampling rate in z dir (for full resolution, use value 1)")
parser.add_argument("--nxoffset", type=int, default=0, required=False, help="offset these many samples in each direction in x dir to set corner of the sampled box")
parser.add_argument("--nyoffset", type=int, default=0, required=False, help="offset these many samples in each direction in x dir to set corner of the sampled box")
parser.add_argument("--nzoffset", type=int, default=0, required=False, help="offset these many samples in each direction in x dir to set corner of the sampled box")
parser.add_argument("--gravity", type=str, default="z", required=False, help="direction of gravity: 'y' or 'z'")
parser.add_argument("--Lh", type=float, default=1.0, required=False, help="Horizontal length of full box")
parser.add_argument("--Lv", type=float, default=0.5, required=False, help="Vertical length of full box")
parser.add_argument("--input_vars", nargs="+", type=str, default=['u' 'v' 'w' 'r'], help="variable name(s) for input to model: 'r' 'u' 'v' 'w'. It can be single or multiple vars. NOTE: change --in_channels accordingly.")
parser.add_argument("--output_vars", nargs="+", type=str, default=['p' 'pv'], help="variable name(s) for model output: 'p' 'pv'. It can be single or multiple vars. NOTE: change --out_channels accordingly.")
parser.add_argument('--output_dir', type=str, default='./snapshots', help='output directory')
parser.add_argument('--plot_dir', type=str, default='./plots', help='plots directory')
parser.add_argument("--saveData", default=False, action='store_true', help="Save data.")

# Parse command-line arguments
args = parser.parse_args()

# Load the YAML config if specified
if args.config_file:
    config = load_config(args.config_file)
else:
    config = {}

# Process the YAML configuration
if config:
    shared_config = config.get("shared", {})
    subsample_config = config.get("subsample")
    train_config = config.get("train") 
    
    # Merge all configurations
    all_config = {**shared_config, **subsample_config, **train_config}
    
    # Set the attributes in args
    for key, value in all_config.items():
        setattr(args, key, value)

# Update derived settings based on configuration
if args.target == 'drag':
    args.field_prediction_type = FieldPredictionType.GLOBAL
elif args.target == 'p_full' or args.method == 'full':
    args.field_prediction_type = FieldPredictionType.FULL
else:
    args.field_prediction_type = FieldPredictionType.LOCAL

if args.method == "full": 
        args.num_samples = args.nxsl*args.nysl*args.nzsl 

if args.arch in ['lstm', 'transformer', 'MLP_transformer', 'CNN_transformer']: 
    args.sequence = True

if args.arch == 'lstm' and (args.overlap > args.window - 1 or args.overlap < 0):
    raise ValueError(f"Invalid arguments: overlap ({args.overlap}) must be >= 0 and <= window - 1 ({args.window - 1})")

# Convert cluster_var to a list if it's a string in the YAML config
if isinstance(args.cluster_var, str): args.cluster_var = [args.cluster_var]

print(args)

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
import importlib
from args import args
from constants import *
from dataloaders import create_sequences
from helpers import scale

fileprefix = f"nxsl{args.nxsl}-nysl{args.nysl}-nzsl{args.nzsl}-ns{args.num_samples}-window{args.window}"
outfilename = f"subsampled_{fileprefix}.npz"

def setup_ddp():
    """
    Initialize the distributed environment for DDP using environment variables
    provided by Slurm.
    """
    rank = int(os.environ['SLURM_PROCID'])  # Global rank of the current process
    world_size = int(os.environ['SLURM_NTASKS'])  # Total number of tasks
    master_addr = os.environ['MASTER_ADDR']  # Address of the master node
    master_port = os.environ['MASTER_PORT']  # Port of the master node

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank % torch.cuda.device_count())  # Assign GPU based on rank

    # Verify GPU setup
    print(f"Rank {rank}: Using GPU {torch.cuda.current_device()} - {torch.cuda.get_device_name()}")

    return rank, world_size


def cleanup_ddp():
    """
    Clean up the distributed process group.
    """
    dist.destroy_process_group()


def main_worker(rank, world_size, args, X_train, Y_train, X_test, Y_test):
    """
    Main worker function for each process. Initializes DDP and runs training.
    """
    setup_ddp()

    device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')
    print(f"Rank {rank}: Device set to {device}")

    # Setup data loaders with DistributedSampler
    train_sampler = DistributedSampler(TensorDataset(X_train, Y_train), num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(TensorDataset(X_test, Y_test), num_replicas=world_size, rank=rank)

    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=args.batch, sampler=train_sampler)
    test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=args.batch, sampler=test_sampler)
    print(f"batch size: {args.batch}")

    # Initialize the model and move it to the correct device
    input_shape = X_train.shape[1:]
    output_shape = Y_train.shape[1:] if len(Y_train.shape) > 1 else 1
    model_module = importlib.import_module('archs.' + args.arch)
    model = model_module.build_model(input_shape, output_shape, window=args.window).to(device)

    print(f"Rank {rank}: Model moved to {device}")

    # Wrap the model with DistributedDataParallel
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device.index])

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)  # Shuffle data for this epoch
        running_loss = 0.0

        for i, (batch_X, batch_Y) in enumerate(train_loader):
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            # print(f"Rank {rank}: Batch moved to {device}")
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Rank {rank}, Epoch {epoch + 1}/{args.epochs}, Loss: {running_loss:.4f}")

    # Save the model only on rank 0
    if rank == 0:
        model_path = f"models/{args.arch}"
        os.makedirs(model_path, exist_ok=True)
        torch.save(model.state_dict(), f"{model_path}/{fileprefix}_model.pth")

    cleanup_ddp()


def main():
    """
    Main function to initialize data, parse arguments, and start the DDP training.
    """
    # Preprocess data
    data = np.load(os.path.join(args.output_dir, outfilename))
    X, Y = data['X'], data['Y']
    print(f"X: {X.shape}; Y: {Y.shape}") # X: [T, [X,Y,Z]-or-NSAMPLES, C]; Y: [T, [X,Y,Z]-or-NSAMPLES, C] 
   
    # make timeseries to sequences
    if args.sequence:
        X, Y = create_sequences(X, Y, args)
    else:
        Y = np.squeeze(Y)
    print(f"After sequence X: {X.shape}; Y: {Y.shape}")

    # transpose shape of X and Y to be [B,T,C,Samples] and [B,T,C,H,W,D]
    if args.method == "full":
        X = X.transpose(0,1,5,2,3,4)
    else:
        X = X.transpose(0,1,3,2)
    if args.field_prediction_type == FieldPredictionType.GLOBAL:
        Y = Y.transpose(0,1,3,2)
    elif args.field_prediction_type == FieldPredictionType.LOCAL:
        Y = Y.transpose(0,1,3,2)
    elif args.field_prediction_type == FieldPredictionType.FULL:
        Y = Y.transpose(0,1,5,2,3,4)
    else:
        raise Exception("Enter a valid `args.target`.")
    print(f"X: {X.shape}; Y: {Y.shape}")
    
    # train:val split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=args.test_frac, shuffle=False)

    # Scale the data
    scaler_x = eval(args.xscaler)()
    X_train = scale(scaler_x.fit_transform, X_train)
    X_test = scale(scaler_x.transform, X_test)
    if args.yscaler != 'None':
        scaler_y = eval(args.yscaler)()
        Y_train = scale(scaler_y.fit_transform, Y_train)
        Y_test = scale(scaler_y.transform, Y_test)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32)
    print(f"X_train: {X_train.shape}; X_test: {X_test.shape}")
    print(f"Y_train: {Y_train.shape}; Y_test: {Y_test.shape}")

    # Determine world size and launch workers
    world_size = int(os.environ['SLURM_NTASKS'])
    rank = int(os.environ['SLURM_PROCID'])

    main_worker(rank, world_size, args, X_train, Y_train, X_test, Y_test)


if __name__ == "__main__":
    main()

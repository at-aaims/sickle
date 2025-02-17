import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from args import args
import dataloader
import importlib
from constants import *
from dataloaders import create_sequences
from helpers import scale

fileprefix = f"nxsl{args.nxsl}-nysl{args.nysl}-nzsl{args.nzsl}-ns{args.num_samples}-window{args.window}_method-{args.method}"
outfilename = f"subsampled_{fileprefix}.npz"  

# Functions to set up the distributed environment
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '3442'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# Function to clean up the distributed environment
def cleanup():
    dist.destroy_process_group()

# The main worker function
def main_worker(rank, world_size, args, X_train, Y_train, X_test, Y_test):
    setup(rank, world_size)

    # Set the device for this rank
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    # Split data using DistributedSampler
    train_sampler = DistributedSampler(TensorDataset(X_train, Y_train), num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(TensorDataset(X_test, Y_test), num_replicas=world_size, rank=rank)

    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=args.batch, sampler=train_sampler)
    test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=args.batch, sampler=test_sampler)
    print(f"batch size: {args.batch}")

    # Initialize the model and move it to the appropriate device
    input_shape = X_train.shape[1:]
    output_shape = Y_train.shape[1:] if len(Y_train.shape) > 1 else 1
    model_module = importlib.import_module('archs.' + args.arch)
    model = model_module.build_model(input_shape, output_shape, window=args.window).to(device)

    # Wrap the model with DistributedDataParallel
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)  # Ensure shuffling is synchronized
        running_loss = 0.0

        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Rank {rank}, Epoch {epoch + 1}/{args.epochs}, Loss: {running_loss:.4f}", flush=True)

    # Save the model (only from rank 0)
    if rank == 0:
        model_path = f"models/{args.arch}"
        if not os.path.exists(model_path): os.makedirs(model_path)
        torch.save(model.state_dict(), f"{model_path}/{fileprefix}_model.pth")

    cleanup()

# The main function
def main():
    world_size = torch.cuda.device_count()  # Total number of GPUs available

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

    # Pass datasets to main_worker
    mp.spawn(main_worker,
             args=(world_size, args, X_train, Y_train, X_test, Y_test),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    main()

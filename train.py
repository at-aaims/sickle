import importlib
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.model_selection import train_test_split

import dataloader
from args import args
from constants import FieldPredictionType
from helpers import scale, print_stats
from plotting import plot_histograms, plot_ML_outputs, plot_learning_curve
import matplotlib.pyplot as plt
import matplotlib.colors as colors

fileprefix = f"nxsl{args.nxsl}-nysl{args.nysl}-nzsl{args.nzsl}-ns{args.num_samples}-window{args.window}_method-{args.method}"

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

outfilename = f"subsampled_{fileprefix}.npz"
data = np.load(os.path.join(args.output_dir, outfilename))
X, Y = data['X'], data['Y']

print(X.shape, Y.shape, len(Y.shape))

if args.sequence:
    print('creating time sequences...')
    X, Y = dataloader.create_sequences(X, Y, args)
    print(X.shape, Y.shape)
    num_sequences, sequence_length, num_features = X.shape
    num_samples = X.shape[0]
    if args.field_prediction_type == FieldPredictionType.GLOBAL:
        Y = Y.reshape(num_sequences, sequence_length)
else:
    Y = np.squeeze(Y)

print('Data shape for network:')
print(X.shape, Y.shape)

# Split data into train/test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=args.test_frac, shuffle=False)

print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

if args.arch == 'fcn' or args.arch == 'fcn_sst':
    # Flatten input so it has only two dimensions: (n_samples, n_features)
    X = X.reshape(X.shape[0], -1)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    print_stats('Train', X_train, Y_train)
    print_stats('Test', X_test, Y_test)

print('train/test shapes:', X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

if args.plot: plot_histograms(X_train, X_test, Y_train, Y_test)

# Scale the data
scaler_x = eval(args.xscaler)()
X_train = scale(scaler_x.fit_transform, X_train)
X_test = scale(scaler_x.transform, X_test)

if args.yscaler != 'None':
    scaler_y = eval(args.yscaler)()
    if args.sequence:
        Y_train = scale(scaler_y.fit_transform, Y_train)
        Y_test = scale(scaler_y.transform, Y_test)
    elif args.arch == 'fcn':
        Y_train = scaler_y.fit_transform(Y_train.reshape(-1, 1))
        Y_test = scaler_y.transform(Y_test.reshape(-1, 1))
    else:
        Y_train = scaler_y.fit_transform(Y_train)
        Y_test = scaler_y.transform(Y_test)

Y_train, Y_test = Y_train / args.yscalefactor, Y_test / args.yscalefactor

print_stats('Train', X_train, Y_train)
print_stats('Test', X_test, Y_test)

if args.plot: plot_histograms(X_train, X_test, Y_train, Y_test)

# Convert data to PyTorch tensors and move to the appropriate device
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
Y_train = torch.tensor(Y_train, dtype=torch.float32).to(device)
Y_test = torch.tensor(Y_test, dtype=torch.float32).to(device)

# Define model
input_shape = X_train.shape[1:]
output_shape = Y_train.shape[1:]
print('**', output_shape)
model_module = importlib.import_module('archs-pt.' + args.arch)
model = model_module.build_model(input_shape, output_shape, window=args.window).to(device)
print(model)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Train model
def train_model(model, optimizer, criterion, X_train, Y_train, X_test, Y_test, args):
    train_loss_history = () # for loss curves
    val_loss_history = () # for loss curves
    model.train()
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=args.batch, shuffle=False)

    running_loss = 0.0 # for saving train loss for curves

    for epoch in range(args.epochs):
        model.train()
        ibatch = 0 # for avg loss over all batches
        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)  # Move batch to the appropriate device
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            ibatch += 1

        train_loss_history = train_loss_history + (running_loss / ibatch,)
        running_loss = 0.0
        print(f'Epoch {epoch+1}/{args.epochs}, Loss: {train_loss_history[-1]:1.3e}')

        model.eval()
        with torch.no_grad():
            val_loss = sum(criterion(model(batch_X.to(device)), batch_Y.to(device)) for batch_X, batch_Y in test_loader) / len(test_loader)
            val_loss_history = val_loss_history + (val_loss.item(),)
        if (epoch + 1) % args.patience == 0:
            print(f'Validation Loss: {val_loss_history[-1]:1.3e}')

    return train_loss_history, val_loss_history

train_loss_history, val_loss_history = train_model(model, optimizer, criterion, X_train, Y_train, X_test, Y_test, args)
plot_learning_curve(train_loss_history, val_loss_history)
plt.savefig(os.path.join(args.plot_dir, f'{fileprefix}_{args.subsample}_ML_loss-curves.png'), dpi=100, bbox_inches='tight')

# Evaluate the model
model.eval()
with torch.no_grad():
    Y_test_ML = model(X_test)
    test_loss = criterion(Y_test_ML, Y_test)
print(f'Loss ({fileprefix}): {test_loss.item():.04f}')
plot_ML_outputs(Y_test_ML[0, :].cpu().numpy().reshape(-1, 1), Y_test[0, :].cpu().numpy().reshape(-1, 1))
plt.savefig(os.path.join(args.plot_dir, f'{fileprefix}_{args.subsample}_ML_output.png'), dpi=100, bbox_inches='tight')

# Save model
model_path = f"models/{args.arch}"
if not os.path.exists(model_path): os.makedirs(model_path)
torch.save(model.state_dict(), f"{model_path}/{fileprefix}_model.pth")
np.savez(os.path.join(args.output_dir, f"{fileprefix}_test.npz"), X_test=X_test.cpu().numpy(), \
                                           Y_test=Y_test.cpu().numpy(), \
                                           X_train=X_train.cpu().numpy(), \
                                           Y_train=Y_train.cpu().numpy())



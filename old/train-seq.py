import importlib
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.model_selection import train_test_split

from args import args
from constants import FieldPredictionType
from dataloaders import create_sequences
from helpers import scale, print_stats
from plotting import plot_histograms, plot_ML_outputs, plot_learning_curve

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

outfilename = f"subsampled_{args.fileprefix}.npz"
data = np.load(os.path.join(args.output_dir, outfilename))
X, Y = data['X'], data['Y']

print(X.shape, Y.shape, len(Y.shape))

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

# Define model
input_shape = X_train.shape[1:]
output_shape = Y_train.shape[1:]
print('**', output_shape)
model_module = importlib.import_module('archs.' + args.arch)
model = model_module.build_model(input_shape, output_shape, window=args.window).to(device)
model.to(device)
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

# Evaluate the model
model.eval()
with torch.no_grad():
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)
    Y_test_ML = model(X_test)
    test_loss = criterion(Y_test_ML, Y_test)
print(f'Loss ({args.fileprefix}): {test_loss.item():.04f}')
plot_ML_outputs(Y_test_ML[0, :].cpu().numpy().reshape(-1, 1), Y_test[0, :].cpu().numpy().reshape(-1, 1))

# Save model
model_path = f"models/{args.arch}"
if not os.path.exists(model_path): os.makedirs(model_path)
torch.save(model.state_dict(), f"{model_path}/{args.fileprefix}_model.pth")
np.savez(os.path.join(args.output_dir, f"{args.fileprefix}_test.npz"), X_test=X_test.cpu().numpy(), \
                                           Y_test=Y_test.cpu().numpy(), \
                                           X_train=X_train.cpu().numpy(), \
                                           Y_train=Y_train.cpu().numpy())

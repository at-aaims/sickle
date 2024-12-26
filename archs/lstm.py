import torch
import torch.nn as nn
import torch.optim as optim

class LSTMModel(nn.Module):
    def __init__(self, input_shape, output_shape='None', units=288, activation='elu', dropout=0.5, window=3):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_shape[-1], units, batch_first=True)
        self.lstm2 = nn.LSTM(units, units, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(units * input_shape[0], units)
        self.fc2 = nn.Linear(units, units // 2)
        self.fc3 = nn.Linear(units // 2, window)
        
        if activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'selu':
            self.activation = nn.SELU()
        else:
            raise ValueError("Unsupported activation function")

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

def build_model(input_shape, units=288, activation='elu', dropout=0.5, lr=0.0003, window=3):
    model = LSTMModel(input_shape, units, activation, dropout, window)
    return model

def get_meta_model(input_shape):
    def meta_model(hp):
        units = hp.Int('units', min_value=32, max_value=512, step=32)
        activation = hp.Choice('activation', ['elu', 'gelu', 'selu'])
        dropout = hp.Float('dropout_rate', 0.0, 0.5, sampling='linear')
        return build_model(input_shape, units, activation, dropout)
    return meta_model

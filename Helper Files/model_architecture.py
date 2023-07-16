from torchsummary import summary
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import wavfile
import numpy as np
import os
import librosa 
import matplotlib.pyplot as plt
import glob 
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

input_size = 13
hidden_size = 256
output_size = 7
print(f"Output Size: {output_size}")
num_layers = 3
device = 'cuda'

class LipSyncNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LipSyncNet, self).__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.2)
#        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size*2, output_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size 
    
    def forward(self, x):

        x = x.permute(0, 2, 1) # permute the dimensions
        x = self.conv1(x)
        x = F.relu(x)

        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)
        
        # Pass through LSTM
        x = x.permute(0, 2, 1) # permute the dimensions

        out, _ = self.lstm(x, (h0, c0))

        # Pass through FC layer to get predictions for each time step
        out = self.fc(out)

        return out

# Assuming your model is named "model" and is moved to the correct device
model = LipSyncNet(input_size, hidden_size, output_size, num_layers).to(device)

# You need to specify the size of the input in the call to summary
# Assuming your input is 1-dimensional MFCC with a sequence length of 1000
summary(model, input_size=13)

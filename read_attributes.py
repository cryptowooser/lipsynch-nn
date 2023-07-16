import torch
from data_classes import LipSyncNet

# Load the model
model = torch.load('model_full_dataset_2layers.pth')

# Print the attributes
print(f'input_size: {model.input_size}')
print(f'hidden_size: {model.hidden_size}')
print(f'output_size: {model.output_size}')
print(f'num_layers: {model.num_layers}')

import torch
from data_classes import LipSyncNet

# Load the model
model = LipSyncNet(input_size=13, hidden_size=256, output_size=7, num_layers=2)
model.load_state_dict(torch.load('model_full_dataset_2layers_backup.pth'))

# Add the parameters as attributes
model.input_size = 13
model.hidden_size = 256
model.output_size = 7
model.num_layers = 2

# Save the entire model, not just the state dictionary
torch.save(model, 'model_full_dataset_2layers.pth')

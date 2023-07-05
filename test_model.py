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

#Constants
sample_rate = 44100  # Your actual sample rate
hop_length = 512  # Your actual hop length




def process_audio(filename):
    # Load the audio file
    y, sr = librosa.load(filename, sr=None)
    # Compute MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    # Transpose to make the time dimension first
    mfcc = mfcc.T
    # Get the total number of frames
    total_frames = mfcc.shape[0]
    return mfcc, total_frames


def process_text(filename, total_frames):
    # Define mapping from letters to integers
    letter_to_int = {chr(i + 65): i for i in range(9)}
    # Read the file
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Initialize an array with the default mouth shape (assuming 'A')
    labels = np.full(total_frames, letter_to_int['A'])

    for line in lines:
        time, shape = line.strip().split('\t')
        time = float(time)
        shape_id = letter_to_int[shape]

        # Convert time to frame index
        index = int(time * sample_rate / hop_length)

        # Set all future frames to this shape, until we find a new shape
        labels[index:] = shape_id

    return labels

class LipSyncDataset(torch.utils.data.Dataset):
    def __init__(self, audio_files, text_files):
        self.audio_files = audio_files
        self.text_files = text_files

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        text_file = self.text_files[idx]
        audio, total_frames = process_audio(audio_file)
        text = process_text(text_file, total_frames)

        return torch.from_numpy(audio).float(), torch.tensor(text, dtype=torch.long)


'''
def pad_sequence(batch):
    # Each element in 'batch' is a tuple (audio, shapes)
    # Sort the batch in the descending order
    sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
    # Get each sequence and pad it
    sequences = [x[0] for x in sorted_batch]
    sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    # Also need to store the length of each sequence
    # This is later needed in order to unpad the sequences
    lengths = torch.LongTensor([len(x) for x in sequences])
    # Don't forget to grab the labels
    labels = torch.LongTensor(list(map(lambda x: x[1], sorted_batch)))
    return sequences_padded, lengths, labels
'''
# Set up DataLoader
audio_files = sorted(glob.glob('wavs/*.wav'))
text_files = sorted(glob.glob('texts/*.txt'))

dataset = LipSyncDataset(audio_files, text_files)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, collate_fn=pad_sequence)

letter_to_int = {chr(i + 65): i for i in range(9)} 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_item(index):
    # Get the first item in the dataset
    audio, labels = dataset[0]

    # We need to detach and convert the tensors to numpy for plotting
    audio = audio.detach().numpy()
    labels = labels.detach().numpy()

    # Plot the MFCC
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axs[0].imshow(audio.T, aspect='auto', origin='lower')
    axs[0].set_title('MFCC')
    axs[0].set_ylabel('MFCC Coefficients')

    print(text_files[0])

    # Plot the labels
    # Get 26 colors from the 'tab20' colormap
    colors = plt.get_cmap('tab20')(np.linspace(0, 1, 9))
    cmap = mcolors.ListedColormap(colors)

    norm = mcolors.BoundaryNorm(np.arange(len(letter_to_int)+1)-0.5, len(letter_to_int))  # map integer labels to colors
    img = axs[1].imshow(labels[None, :], aspect='auto', cmap=cmap, norm=norm)
    axs[1].set_title('Mouth Shapes')
    axs[1].set_ylabel('Mouth Shape ID')
    axs[1].set_xlabel('Frame')

    # Add a colorbar with the mouth shapes
    #cbar = fig.colorbar(img, ax=axs[1], ticks=np.arange(len(letter_to_int)))
    #labels = list(letter_to_int.keys())
    #cbar.ax.set_yticklabels(labels)

    plt.tight_layout()
    plt.show()


def plot_batch(batch):
    # Get the first item in the batch
    audio, labels = batch[0][0], batch[1][0]

    # We need to detach and convert the tensors to numpy for plotting
    audio = audio.detach().numpy()
    labels = labels.detach().numpy()

    # Plot the MFCC
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axs[0].imshow(audio.T, aspect='auto', origin='lower')
    axs[0].set_title('MFCC')
    axs[0].set_ylabel('MFCC Coefficients')

    # Plot the labels
    # Get 26 colors from the 'tab20' colormap
    colors = plt.get_cmap('tab20')(np.linspace(0, 1, 9))
    cmap = mcolors.ListedColormap(colors)

    norm = mcolors.BoundaryNorm(np.arange(len(letter_to_int)+1)-0.5, len(letter_to_int))  # map integer labels to colors
    img = axs[1].imshow(labels[None, :], aspect='auto', cmap=cmap, norm=norm)
    axs[1].set_title('Mouth Shapes')
    axs[1].set_ylabel('Mouth Shape ID')
    axs[1].set_xlabel('Frame')

    plt.tight_layout()
    plt.show()



from sklearn.model_selection import train_test_split

# Pair the audio and text files together
all_files = list(zip(audio_files, text_files))

# Perform the split
train_files, val_files = train_test_split(all_files, test_size=0.2)

# Unzip the file pairs for each set
train_audio_files, train_text_files = zip(*train_files)
val_audio_files, val_text_files = zip(*val_files)

# Create the datasets
train_dataset = LipSyncDataset(train_audio_files, train_text_files)
val_dataset = LipSyncDataset(val_audio_files, val_text_files)


#Define the Collate_Function
def collate_fn(batch):
    # Separate the audio tensors and the label lists
    audios, labels = zip(*batch)

    # Pad the audio tensors and the labels
    # Note: We use 0 padding for audios and -1 for labels, assuming -1 is not a valid label
    audios_padded = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-1)

    # Return the padded audios and labels as a batch
    return audios_padded, labels_padded


from torch.utils.data import DataLoader

batch_size = 32  # You can adjust this value based on your system's memory

# Create the DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

first_batch = next(iter(train_dataloader))

class LipSyncNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LipSyncNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, output_size)

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)
        
        # Pass through LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Pass through FC layer to get predictions for each time step
        out = self.fc(out)

        return out

# Instantiate the model
output_size =9 
input_size = 13
hidden_size = 256
output_size = len(letter_to_int)
print(f"Output Size: {output_size}")
num_layers = 3
device='cuda'

model = LipSyncNet(output_size=output_size, input_size=input_size,hidden_size=hidden_size,num_layers=num_layers)

# Load the saved state dictionary
model.load_state_dict(torch.load('model.pth'))
model = model.to(device)
# Make sure to call .eval() on the model for evaluation!
model.eval()
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

model.eval()
all_outputs = []
all_labels = []

# Get predictions for the validation dataset
with torch.no_grad():
    for batch in val_dataloader:
        audio, labels = batch
        audio = audio.to(device)
        labels = labels.to(device)

        outputs = model(audio)
        
        # The predicted class for each frame is the class with the highest score
        _, preds = torch.max(outputs, 2)
        
        all_outputs.extend(preds.view(-1).cpu().numpy())
        all_labels.extend(labels.view(-1).cpu().numpy())

# Exclude the predictions and labels where the labels are equal to -1 (the padding)
all_outputs = [pred for pred, label in zip(all_outputs, all_labels) if label != -1]
all_labels = [label for label in all_labels if label != -1]

# Calculate the metrics
accuracy = accuracy_score(all_labels, all_outputs)
f1 = f1_score(all_labels, all_outputs, average='macro')  # Set the average parameter as you need
precision = precision_score(all_labels, all_outputs, average='macro')  # Set the average parameter as you need
recall = recall_score(all_labels, all_outputs, average='macro')  # Set the average parameter as you need

print(f'Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')

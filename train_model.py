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
from torch.utils.tensorboard import SummaryWriter
import random
from datetime import datetime


def create_run_name(nouns_file, adjectives_file):
    # Read in the nouns and adjectives
    with open(nouns_file, 'r') as f:
        nouns = [line.strip() for line in f.readlines()]
    with open(adjectives_file, 'r') as f:
        adjectives = [line.strip() for line in f.readlines()]

    # Randomly select a noun and adjective
    noun = random.choice(nouns)
    adjective = random.choice(adjectives)

    # Get current date and time
    now = datetime.now()

    # Format as a string
    now_str = now.strftime("%Y%m%d-%H%M%S")

    # Construct run name
    run_name = f"lipsync_{adjective}_{noun}_{now_str}"

    return run_name

run_name = create_run_name('english-nouns.txt', 'english-adjectives.txt')
writer = SummaryWriter(f'runs/{run_name}')

#Constants
sample_rate = 44100  # Your actual sample rate
hop_length = 512  # Your actual hop length



'''
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
'''
def resample_audio(file_path, target_sr=44100):
    # Load the audio file
    y, sr = librosa.load(file_path, sr=None)

    # If the current sample rate is not the target sample rate, resample
    if sr != target_sr:
        y = librosa.resample(y=y, orig_sr=sr, target_sr=target_sr)

    return y, target_sr


def process_audio(filename, save_dir):
    save_filename = os.path.join(save_dir, os.path.basename(filename) + '.npy')
    # Check if the file is already processed
    if os.path.exists(save_filename):
        # If so, load the MFCC from disk instead of re-computing it
        mfcc = np.load(save_filename)
    else:
        # If not, compute the MFCC and save it to disk
        y, sr = resample_audio(filename,sample_rate)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
        mfcc = mfcc.T
        np.save(save_filename, mfcc)

    # Get the total number of frames
    total_frames = mfcc.shape[0]

    return mfcc, total_frames

def process_text(filename, total_frames):
    # Define mapping from letters to integers
    letter_to_int = {chr(i + 65): i for i in range(7)}
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
        audio, total_frames = process_audio(audio_file, 'mfccs')
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

letter_to_int = {chr(i + 65): i for i in range(7)} 
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

batch_size = 128 # You can adjust this value based on your system's memory

# Create the DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

first_batch = next(iter(train_dataloader))

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Create the model, loss function and optimizer
input_size = 13
hidden_size = 256
output_size = len(letter_to_int)
print(f"Output Size: {output_size}")
num_layers = 3
model = LipSyncNet(input_size, hidden_size, output_size, num_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)


# Number of epochs to train for
num_epochs = 1

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for i, batch in enumerate(train_dataloader):
        # Get the input and target
        audio, labels = batch
        audio = audio.to(device)
        labels = labels.to(device)
        # Clear the gradients
        optimizer.zero_grad()
        # Forward propagation
        outputs = model(audio)
        
        # Create a mask by filtering out all labels values equal to -1 (the pad token)
        mask = (labels.view(-1) != -1).to(device)
        
        # We apply the mask to the outputs and labels tensors
        outputs = outputs.view(-1, output_size)[mask]
        labels = labels.view(-1)[mask]

        # Compute the loss and do backprop
        loss = criterion(outputs, labels)
        loss.backward()
        # Update the weights
        optimizer.step()
        # Accumulate the loss
        train_loss += loss.item()
        scheduler.step()
        writer.add_scalar('Training Loss', loss.item(), global_step=epoch * len(train_dataloader) + i)
    # Compute the average training loss for this epoch
    train_loss /= len(train_dataloader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            # Get the input and target
            audio, labels = batch
            audio = audio.to(device)
            labels = labels.to(device)
            # Forward propagation
            outputs = model(audio)
            
            # Create a mask by filtering out all labels values equal to -1 (the pad token)
            mask = (labels.view(-1) != -1).to(device)
        
            # We apply the mask to the outputs and labels tensors
            outputs = outputs.view(-1, output_size)[mask]
            labels = labels.view(-1)[mask]

            # Compute the loss
            loss = criterion(outputs, labels)
            # Accumulate the loss
            val_loss += loss.item()
            writer.add_scalar('Validation Loss', loss.item(), global_step=epoch * len(train_dataloader) + i)
    # Compute the average validation loss for this epoch
    val_loss /= len(val_dataloader)

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
    torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, f"model_{epoch+1}.pth")

torch.save(model.state_dict(), 'model.pth')
writer.close()

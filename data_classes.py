import os
import torch
import numpy as np
import librosa
import torch.nn as nn
import torch.nn.functional as F

#Constants
SAMPLE_RATE = 44100
HOP_LENGTH = 512


class LipSyncDataset(torch.utils.data.Dataset):

    def __init__(self, audio_files, text_files):
        self.audio_files = audio_files
        self.text_files = text_files

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        text_file = self.text_files[idx]
        audio, total_frames = self.__process_audio(audio_file, 'mfccs')
        text = self.__process_text(text_file, total_frames)
        return torch.from_numpy(audio).float(), torch.tensor(text,
                                                             dtype=torch.long)

    def __process_text(self, filename, total_frames):
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
            index = int(time * SAMPLE_RATE / HOP_LENGTH)

            # Set all future frames to this shape, until we find a new shape
            labels[index:] = shape_id

        return labels

    def __process_audio(self, filename, save_dir):
        save_filename = os.path.join(save_dir,
                                     os.path.basename(filename) + '.npy')
        # Check if the file is already processed
        if os.path.exists(save_filename):
            # If so, load the MFCC from disk instead of re-computing it
            mfcc = np.load(save_filename)
        else:
            # If not, compute the MFCC and save it to disk
            y, sr = self.__resample_audio(filename, SAMPLE_RATE)
            mfcc = librosa.feature.mfcc(y=y,
                                        sr=sr,
                                        n_mfcc=13,
                                        HOP_LENGTH=HOP_LENGTH)
            mfcc = mfcc.T
            np.save(save_filename, mfcc)

        # Get the total number of frames
        total_frames = mfcc.shape[0]

        return mfcc, total_frames

    def __resample_audio(file_path, target_sr=44100):
        # Load the audio file
        y, sr = librosa.load(file_path, sr=None)

        # If the current sample rate is not the target sample rate, resample
        if sr != target_sr:
            y = librosa.resample(y=y, orig_sr=sr, target_sr=target_sr)

        return y, target_sr


class LipSyncNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LipSyncNet, self).__init__()
        self.conv1 = nn.Conv1d(input_size,
                               hidden_size,
                               kernel_size=3,
                               padding=1)
        self.lstm = nn.LSTM(hidden_size,
                            hidden_size,
                            num_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=0.2)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, x):

        x = x.permute(0, 2, 1)  # permute the dimensions
        x = self.conv1(x)
        x = F.relu(x)

        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers * 2, x.size(0),
                         self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0),
                         self.hidden_size).to(x.device)

        # Pass through LSTM
        x = x.permute(0, 2, 1)  # permute the dimensions

        out, _ = self.lstm(x, (h0, c0))

        # Pass through FC layer to get predictions for each time step
        out = self.fc(out)

        return out

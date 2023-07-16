import os
import librosa
from scipy.io.wavfile import write

input_dir = 'newflacs'
output_dir = 'newwavs'

os.makedirs(output_dir, exist_ok=True)

# Walking through the input directory to find all .flac files
for subdir, dirs, files in os.walk(input_dir):
    for file in files:
        filepath = subdir + os.sep + file

        if filepath.endswith(".flac"):
            # Loading the flac file and checking the sample rate
            y, sr = librosa.load(filepath, sr=None)

            # If sample rate is not 44.1kHz, resample it
            if sr != 44100:
                y_resampled = librosa.resample(y=y, orig_sr=sr, target_sr=44100)

            else:
                y_resampled = y

            # Write out as a .wav file in the output directory
            output_file_path = os.path.join(output_dir, file.replace('.flac', '.wav'))
            write(output_file_path, 44100, y_resampled)

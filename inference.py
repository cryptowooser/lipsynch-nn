"""
This module provides an inference pipeline for a lip syncing model. 

Given a .wav audio file, the module extracts MFCC features from the audio 
and feeds them into a trained model to predict the corresponding mouth shapes. 
The module supports both CPU and GPU (CUDA) computation.

The output is a sequence of mouth shapes for each window of audio, which can be 
directly used for animation or further processing.

Functions:
- `preprocess`: Load an audio file and extract MFCC features.
- `postprocess`: Get the most likely mouth shape from the tensor.
- `convert_list_to_RLE`: Convert the list of mouth shapes to a format similar 
  to Run-Length Encoding.
- `convert_to_aegisub_format`: Convert the timestamp-value pairs to aegisub format.
- `infer`: Given a .wav file and a model, load them both and compute the mouth shapes for the .wav.
- `main`: The main function that orchestrates the inference pipeline.

Example usage:

python inference.py --wav_file_name my_audio.wav --model_name my_model.pth
"""

import argparse
from   data_classes import LipSyncNet
from   python_speech_features import mfcc
from   scipy.signal import resample
import soundfile
from timebudget import timebudget
import torch


class LipSyncInference:
    def __init__(self, model_name):
        self.target_sr = 44100
        self.hop_length = 512
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_name)

    # @timebudget
    def load_model(self, model_name):
        model_with_params = torch.load(model_name)
        input_size = model_with_params.input_size
        hidden_size = model_with_params.hidden_size
        output_size = model_with_params.output_size
        num_layers = model_with_params.num_layers
        model = LipSyncNet(input_size, hidden_size, output_size, num_layers)
        model.load_state_dict(model_with_params.state_dict())
        model = model.to(self.device)

        # Apply dynamic quantization - this doesn't make things much faster
        # model = torch.quantization.quantize_dynamic(
        #     model, {torch.nn.Linear}, dtype=torch.qint8
        # )

        return model

    # @timebudget
    def preprocess(self, filename):
        """
        Load an audio file and extract MFCC features for processing.

        Parameters:
        filename (str): The path to the audio file.

        Returns:
        numpy.ndarray: The MFCC features of the audio file.
        """
        y, sr = soundfile.read(filename)
        if sr != self.target_sr:
            y = resample(y, int(len(y) * self.target_sr / sr))
        winstep = self.hop_length / sr
        mfcc_features = mfcc(y, samplerate=sr, numcep=13, nfft=2048, winlen=0.025, winstep=winstep)
        return mfcc_features

    # @timebudget
    def postprocess(self, output_data):
        """
        Get the most likely mouth shape from the tensor.

        Parameters:
        output_data (tensor): The data output by the model 
        (In the form of likelihoods of each possible mouth shape for each window.)

        Returns:
        max_likelihood_list: A list with the most likely mouth shape for each window.
        """
        max_likelihood = torch.argmax(output_data, dim=2)
        max_likelihood_list = torch.flatten(max_likelihood).tolist()
        return max_likelihood_list

    # @timebudget
    def convert_list_to_RLE(self, input_list):
        """
        Converts the list to a format similar to RunLengthEncoding, 
        where instead of a list that gives the mouth shape for every window, 
        it only stores where the mouth shape changes 
        and what it changes to, which is what we need
        This also converts the integers generated by the model 
        into the standard Hannah Barbera mouth shapes. 
        
        
        Parameters: 
        input_list (list) : A list of windows and mouthshapes for each window. 

        Returns:
        timestamps_value_pairs (list) : A list of tuples with timestamps 
        of when the mouth shape changes, 
        and the corresponding new mouth shape. 
        """
        window_length = 1 / self.target_sr * self.hop_length
        timestamps = [round(index * window_length, 2) for index in range(len(input_list))]
        timestamps_value_pairs = zip(timestamps, input_list)
        timestamps_value_pairs = [(t, v) for i, (t, v) in enumerate(timestamps_value_pairs)
                                  if i == 0 or v != input_list[i - 1]]
        for _, mouth_shape in timestamps_value_pairs:
            if mouth_shape == 7:
                mouth_shape = 'X'
            else:
                mouth_shape = chr(ord('A') + mouth_shape)
        timestamps_value_pairs = [(t, v) for i, (t, v) in enumerate(timestamps_value_pairs)
                                  if i == 0 or t - timestamps_value_pairs[i - 1][0] > 0.05]
        return timestamps_value_pairs

    # @timebudget
    def convert_to_aegisub_format(self, timestamps_value_pairs):
        """
        Converts from timestamp_value_pairs format to aegisub format, which is useful for debugging.

        Parameters: 
        timestamps_value_pairs : The times and mouth shapes in timestamp/value pair format.

        Returns:
        A string with all the lines in aegisub .ass format. 
        """
        aegisub_lines = []
        for i in range(len(timestamps_value_pairs)):
            timestamp, text = timestamps_value_pairs[i]
            timestamp_str = '{:.2f}'.format(timestamp)
            if i < len(timestamps_value_pairs) - 1:
                end_timestamp = '{:.2f}'.format(timestamps_value_pairs[i + 1][0])
            else:
                end_timestamp = '{:.2f}'.format(timestamp + 1.0)
            aegisub_line = f'Dialogue: 0,{timestamp_str},{end_timestamp},Default,,0,0,0,,{text}'
            aegisub_lines.append(aegisub_line)
        return '\n'.join(aegisub_lines)

    # @timebudget
    def infer(self, wav_file_name):
        """
        Given a wav_file and a model, loads them both and computes the mouth shapes for the wav.

        Parameters: 
        wav_file_name : Name of a wav file that should be loaded. It should be 41khz.
        model_name : Name of the model you're using. 

        Returns:
        A list of windows and their corresponding mouth shapes.
        """
        input_data = self.preprocess(wav_file_name)
        input_data = torch.from_numpy(input_data)
        input_data = input_data.unsqueeze(0)
        input_data = input_data.float()  # Cast input data to float32
        input_data = input_data.to(self.device)
        with torch.no_grad():
            output_data = self.model(input_data)
        prediction = self.postprocess(output_data)
        return prediction


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process wav and model name.")
    parser.add_argument(
        '--wav_file_name',
        type=str,
        required=True,
        help='The path to the input data file (Should be a 44khz sampled wav).'
    )
    parser.add_argument('--model_name',
                        type=str,
                        default='model_full_dataset_2layers.pth',
                        help='The name of the model file.')
    args = parser.parse_args()

    inference = LipSyncInference(model_name=args.model_name)
    output = inference.infer(wav_file_name=args.wav_file_name)
    rle = inference.convert_list_to_RLE(output)
    int_to_letter = {i: chr(i + 65) for i in range(7)}
    for item1, item2 in rle:
        print(f"{item1}:{int_to_letter[item2]}")

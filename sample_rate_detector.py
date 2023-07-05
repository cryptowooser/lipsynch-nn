import wave

def get_sample_rate(filename):
    with wave.open(filename, 'rb') as wav_file:
        return wav_file.getframerate()

# Test it on an audio file
sample_rate = get_sample_rate('wavs/003.wav')
print(f'Sample rate: {sample_rate} Hz')

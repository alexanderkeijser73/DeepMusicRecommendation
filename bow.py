from __future__ import print_function
import librosa

def bow(audio_signal,sr):
    # Compute MFCC features from the raw signal
    mfcc = librosa.feature.mfcc(y=audio_signal, sr=sr, hop_length=hop_length, n_mfcc=13)

    # And the first- and second-order differences (delta features)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    print("MFCCs calculated with shapes:", mfcc.shape, mfcc_delta.shape, mfcc_delta2.shape)

if __name__ == '__main__':
    # 1. Get the file path to the included audio example
    filename = librosa.util.example_audio_file()

    # 2. Load the audio as a waveform `y`
    #    Store the sampling rate as `sr`
    y, sr = librosa.load(filename)

    # Set the hop length; at 22050 Hz, 512 samples ~= 23ms
    hop_length = 512

    # Calculate BOW representation
    bow(y, sr)
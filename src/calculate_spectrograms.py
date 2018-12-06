import os
from librosa.core import get_duration, load
from librosa.feature import melspectrogram
import numpy as np
import pickle
import time
import glob

def calculate_spectrograms(audio_dir, out_dir, file_type='.mp3'):
    files = glob.glob(os.path.join(audio_dir, '*' + file_type))
    num_files = len(files)

    print(f'{num_files} audio files found')

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for i, file_name in enumerate(sorted(files)):
        start_time = time.time()
        track_name = os.path.basename(file_name)
        track_id = os.path.splitext(track_name)[0]
        try:
            song_name = track_to_song[track_id]
        except KeyError:
            continue
        if song_name in wmf_item2i.keys():
            audio_file = os.path.join(audio_dir,
                                      track_name)
            out_file = os.path.join(out_dir, track_id) + '.npy'
            if not os.path.exists(out_file):
                y, sr = load(audio_file)
                mel_spectrogram = melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=128)

    wmf_item2i = pickle.load(open('../../index_dicts.pkl', 'rb'))['item2i']
    track_to_song = pickle.load(open('../../track_to_song.pkl', 'rb'))
    calculate_spectrograms(audio_dir='../../data/MillionSongSubset/audio', out_dir='../../data/MillionSongSubset/spectrograms')
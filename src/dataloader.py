import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from librosa.core import get_duration, load
from librosa.feature import melspectrogram
import numpy as np
import pickle
import time
import glob

class SpectrogramDataset(Dataset):
    """Dataset with mel-spectrograms     for audio samples"""

    def __init__(self, root_dir, latent_factors, wmf_item2i, track_to_song, file_type='.npy', transform=None):
        """
        Args:
            root_dir (string): Directory with all the audio samples.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.latent_factors = latent_factors
        self.wmf_item2i = wmf_item2i
        self.tra2so = track_to_song
        number = 0
        sample_index = {}
        files = glob.glob(os.path.join(root_dir,'*'+file_type))
        for file_name in sorted(files):
            track_name = os.path.basename(file_name)
            track_id = os.path.splitext(track_name)[0]
            try:
                song_name = track_to_song[track_id]
            except KeyError:
                continue
            if song_name in wmf_item2i.keys():
                sample_index[number] = track_name
                number += 1
        self.files = sample_index

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # start_time = time.time()
        file_name = os.path.join(self.root_dir,
                                  self.files[idx])
        mel_spectrogram = np.load(file_name)[:, :1280]
        assert mel_spectrogram.shape == (128, 1280), f'found shape: {mel_spectrogram.shape} for example: {self.files[idx]}'
        latent_factors = self.latent_factors[idx]
        assert latent_factors.shape == (50,), f'found shape: {latent_factors.shape} for example: {self.files[idx]}'
        sample = {'spectrogram': mel_spectrogram, 'latent_factors': latent_factors}

        if self.transform:
            sample = self.transform(sample)
        return sample

class LogCompress(object):
    """"Applies log-compression to input array"""

    def __init__(self, offset=1e-6):
        self.offset = offset

    def __call__(self, sample):
        spectogram, latent_factors = sample['spectrogram'], sample['latent_factors']
        log_mel_spectrograms = np.log(spectogram + self.offset)

        return {'spectrogram': log_mel_spectrograms, 'latent_factors': latent_factors}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        spectrogram, latent_factors = sample['spectrogram'], sample['latent_factors']

        return {'spectrogram': torch.from_numpy(spectrogram).type(torch.FloatTensor),
                'latent_factors': torch.from_numpy(latent_factors).type(torch.FloatTensor)}


if __name__ == '__main__':
    item_factors = pickle.load(open('../../item_wmf_50.pkl', 'rb'))
    wmf_item2i = pickle.load(open('../../index_dicts.pkl', 'rb'))['item2i']
    track_to_song = pickle.load(open('../../track_to_song.pkl', 'rb'))
    start_time = time.time()
    transformed_dataset = SpectrogramDataset(root_dir='../../data/MillionSongSubset/spectrograms',
                                               latent_factors=item_factors,
                                               wmf_item2i = wmf_item2i,
                                                track_to_song=track_to_song,
                                               transform=transforms.Compose([
                                                   LogCompress(),
                                                   ToTensor()
                                                   ]))
    print("Dataset size:", len(transformed_dataset))

    dataloader = DataLoader(transformed_dataset, batch_size=64,
                            shuffle=True, num_workers=4)
    dataloader_iter = iter(dataloader)
    start_time = time.time()
    batch = dataloader_iter.next()
    print(f"Loading one batch took {time.time() - start_time} seconds")
    print(batch['spectrogram'].size())
    for batch in dataloader:
        print(batch['spectrogram'].size())
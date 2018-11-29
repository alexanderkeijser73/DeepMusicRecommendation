import torch
import torch.utils as utils
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from librosa.core import get_duration, load
from librosa.feature import melspectrogram
import numpy as np
import pickle

class SpectrogramDataset(Dataset):
    """Dataset with mel-spectograms for audio samples"""

    def __init__(self, root_dir, latent_factors, wmf_item2i, track_to_song, transform=None):
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
        files = {}
        for file_name in sorted(os.listdir(root_dir)):
            song_name = track_to_song[os.path.splitext(file_name)[0]]
            if song_name in wmf_item2i.keys():
                files[number] = file_name

                assert get_duration(filename=os.path.join(root_dir, file_name)) == 30
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio_name = os.path.join(self.root_dir,
                                  self.files[idx])
        y, sr = load(audio_name)
        mel_spectrogram = melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=128)
        latent_factors = self.latent_factors[idx]
        sample = {'spectrogram': mel_spectrogram, 'latent_factors': latent_factors}

        if self.transform:
            sample - self.transform(sample)

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
        spectogram, latent_factors = sample['spectrogram'], sample['latent_factors']

        return {'spectrogram': torch.from_numpy(spectrograms),
                'latent_factors': torch.from_numpy(latent_factors)}

if __name__ == '__main__':
    item_factors = pickle.load(open('../../item_wmf_50.pkl', 'rb'))
    wmf_item2i = pickle.load(open('../../index_dicts.pkl', 'rb'))['item2i']
    track_to_song = pickle.load(open('../../track_to_song.pkl', 'rb'))
    transformed_dataset = SpectrogramDataset(root_dir='../../data/MillionSongSubset/audio',
                                               latent_factors=item_factors,
                                               wmf_item2i = wmf_item2i,
                                                track_to_song=track_to_song,
                                               transform=transforms.Compose([
                                                   LogCompress(),
                                                   ToTensor()
                                                   ]))
    print(len(transformed_dataset))
    dataloader = DataLoader(transformed_dataset, batch_size=4,
                            shuffle=True, num_workers=4)

    # # Helper function to show a batch
    def show_batch(sample_batched):
        """Show spectograms for a batch of samples."""
        images_batch = sample_batched['spectograms']
        batch_size = len(images_batch)
        im_size = images_batch.size(2)

        grid = utils.make_grid(images_batch)
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
        plt.title('Batch from dataloader')
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(),
              sample_batched['latent_factors'].size())
        print('Jode')
        # observe 4th batch and stop.
        if i_batch == 3:
            plt.figure()
            show_batch(sample_batched)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break
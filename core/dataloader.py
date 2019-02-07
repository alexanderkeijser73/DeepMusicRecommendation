import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import numpy as np
import pickle
import time
import glob

class SpectrogramDataset(Dataset):
    """Dataset with mel-spectrograms     for audio samples"""

    def __init__(self, root_dir, user_item_matrix, item_factors, user_factors,
                 wmf_item2i, wmf_user2i, track_to_song,
                 track_id_to_info, file_type='.npy', transform=None):
        """
        Args:
            root_dir (string): Directory with all the audio samples.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.item_factors = item_factors
        self.user_item_matrix = user_item_matrix
        self.user_factors = user_factors
        self.wmf_item2i = wmf_item2i
        self.wmf_user2i = wmf_user2i
        self.tra2so = track_to_song
        self.track_id_to_info = track_id_to_info
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
        file_name = os.path.join(self.root_dir,
                                  self.files[idx])
        try:
            mel_spectrogram = np.load(file_name)[:, :1280]
        except Exception:
            raise ValueError(f'unable to load pickle file {file_name}')
        assert mel_spectrogram.shape == (128, 1280), f'found shape: {mel_spectrogram.shape} for example: {self.files[idx]}'
        item_factors = self.item_factors[idx]
        assert item_factors.shape == (50,), f'found shape: {item_factors.shape} for example: {self.files[idx]}'

        item_play_counts = self.user_item_matrix[:, idx]
        # use round and clamp functions to make play counts binary targets
        item_play_counts.data = np.around(np.clip(item_play_counts.data, 0, 1))
        # metadata
        track_name = os.path.basename(file_name)
        track_id = os.path.splitext(track_name)[0]
        track_info = self.track_id_to_info[track_id]
        track_info_str = f"{str(track_info['artist_name'], 'utf-8')} - {str(int(track_info['tempo']))}"
        sample = {'spectrogram': mel_spectrogram,
                  'item_factors': item_factors,
                  'item_play_counts': item_play_counts,
                  'track_info_str': track_info_str
                  }

        if self.transform:
            sample = self.transform(sample)
        return sample

class LogCompress(object):
    """"Applies log-compression to input array"""

    def __init__(self, offset=1e-5):
        self.offset = offset

    def __call__(self, sample):
        spectrogram, item_factors, item_play_counts, track_info_str = sample['spectrogram'], \
                                                                 sample['item_factors'], \
                                                                 sample['item_play_counts'], \
                                                                 sample['track_info_str']

        log_mel_spectrograms = np.log(spectrogram + self.offset)

        return {'spectrogram': log_mel_spectrograms,
                'item_factors': item_factors,
                'item_play_counts': item_play_counts,
                'track_info_str': track_info_str}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        spectrogram, item_factors, item_play_counts, track_info_str = sample['spectrogram'], \
                                                                 sample['item_factors'], \
                                                                 sample['item_play_counts'], \
                                                                 sample['track_info_str']

        return {'spectrogram': torch.from_numpy(spectrogram).type(torch.FloatTensor),
                'item_factors': torch.from_numpy(item_factors).type(torch.FloatTensor),
                'item_play_counts': torch.from_numpy(item_play_counts.toarray()).type(torch.FloatTensor),
                'track_info_str': track_info_str
                }

if __name__ == '__main__':
    item_factors = pickle.load(open('../data/wmf/item_wmf_50.pkl', 'rb'))
    user_factors = pickle.load(open('../data/wmf/user_wmf_50.pkl', 'rb'))
    wmf_item2i = pickle.load(open('../data/wmf/index_dicts.pkl', 'rb'))['item2i']
    wmf_user2i = pickle.load(open('../data/wmf/index_dicts.pkl', 'rb'))['user2i']
    track_to_song = pickle.load(open('../data/wmf/track_to_song.pkl', 'rb'))
    user_item_matrix = pickle.load(open('../data/wmf/user_item_matrix.pkl', 'rb'))
    track_id_to_info = pickle.load(open( '../data/song_metadata/track_id_to_info.pkl', 'rb'))

    start_time = time.time()
    transformed_dataset = SpectrogramDataset(root_dir='../data/spectrograms',
                                             user_item_matrix=user_item_matrix,
                                             item_factors=item_factors,
                                             user_factors=user_factors,
                                             wmf_item2i=wmf_item2i,
                                             wmf_user2i=wmf_user2i,
                                             track_to_song=track_to_song,
                                             track_id_to_info=track_id_to_info,
                                             transform=transforms.Compose([
                                                 LogCompress(),
                                                 ToTensor()
                                             ])
                                             )
    print("Dataset size:", len(transformed_dataset))

    dataloader = DataLoader(transformed_dataset, batch_size=64,
                            shuffle=True, num_workers=4)
    dataloader_iter = iter(dataloader)
    start_time = time.time()
    batch = dataloader_iter.next()

    print(f"Loading one batch took {time.time() - start_time} seconds")
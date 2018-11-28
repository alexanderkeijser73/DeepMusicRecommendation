from torch.utils.data import Dataset, Dataloader
import os
import librosa.load as load
from librosa.core import get_duration
from librosa.feature import melspectrogram
import numpy as np

class SpectrogramDataset(Dataset):
    """Dataset with mel-spectograms for audio samples"""

    def __init__(self, root_dir, latent_factors, wfm_item2i, transform=None):
        """
        Args:
            root_dir (string): Directory with all the audio samples.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.latent_factors = item_latent_factors
        self.wmf_item2i = wfm_item2
        number = 0
        files = {}
        for file_name in sorted(os.listdir(root_dir)):
            if os.path.splitext(file_name)[0] in wmf_item2i.keys():
                files[number] = file_name
                assert get_duration(filename=os.path.join(root_dir, file_name)) == 30
        self.files = files

    def __len__(self):
        return len(files)

    def __getitem__(self, idx):
        audio_name = os.path.join(self.root_dir,
                                  self.files[idx])
        y, sr = load(audio_name)
        mel_spectrogram = melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=128)
        log_offset = 1e-6
        log_mel_spectrogram = np.log(mel_spectrogram + log_offset)
        latent_factors = self.latent_factors[idx]
        sample = {'spectrogram': log_mel_spectrogram, 'latent_factors': latent_factors}

        if self.transform:
            sample - self.transform(sample)

        return sample
import torch.nn as nn
import pickle
import time
from src.dataset import SpectrogramDataset, LogCompress, ToTensor
from torch.utils.data import DataLoader
from torchvision import transforms


if __name__ == '__main__':

    item_factors = pickle.load(open('../item_wmf_50.pkl', 'rb'))
    wmf_item2i = pickle.load(open('../index_dicts.pkl', 'rb'))['item2i']
    track_to_song = pickle.load(open('../track_to_song.pkl', 'rb'))
    start_time = time.time()
    transformed_dataset = SpectrogramDataset(root_dir='../data/MillionSongSubset/audio',
                                               latent_factors=item_factors,
                                               wmf_item2i = wmf_item2i,
                                                track_to_song=track_to_song,
                                               transform=transforms.Compose([
                                                   LogCompress(),
                                                   ToTensor()
                                                   ]))
    print("Dataset size:", len(transformed_dataset))

    dataloader = DataLoader(transformed_dataset, batch_size=4,
                            shuffle=True, num_workers=4)

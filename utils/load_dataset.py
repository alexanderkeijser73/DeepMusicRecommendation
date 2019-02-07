import os
import sys
import pickle
from torchvision import transforms
from core.dataloader import SpectrogramDataset, LogCompress, ToTensor
from utils.DEFAULTS import *

def load_dataset(data_path=data_path,
                 user_item_matrix=user_item_matrix,
                 wmf_item2i=wmf_item2i,
                 wmf_user2i=wmf_user2i,
                 track_to_song=track_to_song,
                 item_factors=item_factors,
                 user_factors=user_factors,
                 track_id_to_info=track_id_to_info):

    user_item_matrix = pickle.load(open(os.path.join(data_path, user_item_matrix), 'rb'))
    wmf_item2i = pickle.load(open(os.path.join(data_path, wmf_item2i), 'rb'))['item2i']
    wmf_user2i = pickle.load(open(os.path.join(data_path, wmf_user2i), 'rb'))['user2i']
    track_to_song = pickle.load(open(os.path.join(data_path, track_to_song), 'rb'))
    item_factors = pickle.load(open(os.path.join(data_path, item_factors), 'rb'))
    user_factors = pickle.load(open(os.path.join(data_path, user_factors), 'rb'))
    track_id_to_info = pickle.load(open(os.path.join(data_path, track_id_to_info), 'rb'))
    spectrogram_path = os.path.join(data_path, 'spectrograms')

    transformed_dataset = SpectrogramDataset(root_dir=spectrogram_path,
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
    return transformed_dataset
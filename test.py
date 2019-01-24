from time import strftime
import pickle
from datetime import datetime
import numpy as np
import time
from src.dataloader import SpectrogramDataset, LogCompress, ToTensor
from src.cnn import AudioCNN
from src.train_parameters import load_train_parameters
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from src.train_utils import *

data_path = 'data/test_spectrograms'

user_item_matrix  = pickle.load(open(os.path.join(data_path, '../wmf/user_item_matrix.pkl'), 'rb'))
wmf_item2i = pickle.load(open(os.path.join(data_path, '../wmf/index_dicts.pkl'), 'rb'))['item2i']
wmf_user2i = pickle.load(open(os.path.join(data_path, '../wmf/index_dicts.pkl'), 'rb'))['user2i']
track_to_song = pickle.load(open(os.path.join(data_path, '../wmf/track_to_song.pkl'), 'rb'))
item_factors = pickle.load(open(os.path.join(data_path,  '../wmf/item_wmf_50.pkl'), 'rb'))
user_factors = pickle.load(open(os.path.join(data_path,  '../wmf/user_wmf_50.pkl'), 'rb'))

start_time = time.time()
print('creating dataset')
transformed_dataset = SpectrogramDataset(root_dir=data_path,
                                        user_item_matrix=user_item_matrix,
                                        item_factors=item_factors,
                                        user_factors=user_factors,
                                        wmf_item2i = wmf_item2i,
                                        wmf_user2i=wmf_user2i,
                                        track_to_song=track_to_song,
                                        transform=transforms.Compose([
                                                       LogCompress(),
                                                       ToTensor()
                                                                    ])
                                        )
print(f"Dataset size: {len(transformed_dataset)}")


test_dl = torch.utils.data.DataLoader(transformed_dataset,
                                           batch_size=len(transformed_dataset))

checkpoint = load_checkpoint('checkpoints_cnn/best_model_auc_0_72.pth.tar')
model = checkpoint['model']
print('checkpoint loaded')
valid_batch = iter(test_dl).next()
valid_data, valid_targets, valid_play_count_targets = valid_batch['spectrogram'], \
                                                      valid_batch['item_factors'], \
                                                      valid_batch['item_play_counts']
print('processing test examples')
item_factor_predictions = model(valid_data)

print('calculating predictions')
# Calculate accuracy
play_count_predictions = calc_play_counts(item_factor_predictions,
                                          user_factors)
print('calculating auc')
valid_auc = calc_auc(play_count_predictions, valid_play_count_targets)
print('valid auc', valid_auc)
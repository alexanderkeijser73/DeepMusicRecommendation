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
batch_size = 16

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
dataset_size = len(transformed_dataset)
print(f"Dataset size: {dataset_size}")


test_dl = torch.utils.data.DataLoader(transformed_dataset,
                                           batch_size=batch_size)

checkpoint = load_checkpoint('checkpoints_cnn/best_model_auc_0_72.pth.tar')
model = AudioCNN()
model.load_state_dict(checkpoint['model'])
model.eval()
print('checkpoint loaded')

predictions = torch.empty((0, user_item_matrix.shape[0]))
targets = torch.empty((0, user_item_matrix.shape[0]))

print('processing test examples')
for i, test_batch in enumerate(test_dl):
    test_data, test_targets, test_play_count_targets = test_batch['spectrogram'], \
                                                          test_batch['item_factors'], \
                                                          test_batch['item_play_counts']
    item_factor_predictions = model(test_data)

    # Calculate accuracy
    play_count_predictions = calc_play_counts(item_factor_predictions,
                                              user_factors)
    predictions = torch.cat((predictions, play_count_predictions), dim=0)
    targets = torch.cat((targets, torch.squeeze(test_play_count_targets)), dim=0)
    print(f'calculated {16*(i+1)}/{dataset_size} predictions')
    if i==2: break

print('calculating AUC')
auc = calc_auc(predictions, targets)
print(auc)
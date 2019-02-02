import torch
import os
from torch.autograd.variable import Variable
from tensorboardX import SummaryWriter
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

config = load_train_parameters()

user_item_matrix = pickle.load(open(os.path.join(config.data_path, '../wmf/user_item_matrix.pkl'), 'rb'))
wmf_item2i = pickle.load(open(os.path.join(config.data_path, '../wmf/index_dicts.pkl'), 'rb'))['item2i']
wmf_user2i = pickle.load(open(os.path.join(config.data_path, '../wmf/index_dicts.pkl'), 'rb'))['user2i']
track_to_song = pickle.load(open(os.path.join(config.data_path, '../wmf/track_to_song.pkl'), 'rb'))
item_factors = pickle.load(open(os.path.join(config.data_path, '../wmf/item_wmf_50.pkl'), 'rb'))
user_factors = pickle.load(open(os.path.join(config.data_path, '../wmf/user_wmf_50.pkl'), 'rb'))

start_time = time.time()
transformed_dataset = SpectrogramDataset(root_dir=config.data_path,
                                         user_item_matrix=user_item_matrix,
                                         item_factors=item_factors,
                                         user_factors=user_factors,
                                         wmf_item2i=wmf_item2i,
                                         wmf_user2i=wmf_user2i,
                                         track_to_song=track_to_song,
                                         transform=transforms.Compose([
                                             LogCompress(),
                                             ToTensor()
                                         ])
                                         )

print(f"Dataset size: {len(transformed_dataset)}")

train_dl = torch.utils.data.DataLoader(transformed_dataset,
                                           batch_size=config.batch_size)
n_batches = len(train_dl)

writer = SummaryWriter(comment='tsne_embedding')

model = AudioCNN()

# Load checkpoint
if config.checkpoint:
    checkpoint = torch.load(config.checkpoint)
    model.load_state_dict(checkpoint['model'])
    print("Checkpoint loaded")

if torch.cuda.is_available():
    print('training on GPU!')
    model.cuda()

model.eval()

features = torch.empty((0, 2048)) #TODO: REMOVE EXPLICIT LAST LAYER SIZE - DETERMINE FROM MODEL

for i, batch in enumerate(train_dl):

    print(f"Processing batch {i}/{n_batches}")

    # load new batch
    batch_data, batch_targets, batch_play_count_targets = batch['spectrogram'], \
                                                          batch['item_factors'], \
                                                          batch['item_play_counts']

    if torch.cuda.is_available():
        batch_data, batch_targets = batch_data.cuda(), batch_targets.cuda()

    # Forward pass to get predicted latent factors
    item_factor_predictions, batch_features = model(batch_data)

    features = torch.cat((features, batch_features), dim=0)

writer.add_embedding(
    item_factor_predictions,
    metadata=batch_targets.data)

writer.close()

# tensorboard --logdir runs
# you should now see a dropdown list with all the timestep,
# last timestep should have a visible separation between the two classes
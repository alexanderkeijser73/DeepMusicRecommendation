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

def train(train_dl, valid_dl, config):
    """ Train the model given the parameters in the config object
    """

    writer = make_summary_writer(config.summary_path, time_now)

    model = AudioCNN()
    if torch.cuda.is_available():
        logger.info('training on GPU!')
        model.cuda()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Load checkpoint
    if config.checkpoint:
        checkpoint = torch.load(config.checkpoint)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("Checkpoint loaded")

    total_loss = 0
    best_auc = 0
    for epoch in range(config.num_epochs):
        for i, batch in enumerate(train_dl):
            t1 = time.time()

            # load new batch
            batch_data, batch_targets, batch_play_count_targets = batch['spectrogram'], \
                                                                batch['item_factors'], \
                                                                batch['item_play_counts']

            if torch.cuda.is_available():
                batch_data, batch_targets = batch_data.cuda(), batch_targets.cuda()

            optimizer.zero_grad()


            # Forward pass to get predicted latent factors
            item_factor_predictions = model(batch_data)

            # Calculate MSE loss
            loss = criterion(item_factor_predictions, batch_targets)
            total_loss += loss.item()

            # Write the outcomes to the tensorboard
            n_iter = (epoch * len(train_dl)) + i
            writer.add_scalar('train_loss', loss.item(), n_iter)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()
            t2 = time.time()
            examples_per_second = config.batch_size / float(t2 - t1)

            if i % config.print_every == 0:

                logger.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M")}]\t '
                            f'Epoch {epoch}\t '
                            f'Batch {i}\t '
                            f'Loss {loss.item():.2g} \t '
                            f'Examples/Sec = {examples_per_second:.2f},'
                            )
                total_loss = 0

            if config.validate_every:
                if i % config.validate_every == 0:
                    valid_batch = iter(valid_dl).next()
                    valid_data, valid_targets, valid_play_count_targets = valid_batch['spectrogram'], \
                                                                   valid_batch['item_factors'], \
                                                                   valid_batch['item_play_counts']
                    item_factor_predictions = model(valid_data)

                    # Calculate MSE loss
                    valid_loss = criterion(item_factor_predictions, valid_targets)
                    writer.add_scalar('validation_loss', valid_loss.item(), n_iter)

                    # Calculate accuracy
                    play_count_predictions = calc_play_counts(item_factor_predictions,
                                                              user_factors)
                    valid_auc = calc_auc(play_count_predictions, valid_play_count_targets)
                    writer.add_scalar('validation_auc', valid_auc, n_iter)

                    logger.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M")}]\t '
                                f'Epoch {epoch}\t '
                                f'Batch {i}\t '
                                f'Loss {valid_loss.item():.2g} \t '
                                f'Valid auc {valid_auc:.2f} \t'
                                f'Examples/Sec = {examples_per_second:.2f},'
                                )
                    if valid_auc > best_auc:
                        best_auc = valid_auc
                        save_checkpoint(model,
                                        optimizer,
                                        config.checkpoint_path,
                                        filename=f'best_model_{time_now}.pth.tar',
                                        auc=valid_auc)

            if config.save_every:
                if i % config.save_every == 0:
                    save_checkpoint(model, optimizer, config.checkpoint_path)

if __name__ == "__main__":

    global time_now
    global logger
    config = load_train_parameters()
    time_now = strftime('%d_%b_%H_%M_%S')
    logger = make_logger(time_now)
    print_flags(config, logger)

    user_item_matrix = pickle.load(open(os.path.join(config.data_path, '../wmf/user_item_matrix.pkl'), 'rb'))
    wmf_item2i = pickle.load(open(os.path.join(config.data_path, '../wmf/index_dicts.pkl'), 'rb'))['item2i']
    wmf_user2i = pickle.load(open(os.path.join(config.data_path, '../wmf/index_dicts.pkl'), 'rb'))['user2i']
    track_to_song = pickle.load(open(os.path.join(config.data_path, '../wmf/track_to_song.pkl'), 'rb'))
    item_factors = pickle.load(open(os.path.join(config.data_path,  '../wmf/item_wmf_50.pkl'), 'rb'))
    user_factors = pickle.load(open(os.path.join(config.data_path,  '../wmf/user_wmf_50.pkl'), 'rb'))

    start_time = time.time()
    transformed_dataset = SpectrogramDataset(root_dir=config.data_path,
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
    logger.info(f"Dataset size: {len(transformed_dataset)}")

    validation_split = .2
    shuffle_dataset = True
    random_seed =42

    # Creating data indices for training and validation splits:
    dataset_size = len(transformed_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(transformed_dataset,
                                               batch_size=config.batch_size,
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(transformed_dataset,
                                                    batch_size=config.val_batch_size,
                                                    sampler=valid_sampler)

    train(train_loader, validation_loader, config)
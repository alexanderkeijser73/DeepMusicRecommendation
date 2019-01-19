import os
from time import strftime
import pickle
import argparse
from datetime import datetime
from tensorboardX import SummaryWriter
import numpy as np
import torch.nn as nn
import time
from src.dataloader import SpectrogramDataset, LogCompress, ToTensor
from src.cnn import AudioCNN
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import logging
from sklearn.metrics import roc_auc_score

def calc_play_counts(item_factor_prediction, user_factors, play_count_targets):
    """Calculate play_counts from predicted item factors
    """
    play_count_predictions = item_factor_prediction @ torch.t(torch.from_numpy(user_factors))
    binary_predictions = torch.round(torch.clamp(play_count_predictions, 0, 1))
    return binary_predictions

def calc_accuracy(predictions, play_count_targets):
    """Calculate accuracy of play_counts calculated from predicted item factors
    """
    predictions = torch.flatten(predictions)
    targets = torch.flatten(play_count_targets)
    accuracy = float(torch.sum(targets==predictions))/targets.numel()
    return accuracy

def calc_auc(predictions, play_count_targets):
    """Calculate auc of play_counts calculated from predicted item factors
    """
    predictions = torch.flatten(predictions)
    targets = torch.flatten(play_count_targets)
    auc = roc_auc_score(targets, predictions)
    return auc

def make_logger():
    if not os.path.exists('../logs'):
        os.mkdir('../logs')
    logging.basicConfig(filename=f'../logs//{time_now}.log', filemode='w', format='%(message)s', level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    return logger

def make_summary_writer(base_dir):
    summary_path = os.path.join(base_dir, time_now)
    if not os.path.exists(summary_path):
        os.mkdir(summary_path)
        writer = SummaryWriter(summary_path)
    else:
        raise OSError('Summary directory already exists.')
    return writer


def train(train_dl, valid_dl, config):
    """ Train the model given the parameters in the config object
    """

    writer = make_summary_writer(config.summary_path)

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
        # epoch = checkpoint['epoch']
        logger.info("Checkpoint loaded")

    total_loss = 0
    best_loss = 1e10
    for epoch in range(config.num_epochs):
        for i, batch in enumerate(train_dl):
            t1 = time.time()

            # load new batch
            batch_data, batch_targets, batch_item_play_counts = batch['spectrogram'], \
                                                                batch['item_factors'], \
                                                                batch['item_play_counts']

            if torch.cuda.is_available():
                batch_data, batch_targets = batch_data.cuda(), batch_targets.cuda()

            optimizer.zero_grad()


            # Forward pass to get predicted latent factors
            outputs = model(batch_data)

            # Calculate MSE loss
            loss = criterion(outputs, batch_targets)
            total_loss += loss.item()

            n_iter = (epoch * len(train_dl)) + i
            # Write the outcomes to the tensorboard
            writer.add_scalar('train_loss', loss.item(), n_iter)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()
            t2 = time.time()
            examples_per_second = config.batch_size / float(t2 - t1)

            if i % config.print_every == 0:
                logger.info('[{}]\t Epoch {}\t Batch {}\t Loss {} \t Examples/Sec = {:.2f},'.format(datetime.now().strftime("%Y-%m-%d %H:%M"),
                                                                                  epoch, i,  loss.item(),
                                                                                  examples_per_second))
                total_loss = 0

            if config.validate_every:
                if i % config.validate_every == 0:
                    valid_batch = iter(valid_dl).next()
                    valid_data, valid_targets, valid_play_counts = valid_batch['spectrogram'], \
                                                                   valid_batch['item_factors'], \
                                                                   valid_batch['item_play_counts']
                    outputs = model(valid_data)

                    # Calculate MSE loss
                    valid_loss = criterion(outputs, valid_targets)
                    writer.add_scalar('validation_loss', valid_loss.item(), n_iter)
                    logger.info('[{}]\t Epoch {}\t Batch {}\t Validation Loss {} \t'.format(datetime.now().strftime("%Y-%m-%d %H:%M"),
                                                                                      epoch, i,  valid_loss.item(),
                                                                                      examples_per_second))
                    if valid_loss.item() < best_loss:
                        best_loss = valid_loss.item()
                        checkpoint = {
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                        }
                        torch.save(checkpoint, f'best_model_{time_now}.pt')

            if config.save_every:
                if i % config.save_every == 0:
                    save_checkpoint(model, optimizer, config.checkpoint_path)

def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    logger.info('\n')
    for key, value in vars(config).items():
        logger.info(key + ' : ' + str(value))
    logger.info('\n')


def save_checkpoint(model, optimizer, path):
    """ Save the trained model checkpoint
    """
    if not os.path.exists(path):
        os.mkdir(path)
    filename = "{}/checkpoint_{}.pth.tar".format(path, datetime.now().strftime("%d_%m_%H_%M"),)
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        }
    torch.save(checkpoint, filename)


def load_checkpoint(filepath):
    """ Load a previously trained checkpoint
    """
    checkpoint = torch.load(filepath)
    return checkpoint

if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--data_path', type=str, default='data/spectrograms')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints_cnn', help='Path to checkpoint file')

    # Training params
    parser.add_argument('--batch_size', type=int, default=16, help='Number of examples to process in a batch')
    parser.add_argument('--val_batch_size', type=int, default=256, help='Number of examples to use in validation step')
    parser.add_argument('--learning_rate', type=float, default=0.03, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=25, help='Number of epochs')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="../tensorboard_summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=10, help='How often to print training progress')
    parser.add_argument('--save_every', type=int, default=250, help='How often to save checkpoint')
    parser.add_argument('--validate_every', type=int, default=10, help='How often to evaluate on validation set')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file')
    parser.add_argument('--test_size', type=int, default=1000, help='Number of samples in the test')

    # Test Args
    parser.add_argument('--testing', type=int, default=0, help='Will the network train or only perform a test')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model to test')

    config = parser.parse_args()

    global time_now
    time_now = strftime('%d_%b_%H_%M_%S')
    global logger
    logger = make_logger()

    print_flags()

    # Train the model
    user_item_matrix = pickle.load(open(os.path.join(config.data_path, '../wmf/user_item_matrix.pkl'), 'rb'))
    item_factors = pickle.load(open(os.path.join(config.data_path, '../wmf/item_wmf_50.pkl'), 'rb'))
    user_factors = pickle.load(open(os.path.join(config.data_path, '../wmf/user_wmf_50.pkl'), 'rb'))
    print(user_factors.shape, item_factors.shape)
    wmf_item2i = pickle.load(open(os.path.join(config.data_path, '../wmf/index_dicts.pkl'), 'rb'))['item2i']
    wmf_user2i = pickle.load(open(os.path.join(config.data_path, '../wmf/index_dicts.pkl'), 'rb'))['user2i']
    track_to_song = pickle.load(open(os.path.join(config.data_path, '../wmf/track_to_song.pkl'), 'rb'))

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
                                                   ]))
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

    train_loader = torch.utils.data.DataLoader(transformed_dataset, batch_size=config.batch_size,
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(transformed_dataset, batch_size=config.val_batch_size,
                                                    sampler=valid_sampler)

    train(train_loader, validation_loader, config)
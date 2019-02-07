import torch
import logging
from sklearn.metrics import roc_auc_score
from tensorboardX import SummaryWriter
import os
from datetime import datetime
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import argparse
from utils.DEFAULTS import *

def calc_play_counts(item_factor_prediction, user_factors):
    """Calculate play_counts from predicted item factors
    """
    play_count_predictions = item_factor_prediction @ torch.t(torch.from_numpy(user_factors))
    predictions = torch.clamp(play_count_predictions, 0, 1)
    return predictions

def calc_accuracy(predictions, play_count_targets):
    """Calculate accuracy of play_counts calculated from predicted item factors
    """
    predictions = torch.flatten(predictions)
    targets = torch.flatten(play_count_targets)
    accuracy = float(torch.sum(torch.round(targets)==predictions))/targets.numel()
    return accuracy

def calc_auc(predictions, play_count_targets):
    """Calculate auc of play_counts calculated from predicted item factors
    """
    predictions = torch.flatten(predictions)
    targets = torch.flatten(play_count_targets)
    auc = roc_auc_score(targets.detach().numpy(), predictions.detach().numpy())
    return auc

def make_logger(time_now):
    if not os.path.exists('../logs'):
        os.mkdir('../logs')
    logging.basicConfig(filename=f'../logs//{time_now}.log', filemode='w', format='%(message)s', level=logging.DEBUG)
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
    logger = logging.getLogger(__name__)
    return logger

def make_summary_writer(base_dir, time_now):
    summary_path = os.path.join(base_dir, time_now)
    if not os.path.exists(summary_path):
        os.mkdir(summary_path)
        writer = SummaryWriter(summary_path)
    else:
        raise OSError('Summary directory already exists.')
    return writer

def print_flags(config, logger):
    """
    Prints all entries in FLAGS variable.
    """
    logger.info('\n')
    for key, value in vars(config).items():
        logger.info(key + ' : ' + str(value))
    logger.info('\n')


def save_checkpoint(model, optimizer, rootdir, filename=None, auc=None):
    """ Save the trained model checkpoint
    """
    if not os.path.exists(rootdir):
        os.mkdir(rootdir)
    if filename is None:
        time_stamp = datetime.now().strftime("%d_%m_%H_%M")
        filename = f'checkpoint_{time_stamp}.pth.tar'
    path = os.path.join(rootdir, filename)
    checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'auc' : auc
    }
    torch.save(checkpoint, path)


def load_checkpoint(filepath):
    """ Load a previously trained checkpoint
    """
    checkpoint = torch.load(filepath)
    return checkpoint

def split_train_valid(dataset,
                      train_batch_size=16,
                      valid_batch_size=256,
                      shuffle_dataset=True,
                      validation_split=0.2):
    random_seed =42

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=train_batch_size,
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset,
                                                    batch_size=valid_batch_size,
                                                    sampler=valid_sampler)
    return train_loader, validation_loader

def load_train_parameters():
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--data_path', type=str, default=data_path, help='Path to spectrogram data directory')
    parser.add_argument('--checkpoint_path', type=str, default=checkpoint_path, help='Path to checkpoint file')

    # Training params
    parser.add_argument('--batch_size', type=int, default=batch_size, help='Number of examples to process in a batch')
    parser.add_argument('--valid_batch_size', type=int, default=valid_batch_size, help='Number of examples to use in validation step')
    parser.add_argument('--learning_rate', type=float, default=learning_rate, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=num_epochs, help='Number of epochs')
    parser.add_argument('--max_norm', type=float, default=max_norm, help='Max allowed gradient norm before clipping value')

    # Misc params
    parser.add_argument('--summary_path', type=str, default=summary_path, help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=print_every, help='How often to print training progress')
    parser.add_argument('--save_every', type=int, default=save_every, help='How often to save checkpoint')
    parser.add_argument('--validate_every', type=int, default=validate_every, help='How often to evaluate on validation set')
    parser.add_argument('--checkpoint', type=str, default=prev_checkpoint_path, help='Path to checkpoint file')

    config = parser.parse_args()
    return config
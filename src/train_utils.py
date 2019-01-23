import torch
import logging
from sklearn.metrics import roc_auc_score
from tensorboardX import SummaryWriter
import os
from datetime import datetime

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
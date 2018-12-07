import os
import time
import pickle
import argparse
from datetime import datetime
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import time
from src.dataloader import SpectrogramDataset, LogCompress, ToTensor
from src.cnn import AudioCNN
import torch
from torch.utils.data import DataLoader
from torchvision import transforms



def calc_accuracy(output, batch_targets):
    """ Calculate the accuracy of a prediction given labels
    """
    predictions = torch.argmax(output, dim=1)
    correct_predictions = torch.eq(predictions, batch_targets).sum()
    return float(correct_predictions.item() / len(batch_targets)) * 100


def train(train_dl, valid_dl, config):
    """ Train the model given the parameters in the config object
    """
    writer = SummaryWriter(config.summary_path)

    model = AudioCNN()
    if torch.cuda.is_available():
        print('kakoe')
        model.cuda()

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)

    # Load checkpoint
    if config.checkpoint:
        checkpoint = torch.load(config.checkpoint)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # epoch = checkpoint['epoch']
        print("Checkpoint loaded")

    total_loss = 0
    best_loss = 1e10
    for epoch in range(config.num_epochs):
        for i, batch in enumerate(train_dl):
            t1 = time.time()

            # load new batch
            batch_data, batch_targets = batch['spectrogram'], batch['latent_factors']

            if torch.cuda.is_available():
                batch_data, batch_targets = batch_data.cuda(), batch_targets.cuda()

            optimizer.zero_grad()


            # Forward pass to get predicted latent factors
            outputs = model(batch_data)

            # Calculate MSE loss
            loss = criterion(outputs, batch_targets)
            total_loss += loss.item()

            n_iter = (epoch * len(dataloader)) + i
            # Write the outcomes to the tensorboard
            writer.add_scalar('train_loss', loss.item(), n_iter)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()
            t2 = time.time()
            examples_per_second = config.batch_size / float(t2 - t1)

            if i % config.print_every == 0:
                print('[{}]\t Epoch {}\t Batch {}\t Loss {} \t Examples/Sec = {:.2f},'.format(datetime.now().strftime("%Y-%m-%d %H:%M"),
                                                                                  epoch, i,  loss.item(),
                                                                                  examples_per_second))
                total_loss = 0

            if i % config.validation_freq == 0:
                valid_batch = next(iter(valid_dl()))
                valid_data, valid_targets = valid_batch['spectrogram'], valid_batch['latent_factors']
                outputs = model(valid_data)

                # Calculate MSE loss
                valid_loss = criterion(outputs, valid_targets)
                writer.add_scalar('validation_loss', valid_loss.item(), n_iter)
                print('[{}]\t Epoch {}\t Batch {}\t Validation Loss {} \t'.format(datetime.now().strftime("%Y-%m-%d %H:%M"),
                                                                                  epoch, i,  valid_loss.item(),
                                                                                  examples_per_second))
                if valid_loss.item() < best_loss:
                    best_loss = valid_loss.item()
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                    torch.save(checkpoint, 'best_model.pt')

            if config.save_every:
                if i % config.save_every == 0:
                    save_checkpoint(model, optimizer, config.checkpoint_path)





def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    print('\n')
    for key, value in vars(config).items():
        print(key + ' : ' + str(value))
    print('\n')


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
    parser.add_argument('--data_path', type=str, default='../data/MillionSongSubset/spectrograms')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints_cnn', help='Path to checkpoint file')

    # Training params
    parser.add_argument('--batch_size', type=int, default=16, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=25, help='Number of epochs')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./tensorboard_summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=10, help='How often to print training progress')
    parser.add_argument('--test_every', type=int, default=100, help='How often to test the model')
    parser.add_argument('--save_every', type=int, default=None, help='How often to save checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file')
    parser.add_argument('--test_size', type=int, default=1000, help='Number of samples in the test')

    # Test Args
    parser.add_argument('--testing', type=int, default=0, help='Will the network train or only perform a test')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model to test')

    config = parser.parse_args()

    # print_flags()

    # Train the model
    item_factors = pickle.load(open('../item_wmf_50.pkl', 'rb'))
    wmf_item2i = pickle.load(open('../index_dicts.pkl', 'rb'))['item2i']
    track_to_song = pickle.load(open('../track_to_song.pkl', 'rb'))

    start_time = time.time()
    transformed_dataset = SpectrogramDataset(root_dir=config.data_path,
                                               latent_factors=item_factors,
                                               wmf_item2i = wmf_item2i,
                                                track_to_song=track_to_song,
                                               transform=transforms.Compose([
                                                   LogCompress(),
                                                   ToTensor()
                                                   ]))
    print("Dataset size:", len(transformed_dataset))

    train_size = int(0.6 * len(transformed_dataset))
    test_size = int(0.2 * len(transformed_dataset))
    validation_size = len(transformed_dataset) - train_size - test_size
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(
        transformed_dataset,
        [train_size, validation_size, test_size]
    )

    train_dl = DataLoader(train_dataset, batch_size=config.batch_size,
                            shuffle=True, num_workers=4)
    valid_dl = DataLoader(validation_dataset, batch_size=validation_size,
                            shuffle=False, num_workers=4)

    train(train_dl, valid_dl, config)
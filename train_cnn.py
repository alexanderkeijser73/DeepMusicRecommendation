import time
from core.cnn import AudioCNN
import torch.nn as nn
from utils.train_utils import *
from utils.load_dataset import load_dataset

def train(train_dl, valid_dl, config, logger=None):
    """ Train the model given the parameters in the config object
    """

    writer = make_summary_writer(config.summary_path, time_now)

    model = AudioCNN()
    if torch.cuda.is_available():
        if logger: logger.info('training on GPU!')
        model.cuda()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Load checkpoint
    if config.checkpoint:
        checkpoint = torch.load(config.checkpoint)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if logger: logger.info("Checkpoint loaded")

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
            item_factor_predictions, _ = model(batch_data)

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

                if logger: logger.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M")}]\t '
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
                    if torch.cuda.is_available():
                        valid_data, valid_targets, valid_play_count_targets = valid_data.cuda(), \
                                                                              valid_targets.cuda(), \
                                                                              valid_play_count_targets.cuda()

                    item_factor_predictions, _ = model(valid_data)

                    # Calculate MSE loss
                    valid_loss = criterion(item_factor_predictions, valid_targets)
                    writer.add_scalar('validation_loss', valid_loss.item(), n_iter)

                    # Calculate accuracy
                    play_count_predictions = calc_play_counts(item_factor_predictions,
                                                              user_factors)
                    valid_auc = calc_auc(play_count_predictions, valid_play_count_targets)
                    writer.add_scalar('validation_auc', valid_auc, n_iter)

                    if logger: logger.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M")}]\t '
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
    time_now = strftime('%d_%b_%H_%M_%S')
    logger = make_logger(time_now)

    config = load_train_parameters()
    print_flags(config, logger)
    dataset = load_dataset()
    logger.info(f"Dataset size: {len(dataset)}")

    train_loader, valid_loader = split_train_valid(dataset,
                                                   train_batch_size=config.batch_size,
                                                   valid_batch_size=config.valid_batch_size,
                                                   validation_split=0.2,
                                                   shuffle_dataset=True)

    train(train_loader, valid_loader, config, logger)
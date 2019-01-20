import argparse

def load_train_parameters():
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
    return config
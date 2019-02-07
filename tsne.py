import time
from core.cnn import AudioCNN
from torch.utils.data import DataLoader
from utils.train_utils import *
from utils.load_dataset import load_dataset

MAX_TRACKS = 1000

config = load_train_parameters()
start_time = time.time()

transformed_dataset = load_dataset(config.data_path)

print(f"Dataset size: {len(transformed_dataset)}")

train_dl = torch.utils.data.DataLoader(transformed_dataset,
                                           batch_size=config.batch_size, shuffle=True)
n_batches = len(train_dl)

writer = SummaryWriter('tsne_embedding_runs', comment='tsne_embedding')

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
metadata = []

for i, batch in enumerate(train_dl):

    print(f"Processing batch {i}/{n_batches}")

    # load new batch
    batch_data, batch_targets, batch_play_count_targets, batch_song_info = batch['spectrogram'], \
                                                          batch['item_factors'], \
                                                          batch['item_play_counts'], \
                                                          batch['track_info_str']
    if torch.cuda.is_available():
        batch_data, batch_targets = batch_data.cuda(), batch_targets.cuda()

    # Forward pass to get predicted latent factors
    _, batch_features = model(batch_data)

    features = torch.cat((features, batch_features), dim=0)
    metadata += batch_song_info

    if i * config.batch_size >= MAX_TRACKS: break


writer.add_embedding(
    features,
    metadata=metadata)

writer.close()

# tensorboard --logdir runs
# you should now see a dropdown list with all the timestep,
# last timestep should have a visible separation between the two classes
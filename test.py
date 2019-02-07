import pickle
import time
from core.dataloader import SpectrogramDataset, LogCompress, ToTensor
from core.cnn import AudioCNN
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.train_utils import *

# root_dir = '/var/scratch/akeijser/'
root_dir = ''
data_path = root_dir + 'data/test_spectrograms'
checkpoint = load_checkpoint(root_dir + 'checkpoints_cnn/best_model_auc_0_72.pth.tar')
batch_size = 16

user_item_matrix  = pickle.load(open(os.path.join(data_path, '../wmf/user_item_matrix.pkl'), 'rb'))
wmf_item2i = pickle.load(open(os.path.join(data_path, '../wmf/index_dicts.pkl'), 'rb'))['item2i']
wmf_user2i = pickle.load(open(os.path.join(data_path, '../wmf/index_dicts.pkl'), 'rb'))['user2i']
track_to_song = pickle.load(open(os.path.join(data_path, '../wmf/track_to_song.pkl'), 'rb'))
item_factors = pickle.load(open(os.path.join(data_path,  '../wmf/item_wmf_50.pkl'), 'rb'))
user_factors = pickle.load(open(os.path.join(data_path,  '../wmf/user_wmf_50.pkl'), 'rb'))
track_id_to_info = pickle.load(open(os.path.join(data_path, '../song_metadata/track_id_to_info.pkl'), 'rb'))

start_time = time.time()
print('creating dataset')
transformed_dataset = SpectrogramDataset(root_dir=data_path,
                                        user_item_matrix=user_item_matrix,
                                        item_factors=item_factors,
                                        user_factors=user_factors,
                                        wmf_item2i = wmf_item2i,
                                        wmf_user2i=wmf_user2i,
                                        track_to_song=track_to_song,
                                         track_id_to_info=track_id_to_info,
                                        transform=transforms.Compose([
                                                       LogCompress(),
                                                       ToTensor()
                                                                    ])
                                         )
dataset_size = len(transformed_dataset)
print(f"Dataset size: {dataset_size}")


test_dl = torch.utils.data.DataLoader(transformed_dataset,
                                           batch_size=batch_size)


model = AudioCNN()
model.load_state_dict(checkpoint['model'])
model.eval()
print('checkpoint loaded')

predictions = torch.empty((0, user_item_matrix.shape[0]))
targets = torch.empty((0, user_item_matrix.shape[0]))

print('processing test examples')

mean_auc = 0
for i, test_batch in enumerate(test_dl):
    test_data, test_targets, test_play_count_targets = test_batch['spectrogram'], \
                                                          test_batch['item_factors'], \
                                                          test_batch['item_play_counts']
    item_factor_predictions, _ = model(test_data)

    # Calculate accuracy
    play_count_predictions = calc_play_counts(item_factor_predictions,
                                              user_factors)

    batch_auc = calc_auc(play_count_predictions, test_play_count_targets)
    mean_auc = mean_auc + float(batch_auc - mean_auc) / (i + 1)
    print(f'calculated {batch_size*(i+1)}/{dataset_size} predictions '
          f'- batch AUC: {batch_auc:.2f} '
          f'- average AUC: {mean_auc:.2f}')

print('------------------------------------------------------\nMEAN AUC:', mean_auc)
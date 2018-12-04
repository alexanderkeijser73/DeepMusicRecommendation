import torch.nn as nn
import pickle
import time
from src.dataloader import SpectrogramDataset, LogCompress, ToTensor
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F

class AudioCNN(nn.Module):

    def __init__(self):
        super(AudioCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv1d(128, 256, 4)
        self.conv2 = nn.Conv1d(256, 256, 4)
        self.conv3 = nn.Conv1d(256, 512, 4)
        self.conv4 = nn.Conv1d(512, 512, 4)
        # Pooling layers
        self.max_pool4 = nn.MaxPool1d(4)
        self.max_pool2 = nn.MaxPool1d(2)
        self.max_pool_global = nn.MaxPool1d(74)
        self.avg_pool_global = nn.AvgPool1d(74)
        # Fully connected layers
        self.fc1 = nn.Linear(1536, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 50)

    def forward(self, input):
        # Convolutional layers
        out = F.relu(self.conv1(input))
        out = self.max_pool4(out)
        out = F.relu(self.conv2(out))
        out = self.max_pool2(out)
        out = F.relu(self.conv3(out))
        out = self.max_pool2(out)
        out = F.relu(self.conv4(out))
        # Global temporal pooling layer
        max_p = self.max_pool_global(out)
        avg_p = self.avg_pool_global(out)
        L2_p = torch.sqrt(self.avg_pool_global(out.pow(2)))
        global_temporal = torch.cat((max_p, avg_p, L2_p), dim=1).squeeze()
        # Fully connected layers
        out = F.relu(self.fc1(global_temporal))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


if __name__ == '__main__':

    item_factors = pickle.load(open('../item_wmf_50.pkl', 'rb'))
    wmf_item2i = pickle.load(open('../index_dicts.pkl', 'rb'))['item2i']
    track_to_song = pickle.load(open('../track_to_song.pkl', 'rb'))

    start_time = time.time()
    transformed_dataset = SpectrogramDataset(root_dir='../data/MillionSongSubset/audio',
                                               latent_factors=item_factors,
                                               wmf_item2i = wmf_item2i,
                                                track_to_song=track_to_song,
                                               transform=transforms.Compose([
                                                   LogCompress(),
                                                   ToTensor()
                                                   ]))
    # print(f"Time for building dataset: {time.time() - start_time}")
    # print("Dataset size:", len(transformed_dataset))

    start_time = time.time()
    dataloader = DataLoader(transformed_dataset, batch_size=64,
                            shuffle=True, num_workers=4)
    # print(f"Time for building data loader: {time.time() - start_time}")

    # start_time = time.time()
    model = AudioCNN()
    # print(f"Time for building CNN: {time.time() - start_time}")
    start_time_bla = time.time()
    for i, batch in enumerate(dataloader):
        start_time = time.time()
        # print(model(batch['spectrogram']).size())
        # print(f"Time for loading samples: {time.time() - start_time}")
        if i==3: break
    # print(f"Time voor die hele ding: {time.time() - start_time_bla}")
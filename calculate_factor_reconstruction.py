import torch
import pickle
from src.train_utils import calc_auc

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = '/var/scratch/akeijser/'

item_factors = pickle.load(open(data_dir + 'data/wmf/item_wmf_50.pkl', 'rb'))
user_factors = pickle.load(open(data_dir + 'data/wmf/user_wmf_50.pkl', 'rb'))
user_item_matrix = pickle.load(open(data_dir + 'data/wmf/user_item_matrix.pkl', 'rb'))

item_factors = torch.tensor(item_factors, device=device)
user_factors = torch.tensor(user_factors, device=device)

play_count_reconstruction =  user_factors @ torch.t(item_factors)
binary_reconstruction = torch.round(torch.clamp(play_count_reconstruction, 0, 1))
user_item_matrix.data = np.around(np.clip(user_item_matrix.data, 0, 1))

auc = roc_auc_score(user_item_matrix, binary_reconstruction.detach().numpy())
with open('outcome.txt', 'w') as f:
    f.write(f'Calculated using {device}\n'
            f'AUC: {reconstruction_auc:.2f}')

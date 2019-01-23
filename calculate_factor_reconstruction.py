import torch
import pickle
from sklearn.metrics import roc_auc_score
import numpy as np
import torch
from scipy.sparse import csr_matrix

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data_dir = '/var/scratch/akeijser/'
data_dir = ''

item_factors = pickle.load(open(data_dir + 'data/wmf/item_wmf_50.pkl', 'rb'))
user_factors = pickle.load(open(data_dir + 'data/wmf/user_wmf_50.pkl', 'rb'))
user_item_matrix = pickle.load(open(data_dir + 'data/wmf/user_item_matrix.pkl', 'rb'))


item_factors = torch.tensor(item_factors, device=device)
user_factors = torch.tensor(user_factors, device=device)

num_users = user_factors.size(0)

for i in range(num_users):
    play_count_reconstruction =  user_factors[i] @ torch.t(item_factors)
    binary_reconstruction = torch.clamp(play_count_reconstruction, 0, 1)
    print(binary_reconstruction[:20])
    # user_item_matrix[i].data = np.around(np.clip(user_item_matrix[i].data, 0, 1))
    print(user_item_matrix[i].toarray()[:20])
    targets = np.squeeze(np.clip(np.around(user_item_matrix[i].toarray()), 0, 1))
    print(targets)
    auc = roc_auc_score(targets, binary_reconstruction.detach().numpy())
    print(auc)
    break

    with open('outcome.txt', 'a') as f:
        f.write(f'Calculated using {device}\n'
                f'AUC: {reconstruction_auc:.2f}')

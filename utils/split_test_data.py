import os
import numpy as np
import glob

random_seed = 42
np.random.seed(random_seed)

test_split = 0.2
root_dir = '../data/spectrograms'
test_dir = '../data/test_spectrograms'
files = glob.glob(os.path.join(root_dir,'*npy'))
files = sorted(files)

total_files = len(files)
n_test = int(np.floor(test_split * total_files))
print('moving', n_test, 'files')
indices = list(range(total_files))
np.random.shuffle(indices)
test_indices = indices[:n_test]

if not os.path.exists(test_dir):
    os.mkdir(test_dir)

for idx in test_indices:
    file = files[idx]
    os.rename(file, os.path.join(test_dir, os.path.basename(file)))
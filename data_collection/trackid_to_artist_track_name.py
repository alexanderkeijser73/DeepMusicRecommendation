import os
from utils import hdf5_utils
from utils import hdf5_getters as GETTERS
import pickle
import glob
from tqdm import tqdm

h5path = '../data/song_metadata/msd_summary_file.h5'
test_dir = '../data/test_spectrograms'
train_dir = '../data/spectrograms'
file_type = '.npy'

track_id_to_info = {}

test_files = glob.glob(os.path.join(test_dir, '*' + file_type))
train_files = glob.glob(os.path.join(train_dir, '*' + file_type))


print('Indexing track ids from audio file names...')
for file_name in tqdm(sorted(train_files)):
    track_name = os.path.basename(file_name)
    track_id = os.path.splitext(track_name)[0]
    track_id_to_info[track_id] = None
for file_name in tqdm(sorted(test_files)):
    track_name = os.path.basename(file_name)
    track_id = os.path.splitext(track_name)[0]
    track_id_to_info[track_id] = None

print(len(track_id_to_info))

h5 = hdf5_utils.open_h5_file_read(h5path)
num_songs = GETTERS.get_num_songs(h5)

print('Retrieving meta data from hdf5 file...')

for i in tqdm(range(num_songs)):
    track_id = GETTERS.get_track_id(h5, songidx=i).decode('utf-8')

    if track_id in track_id_to_info:
        artist_name = GETTERS.get_artist_name(h5, songidx=i)
        track_name = GETTERS.get_title(h5, songidx=i)
        year = GETTERS.get_year(h5, songidx=i)
        tempo = GETTERS.get_tempo(h5, songidx=i)

        info_dict = {'artist_name': artist_name,
                    'track_name': track_name,
                    'year': year,
                    'tempo': tempo
                    }

        track_id_to_info[track_id] = info_dict

pickle.dump(track_id_to_info, open('../track_id_to_info.pkl', 'wb'))
h5.close()


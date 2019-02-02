import os
import hdf5_utils
import hdf5_descriptors
import hdf5_getters as GETTERS
import pickle
import glob
from tqdm import tqdm

h5path = '../data/song_metadata/msd_summary_file.h5'
audio_dir = '../data/audio'
file_type = '.mp3'

track_id_to_info = {}

files = glob.glob(os.path.join(audio_dir, '*' + file_type))

print('Indexing track ids from audio file names...')
for file_name in tqdm(sorted(files)):
    track_name = os.path.basename(file_name)
    track_id = os.path.splitext(track_name)[0]
    track_id_to_info[track_id] = None

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


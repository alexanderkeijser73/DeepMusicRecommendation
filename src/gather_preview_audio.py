import os
import fnmatch
from get_preview_url import get_trackid_from_text_search
import hdf5_utils
import hdf5_getters as GETTERS

for dirpath, dirs, files in os.walk('../data/MillionSongSubset/data'):
    for filename in fnmatch.filter(files, '*.h5'):
        h5path = os.path.join(dirpath, filename)
        # open h5 song, get all we know about the song
        h5 = hdf5_utils.open_h5_file_read(h5path)
        artist_name = GETTERS.get_artist_name(h5).decode('utf-8')
        track_name = GETTERS.get_title(h5).decode('utf-8')
        h5.close()

        print('Searching for track: ', artist_name, ' - ', track_name)
        # search by artist name + track title
        res = get_trackid_from_text_search(track_name, artistname=artist_name)
        if res is None:
            print('Did not find track using artist name and track title')
        else:
            name, preview_url = res
            print(res)
            # print(preview_url)

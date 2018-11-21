import os
import fnmatch
from get_preview_url import get_trackid_from_text_search
import hdf5_utils
import hdf5_getters as GETTERS

total_count = 0
unavailable_count = 0
unfound_count = 0


for dirpath, dirs, files in os.walk('../data/MillionSongSubset/data'):
    for i, filename in enumerate(fnmatch.filter(files, '*.h5')):
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
            unfound_count +=1
        else:
            name, preview_url = res
            if preview_url is None:
                unavailable_count += 1
            # print(preview_url)

        total_count +=1
        if i %100 == 0:
            print("Total tracks: {} Unfound tracks: {} Unavailable tracks {}".format(total_count, unfound_count,
                                                                                     unavailable_count))

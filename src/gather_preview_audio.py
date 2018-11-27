import os
import fnmatch
from get_preview_url import get_trackid_from_text_search, get_preview_from_trackid
import hdf5_utils
import hdf5_getters as GETTERS
import re
import wget

total_count = 0
unavailable_count = 0
unfound_count = 0

OUTDIR = '../../data/MillionSongSubset/audio'

if not os.path.isdir(OUTDIR):
    os.mkdir(OUTDIR)

for dirpath, dirs, files in os.walk('../../data/MillionSongSubset/data'):
    for i, filename in enumerate(fnmatch.filter(files, '*.h5')):
        h5path = os.path.join(dirpath, filename)
        # open h5 song, get all we know about the song
        h5 = hdf5_utils.open_h5_file_read(h5path)
        artist_name = GETTERS.get_artist_name(h5).decode('utf-8')
        track_name = GETTERS.get_title(h5).decode('utf-8')
        h5.close()

        artist_name_re = re.sub(' *[\[\(]*featuring*.*[\]\)]*', '', artist_name, flags=re.IGNORECASE)
        track_name_re = re.sub(' *[\[\(]+.*[\]\)]+', '', track_name)

        # print('Searching for track: ', artist_name, ' - ', track_name)
        # search by artist name + track title
        res = get_trackid_from_text_search(track_name_re, artistname=artist_name_re)
        if res is None:
            # print('Did not find track using artist name and track title')
                unfound_count +=1
                with open('../../data/MillionSongSubset/unfound_tracks.txt', 'a+') as f:
                    f.write('\t'.join([os.path.splitext(filename)[0], artist_name, track_name]) + '\n')
        else:
            name, preview_url, id = res
            if preview_url is None:
                preview_url = get_preview_from_trackid(id)
                if preview_url is None:
                    unavailable_count += 1
                    with open('../../data/MillionSongSubset/unavailable_tracks.txt', 'a+') as f:
                        f.write('\t'.join([os.path.splitext(filename)[0], artist_name, track_name]) + '\n')
            if preview_url is not None:
                out_path = os.path.join(OUTDIR, os.path.splitext(filename)[0]) + '.mp3'
                if not os.path.exists(out_path):
                    wget.download(preview_url, out=out_path)
            # print(preview_url)

        total_count +=1
        if i % 100 == 0:
            print("Total tracks: {} Unfound tracks: {} Unavailable tracks {}".format(total_count, unfound_count,
                                                                                     unavailable_count))

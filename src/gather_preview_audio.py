import os
from get_preview_url import get_trackid_from_text_search, get_preview_from_trackid
import hdf5_utils
import hdf5_getters as GETTERS
import re
import wget
import pickle


total_count = 0
unavailable_count = 0
unfound_count = 0

OUTDIR = '../../data/MillionSongSubset/audio'
wmf_item2i = pickle.load(open('../../index_dicts.pkl', 'rb'))['item2i']
track_to_song = pickle.load(open('../../track_to_song.pkl', 'rb'))

h5path = '../../data/msd_summary_file.h5'


if not os.path.isdir(OUTDIR):
    os.mkdir(OUTDIR)

h5 = hdf5_utils.open_h5_file_read(h5path)
num_songs = GETTERS.get_num_songs(h5)


for i in range(16200, num_songs):
    artist_name = GETTERS.get_artist_name(h5, songidx=i).decode('utf-8')
    track_name = GETTERS.get_title(h5, songidx=i).decode('utf-8')
    track_id = GETTERS.get_track_id(h5, songidx=i).decode('utf-8')

    out_path = os.path.join(OUTDIR, os.path.splitext(track_id)[0]) + '.mp3'
    if os.path.exists(out_path) or not track_to_song[track_id] in wmf_item2i.keys():
        continue

    track_name = re.sub('_', '', track_name)
    artist_name_re = re.sub(' *([;_/&,*]|(feat))+.*', '', artist_name)
    artist_name_re = re.sub(' *[\[\(]*feat*.*[\]\)]*', '', artist_name_re, flags=re.IGNORECASE)
    track_name_re = re.sub(' *[\[\(]+.*[\]\)]+', '', track_name)
    artist_name_re = re.sub(' *[\[\(]*featuring*.*[\]\)]*', '', artist_name_re, flags=re.IGNORECASE)
    track_name_re = re.sub(' *[\[\(]+.*[\]\)]+', '', track_name_re)

    # print('Searching for track: ', artist_name, ' - ', track_name)
    # search by artist name + track title
    res = get_trackid_from_text_search(track_name_re, artistname=artist_name_re)
    if res is None:
        # print('Did not find track using artist name and track title')
            unfound_count +=1
            with open('../../data/MillionSongSubset/unfound_tracks.txt', 'a+') as f:
                f.write('\t'.join([os.path.splitext(track_id)[0], artist_name, track_name]) + '\n')
    else:
        name, preview_url, id = res
        if preview_url is None:
            preview_url = get_preview_from_trackid(id)
            if preview_url is None:
                unavailable_count += 1
                with open('../../data/MillionSongSubset/unavailable_tracks.txt', 'a+') as f:
                    f.write('\t'.join([os.path.splitext(track_id)[0], artist_name, track_name]) + '\n')
        if preview_url is not None:
            out_path = os.path.join(OUTDIR, os.path.splitext(track_id)[0]) + '.mp3'
            if not os.path.exists(out_path):
                wget.download(preview_url, out=out_path)
        # print(preview_url)

    total_count +=1
    if i % 100 == 0:
        print("Total tracks: {} Unfound tracks: {} Unavailable tracks {}".format(i, unfound_count,
                                                                                 unavailable_count))
h5.close()

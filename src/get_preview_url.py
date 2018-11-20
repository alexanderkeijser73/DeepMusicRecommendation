"""
Thierry Bertin-Mahieux (2010) Columbia University
tb2332@columbia.edu


This code uses 7digital API and info contained in HDF5 song
file to get a preview URL.

This is part of the Million Song Dataset project from
LabROSA (Columbia University) and The Echo Nest.


Copyright 2010, Thierry Bertin-Mahieux

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import os
import sys
import urllib.request, urllib.error, urllib.parse
import json, pprint
try:
    from numpy import argmin
except ImportError:
    from scipy import argmin
except ImportError:
    print('no argmin function (no numpy or scipy), might cause problems')
from xml.dom import minidom

# Million Song Dataset imports, works under Linux
# otherwise, put the PythonSrc directory in the PYTHONPATH!
pythonsrc = os.path.abspath('__file__')
pythonsrc = os.path.join(pythonsrc,'../../../PythonSrc')
pythonsrc = os.path.abspath( pythonsrc )
sys.path.append( pythonsrc )
import hdf5_utils
import hdf5_getters as GETTERS

# try to get 7digital API key
DIGITAL7_API_KEY = "BQBCemMMflDmmI8_Z5XrpAoukXlsso5qZjVR8jvg_hcjBr47Pj2Eqo0oX-pi7ShzfEUsdLWuCLe8xpynIry4sQf3bmN1IbhyKPgiZFqaX82zFtmWC5T657wJTmsALBUZnL0C0ZVxVvk6YDs1I99dv_-94H0R-rhN-Qd8mJdJJIBHE61nuKAS-ZU3O8ZHvV_0g-NY8vRki-1A1SkX_sr_P60zZlvf5FeJtJTQYbGGah0b3AoSX_tNEN69s1SjylVoTHl_PH_y0cM1uSAaYNGJyS5l6Gc"


def url_call(url):
    """
    Do a simple request to the 7digital API
    We assume we don't do intense querying, this function is not
    robust
    Return the answer as na xml document
    """
    stream = urllib.request.urlopen(url)
    xmldoc = minidom.parse(stream).documentElement
    stream.close()
    return xmldoc


def levenshtein(s1, s2):
    """
    Levenstein distance, or edit distance, taken from Wikibooks:
    http://en.wikibooks.org/wiki/Algorithm_implementation/Strings/Levenshtein_distance#Python
    """
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if not s1:
        return len(s2)
 
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
 
    return previous_row[-1]


def get_closest_track(tracklist,target):
    """
    Find the closest track based on edit distance
    Might not be an exact match, you should check!
    """
    dists = [levenshtein(x,target) for x in tracklist]
    best = argmin(dists)
    return tracklist[best]


def get_trackid_from_text_search(title,artistname=''):
    """
    Search for an artist + title using 7digital search API
    Return None if there is a problem, or tuple (title,trackid)
    """
    url = 'https://api.spotify.com/v1/search?'
    """
    q=artist%3Ashakira%20track%3Awhenever%20wherever&type=track" -H "Accept: application/json" 
    -H "Content-Type: application/json" -H "Authorization: 
    Bearer BQC5vZuLBakaav1kVx8-RVnjvL2aKHYXEdNyUqJzaTIwhWDYs-z_C68qH0NGpuVyOKUiQqwosJPwbO789lqf3Pys01tPxH_Hr"
    """
    query = 'track: ' + str(title)
    if artistname != '':
        query = query + ' artist: ' + str(artistname)
    query = urllib.parse.quote(query)
    url += 'q='+query
    url += '&amp;type=track'
    req = urllib.request.Request(url)  # POST request doesn't not work
    req.add_header("Accept", "application/json")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", "Bearer " + DIGITAL7_API_KEY)
    stream = urllib.request.urlopen(req)
    string = stream.read().decode('utf-8')
    with open('./string.txt', 'w+') as f:
        f.write(string)
    json_obj = json.loads(string)
    print(json_obj)
    # tracks = json_obj['tracks']
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(tracks)

    # xmldoc = minidom.parse(stream).documentElement
    # stream.close()
    # print(xmldoc)
    # response = urllib.request.urlopen(req)
    #
    # the_page = response.read()
    # print(the_page[0])
    # # print(the_page)
    # # print(str(req))
    # xmldoc = url_call(req)
    # status = xmldoc.getAttribute('status')
    # if status != 'ok':
    #     return None
    # resultelem = xmldoc.getElementsByTagName('searchResult')
    # if len(resultelem) == 0:
    #     return None
    # track = resultelem[0].getElementsByTagName('track')[0]
    # tracktitle = track.getElementsByTagName('title')[0].firstChild.data
    # trackid = int(track.getAttribute('id'))
    return (tracktitle,trackid)

    
def get_tracks_from_artistid(artistid):
    """
    We get a list of release from artists.
    For each of these, get release.
    After calling the API with a given release ID, we receive a list of tracks.
    We return a map of <track name> -> <track id>
    or None if there is a problem
    """
    url = 'http://api.7digital.com/1.2/artist/releases?'
    url += '&artistid='+str(artistid)
    url += '&oauth_consumer_key='+DIGITAL7_API_KEY
    xmldoc = url_call(url)
    status = xmldoc.getAttribute('status')
    if status != 'ok':
        return None
    releaseselem = xmldoc.getElementsByTagName('releases')[0]
    releases = releaseselem.getElementsByTagName('release')
    if len(releases) == 0:
        return None
    releases_ids = [int(x.getAttribute('id')) for x in releases]
    res = {}
    for rid in releases_ids:
        tmpres = get_tracks_from_releaseid(rid)
        if tmpres is not None:
            res.update(tmpres)
    return res


def get_tracks_from_releaseid(releaseid):
    """
    After calling the API with a given release ID, we receive a list of tracks.
    We return a map of <track name> -> <track id>
    or None if there is a problem
    """
    url = 'http://api.7digital.com/1.2/release/tracks?'
    url += 'releaseid='+str(releaseid)
    url += '&oauth_consumer_key='+DIGITAL7_API_KEY
    xmldoc = url_call(url)
    status = xmldoc.getAttribute('status')
    if status != 'ok':
        return None
    tracks = xmldoc.getElementsByTagName('track')
    if len(tracks)==0:
        return None
    res = {}
    for t in tracks:
        tracktitle = t.getElementsByTagName('title')[0].firstChild.data
        trackid = int(t.getAttribute('id'))
        res[tracktitle] = trackid
    return res
    

def get_preview_from_trackid(trackid):
    """
    Ask for the preview to a particular track, get the XML answer
    After calling the API with a given track id,
    we get an XML response that looks like:
    
    <response status="ok" version="1.2" xsi:noNamespaceSchemaLocation="http://api.7digital.com/1.2/static/7digitalAPI.xsd">
      <url>
        http://previews.7digital.com/clips/34/6804688.clip.mp3
      </url>
    </response>

    We parse it for the URL that we return, or '' if a problem
    """
    url = 'http://api.7digital.com/1.2/track/preview?redirect=false'
    url += '&trackid='+str(trackid)
    url += '&oauth_consumer_key='+DIGITAL7_API_KEY
    xmldoc = url_call(url)
    status = xmldoc.getAttribute('status')
    if status != 'ok':
        return ''
    urlelem = xmldoc.getElementsByTagName('url')[0]
    preview = urlelem.firstChild.nodeValue
    return preview


def die_with_usage():
    """ HELP MENU """
    print( 'get_preview_url.py')
    print( '    by T. Bertin-Mahieux (2010) Columbia University')
    print( 'HELP MENU')
    print( 'usage:')
    print( '    python get_preview_url.py [FLAG] <SONGFILE>')
    print( 'PARAMS:')
    print( '  <SONGFILE>  - a Million Song Dataset file TRABC...123.h5')
    print( 'FLAGS:')
    print( '  -7digitalkey KEY - API key from 7 digital, we recomment you put it')
    print( '                     under environment variable: DIGITAL7_API_KEY')
    print( 'OUTPUT:')
    print( '  url from 7digital that should play a clip of the song.')
    print( '  No guarantee that this is the exact audio used for the analysis')
    sys.exit(0)


if __name__ == '__main__':

    # help menu
    if len(sys.argv) < 2:
        die_with_usage()


    # params
    h5path = sys.argv[1]

    # sanity checks
    if not os.path.isfile(h5path):
        print(( 'invalid path (not a file):',h5path))
        sys.exit(0)


    # open h5 song, get all we know about the song
    h5 = hdf5_utils.open_h5_file_read(h5path)
    track_7digitalid = GETTERS.get_track_7digitalid(h5)
    release_7digitalid = GETTERS.get_release_7digitalid(h5)
    artist_7digitalid = GETTERS.get_artist_7digitalid(h5)
    artist_name = GETTERS.get_artist_name(h5).decode('utf-8')
    release_name = GETTERS.get_release(h5)
    track_name = GETTERS.get_title(h5).decode('utf-8')
    h5.close()


    print('search track: ', track_name, artist_name)
    #search by artist name + track title
    res = get_trackid_from_text_search(track_name,artistname=artist_name)
    if res is None:
        print( 'something went wrong when doing text search with artist and track name, no more ideas')
        sys.exit(0)
    closest_track,trackid = res
    if closest_track != track_name:
        print(( 'we approximate your song title:',track_name,'by:',closest_track))
    preview = get_preview_from_trackid(trackid)
    if preview == '':
        print( 'something went wrong when looking by track id after text searching by artist and track name')
    else:
        print( preview)
        sys.exit(0)



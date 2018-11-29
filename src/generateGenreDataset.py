import sys
import numpy as np
from scipy.sparse import *
CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'
'''
# musiXmatch dataset - list of matches
#     matches provided by musiXmatch based on artist names
#     and song titles from the Million Song Dataset.
# MORE INFORMATION:
#     http://labrosa.ee.columbia.edu/millionsong/musixmatch
# FORMAT:
#     #    -> comment, ignore
#     tid|artist name|title|mxm tid|artist_name|title
#        tid          -> Million Song Dataset track ID
#        artist name  -> artist name in the MSD
#        title        -> title in the MSD
#        mxm tid      -> musiXmatch track ID
#        artist name  -> artist name for mXm
#        title        -> title for mXm
#        |            -> actual separator: <SEP>
# QUESTIONS / COMMENTS
#     tb2332@columbia.edu
# Enjoy!
'''
def generate_mapping(): # create mapping from mxm to msd datasets
    mxm_msd_map = {}
    with open("../data/lyric_data/mxm_to_msd_mapping.txt", 'r') as input:
        for i,line in enumerate(input):
            line = line.split("<SEP>")
            mxm_msd_map[line[0]] = line[3]
            print("Processing Mapping Data Item #{}".format(i+1))
            sys.stdout.write(CURSOR_UP_ONE)
            sys.stdout.write(ERASE_LINE)
    return mxm_msd_map        
    
def read_in_top_genres(): #load in the mapping for tracks belonging to ONLY the top 13 genres 
    trackID_Genre = {}
    with open("../data/lyric_data/msd-topMAGD-genreAssignment.txt", 'r') as input:
        for i,line in enumerate(input):
            line = line.strip('\n')
            temp = line.split('\t')
            trackID_Genre[temp[0]] = temp[1]
            print("Processing Genre Data Item #{}".format(i+1))
            sys.stdout.write(CURSOR_UP_ONE)
            sys.stdout.write(ERASE_LINE)
    return trackID_Genre
            
def read_in_all_genres(): #load in the mapping for tracks belonging to ALL 21 genres
    trackID_Genre = {}
    with open("../data/lyric_data/msd-MAGD-genreAssignment.txt", 'r') as input:
        for i,line in enumerate(input):
            line = line.strip('\n')
            temp = line.split('\t')
            trackID_Genre[temp[0]] = temp[1]
            print("Reading Genre Data Item #{}".format(i+1))
            sys.stdout.write(CURSOR_UP_ONE)
            sys.stdout.write(ERASE_LINE)
    return trackID_Genre
    
def load_lyric_data():
    trackID = []
    lyrics
    with open("../data/lyric_data/mxm.data", 'r') as input:
        for i,line in enumerate(input):
            line = line.strip('\n')
            temp = line.split(",")
            words = temp[2:]
            vector = [0]*5000#dok_matrix((1,5000),dtype=np.int8)
            for word in words:
                item = word.split(":")
                vector[int(item[0])-1] = int(item[1])
            trackID_Lyrics.append(temp[0],lil_matrix([vector]))
            print("Reading Lyric Data Item #{}".format(i+1))
            sys.stdout.write(CURSOR_UP_ONE)
            sys.stdout.write(ERASE_LINE) 
    return trackIDs,lyrics
    
if __name__ == '__main__':
    
    genremap = read_in_top_genres()
    data_unlabeled = load_lyric_data()
    
    
        
    
    
    
    
    
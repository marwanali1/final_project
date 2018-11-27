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

def generate_mapping():
    mxm_msd_map = {}
    with open("../data/lyric_data/mxm_to_msd_mapping.txt", 'r') as input:
        for i,line in enumerate(input):
            line = line.split("<SEP>")
            mxm_msd_map[line[0]] = line[3]
            print("Processing Item #{}".format(i+1))
            
    print("TRMMMKD128F425225D -> {}     TRYYYVU12903CD01E3 -> {}".format(mxm_msd_map['TRMMMKD128F425225D'], mxm_msd_map['TRYYYVU12903CD01E3']))
    print("TRMMMKD128F425225D -> 4418550     TRYYYVU12903CD01E3 -> 4116012")


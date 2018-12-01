import sys
import numpy as np
from scipy.sparse import *
import pickle
import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from scipy.sparse import *
from sklearn.svm import *

CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'
    
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
    
def load_train_data():
        ######## Flags ########
        reloadGenreMap = False
        reloadTrainData = False
        ######## Flags ########
        train_labels = []
        train_validTracks = []
        train_validTrackIDs = []
        
        if(reloadTrainData):
            print("Regenerating Pickled Training Data")
            trackIDs = []
            lyrics = []
            with open("../data/lyric_data/ancillary_data/train_raw.dat", 'r') as input:
                for i,line in enumerate(input):
                    line = line.strip('\n')
                    temp = line.split(",")
                    words = temp[2:]
                    vector = [0]*5000 #dok_matrix((1,5000),dtype=np.int8)
                    for word in words:
                        item = word.split(":")
                        vector[int(item[0])-1] = int(item[1])
                    lyrics.append(lil_matrix([vector]))
                    trackIDs.append(temp[0])
                    print("Reading Raw Train Data Item #{}".format(i+1))
                    sys.stdout.write(CURSOR_UP_ONE)
                    sys.stdout.write(ERASE_LINE) 
            if(reloadGenreMap):
                print("Regenerating Genre Map.")
                genremap = read_in_top_genres()
                pickle.dump(genremap,open("../data/lyric_data/ancillary_data/genremap.dat",'wb'))
            else:
                print("Pregenerated Genre Map Found.")
                genremap = pickle.load(open("../data/lyric_data/ancillary_data/genremap.dat",'rb'))
                print("Genre Map Loaded.")
            for i,track in enumerate(trackIDs):
                try:
                    print("Checking for track #{}.".format(i+1))
                    train_labels.append(genremap[track])
                    train_validTracks.append(lyrics[i])
                    train_validTrackIDs.append(track)
                    print("Track found in mapping.")
                except:
                    print("Track not found in mapping.")
            pickle.dump(train_labels,open("../data/lyric_data/train_labels.dat",'wb'))
            pickle.dump(train_validTracks,open("../data/lyric_data/train_tracks.dat",'wb'))
            pickle.dump(train_validTrackIDs,open("../data/lyric_data/ancillary_data/train_track_ids.dat",'wb'))
        else:
            print("Pickled Training Data File Found.")
            train_labels = pickle.load(open("../data/lyric_data/train_labels.dat",'rb'))
            train_validTracks = pickle.load(open("../data/lyric_data/train_tracks.dat",'rb'))
            train_validTrackIDs = pickle.load(open("../data/lyric_data/ancillary_data/train_track_ids.dat",'rb'))
            print("Training Data Loaded.")
        return train_labels,train_validTracks,train_validTrackIDs

def load_test_data():
    ######## Flags ########
    reloadGenreMap = False
    reloadtestData = False
    ######## Flags ########
    test_labels = []
    test_validTracks = []
    test_validTrackIDs = []
    
    if(reloadtestData):
        print("Regenerating Pickled testing Data")
        trackIDs = []
        lyrics = []
        with open("../data/lyric_data/ancillary_data/test_raw.dat", 'r') as input:
            for i,line in enumerate(input):
                line = line.strip('\n')
                temp = line.split(",")
                words = temp[2:]
                vector = [0]*5000 #dok_matrix((1,5000),dtype=np.int8)
                for word in words:
                    item = word.split(":")
                    vector[int(item[0])-1] = int(item[1])
                lyrics.append(lil_matrix([vector]))
                trackIDs.append(temp[0])
                print("Reading Raw test Data Item #{}".format(i+1))
                sys.stdout.write(CURSOR_UP_ONE)
                sys.stdout.write(ERASE_LINE) 
        if(reloadGenreMap):
            print("Regenerating Genre Map.")
            genremap = read_in_top_genres()
            pickle.dump(genremap,open("../data/lyric_data/ancillary_data/genremap.dat",'wb'))
        else:
            print("Pregenerated Genre Map Found.")
            genremap = pickle.load(open("../data/lyric_data/ancillary_data/genremap.dat",'rb'))
            print("Genre Map Loaded.")
        for i,track in enumerate(trackIDs):
            try:
                print("Checking for track #{}.".format(i+1))
                test_labels.append(genremap[track])
                test_validTracks.append(lyrics[i])
                test_validTrackIDs.append(track)
                print("Track found in mapping.")
            except:
                print("Track not found in mapping.")
        pickle.dump(test_labels,open("../data/lyric_data/test_labels.dat",'wb'))
        pickle.dump(test_validTracks,open("../data/lyric_data/test_tracks.dat",'wb'))
        pickle.dump(test_validTrackIDs,open("../data/lyric_data/ancillary_data/test_track_ids.dat",'wb'))
    else:
        print("Pickled testing Data File Found.")
        test_labels = pickle.load(open("../data/lyric_data/test_labels.dat",'rb'))
        test_validTracks = pickle.load(open("../data/lyric_data/test_tracks.dat",'rb'))
        test_validTrackIDs = pickle.load(open("../data/lyric_data/ancillary_data/test_track_ids.dat",'rb'))
        print("testing Data Loaded.")
    return test_labels,test_validTracks,test_validTrackIDs

if __name__ == '__main__':
    writeout_start = "===== " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " =====\n"
    print(writeout_start)
    
    train_labels,train_tracks,train_track_ids = load_train_data()
    test_labels,test_tracks,test_track_ids = load_test_data()

    knn = KNeighborsClassifier(n_neighbors = 3, algorithm = 'ball_tree',n_jobs = -1)
    linearSVC = LinearSVC(verbose = 20, random_state = 484)

    classifier = linearSVC
    classifier.fit(vstack(train_tracks), train_labels)
    predictions = classifier.predict(vstack(test_tracks))
    
    print(classification_report(test_labels,predictions, target_names = classifier.classes_))
    
    writeout_end = "===== " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " =====\n"
    print(writeout_end)
    
"""LinearSVC NO PARAMETER TUNING YET
               precision    recall  f1-score   support

        Blues       0.11      0.08      0.09        90
      Country       0.46      0.16      0.23       647
   Electronic       0.02      0.00      0.00       446
         Folk       0.02      0.01      0.01       152
International       0.10      0.06      0.07       248
         Jazz       0.03      0.02      0.03        85
        Latin       0.61      0.67      0.64       520
      New_Age       0.00      0.00      0.00        18
     Pop_Rock       0.82      0.93      0.87     10978
          Rap       0.70      0.54      0.61       785
       Reggae       0.12      0.14      0.13        65
          RnB       0.36      0.20      0.25       509
        Vocal       0.06      0.03      0.04       195

  avg / total       0.71      0.76      0.73     14738

===== 2018-12-01 02:45:59 ===== (~6 minutes runtime)
"""    
    
    
    
    